"""Live LLM-provider connection probes (RAV-1297).

An explicit, operator-triggered check per catalog provider that reports
reachable / unauthorized / entitlement-refused, reusing the RAV-1284
decoded-403 semantics for xAI (an ambiguous 403 usually means the account
lacks a standalone SuperGrok subscription, not a bad token). Results never
echo credentials — only status words and provider-side error text.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import httpx
import structlog

from embry0.agents.providers import XAI, ModelProvider, provider_named
from embry0.execution.xai_credential import _XAI_403_HINT

logger = structlog.get_logger(__name__)

PROBE_TIMEOUT_SECONDS = 10.0

# Cheap authenticated read endpoints — prove reachability + auth without
# spending tokens.
_ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
_XAI_MODELS_URL = f"{XAI.openai_base_url}{XAI.probe_path}"

STATUS_OK = "ok"
STATUS_UNAUTHORIZED = "unauthorized"
STATUS_ENTITLEMENT_REFUSED = "entitlement_refused"
STATUS_UNREACHABLE = "unreachable"
STATUS_NOT_CONFIGURED = "not_configured"
STATUS_ERROR = "error"


@dataclass(frozen=True, slots=True)
class ProbeCheck:
    """One credential source's live result."""

    source: str
    status: str
    message: str


async def _http_check(source: str, url: str, headers: dict[str, str], hint_403: str = "") -> ProbeCheck:
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS) as client:
            resp = await client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        return ProbeCheck(source, STATUS_UNREACHABLE, f"request failed: {type(exc).__name__}")
    if resp.status_code == 200:
        return ProbeCheck(source, STATUS_OK, "authenticated and reachable")
    if resp.status_code == 401:
        return ProbeCheck(source, STATUS_UNAUTHORIZED, "credential rejected (HTTP 401)")
    if resp.status_code == 403:
        return ProbeCheck(source, STATUS_ENTITLEMENT_REFUSED, f"HTTP 403.{hint_403}")
    return ProbeCheck(source, STATUS_ERROR, f"unexpected HTTP {resp.status_code}")


def probe_url(provider: ModelProvider, base_url: str = "") -> str:
    """The OpenAI-compatible family's readiness probe: ``GET <root><probe_path>``.

    ``base_url`` is the operator's root for a provider with none of its own (custom).
    """
    root = (base_url or provider.openai_base_url).rstrip("/")
    return f"{root}{provider.probe_path}" if root else ""


async def probe_openai_compatible(source: str, provider: ModelProvider, secret: str, base_url: str = "") -> ProbeCheck:
    """Probe one key of an OpenAI-compatible provider: its models list (OpenRouter: its key info)."""
    url = probe_url(provider, base_url)
    if not url:
        return ProbeCheck(source, STATUS_NOT_CONFIGURED, f"no base URL is configured for {provider.name}")
    hint = _XAI_403_HINT if provider.name == "xai" else ""
    return await _http_check(source, url, {"Authorization": f"Bearer {secret}"}, hint_403=hint)


async def probe_llm_secret(provider: str, kind: str, secret: str, *, base_url: str = "") -> ProbeCheck:
    """Probe one secret. For xAI ``llm_oauth``, ``secret`` is the minted access token.

    ``base_url`` is a custom credential's own API root (its ``metadata.base_url``).
    """
    if provider == "anthropic" and kind == "llm_api_key":
        return await _http_check(
            "api_key",
            _ANTHROPIC_MODELS_URL,
            {"x-api-key": secret, "anthropic-version": "2023-06-01"},
        )
    if provider == "anthropic" and kind == "llm_oauth":
        return await _http_check(
            "oauth",
            _ANTHROPIC_MODELS_URL,
            {
                "Authorization": f"Bearer {secret}",
                "anthropic-beta": "oauth-2025-04-20",
                "anthropic-version": "2023-06-01",
            },
        )
    if provider == "xai" and kind == "llm_api_key":
        return await probe_openai_compatible("api_key", XAI, secret)
    if provider == "xai" and kind == "llm_oauth":
        return await _http_check(
            "oauth",
            _XAI_MODELS_URL,
            {"Authorization": f"Bearer {secret}"},
            hint_403=_XAI_403_HINT,
        )
    catalog = provider_named(provider)
    if catalog is not None and catalog.runtime == "chat_completions" and kind == "llm_api_key":
        return await probe_openai_compatible("api_key", catalog, secret, base_url)
    raise ValueError(f"cannot probe {provider} {kind}")


async def _probe_anthropic_env() -> list[ProbeCheck]:
    """Each Anthropic credential the process environment carries, probed independently."""
    checks: list[ProbeCheck] = []
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
    if api_key:
        checks.append(await probe_llm_secret("anthropic", "llm_api_key", api_key))
    if oauth_token:
        # Claude Code OAuth tokens authenticate as a bearer with the OAuth
        # beta header — a header change, not a key swap.
        checks.append(await probe_llm_secret("anthropic", "llm_oauth", oauth_token))
    return checks


async def _probe_resolved_anthropic(cred: Any) -> ProbeCheck:
    """Probe the account the resolver chose, labelled by where the resolver found it."""
    source = "registry" if cred.source == "registry" else ("oauth" if cred.kind == "llm_oauth" else "api_key")
    if cred.kind not in ("llm_api_key", "llm_oauth"):
        return ProbeCheck(source, STATUS_ERROR, f"cannot probe an anthropic {cred.kind} credential")
    check = await probe_llm_secret("anthropic", cred.kind, cred.secret)
    return ProbeCheck(source, check.status, check.message)


async def probe_anthropic(resolver: Any = None) -> list[ProbeCheck]:
    """Probe the Anthropic account a job with no repository binding would run with.

    With a resolver the registry answers first, in the order a job resolves
    (the global pool, then the environment), so the test covers the account
    that actually runs: before #7 this read ``os.environ`` alone, so a
    deployment whose only account is in the registry was told
    ``not_configured`` while /config/models called the same provider available
    and jobs ran on that very account. The result names the source it tested
    and never carries the secret. No use is recorded (no job id).

    A registry account is probed by its own kind. When the resolver's answer
    comes from the environment, each environment source is probed on its own
    as before — except where the two variables carry none, which is the
    resolver's remaining env source, the orchestrator's Claude credentials
    file. Without a resolver the environment alone is probed.
    """
    from embry0.services.credentials.resolver import CredentialMissingError, CredentialUnavailableError

    resolved: Any = None
    if resolver is not None:
        try:
            resolved = await resolver.resolve_llm("", "anthropic")
        except CredentialMissingError:
            resolved = None
        except CredentialUnavailableError as exc:
            # A bound account the registry cannot serve (quarantined, disabled,
            # undecryptable) is the operator's answer, not a silent env fallback.
            # The message names labels and reasons, never a secret.
            return [ProbeCheck("registry", STATUS_ERROR, str(exc))]
        except Exception as exc:  # noqa: BLE001 — a registry outage must not hide a working env account
            logger.warning("provider_probe_resolver_error", provider="anthropic", error=str(exc)[:200])
            resolved = None
        if resolved is not None and resolved.source == "registry":
            return [await _probe_resolved_anthropic(resolved)]
    checks = await _probe_anthropic_env()
    if not checks and resolved is not None:
        checks.append(await _probe_resolved_anthropic(resolved))
    if not checks:
        checks.append(
            ProbeCheck(
                "none",
                STATUS_NOT_CONFIGURED,
                "No Anthropic account: bind one in the credential registry, "
                "or set CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY.",
            )
        )
    return checks


async def probe_xai(refresher: Any = None) -> list[ProbeCheck]:
    """Probe xAI over whichever path is configured.

    Prefers the console API key; falls back to the SuperGrok OAuth
    refresher (the EMB-46 default runtime path). A 403 carries the
    RAV-1284 hint — X Premium+ is NOT API access.
    """
    api_key = os.environ.get("XAI_API_KEY", "")
    if api_key:
        return [await probe_llm_secret("xai", "llm_api_key", api_key)]
    if refresher is not None:
        try:
            access_token = await refresher.ensure_fresh()
        except Exception as exc:
            # XaiCredentialError text names the token endpoint, never the token.
            return [ProbeCheck("oauth_proxy", STATUS_UNAUTHORIZED, f"SuperGrok refresh failed: {exc}")]
        return [
            await _http_check(
                "oauth_proxy",
                _XAI_MODELS_URL,
                {"Authorization": f"Bearer {access_token}"},
                hint_403=_XAI_403_HINT,
            )
        ]
    return [
        ProbeCheck(
            "none", STATUS_NOT_CONFIGURED, "Neither XAI_API_KEY nor the SuperGrok OAuth credential is configured."
        )
    ]


async def probe_chat_completions_provider(provider: ModelProvider, resolver: Any = None) -> list[ProbeCheck]:
    """Probe the key a job with no repository binding runs ``provider`` with: the global pool, then the env.

    The resolver's own answer, so the test covers what a job would get; no use is recorded.
    Without a resolver the env key alone is probed.
    """
    from embry0.services.credentials.resolver import CredentialMissingError, CredentialUnavailableError

    if resolver is not None:
        try:
            cred = await resolver.resolve_llm("", provider.name)
        except CredentialMissingError:
            return [_not_configured(provider)]
        except CredentialUnavailableError as exc:
            return [ProbeCheck("registry", STATUS_ERROR, str(exc))]
        source = "registry" if cred.source == "registry" else "api_key"
        return [await probe_openai_compatible(source, provider, cred.secret, str(cred.metadata.get("base_url") or ""))]
    secret = os.environ.get(provider.api_key_env, "").strip()
    base_url = os.environ.get(provider.base_url_env, "").strip() if provider.base_url_env else ""
    if not secret or (provider.base_url_env and not base_url):
        return [_not_configured(provider)]
    return [await probe_openai_compatible("api_key", provider, secret, base_url)]


def _not_configured(provider: ModelProvider) -> ProbeCheck:
    needs = provider.api_key_env + (f" with {provider.base_url_env}" if provider.base_url_env else "")
    return ProbeCheck(
        "none", STATUS_NOT_CONFIGURED, f"No {provider.name} credential: bind one in the registry or set {needs}."
    )


def overall_status(checks: list[ProbeCheck]) -> str:
    """One runnable path is enough — ok wins; else surface the first failure."""
    if any(c.status == STATUS_OK for c in checks):
        return STATUS_OK
    return checks[0].status if checks else STATUS_NOT_CONFIGURED


def serialize_checks(checks: list[ProbeCheck]) -> list[dict[str, str]]:
    return [asdict(c) for c in checks]
