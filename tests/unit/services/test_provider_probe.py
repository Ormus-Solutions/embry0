"""RAV-1297 — live provider probe unit tests (httpx faked at the module seam)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from embry0.services import provider_probe as probe


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _FakeClient:
    def __init__(self, status: int | None = None, exc: Exception | None = None) -> None:
        self._status = status
        self._exc = exc
        self.last_headers: dict[str, str] | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url, headers=None):
        self.last_headers = headers
        if self._exc is not None:
            raise self._exc
        return _FakeResponse(self._status or 200)


def _patch_httpx(monkeypatch, *, status: int | None = None, exc: Exception | None = None) -> _FakeClient:
    client = _FakeClient(status=status, exc=exc)
    fake = SimpleNamespace(AsyncClient=lambda **_kw: client, HTTPError=httpx.HTTPError)
    monkeypatch.setattr(probe, "httpx", fake)
    return client


@pytest.mark.asyncio
async def test_anthropic_not_configured(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    checks = await probe.probe_anthropic()
    assert [c.status for c in checks] == [probe.STATUS_NOT_CONFIGURED]


@pytest.mark.asyncio
async def test_anthropic_ok_via_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-not-real")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    client = _patch_httpx(monkeypatch, status=200)
    checks = await probe.probe_anthropic()
    assert checks[0].source == "api_key"
    assert checks[0].status == probe.STATUS_OK
    assert client.last_headers["x-api-key"] == "key-not-real"


@pytest.mark.asyncio
async def test_anthropic_oauth_uses_bearer_and_beta_header(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oat-not-real")
    client = _patch_httpx(monkeypatch, status=401)
    checks = await probe.probe_anthropic()
    assert checks[0].source == "oauth"
    assert checks[0].status == probe.STATUS_UNAUTHORIZED
    assert client.last_headers["Authorization"] == "Bearer oat-not-real"
    assert client.last_headers["anthropic-beta"] == "oauth-2025-04-20"


@pytest.mark.asyncio
async def test_unreachable_maps_transport_errors(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-not-real")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    _patch_httpx(monkeypatch, exc=httpx.ConnectError("boom"))
    checks = await probe.probe_anthropic()
    assert checks[0].status == probe.STATUS_UNREACHABLE
    # No credential text in the message.
    assert "key-not-real" not in checks[0].message


@pytest.mark.asyncio
async def test_xai_403_carries_entitlement_hint(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-not-real")
    _patch_httpx(monkeypatch, status=403)
    checks = await probe.probe_xai()
    assert checks[0].status == probe.STATUS_ENTITLEMENT_REFUSED
    # RAV-1284 decoded semantics: X Premium+ is not API access.
    assert "SuperGrok" in checks[0].message


@pytest.mark.asyncio
async def test_xai_oauth_proxy_path_uses_refresher_token(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    client = _patch_httpx(monkeypatch, status=200)
    refresher = MagicMock()
    refresher.ensure_fresh = AsyncMock(return_value="access-not-real")
    checks = await probe.probe_xai(refresher)
    assert checks[0].source == "oauth_proxy"
    assert checks[0].status == probe.STATUS_OK
    assert client.last_headers["Authorization"] == "Bearer access-not-real"


@pytest.mark.asyncio
async def test_xai_refresh_failure_reports_unauthorized(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    refresher = MagicMock()
    refresher.ensure_fresh = AsyncMock(side_effect=RuntimeError("token endpoint said no"))
    checks = await probe.probe_xai(refresher)
    assert checks[0].status == probe.STATUS_UNAUTHORIZED
    assert "SuperGrok refresh failed" in checks[0].message


@pytest.mark.asyncio
async def test_xai_not_configured(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    checks = await probe.probe_xai(None)
    assert checks[0].status == probe.STATUS_NOT_CONFIGURED


@pytest.mark.asyncio
async def test_probe_llm_secret_anthropic_api_key_sends_x_api_key(monkeypatch):
    client = _patch_httpx(monkeypatch, status=200)
    check = await probe.probe_llm_secret("anthropic", "llm_api_key", "sk-ant-secret")
    assert check.status == probe.STATUS_OK
    assert client.last_headers["x-api-key"] == "sk-ant-secret"
    assert client.last_headers["anthropic-version"] == "2023-06-01"


@pytest.mark.asyncio
async def test_probe_llm_secret_anthropic_oauth_sends_bearer_and_beta(monkeypatch):
    client = _patch_httpx(monkeypatch, status=200)
    check = await probe.probe_llm_secret("anthropic", "llm_oauth", "oat-secret")
    assert check.source == "oauth" and check.status == probe.STATUS_OK
    assert client.last_headers["Authorization"] == "Bearer oat-secret"
    assert client.last_headers["anthropic-beta"] == "oauth-2025-04-20"


@pytest.mark.asyncio
async def test_probe_llm_secret_xai_api_key_sends_bearer(monkeypatch):
    client = _patch_httpx(monkeypatch, status=401)
    check = await probe.probe_llm_secret("xai", "llm_api_key", "xai-secret")
    assert check.status == probe.STATUS_UNAUTHORIZED
    assert client.last_headers["Authorization"] == "Bearer xai-secret"
    assert "xai-secret" not in check.message


@pytest.mark.asyncio
async def test_probe_llm_secret_xai_oauth_sends_the_access_token(monkeypatch):
    """The caller refreshed first. This probe sends that access token, not the refresh payload."""
    client = _patch_httpx(monkeypatch, status=200)
    check = await probe.probe_llm_secret("xai", "llm_oauth", "access-token-value")
    assert check.source == "oauth" and check.status == probe.STATUS_OK
    assert client.last_headers["Authorization"] == "Bearer access-token-value"
    assert "access-token-value" not in check.message


def test_overall_status_ok_wins():
    checks = [
        probe.ProbeCheck("api_key", probe.STATUS_UNAUTHORIZED, "x"),
        probe.ProbeCheck("oauth", probe.STATUS_OK, "y"),
    ]
    assert probe.overall_status(checks) == probe.STATUS_OK
    assert probe.overall_status([checks[0]]) == probe.STATUS_UNAUTHORIZED
    assert probe.overall_status([]) == probe.STATUS_NOT_CONFIGURED


# --- #7: the Anthropic probe must test the account a job would get ----------
# Before this, probe_anthropic() read os.environ only, so a deployment whose
# only Anthropic account lives in the credential registry was told
# "not_configured" while GET /config/models called the same provider available
# and jobs ran on that very account.


def _resolved(kind: str, secret: str, source: str = "registry", label: str = "prod-claude"):
    from embry0.services.credentials.resolver import ResolvedCredential

    return ResolvedCredential(
        id="cred-1" if source == "registry" else None,
        kind=kind,
        provider="anthropic",
        host="api.anthropic.com",
        label=label,
        secret=secret,
        git_author_name=None,
        git_author_email=None,
        source=source,
        trace=("global: -> prod-claude",),
    )


def _resolver(result=None, error: Exception | None = None):
    """A stand-in CredentialResolver: resolve_llm returns ``result`` or raises ``error``."""
    resolver = SimpleNamespace(calls=[])

    async def resolve_llm(repo, provider, **kwargs):
        resolver.calls.append((repo, provider, kwargs))
        if error is not None:
            raise error
        return result

    resolver.resolve_llm = resolve_llm
    return resolver


def _no_anthropic_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)


@pytest.mark.asyncio
async def test_anthropic_registry_only_account_is_probed(monkeypatch):
    """A registry-only account must be probed, not reported not_configured (#7)."""
    _no_anthropic_env(monkeypatch)
    client = _patch_httpx(monkeypatch, status=200)
    resolver = _resolver(_resolved("llm_api_key", "sk-registry-not-real"))

    checks = await probe.probe_anthropic(resolver)

    assert [c.status for c in checks] != [probe.STATUS_NOT_CONFIGURED], (
        f"a registry-bound Anthropic account must be probed, got {checks}"
    )
    assert [c.status for c in checks] == [probe.STATUS_OK]
    assert checks[0].source == "registry", f"the result must name the source it tested, got {checks[0].source!r}"
    assert client.last_headers["x-api-key"] == "sk-registry-not-real"
    # The resolver is asked the question a job with no repository binding asks,
    # and records no use (no job_id).
    assert resolver.calls == [("", "anthropic", {})]


@pytest.mark.asyncio
async def test_anthropic_registry_oauth_account_uses_bearer_and_beta_header(monkeypatch):
    """A registry llm_oauth account is probed by its own kind, not as an API key."""
    _no_anthropic_env(monkeypatch)
    client = _patch_httpx(monkeypatch, status=200)

    checks = await probe.probe_anthropic(_resolver(_resolved("llm_oauth", "oat-registry-not-real")))

    assert checks[0].source == "registry" and checks[0].status == probe.STATUS_OK
    assert client.last_headers["Authorization"] == "Bearer oat-registry-not-real"
    assert client.last_headers["anthropic-beta"] == "oauth-2025-04-20"


@pytest.mark.asyncio
async def test_anthropic_registry_secret_never_appears_in_the_result(monkeypatch):
    """No raw secret in status, message or source — the serialized body is what the dashboard shows."""
    _no_anthropic_env(monkeypatch)
    _patch_httpx(monkeypatch, status=401)
    secret = "sk-registry-super-secret-not-real"

    checks = await probe.probe_anthropic(_resolver(_resolved("llm_api_key", secret)))

    assert secret not in repr(probe.serialize_checks(checks))


@pytest.mark.asyncio
async def test_anthropic_resolver_env_answer_keeps_both_environment_checks(monkeypatch):
    """When the resolver falls through to the environment, each env source is still probed on its own."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-not-real")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oat-not-real")
    _patch_httpx(monkeypatch, status=200)

    checks = await probe.probe_anthropic(
        _resolver(_resolved("llm_oauth", "oat-not-real", source="env", label="CLAUDE_CODE_OAUTH_TOKEN"))
    )

    assert [c.source for c in checks] == ["api_key", "oauth"]


@pytest.mark.asyncio
async def test_anthropic_resolver_env_step_covers_the_claude_credentials_file(monkeypatch):
    """The resolver's env step reads one source the two variables do not carry: the credentials file."""
    _no_anthropic_env(monkeypatch)
    client = _patch_httpx(monkeypatch, status=200)

    checks = await probe.probe_anthropic(
        _resolver(_resolved("llm_oauth", "file-token-not-real", source="env", label="claude-credentials-file"))
    )

    assert [c.status for c in checks] == [probe.STATUS_OK]
    assert checks[0].source == "oauth"
    assert client.last_headers["Authorization"] == "Bearer file-token-not-real"


@pytest.mark.asyncio
async def test_anthropic_not_configured_when_resolver_has_none_either(monkeypatch):
    from embry0.services.credentials.resolver import CredentialMissingError

    _no_anthropic_env(monkeypatch)
    checks = await probe.probe_anthropic(_resolver(error=CredentialMissingError("no anthropic credential for ''")))
    assert [c.status for c in checks] == [probe.STATUS_NOT_CONFIGURED]
    assert checks[0].source == "none"


@pytest.mark.asyncio
async def test_anthropic_unusable_registry_account_reports_the_registry_error(monkeypatch):
    """A bound account the registry cannot serve is the operator's answer, not a silent env fallback."""
    from embry0.services.credentials.resolver import CredentialUnavailableError

    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-not-real")
    _patch_httpx(monkeypatch, status=200)

    checks = await probe.probe_anthropic(_resolver(error=CredentialUnavailableError("'prod-claude' is quarantined")))

    assert [c.source for c in checks] == ["registry"]
    assert checks[0].status == probe.STATUS_ERROR
    assert "quarantined" in checks[0].message


@pytest.mark.asyncio
async def test_anthropic_registry_outage_falls_back_to_the_environment(monkeypatch):
    """A broken registry read must not hide a working environment account."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-not-real")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    _patch_httpx(monkeypatch, status=200)

    checks = await probe.probe_anthropic(_resolver(error=RuntimeError("connection reset")))

    assert [c.source for c in checks] == ["api_key"]
    assert checks[0].status == probe.STATUS_OK
