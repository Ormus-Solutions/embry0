"""SSO callback id_token validation (#506).

The callback is driven end to end: a stub token endpoint hands back an
RS256 id_token signed with a locally generated key, and a stub JWKS
endpoint serves the public half. Pins accept / reject / key-rotation so a
JOSE library change cannot break SSO login with the suite still green.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from joserfc import jwt
from joserfc.jwk import RSAKey

from embry0.api.session import STATE_COOKIE, encode_state
from tests.unit.api.test_sso_auth import SECRET, _app, _config

ISSUER = "https://tenant.example.com/"
CLIENT_ID = "client-id"
NONCE = "nonce-123"
TOKEN_URL = "https://tenant.example.com/oauth/token"
JWKS_URL = "https://tenant.example.com/.well-known/jwks.json"

KEY = RSAKey.generate_key(2048, parameters={"kid": "key-1"}, private=True)
ROTATED = RSAKey.generate_key(2048, parameters={"kid": "key-2"}, private=True)
OTHER = RSAKey.generate_key(2048, parameters={"kid": "key-1"}, private=True)  # same kid, wrong key


def _jwks(*keys: RSAKey) -> dict[str, Any]:
    return {"keys": [k.as_dict(private=False) for k in keys]}


def _id_token(key: RSAKey = KEY, **overrides: Any) -> str:
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "aud": CLIENT_ID,
        "sub": "auth0|u1",
        "email": "u@example.com",
        "name": "U",
        "nonce": NONCE,
        "iat": now,
        "exp": now + 600,
        "https://example.com/roles": ["embry0-admin"],
    }
    claims.update(overrides)
    return jwt.encode({"alg": "RS256", "kid": key.kid}, claims, key)


class _Idp:
    """The two IdP endpoints the callback calls, served from memory."""

    def __init__(self, id_token: str, jwks: dict[str, Any]) -> None:
        self.id_token = id_token
        self.jwks = jwks
        self.jwks_fetches = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            return httpx.Response(200, json={"id_token": self.id_token})
        if str(request.url) == JWKS_URL:
            self.jwks_fetches += 1
            return httpx.Response(200, json=self.jwks)
        return httpx.Response(404)


def _callback(
    monkeypatch,
    idp: _Idp,
    cached_jwks: dict[str, Any] | None = None,
    *,
    issuer: str | None = None,
    configured_issuer: str = ISSUER,
) -> httpx.Response:
    """Drive the callback; ``issuer`` is the discovery document's, absent unless given."""
    import embry0.api.v1.auth_session as auth_session

    real_client = httpx.AsyncClient

    def _client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return real_client(*args, transport=httpx.MockTransport(idp.handler), **kwargs)

    monkeypatch.setattr(auth_session.httpx, "AsyncClient", _client)
    app = _app(_config(auth0_issuer=configured_issuer))
    app.state.oidc_metadata = {"token_endpoint": TOKEN_URL, "jwks_uri": JWKS_URL}
    if issuer is not None:
        app.state.oidc_metadata["issuer"] = issuer
    if cached_jwks is not None:
        app.state.oidc_jwks = cached_jwks
    client = TestClient(app)
    client.cookies.set(STATE_COOKIE, encode_state(SECRET, {"state": "st", "nonce": NONCE, "verifier": "v"}))
    return client.get("/api/v1/auth/callback?code=c&state=st", follow_redirects=False)


def test_a_valid_rs256_id_token_logs_in(monkeypatch):
    resp = _callback(monkeypatch, _Idp(_id_token(), _jwks(KEY)))
    assert resp.status_code == 303, f"valid id_token refused: {resp.status_code} {resp.text}"
    assert resp.headers["location"] == "/"
    assert "embry0_session=" in resp.headers.get("set-cookie", "")


@pytest.mark.parametrize(
    "token",
    [
        pytest.param(lambda: _id_token(aud="someone-else"), id="wrong-aud"),
        pytest.param(lambda: _id_token(iss="https://evil.example.com/"), id="wrong-iss"),
        pytest.param(lambda: _id_token(exp=int(time.time()) - 60, iat=int(time.time()) - 660), id="expired"),
        pytest.param(lambda: _id_token(key=OTHER), id="bad-signature"),
        pytest.param(
            lambda: _id_token().rsplit(".", 1)[0] + "." + _id_token(sub="auth0|other").rsplit(".", 1)[1],
            id="tampered-signature",
        ),
    ],
)
def test_an_invalid_id_token_is_refused(monkeypatch, token):
    resp = _callback(monkeypatch, _Idp(token(), _jwks(KEY)))
    assert resp.status_code == 400, f"invalid id_token accepted: {resp.status_code}"
    assert resp.json()["detail"] == "ID token validation failed"
    assert "embry0_session=" not in resp.headers.get("set-cookie", "")


def test_a_rotated_key_is_picked_up_by_one_jwks_refresh(monkeypatch):
    """The cached JWKS lacks the new kid; the refreshed one has it."""
    idp = _Idp(_id_token(key=ROTATED), _jwks(KEY, ROTATED))
    resp = _callback(monkeypatch, idp, cached_jwks=_jwks(KEY))
    assert resp.status_code == 303, f"rotated key refused: {resp.status_code} {resp.text}"
    assert idp.jwks_fetches == 1


def test_a_key_missing_after_refresh_is_refused(monkeypatch):
    idp = _Idp(_id_token(key=ROTATED), _jwks(KEY))
    resp = _callback(monkeypatch, idp, cached_jwks=_jwks(KEY))
    assert resp.status_code == 400
    assert idp.jwks_fetches == 1


def test_a_nonce_mismatch_is_refused(monkeypatch):
    resp = _callback(monkeypatch, _Idp(_id_token(nonce="replayed"), _jwks(KEY)))
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Nonce mismatch"


def test_a_cached_jwks_that_has_the_key_is_not_refetched(monkeypatch):
    idp = _Idp(_id_token(), _jwks(KEY))
    resp = _callback(monkeypatch, idp, cached_jwks=_jwks(KEY))
    assert resp.status_code == 303
    assert idp.jwks_fetches == 0, "the JWKS was fetched although the cached set had the key"


# An OIDC provider names its issuer in its discovery document, and an ID token's
# ``iss`` must equal it exactly (OIDC Discovery 1.0, section 4.3). Auth0 writes it
# with a trailing slash; Keycloak (``https://<host>/realms/<realm>``) and most others
# without one.
SLASHLESS = "https://idp.example.com/realms/team"


def test_an_issuer_without_a_trailing_slash_logs_in(monkeypatch):
    resp = _callback(
        monkeypatch,
        _Idp(_id_token(iss=SLASHLESS), _jwks(KEY)),
        issuer=SLASHLESS,
        configured_issuer=SLASHLESS,
    )
    assert resp.status_code == 303, f"an ID token from a slashless issuer was refused: {resp.status_code} {resp.text}"


def test_a_slashless_issuer_still_refuses_a_token_with_the_slash(monkeypatch):
    resp = _callback(
        monkeypatch,
        _Idp(_id_token(iss=SLASHLESS + "/"), _jwks(KEY)),
        issuer=SLASHLESS,
        configured_issuer=SLASHLESS,
    )
    assert resp.status_code == 400, f"iss {SLASHLESS}/ was accepted for issuer {SLASHLESS}: {resp.status_code}"


def test_an_issuer_with_a_trailing_slash_still_logs_in(monkeypatch):
    """The Auth0 form: the discovery document and the token both end in a slash."""
    resp = _callback(monkeypatch, _Idp(_id_token(), _jwks(KEY)), issuer=ISSUER)
    assert resp.status_code == 303, f"an Auth0-form issuer was refused: {resp.status_code} {resp.text}"


def test_a_discovery_document_naming_another_issuer_is_refused(monkeypatch):
    idp = _Idp(_id_token(iss="https://other.example.com/"), _jwks(KEY))
    resp = _callback(monkeypatch, idp, issuer="https://other.example.com/")
    assert resp.status_code == 502, f"a discovery document for another issuer was trusted: {resp.status_code}"
    assert "embry0_session=" not in resp.headers.get("set-cookie", "")
