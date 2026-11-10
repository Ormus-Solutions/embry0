"""`embry0 health` reaches the stack wherever PROD_PORT publishes it (#1).

PROD_PORT is the compose file's published-ports value
(`"${PROD_PORT:-8200}:80"`), so it may carry a bind address: `127.0.0.1:8200`
publishes the dashboard on loopback only, which is how a stack reached through
a tunnel is run. Read as a bare port it builds
`http://localhost:127.0.0.1:8200/`, which answers nothing. `infra/deploy.sh`
splits the address off the same way; the two must agree.
"""

from __future__ import annotations

import argparse

import httpx
import pytest

from embry0.cli import __main__ as cli

# (PROD_PORT, the base URL the CLI must dial)
_PUBLISHED = [
    ("8200", "http://localhost:8200"),
    ("8300", "http://localhost:8300"),
    ("127.0.0.1:8200", "http://127.0.0.1:8200"),
    ("192.168.1.10:8200", "http://192.168.1.10:8200"),
    ("[::1]:8200", "http://[::1]:8200"),
    # A wildcard bind is not an address to dial: reach it on loopback.
    ("0.0.0.0:8200", "http://localhost:8200"),
    ("[::]:8200", "http://localhost:8200"),
]


@pytest.mark.parametrize(("prod_port", "base"), _PUBLISHED)
def test_prod_base_url_splits_an_optional_bind_address(prod_port: str, base: str) -> None:
    assert cli.prod_base_url(prod_port) == base, f"PROD_PORT={prod_port}"


@pytest.mark.parametrize("unset", [None, "", "   "])
def test_prod_base_url_falls_back_to_the_compose_default(unset: str | None) -> None:
    assert cli.prod_base_url(unset) == "http://localhost:8200"


@pytest.mark.parametrize(("prod_port", "base"), _PUBLISHED)
def test_health_asks_the_address_prod_port_publishes(monkeypatch, capsys, prod_port: str, base: str) -> None:
    """The ready check must land on PROD_PORT's address, not on localhost:<the whole value>."""
    asked: list[str] = []

    def fake_get(url: str, **kwargs: object) -> httpx.Response:
        asked.append(url)
        return httpx.Response(200, json={"status": "ok", "checks": {"database": "ok"}})

    monkeypatch.setenv("PROD_PORT", prod_port)
    monkeypatch.delenv("EMBRY0_URL", raising=False)
    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(SystemExit) as exit_info:
        cli.cmd_health(argparse.Namespace(url=None))

    assert exit_info.value.code == 0, capsys.readouterr().err
    assert asked == [f"{base}/api/v1/health/ready"], f"PROD_PORT={prod_port}: asked {asked}"


def test_an_explicit_url_still_wins_over_prod_port(monkeypatch) -> None:
    asked: list[str] = []
    monkeypatch.setenv("PROD_PORT", "127.0.0.1:8200")
    monkeypatch.delenv("EMBRY0_URL", raising=False)
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kwargs: (asked.append(url), httpx.Response(200, json={"status": "ok"}))[1],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.cmd_health(argparse.Namespace(url="http://other.example:9000/"))

    assert exit_info.value.code == 0
    assert asked == ["http://other.example:9000/api/v1/health/ready"], asked


def test_embry0_url_still_wins_over_prod_port(monkeypatch) -> None:
    asked: list[str] = []
    monkeypatch.setenv("PROD_PORT", "127.0.0.1:8200")
    monkeypatch.setenv("EMBRY0_URL", "http://other.example:9000")
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kwargs: (asked.append(url), httpx.Response(200, json={"status": "ok"}))[1],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.cmd_health(argparse.Namespace(url=None))

    assert exit_info.value.code == 0
    assert asked == ["http://other.example:9000/api/v1/health/ready"], asked
