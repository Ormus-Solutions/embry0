"""The tier models in ``.env.example`` are the code defaults, and the catalogue lists them.

A deployment's root ``.env`` starts as a copy of ``.env.example``, and a value
there wins over the default in ``Embry0Config``. A tier model in the example
that is not the default pins every new deployment to it, and one the Anthropic
catalogue does not list is not offered by the model pickers or priced.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from embry0.agents.providers import ANTHROPIC
from embry0.config import Embry0Config

ROOT = Path(__file__).resolve().parents[3]
TIERS = ("MODEL_HEAVY", "MODEL_MEDIUM", "MODEL_LIGHT")


def _example_values() -> dict[str, str]:
    text = (ROOT / ".env.example").read_text()
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"^(MODEL_[A-Z]+)=(.*)$", text, re.M)}


@pytest.mark.parametrize("name", TIERS)
def test_example_tier_model_is_the_code_default(name: str) -> None:
    values = _example_values()
    assert name in values, f"{name} is not set in .env.example"
    default = Embry0Config.model_fields[name.lower()].default
    assert values[name] == default, f".env.example sets {name}={values[name]}, the code default is {default}"


@pytest.mark.parametrize("name", TIERS)
def test_example_tier_model_is_in_the_anthropic_catalogue(name: str) -> None:
    value = _example_values()[name]
    assert value in ANTHROPIC.models, f".env.example sets {name}={value}, not in {ANTHROPIC.models}"
