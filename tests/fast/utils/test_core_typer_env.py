"""Environment parsing contracts used by upstream's new launchers."""

from dataclasses import dataclass
from enum import Enum

from miles.utils import typer_utils


class Choice(str, Enum):
    A = "a"
    B = "b"


@dataclass
class Config:
    required: str
    count: int = 2
    flags: tuple[Choice, ...] = (Choice.A,)
    enabled: bool = False


def test_environment_conversion_and_explicit_override(monkeypatch):
    monkeypatch.setenv("MILES_SCRIPT_COUNT", "7")
    monkeypatch.setenv("MILES_SCRIPT_FLAGS", "a b")
    monkeypatch.setenv("MILES_SCRIPT_ENABLED", "true")
    monkeypatch.setenv("MILES_SCRIPT_REQUIRED", "ignored")
    result = typer_utils.dataclass_from_env(Config, overrides={"required": "explicit"})
    assert result == Config("explicit", count=7, flags=(Choice.A, Choice.B), enabled=True)
    assert all(isinstance(flag, Choice) for flag in result.flags)


def test_required_override_needs_no_environment(monkeypatch):
    monkeypatch.delenv("MILES_SCRIPT_REQUIRED", raising=False)
    assert typer_utils.dataclass_from_env(Config, overrides={"required": "provided"}) == Config("provided")
