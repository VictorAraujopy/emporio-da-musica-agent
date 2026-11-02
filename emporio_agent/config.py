"""Settings read from the environment (locally, from .env), and the rules fixed in code."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
WHERE_TO_SET = (
    "Set it in .env (see .env.example); a variable set in the environment"
    " takes precedence over .env."
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ALLOW_PROVIDER_FALLBACKS = False
REQUIRE_ZERO_DATA_RETENTION = True
PROVIDER_DATA_COLLECTION = "deny"
OPENROUTER_TIMEOUT_MS = 10_000


class SettingsError(Exception):
    """A setting is missing or has no usable value."""


@dataclass(frozen=True)
class Settings:
    """Everything the agent reads from the environment."""

    openrouter_api_key: str = field(repr=False)
    chat_model: str
    chat_providers: tuple[str, ...]
    chat_quantizations: tuple[str, ...]
    embedding_model: str
    embedding_providers: tuple[str, ...]


def load_settings() -> Settings:
    """Read the environment and build the settings from it."""
    from_env = read_environment()
    return build_settings(from_env)


def read_environment() -> dict[str, str]:
    """Return the .env file's values, overridden by variables set in the process environment."""
    from_env_file = dotenv_values(ENV_FILE)
    environment = {name: value for name, value in from_env_file.items() if value is not None}
    environment.update(os.environ)
    return environment


def build_settings(from_env: Mapping[str, str]) -> Settings:
    """Build the settings from environment values, validating each one."""
    return Settings(
        openrouter_api_key=read_required(from_env, "OPENROUTER_API_KEY"),
        chat_model=read_required(from_env, "CHAT_MODEL"),
        chat_providers=read_required_list(from_env, "CHAT_PROVIDERS"),
        chat_quantizations=split_list(read_allowing_empty(from_env, "CHAT_QUANTIZATIONS")),
        embedding_model=read_required(from_env, "EMBEDDING_MODEL"),
        embedding_providers=read_required_list(from_env, "EMBEDDING_PROVIDERS"),
    )


def read_allowing_empty(from_env: Mapping[str, str], name: str) -> str:
    """Return a setting's trimmed value, which may be empty; raise SettingsError if missing."""
    if name not in from_env:
        raise SettingsError(f"Setting {name} is missing. {WHERE_TO_SET}")
    return from_env[name].strip()


def read_required(from_env: Mapping[str, str], name: str) -> str:
    """Return a setting's trimmed value; raise SettingsError if it is missing or empty."""
    value = read_allowing_empty(from_env, name)
    if not value:
        raise SettingsError(f"Setting {name} is empty. {WHERE_TO_SET}")
    return value


def read_required_list(from_env: Mapping[str, str], name: str) -> tuple[str, ...]:
    """Return a comma-separated setting's items; raise SettingsError if there are none."""
    items = split_list(read_required(from_env, name))
    if not items:
        raise SettingsError(f"Setting {name} has no items: list them separated by commas.")
    return items


def split_list(value: str) -> tuple[str, ...]:
    """Split a comma-separated setting into its non-empty, trimmed items."""
    items = [item.strip() for item in value.split(",")]
    return tuple(item for item in items if item)
