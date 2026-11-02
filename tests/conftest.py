"""Setup shared by every test: no test may send data to LangSmith."""

from collections.abc import Iterator

import langsmith.utils
import pytest

SHELL_TRACING_VARIABLES = (
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
)


@pytest.fixture(autouse=True)
def no_langsmith_tracing_from_the_shell(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run each test as if the shell had no LangSmith variables, with a dead local endpoint."""
    for variable in SHELL_TRACING_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "http://127.0.0.1:9")
    langsmith.utils.get_env_var.cache_clear()
    yield
    langsmith.utils.get_env_var.cache_clear()
