"""Tests for the OpenRouter client: limits and settings fixed in code, a fake server, no API."""

import time

import httpx
import pytest
from openrouter.utils.logger import NoOpLogger

from emporio_agent.config import Settings
from emporio_agent.openrouter_client import create_openrouter_client

TEST_SETTINGS = Settings(
    openrouter_api_key="test-key",
    chat_model="test/chat-model",
    chat_providers=("test-chat-provider",),
    chat_quantizations=(),
    embedding_model="test/embedding-model",
    embedding_providers=("test-embedding-provider",),
)


def test_the_client_ignores_openrouter_variables_set_in_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stray OPENROUTER_API_KEY or OPENROUTER_DEBUG cannot change the key or turn on logging."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-from-the-shell")
    monkeypatch.setenv("OPENROUTER_DEBUG", "1")
    client_config = create_openrouter_client(TEST_SETTINGS, timeout_ms=10_000).sdk_configuration
    assert client_config.security.api_key == "test-key"
    assert isinstance(client_config.debug_logger, NoOpLogger)
    assert client_config.server_url == "https://openrouter.ai/api/v1"
    assert client_config.timeout_ms == 10_000


def test_a_server_error_fails_at_once_even_when_the_server_asks_to_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One attempt only: a 503 asking for an hour's wait is not obeyed, and nothing is retried."""
    attempts = []
    waits = []

    def unavailable_for_an_hour(request: httpx.Request) -> httpx.Response:
        """Answer the first request with 503 and a one-hour Retry-After; any retry succeeds."""
        attempts.append(request)
        if len(attempts) > 1:
            return httpx.Response(200, json={"object": "list", "data": [], "model": "test"})
        error_body = {"error": {"code": 503, "message": "unavailable"}}
        return httpx.Response(503, headers={"Retry-After": "3600"}, json=error_body)

    monkeypatch.setattr(time, "sleep", waits.append)
    client = create_openrouter_client(TEST_SETTINGS, timeout_ms=10_000)
    client.sdk_configuration.client = httpx.Client(
        transport=httpx.MockTransport(unavailable_for_an_hour)
    )
    with pytest.raises(Exception):
        client.embeddings.generate(input=["pergunta do cliente"], model="test/embedding-model")
    assert len(attempts) == 1
    assert waits == []
