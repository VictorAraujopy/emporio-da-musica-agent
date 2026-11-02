"""Tests for the embeddings client; every request is answered by a fake server, not the API."""

import json
import time
from collections.abc import Callable

import httpx
import numpy as np
import pytest
from openrouter import OpenRouter
from openrouter.utils.logger import NoOpLogger

from emporio_agent.config import Settings
from emporio_agent.policies.embeddings import create_embedding_client, embed_texts

TEST_SETTINGS = Settings(
    openrouter_api_key="test-key",
    chat_model="test/chat-model",
    chat_providers=("test-chat-provider",),
    chat_quantizations=(),
    embedding_model="test/embedding-model",
    embedding_providers=("google-vertex/us-central1",),
)


def fake_openrouter(answer: Callable[[httpx.Request], httpx.Response]) -> OpenRouter:
    """Return an OpenRouter client whose requests are answered by a function, not the network."""
    fake_network = httpx.Client(transport=httpx.MockTransport(answer))
    return OpenRouter(api_key="test-key", client=fake_network)


def embeddings_response(vectors: list[list[float]]) -> httpx.Response:
    """Build the answer the embeddings endpoint gives for these vectors."""
    data = [
        {"object": "embedding", "embedding": vector, "index": position}
        for position, vector in enumerate(vectors)
    ]
    usage = {"prompt_tokens": 1, "total_tokens": 1}
    body = {"id": "test", "object": "list", "model": "test", "data": data, "usage": usage}
    return httpx.Response(200, json=body)


def test_every_request_carries_the_fixed_privacy_rules() -> None:
    """The request pins the providers: no fallback, zero retention, no data collection."""
    sent_bodies = []

    def record_and_answer(request: httpx.Request) -> httpx.Response:
        """Keep the request body and answer with one vector."""
        sent_bodies.append(json.loads(request.content))
        return embeddings_response([[1.0, 0.0]])

    embed_texts(fake_openrouter(record_and_answer), ["pergunta do cliente"], TEST_SETTINGS)
    expected_provider_rules = {
        "order": ["google-vertex/us-central1"],
        "allow_fallbacks": False,
        "zdr": True,
        "data_collection": "deny",
    }
    assert sent_bodies[0]["provider"] == expected_provider_rules
    assert sent_bodies[0]["model"] == "test/embedding-model"
    assert sent_bodies[0]["input"] == ["pergunta do cliente"]


def test_vectors_come_back_unit_length_in_the_order_of_the_texts() -> None:
    """Each row is scaled to length 1, and row i belongs to text i."""
    client = fake_openrouter(lambda request: embeddings_response([[3.0, 4.0], [0.0, 2.0]]))
    vectors = embed_texts(client, ["primeiro", "segundo"], TEST_SETTINGS)
    assert np.allclose(vectors, [[0.6, 0.8], [0.0, 1.0]])


def test_fewer_vectors_than_texts_is_an_error() -> None:
    """A missing vector would shift every row onto the wrong text, so it raises instead."""
    client = fake_openrouter(lambda request: embeddings_response([[1.0, 0.0]]))
    with pytest.raises(ValueError):
        embed_texts(client, ["primeiro", "segundo"], TEST_SETTINGS)


def test_the_client_ignores_openrouter_variables_set_in_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stray OPENROUTER_API_KEY or OPENROUTER_DEBUG cannot change the key or turn on logging."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-from-the-shell")
    monkeypatch.setenv("OPENROUTER_DEBUG", "1")
    client_config = create_embedding_client(TEST_SETTINGS).sdk_configuration
    assert client_config.security.api_key == "test-key"
    assert isinstance(client_config.debug_logger, NoOpLogger)
    assert client_config.server_url == "https://openrouter.ai/api/v1"


def test_a_server_error_fails_at_once_even_when_the_server_asks_to_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One attempt with a 10 s time limit: a 503 asking for an hour's wait is not obeyed."""
    attempts = []
    waits = []

    def unavailable_for_an_hour(request: httpx.Request) -> httpx.Response:
        """Answer the first request with 503 and a one-hour Retry-After; any retry succeeds."""
        attempts.append(request)
        if len(attempts) > 1:
            return embeddings_response([[1.0, 0.0]])
        error_body = {"error": {"code": 503, "message": "unavailable"}}
        return httpx.Response(503, headers={"Retry-After": "3600"}, json=error_body)

    monkeypatch.setattr(time, "sleep", waits.append)
    client = create_embedding_client(TEST_SETTINGS)
    client.sdk_configuration.client = httpx.Client(
        transport=httpx.MockTransport(unavailable_for_an_hour)
    )
    with pytest.raises(Exception):
        embed_texts(client, ["pergunta do cliente"], TEST_SETTINGS)
    assert len(attempts) == 1
    assert waits == []
    assert client.sdk_configuration.timeout_ms == 10_000


def test_vectors_returned_out_of_order_are_an_error() -> None:
    """A vector whose index is not its position would land on the wrong text, so it raises."""
    swapped_data = [
        {"object": "embedding", "embedding": [0.0, 1.0], "index": 1},
        {"object": "embedding", "embedding": [1.0, 0.0], "index": 0},
    ]
    usage = {"prompt_tokens": 1, "total_tokens": 1}
    body = {"id": "test", "object": "list", "model": "test", "data": swapped_data, "usage": usage}
    client = fake_openrouter(lambda request: httpx.Response(200, json=body))
    with pytest.raises(ValueError):
        embed_texts(client, ["primeiro", "segundo"], TEST_SETTINGS)
