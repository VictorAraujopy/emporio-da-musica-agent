"""Embeddings through OpenRouter: texts become unit-length vectors for the policy search."""

import numpy as np
from openrouter import OpenRouter
from openrouter.components import ProviderPreferences
from openrouter.utils.logger import NoOpLogger
from openrouter.utils.retries import BackoffStrategy, RetryConfig

from emporio_agent.config import (
    ALLOW_PROVIDER_FALLBACKS,
    OPENROUTER_BASE_URL,
    OPENROUTER_TIMEOUT_MS,
    PROVIDER_DATA_COLLECTION,
    REQUIRE_ZERO_DATA_RETENTION,
    Settings,
)


def create_embedding_client(settings: Settings) -> OpenRouter:
    """Create the OpenRouter client; key, URL, time limit and a silent logger are set here."""
    return OpenRouter(
        api_key=settings.openrouter_api_key,
        server_url=OPENROUTER_BASE_URL,
        retry_config=single_attempt(),
        timeout_ms=OPENROUTER_TIMEOUT_MS,
        debug_logger=NoOpLogger(),
    )


def single_attempt() -> RetryConfig:
    """Never retry: the SDK's retries wait as long as a server's Retry-After header asks."""
    unused_backoff = BackoffStrategy(
        initial_interval=0, max_interval=0, exponent=1.0, max_elapsed_time=0
    )
    return RetryConfig("none", unused_backoff, retry_connection_errors=False)


def embed_texts(client: OpenRouter, texts: list[str], settings: Settings) -> np.ndarray:
    """Return one unit-length vector per text, row i belonging to texts[i]."""
    embeddings = request_embeddings(client, texts, settings)
    vectors = np.array(embeddings, dtype=np.float32)
    return normalize_rows(vectors)


def request_embeddings(
    client: OpenRouter, texts: list[str], settings: Settings
) -> list[list[float]]:
    """Ask the embedding model for one vector per text, under the fixed provider rules."""
    response = client.embeddings.generate(
        input=texts,
        model=settings.embedding_model,
        provider=embedding_provider_rules(settings),
        encoding_format="float",
    )
    if len(response.data) != len(texts):
        raise ValueError(f"Asked for {len(texts)} embeddings, received {len(response.data)}.")
    for position, item in enumerate(response.data):
        if item.index is not None and item.index != position:
            raise ValueError(f"Embedding {item.index} came back at position {position}.")
    return [item.embedding for item in response.data]


def embedding_provider_rules(settings: Settings) -> ProviderPreferences:
    """Send requests only to the configured providers, with the privacy rules fixed in config."""
    return ProviderPreferences(
        order=list(settings.embedding_providers),
        allow_fallbacks=ALLOW_PROVIDER_FALLBACKS,
        zdr=REQUIRE_ZERO_DATA_RETENTION,
        data_collection=PROVIDER_DATA_COLLECTION,
    )


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to length 1, so the dot product of two rows is their cosine similarity."""
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / lengths
