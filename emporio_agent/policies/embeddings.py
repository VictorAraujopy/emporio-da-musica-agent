"""Embeddings through OpenRouter: texts become unit-length vectors for the policy search."""

import numpy as np
from openrouter import OpenRouter
from openrouter.components import ProviderPreferences

from emporio_agent.config import (
    ALLOW_PROVIDER_FALLBACKS,
    PROVIDER_DATA_COLLECTION,
    REQUIRE_ZERO_DATA_RETENTION,
    Settings,
)


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
