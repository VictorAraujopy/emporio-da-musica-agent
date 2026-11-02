"""Tests for the policy index: cache rules with a stub client, and search on test vectors."""

import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from emporio_agent.config import Settings
from emporio_agent.policies import index as index_module
from emporio_agent.policies.chunker import Chunk
from emporio_agent.policies.index import (
    CHUNKS_FILE_NAME,
    MANUAL_FILE,
    VECTORS_FILE_NAME,
    PolicyIndex,
    load_policy_index,
    search_policies,
)

TEST_SETTINGS = Settings(
    openrouter_api_key="test-key",
    chat_model="test/chat-model",
    chat_providers=("test-chat-provider",),
    chat_quantizations=(),
    embedding_model="test/embedding-model",
    embedding_providers=("test-embedding-provider",),
)
QUERY_VECTOR = np.array([1.0, 0.0], dtype=np.float32)


class StubEmbeddingsEndpoint:
    """Stands in for client.embeddings: counts the requests and returns one vector per text."""

    def __init__(self) -> None:
        """Start with no requests made."""
        self.requests = 0

    def generate(self, input: list[str], **request_options: object) -> SimpleNamespace:
        """Return a different two-number vector for each text, shaped like the real response."""
        self.requests += 1
        vectors = [[1.0, float(position)] for position in range(len(input))]
        data = [
            SimpleNamespace(embedding=vector, index=position)
            for position, vector in enumerate(vectors)
        ]
        return SimpleNamespace(data=data)


class StubEmbeddingClient:
    """Stands in for the OpenRouter client, with only its embeddings endpoint."""

    def __init__(self) -> None:
        """Give the client a stub embeddings endpoint."""
        self.embeddings = StubEmbeddingsEndpoint()


def requests_made_by_a_fresh_start(storage_dir: Path, settings: Settings = TEST_SETTINGS) -> int:
    """Load the index as a new start would, and return how many embedding requests it made."""
    client = StubEmbeddingClient()
    load_policy_index(client, settings, storage_dir=storage_dir)
    return client.embeddings.requests


def test_without_a_cache_the_index_is_built_and_saved(tmp_path: Path) -> None:
    """The first start embeds the chunks once and writes both cache files."""
    client = StubEmbeddingClient()
    index = load_policy_index(client, TEST_SETTINGS, storage_dir=tmp_path)
    assert client.embeddings.requests == 1
    assert len(index.chunks) == len(index.vectors) > 0
    assert (tmp_path / VECTORS_FILE_NAME).exists()
    assert (tmp_path / CHUNKS_FILE_NAME).exists()


def test_a_second_start_loads_the_same_index_without_the_api(tmp_path: Path) -> None:
    """With a valid cache, no request is made and the same chunks and vectors come back."""
    built_index = load_policy_index(StubEmbeddingClient(), TEST_SETTINGS, storage_dir=tmp_path)
    client = StubEmbeddingClient()
    loaded_index = load_policy_index(client, TEST_SETTINGS, storage_dir=tmp_path)
    assert client.embeddings.requests == 0
    assert loaded_index.chunks == built_index.chunks
    assert np.array_equal(loaded_index.vectors, built_index.vectors)


def test_another_embedding_model_rebuilds_the_index(tmp_path: Path) -> None:
    """Vectors from different models cannot be compared, so a model change rebuilds."""
    requests_made_by_a_fresh_start(tmp_path)
    other_model = dataclasses.replace(TEST_SETTINGS, embedding_model="test/another-model")
    assert requests_made_by_a_fresh_start(tmp_path, other_model) == 1


def test_a_changed_manual_rebuilds_the_index(tmp_path: Path) -> None:
    """A manual whose bytes changed has another hash, so the cache is rebuilt."""
    manual_copy = tmp_path / "manual.pdf"
    manual_copy.write_bytes(MANUAL_FILE.read_bytes())
    storage_dir = tmp_path / "storage"
    load_policy_index(StubEmbeddingClient(), TEST_SETTINGS, manual_copy, storage_dir)
    manual_copy.write_bytes(MANUAL_FILE.read_bytes() + b"\n")
    client = StubEmbeddingClient()
    load_policy_index(client, TEST_SETTINGS, manual_copy, storage_dir)
    assert client.embeddings.requests == 1


def test_a_chunks_file_that_is_not_json_is_rebuilt(tmp_path: Path) -> None:
    """A chunks file cut in the middle or overwritten is rebuilt, not a crash."""
    requests_made_by_a_fresh_start(tmp_path)
    cut_json = '{"manual_sha256": "cut in the mid'
    (tmp_path / CHUNKS_FILE_NAME).write_text(cut_json, encoding="utf-8")
    assert requests_made_by_a_fresh_start(tmp_path) == 1


def test_a_chunks_file_with_another_structure_is_rebuilt(tmp_path: Path) -> None:
    """Valid JSON shaped differently (a list) is rebuilt, not a crash."""
    requests_made_by_a_fresh_start(tmp_path)
    (tmp_path / CHUNKS_FILE_NAME).write_text("[]", encoding="utf-8")
    assert requests_made_by_a_fresh_start(tmp_path) == 1


def test_a_chunks_file_missing_its_chunks_is_rebuilt(tmp_path: Path) -> None:
    """A chunks file that still matches the manual and model but lost its chunks is rebuilt."""
    requests_made_by_a_fresh_start(tmp_path)
    chunks_file = tmp_path / CHUNKS_FILE_NAME
    cache_info = json.loads(chunks_file.read_text(encoding="utf-8"))
    del cache_info["chunks"]
    chunks_file.write_text(json.dumps(cache_info), encoding="utf-8")
    assert requests_made_by_a_fresh_start(tmp_path) == 1


def test_a_cut_vectors_file_is_rebuilt(tmp_path: Path) -> None:
    """A vectors file cut in the middle is rebuilt, not a crash."""
    requests_made_by_a_fresh_start(tmp_path)
    vectors_file = tmp_path / VECTORS_FILE_NAME
    vectors_file.write_bytes(vectors_file.read_bytes()[:100])
    assert requests_made_by_a_fresh_start(tmp_path) == 1


def test_a_vectors_file_with_another_number_of_rows_is_rebuilt(tmp_path: Path) -> None:
    """Every chunk needs its own vector, so a row count that differs is rebuilt."""
    requests_made_by_a_fresh_start(tmp_path)
    np.save(tmp_path / VECTORS_FILE_NAME, np.ones((2, 2), dtype=np.float32), allow_pickle=False)
    assert requests_made_by_a_fresh_start(tmp_path) == 1


def test_a_crash_between_the_two_writes_leaves_no_valid_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New vectors saved without their chunks file are never paired with the old chunks file."""
    requests_made_by_a_fresh_start(tmp_path)

    def crash(*args: object, **kwargs: object) -> str:
        """Stop the program while the chunks file is being written."""
        raise RuntimeError("crash between the two writes")

    other_model = dataclasses.replace(TEST_SETTINGS, embedding_model="test/another-model")
    monkeypatch.setattr(index_module.json, "dumps", crash)
    with pytest.raises(RuntimeError):
        load_policy_index(StubEmbeddingClient(), other_model, storage_dir=tmp_path)
    monkeypatch.undo()
    assert requests_made_by_a_fresh_start(tmp_path) == 1


def test_an_empty_vectors_file_is_rebuilt(tmp_path: Path) -> None:
    """A vectors file left empty, as after a power loss, is rebuilt, not a crash."""
    requests_made_by_a_fresh_start(tmp_path)
    (tmp_path / VECTORS_FILE_NAME).write_bytes(b"")
    assert requests_made_by_a_fresh_start(tmp_path) == 1


def test_a_chunks_file_whose_chunks_are_null_is_rebuilt(tmp_path: Path) -> None:
    """Chunks of the wrong type in a chunks file that still matches are rebuilt, not a crash."""
    requests_made_by_a_fresh_start(tmp_path)
    chunks_file = tmp_path / CHUNKS_FILE_NAME
    cache_info = json.loads(chunks_file.read_text(encoding="utf-8"))
    cache_info["chunks"] = None
    chunks_file.write_text(json.dumps(cache_info), encoding="utf-8")
    assert requests_made_by_a_fresh_start(tmp_path) == 1


def test_a_crash_while_saving_the_vectors_leaves_no_valid_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new chunks file is written only after the vectors, so it never meets old vectors."""
    requests_made_by_a_fresh_start(tmp_path)

    def crash(*args: object, **kwargs: object) -> None:
        """Stop the program while the vectors are being saved."""
        raise RuntimeError("crash while saving the vectors")

    other_model = dataclasses.replace(TEST_SETTINGS, embedding_model="test/another-model")
    monkeypatch.setattr(index_module.np, "save", crash)
    with pytest.raises(RuntimeError):
        load_policy_index(StubEmbeddingClient(), other_model, storage_dir=tmp_path)
    monkeypatch.undo()
    assert requests_made_by_a_fresh_start(tmp_path, other_model) == 1


def index_with_similarities(similarities: list[float]) -> PolicyIndex:
    """Make an index whose chunk i has cosine similarities[i] with the query (1, 0)."""
    vectors = [[similarity, math.sqrt(1 - similarity**2)] for similarity in similarities]
    chunks = tuple(Chunk(str(position), f"texto {position}") for position in range(len(vectors)))
    return PolicyIndex(chunks, np.array(vectors, dtype=np.float32))



def test_search_returns_the_most_similar_chunks_first() -> None:
    """Results come best first, with their similarity."""
    index = index_with_similarities([0.70, 0.95, 0.80])
    retrieved = search_policies(index, QUERY_VECTOR, top_k=3, min_similarity=0.0)
    assert [found.chunk.section_id for found in retrieved] == ["1", "2", "0"]
    assert retrieved[0].similarity == pytest.approx(0.95)


def test_search_keeps_only_top_k_chunks() -> None:
    """No more than top_k chunks come back."""
    index = index_with_similarities([0.70, 0.95, 0.80])
    retrieved = search_policies(index, QUERY_VECTOR, top_k=2, min_similarity=0.0)
    assert [found.chunk.section_id for found in retrieved] == ["1", "2"]


def test_search_drops_chunks_below_the_cutoff() -> None:
    """Chunks less similar than the cutoff are left out, even inside the top_k."""
    index = index_with_similarities([0.70, 0.95, 0.80])
    retrieved = search_policies(index, QUERY_VECTOR, top_k=3, min_similarity=0.75)
    assert [found.chunk.section_id for found in retrieved] == ["1", "2"]


def test_search_defaults_to_five_chunks_at_or_above_0_60() -> None:
    """By default at most 5 chunks come back, none below a similarity of 0.60."""
    seven_above_cutoff = index_with_similarities([0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65])
    three_above_cutoff = index_with_similarities([0.95, 0.90, 0.61, 0.59, 0.30])
    assert len(search_policies(seven_above_cutoff, QUERY_VECTOR)) == 5
    assert len(search_policies(three_above_cutoff, QUERY_VECTOR)) == 3


def test_search_keeps_a_chunk_exactly_at_the_cutoff() -> None:
    """The cutoff is inclusive: a chunk as similar as the cutoff stays."""
    index = index_with_similarities([0.75, 0.50])
    retrieved = search_policies(index, QUERY_VECTOR, top_k=2, min_similarity=0.75)
    assert [found.chunk.section_id for found in retrieved] == ["0"]
