"""Policy index: the manual's chunks and their vectors, cached on disk, searched by cosine."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openrouter import OpenRouter

from emporio_agent.config import PROJECT_ROOT, Settings
from emporio_agent.policies.chunker import Chunk, split_manual_into_chunks
from emporio_agent.policies.embeddings import embed_texts
from emporio_agent.policies.extract import extract_manual_text

MANUAL_FILE = PROJECT_ROOT / "data" / "store_policies.pdf"
STORAGE_DIR = PROJECT_ROOT / "storage"
VECTORS_FILE_NAME = "policy_vectors.npy"
CHUNKS_FILE_NAME = "policy_chunks.json"
UNREADABLE_CACHE_ERRORS = (OSError, EOFError, ValueError, KeyError, TypeError, AttributeError)
POLICY_TOP_K = 5
POLICY_MIN_SIMILARITY = 0.60


@dataclass(frozen=True)
class PolicyIndex:
    """The manual's chunks and their unit-length vectors; row i belongs to chunks[i]."""

    chunks: tuple[Chunk, ...]
    vectors: np.ndarray


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk found by a search, with its cosine similarity to the query."""

    chunk: Chunk
    similarity: float


def load_policy_index(
    client: OpenRouter,
    settings: Settings,
    manual_file: Path = MANUAL_FILE,
    storage_dir: Path = STORAGE_DIR,
) -> PolicyIndex:
    """Load the cached index if built from this manual and model; else build and cache it."""
    manual_sha256 = file_sha256(manual_file)
    cached_index = read_cached_index(storage_dir, manual_sha256, settings.embedding_model)
    if cached_index is not None:
        return cached_index
    index = build_index(client, settings, manual_file)
    write_cached_index(index, storage_dir, manual_sha256, settings.embedding_model)
    return index


def build_index(client: OpenRouter, settings: Settings, manual_file: Path) -> PolicyIndex:
    """Extract the manual's text, split it into chunks and embed every chunk."""
    manual_text = extract_manual_text(manual_file)
    chunks = split_manual_into_chunks(manual_text)
    chunk_texts = [chunk.text for chunk in chunks]
    vectors = embed_texts(client, chunk_texts, settings)
    return PolicyIndex(tuple(chunks), vectors)


def file_sha256(file: Path) -> str:
    """Return the SHA-256 of a file's bytes, in hex."""
    return hashlib.sha256(file.read_bytes()).hexdigest()


def read_cached_index(
    storage_dir: Path, manual_sha256: str, embedding_model: str
) -> PolicyIndex | None:
    """Return the cached index, or None if missing, unreadable or built from something else."""
    try:
        cache_info, vectors = read_cache_files(storage_dir)
        if not cache_matches(cache_info, manual_sha256, embedding_model):
            return None
        chunks = chunks_from_cache(cache_info)
    except UNREADABLE_CACHE_ERRORS:
        return None
    if vectors.ndim != 2 or len(vectors) != len(chunks):
        return None
    return PolicyIndex(chunks, vectors)


def read_cache_files(storage_dir: Path) -> tuple[dict, np.ndarray]:
    """Read the chunks file and the vectors file as they are on disk."""
    chunks_file = storage_dir / CHUNKS_FILE_NAME
    vectors_file = storage_dir / VECTORS_FILE_NAME
    cache_info = json.loads(chunks_file.read_text(encoding="utf-8"))
    vectors = np.load(vectors_file, allow_pickle=False)
    return cache_info, vectors


def cache_matches(cache_info: dict, manual_sha256: str, embedding_model: str) -> bool:
    """Tell whether the cache was built from this manual's bytes and this embedding model."""
    same_manual = cache_info.get("manual_sha256") == manual_sha256
    same_model = cache_info.get("embedding_model") == embedding_model
    return same_manual and same_model


def chunks_from_cache(cache_info: dict) -> tuple[Chunk, ...]:
    """Rebuild the chunks saved in the chunks file, in their order."""
    return tuple(Chunk(item["section_id"], item["text"]) for item in cache_info["chunks"])


def write_cached_index(
    index: PolicyIndex, storage_dir: Path, manual_sha256: str, embedding_model: str
) -> None:
    """Delete the old chunks file, save the vectors, then write the chunks file as the seal."""
    chunks_file = storage_dir / CHUNKS_FILE_NAME
    storage_dir.mkdir(parents=True, exist_ok=True)
    chunks_file.unlink(missing_ok=True)
    np.save(storage_dir / VECTORS_FILE_NAME, index.vectors, allow_pickle=False)
    saved_chunks = [
        {"section_id": chunk.section_id, "text": chunk.text} for chunk in index.chunks
    ]
    cache_info = {
        "manual_sha256": manual_sha256,
        "embedding_model": embedding_model,
        "chunks": saved_chunks,
    }
    chunks_json = json.dumps(cache_info, ensure_ascii=False, indent=2)
    chunks_file.write_text(chunks_json, encoding="utf-8")


def search_policies(
    index: PolicyIndex,
    query_vector: np.ndarray,
    top_k: int = POLICY_TOP_K,
    min_similarity: float = POLICY_MIN_SIMILARITY,
) -> list[RetrievedChunk]:
    """Return up to top_k chunks, best first, keeping only those at or above min_similarity."""
    similarities = index.vectors @ query_vector
    positions_best_first = np.argsort(similarities)[::-1]
    top_positions = positions_best_first[:top_k]
    top_chunks = [RetrievedChunk(index.chunks[i], float(similarities[i])) for i in top_positions]
    return [retrieved for retrieved in top_chunks if retrieved.similarity >= min_similarity]
