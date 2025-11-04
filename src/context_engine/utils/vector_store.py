"""
Vector store for semantic memory storage and retrieval.

Implements efficient similarity search using sentence transformers
for embeddings and numpy/faiss for vector operations.
"""

import os
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from .schemas import MemoryEntry


class EmbeddingModel:
    """Wrapper for embedding generation."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.

        Default model: all-MiniLM-L6-v2
        - Fast and efficient
        - 384 dimensions
        - Good for semantic search
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


class VectorStore:
    """
    Vector store for semantic memory storage and retrieval.

    Features:
    - Semantic similarity search
    - Persistence to disk
    - Filtering by metadata
    - Recency and frequency weighting
    """

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        storage_path: Optional[str] = None
    ):
        """Initialize vector store."""
        self.embedding_model = embedding_model or EmbeddingModel()
        self.storage_path = storage_path
        self.entries: Dict[str, MemoryEntry] = {}
        self.vectors: Optional[np.ndarray] = None
        self.index_map: List[str] = []  # Maps vector index to entry ID

        # Load from disk if path exists
        if storage_path and os.path.exists(storage_path):
            self.load()

    def add(
        self,
        entry: MemoryEntry,
        generate_embedding: bool = True
    ) -> None:
        """
        Add a memory entry to the store.

        Args:
            entry: Memory entry to add
            generate_embedding: If True, generate embedding from content
        """
        # Generate embedding if needed
        if generate_embedding and entry.embedding is None:
            entry.embedding = self.embedding_model.embed(entry.content)

        # Store entry
        self.entries[entry.id] = entry

        # Update vector index
        if entry.embedding:
            embedding_array = np.array(entry.embedding).reshape(1, -1)

            if self.vectors is None:
                self.vectors = embedding_array
                self.index_map = [entry.id]
            else:
                self.vectors = np.vstack([self.vectors, embedding_array])
                self.index_map.append(entry.id)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_fn: Optional[callable] = None,
        rerank_by_recency: bool = True,
        rerank_by_frequency: bool = True
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Search for similar entries.

        Implements "Attention Before Attention" pattern from the paper:
        - Semantic relevance (vector similarity)
        - Recency weighting (newer memories prioritized)
        - Frequency weighting (often-accessed memories prioritized)

        Args:
            query: Search query
            k: Number of results to return
            filter_fn: Optional filter function(entry) -> bool
            rerank_by_recency: Weight by recency
            rerank_by_frequency: Weight by access count

        Returns:
            List of (entry, score) tuples, sorted by relevance
        """
        if self.vectors is None or len(self.entries) == 0:
            return []

        # Generate query embedding
        query_embedding = np.array(self.embedding_model.embed(query))

        # Compute cosine similarity
        similarities = self._cosine_similarity(query_embedding, self.vectors)

        # Get top k candidates (before filtering)
        candidate_indices = np.argsort(similarities)[::-1][:k * 3]  # Get 3x for filtering

        # Build results with reranking
        results = []
        for idx in candidate_indices:
            entry_id = self.index_map[idx]
            entry = self.entries[entry_id]

            # Apply filter
            if filter_fn and not filter_fn(entry):
                continue

            # Calculate composite score
            base_score = float(similarities[idx])
            score = base_score

            # Recency weighting
            if rerank_by_recency and entry.timestamp:
                # Decay factor based on age (newer = higher score)
                from datetime import datetime
                age_seconds = (datetime.now() - entry.timestamp).total_seconds()
                age_days = age_seconds / 86400
                recency_factor = 1.0 / (1.0 + age_days * 0.1)  # Slow decay
                score *= recency_factor

            # Frequency weighting
            if rerank_by_frequency:
                # Log scale for access count
                frequency_factor = 1.0 + np.log1p(entry.access_count) * 0.1
                score *= frequency_factor

            results.append((entry, score))

            if len(results) >= k:
                break

        # Sort by final score
        results.sort(key=lambda x: x[1], reverse=True)

        # Mark as accessed
        for entry, _ in results:
            entry.mark_accessed()

        return results

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get entry by ID."""
        return self.entries.get(entry_id)

    def delete(self, entry_id: str) -> bool:
        """Delete entry by ID."""
        if entry_id not in self.entries:
            return False

        # Remove from entries
        del self.entries[entry_id]

        # Rebuild vector index (inefficient but simple)
        if entry_id in self.index_map:
            idx = self.index_map.index(entry_id)
            self.vectors = np.delete(self.vectors, idx, axis=0)
            self.index_map.pop(idx)

        return True

    def save(self, path: Optional[str] = None) -> None:
        """Save vector store to disk."""
        save_path = path or self.storage_path
        if not save_path:
            raise ValueError("No storage path specified")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Save entries as JSON
        entries_dict = {
            entry_id: entry.model_dump()
            for entry_id, entry in self.entries.items()
        }

        with open(f"{save_path}.json", "w") as f:
            json.dump(entries_dict, f, default=str)

        # Save vectors as numpy array
        if self.vectors is not None:
            np.save(f"{save_path}.npy", self.vectors)

        # Save index map
        with open(f"{save_path}.idx", "wb") as f:
            pickle.dump(self.index_map, f)

    def load(self, path: Optional[str] = None) -> None:
        """Load vector store from disk."""
        load_path = path or self.storage_path
        if not load_path:
            raise ValueError("No storage path specified")

        # Load entries
        with open(f"{load_path}.json", "r") as f:
            entries_dict = json.load(f)
            self.entries = {
                entry_id: MemoryEntry(**entry_data)
                for entry_id, entry_data in entries_dict.items()
            }

        # Load vectors
        if os.path.exists(f"{load_path}.npy"):
            self.vectors = np.load(f"{load_path}.npy")

        # Load index map
        if os.path.exists(f"{load_path}.idx"):
            with open(f"{load_path}.idx", "rb") as f:
                self.index_map = pickle.load(f)

    def _cosine_similarity(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all vectors."""
        # Normalize vectors
        query_norm = query / np.linalg.norm(query)
        vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        # Compute dot product
        return np.dot(vectors_norm, query_norm)

    def __len__(self) -> int:
        """Return number of entries in store."""
        return len(self.entries)
