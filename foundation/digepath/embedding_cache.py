"""
Embedding Caching Module for Digepath Visual Representations.
Provides both in-memory and persistent on-disk caching.
"""

import hashlib
from pathlib import Path
from typing import Optional, Dict, Union
import numpy as np


class EmbeddingCache:
    """
    Two-tier (Memory + Disk) embedding cache for fast feature retrieval.
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None, max_memory_entries: int = 1000):
        if cache_dir is None:
            self.cache_dir = Path(__file__).resolve().parents[2] / ".cache" / "embeddings"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_entries = max_memory_entries
        self._memory_cache: Dict[str, np.ndarray] = {}

    @staticmethod
    def _compute_key(identifier_or_path: Union[str, Path, bytes]) -> str:
        """
        Generates a stable MD5 key from an identifier, path, or raw bytes.
        """
        if isinstance(identifier_or_path, Path):
            identifier_or_path = str(identifier_or_path.resolve())

        if isinstance(identifier_or_path, str):
            return hashlib.md5(identifier_or_path.encode("utf-8")).hexdigest()
        elif isinstance(identifier_or_path, bytes):
            return hashlib.md5(identifier_or_path).hexdigest()
        else:
            return hashlib.md5(str(identifier_or_path).encode("utf-8")).hexdigest()

    def get(self, identifier: Union[str, Path, bytes]) -> Optional[np.ndarray]:
        """
        Retrieves embedding from memory or disk cache. Returns None if not cached.
        """
        key = self._compute_key(identifier)

        # 1. Check memory cache
        if key in self._memory_cache:
            return self._memory_cache[key]

        # 2. Check disk cache
        disk_file = self.cache_dir / f"{key}.npy"
        if disk_file.exists():
            try:
                emb = np.load(disk_file)
                # Store in memory cache
                if len(self._memory_cache) < self.max_memory_entries:
                    self._memory_cache[key] = emb
                return emb
            except Exception:
                return None

        return None

    def put(self, identifier: Union[str, Path, bytes], embedding: np.ndarray) -> None:
        """
        Stores embedding in both memory and disk cache.
        """
        key = self._compute_key(identifier)
        emb_array = np.asarray(embedding, dtype=np.float32)

        # Store in memory
        if len(self._memory_cache) >= self.max_memory_entries:
            # Simple eviction: pop first key
            first_key = next(iter(self._memory_cache))
            self._memory_cache.pop(first_key)
        self._memory_cache[key] = emb_array

        # Save to disk
        disk_file = self.cache_dir / f"{key}.npy"
        np.save(disk_file, emb_array)

    def contains(self, identifier: Union[str, Path, bytes]) -> bool:
        key = self._compute_key(identifier)
        if key in self._memory_cache:
            return True
        return (self.cache_dir / f"{key}.npy").exists()

    def clear(self) -> None:
        self._memory_cache.clear()
        for f in self.cache_dir.glob("*.npy"):
            try:
                f.unlink()
            except Exception:
                pass
