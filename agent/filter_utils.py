"""Bloom filter for user history (read/block). Demo: in-memory bitarray; production: Redis/Rebloom."""
from __future__ import annotations

import hashlib
import logging
import math
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from bitarray import bitarray
except ImportError:
    bitarray = None  # type: ignore


class UserHistoryFilter:
    """
    Bloom filter: record (user_id, action_type, paper_id). No false negatives; low false positive rate.
    Used to filter out already-read or blocked papers in recall (discovery path).
    """

    def __init__(self, capacity: int = 1_000_000, error_rate: float = 0.001):
        if bitarray is None:
            raise ImportError("UserHistoryFilter requires bitarray: pip install bitarray")
        self.capacity = capacity
        self.error_rate = error_rate
        self.num_bits = max(1024, int(-(capacity * math.log(error_rate)) / (math.log(2) ** 2)))
        self.num_hashes = max(1, int((self.num_bits / capacity) * math.log(2)))
        self._bit_array = bitarray(self.num_bits)
        self._bit_array.setall(0)
        logger.info(
            "UserHistoryFilter: capacity=%s error_rate=%s num_bits=%s num_hashes=%s",
            capacity, error_rate, self.num_bits, self.num_hashes,
        )

    def _get_hashes(self, item: str) -> List[int]:
        h = hashlib.sha256(item.encode("utf-8")).hexdigest()
        base = int(h[:16], 16)
        return [(base + i * 12345) % self.num_bits for i in range(self.num_hashes)]

    def add(self, user_id: str, paper_id: str, action_type: str = "read") -> None:
        """Record user action: read, block."""
        key = f"{user_id}:{action_type}:{paper_id}"
        for index in self._get_hashes(key):
            self._bit_array[index] = 1

    def exists(self, user_id: str, paper_id: str, action_type: str = "read") -> bool:
        """True if (user_id, action_type, paper_id) may have been added (no false negatives)."""
        key = f"{user_id}:{action_type}:{paper_id}"
        for index in self._get_hashes(key):
            if not self._bit_array[index]:
                return False
        return True

    def filter_candidates(
        self,
        user_id: str,
        candidates: List,
        filter_read: bool = True,
        filter_blocked: bool = True,
    ) -> List:
        """Return papers that are not (read or blocked). Items must have paper_id attribute."""
        valid = []
        for paper in candidates:
            pid = getattr(paper, "paper_id", None) or str(getattr(paper, "id", ""))
            if not pid:
                valid.append(paper)
                continue
            if filter_read and self.exists(user_id, pid, "read"):
                continue
            if filter_blocked and self.exists(user_id, pid, "block"):
                continue
            valid.append(paper)
        return valid


# Global singleton for demo. Production should use Redis BitMap / Rebloom per user or sharded.
_global_bloom: Optional[UserHistoryFilter] = None


def get_global_bloom_filter(capacity: int = 1_000_000, error_rate: float = 0.001) -> UserHistoryFilter:
    """Lazy-init global Bloom filter (in-memory). For production, replace with Redis-backed store."""
    global _global_bloom
    if _global_bloom is None:
        if bitarray is None:
            logger.warning("bitarray not installed; Bloom filter disabled. pip install bitarray")
            raise ImportError("bitarray required for Bloom filter")
        _global_bloom = UserHistoryFilter(capacity=capacity, error_rate=error_rate)
    return _global_bloom
