"""Divergence between flow maps: Jensen-Shannon in bits, 0 for the same flow, 1 for flow on disjoint cells and headings."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def occupancy(counts: np.ndarray) -> np.ndarray:
    """The map with its headings summed: pedestrian-seconds per cell."""
    return counts.sum(axis=-1, keepdims=True)


def normalized(counts: np.ndarray) -> np.ndarray | None:
    """The map as a distribution over cells and bins, None when it holds no time."""
    total = float(counts.sum())
    return None if total <= 0 else counts / total


def jsd(p: np.ndarray, q: np.ndarray) -> float | None:
    """Jensen-Shannon divergence (base 2) of two maps of one grid, None when either holds no time."""
    a, b = normalized(p), normalized(q)
    if a is None or b is None:
        return None
    a, b = a.ravel(), b.ravel()
    total = a + b

    def kl(x: np.ndarray) -> float:
        # KL(x || (a + b) / 2) as sum x log2(2x / (a + b)): a + b >= x > 0, where halving a denormal x could round to 0
        k = x > 0
        return float(np.sum(x[k] * np.log2(2 * x[k] / total[k])))

    return max(0.0, (kl(a) + kl(b)) / 2)


def pooled(maps: Sequence[np.ndarray]) -> np.ndarray | None:
    """The mean of the normalized maps (one per seed): the flow the crowd settles to, None when none holds time."""
    shares = [n for m in maps if (n := normalized(m)) is not None]
    return None if not shares else np.mean(shares, axis=0)


__all__ = ["jsd", "normalized", "occupancy", "pooled"]
