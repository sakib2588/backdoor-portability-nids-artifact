"""Backdoor poison-identification detectors (vision control and NIDS)."""
from __future__ import annotations

import numpy as np


def poison_recall(flagged: np.ndarray, is_poison: np.ndarray) -> float:
    """Fraction of truly-poisoned samples that were flagged. NaN if none poisoned."""
    flagged = np.asarray(flagged, dtype=bool)
    is_poison = np.asarray(is_poison, dtype=bool)
    n_poison = int(is_poison.sum())
    if n_poison == 0:
        return float("nan")
    return float((flagged & is_poison).sum() / n_poison)
