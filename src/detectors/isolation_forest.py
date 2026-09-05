"""Isolation Forest, as a tabular-native comparator.

WHY THIS EXISTS. Every detector in this study is vision-built. The only non-vision
comparator on the board is a univariate z-filter
(scripts/37_univariate_zfilter_baseline.py), which is a deliberately trivial baseline.
Severi et al. 2021's own most-effective mitigation against explanation-guided poisoning
is an isolation-based outlier filter, and this paper cites that work while never running
its defense. A reviewer can name that gap in one sentence, so this closes it.

SIGN CONVENTION, AND IT IS THE EASY THING TO GET WRONG. scikit-learn's
`IsolationForest.score_samples` returns the opposite of what this project expects: its
values are HIGHER for rows the forest considers NORMAL. Every decision rule here reads the
UPPER tail as suspicious -- `flag_by_scores` takes the top-k
(src/detectors/spectral.py:35) and `flag_by_mad_threshold` flags the upper tail
(src/detectors/spectral.py:48). Returning sklearn's score unmodified would turn this into
a perfect anti-detector whose recall looked like a finding rather than a bug, so
`isolation_scores` negates, and tests/detectors/test_isolation_forest.py asserts the
direction on planted outliers.

WHAT THIS IS NOT. Isolation Forest is an unsupervised outlier detector, not a backdoor
detector. It knows nothing about triggers, target classes or poison fractions. It is here
as a comparator that is native to tabular data, which is the axis the paper is missing,
and its result should be read that way.
"""

from typing import Dict

import numpy as np
from sklearn.ensemble import IsolationForest

from src import config

__all__ = ["isolation_scores", "isolation_diagnostics"]

# Fixed before any run. n_estimators is sklearn's default; max_samples 256 is the value
# in Liu et al.'s original paper and sklearn's "auto". contamination is left at "auto"
# deliberately: this detector is scored through the project's own two decision rules, so
# letting sklearn pick its own cut would apply a third, undeclared rule.
N_ESTIMATORS = 100
MAX_SAMPLES = 256
CONTAMINATION = "auto"


def isolation_scores(x: np.ndarray, seed: int = config.SEEDS[0]) -> np.ndarray:
    """Per-row suspicion score. HIGHER means more anomalous, matching this project's rules.

    `x` is the benign-conditioned block in standardized space, the same matrix the
    activation-based detectors score, so the comparator sees the data the others see.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D block, got shape {x.shape}")
    if x.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)

    forest = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=min(MAX_SAMPLES, x.shape[0]),
        contamination=CONTAMINATION,
        random_state=seed,
        n_jobs=1,          # determinism, and this runs beside other CPU work
    )
    forest.fit(x)
    # Negation is the whole point. See the module docstring.
    return -np.asarray(forest.score_samples(x), dtype=np.float64)


def isolation_diagnostics(scores: np.ndarray, is_poison: np.ndarray) -> Dict:
    """Separation summary, for the note rather than for any decision rule."""
    scores = np.asarray(scores, dtype=np.float64)
    is_poison = np.asarray(is_poison, dtype=bool)
    if scores.shape[0] != is_poison.shape[0]:
        raise ValueError("scores and is_poison must align")
    if not is_poison.any() or is_poison.all():
        return {"mean_poison": None, "mean_clean": None, "separation": None}
    mp = float(scores[is_poison].mean())
    mc = float(scores[~is_poison].mean())
    return {"mean_poison": mp, "mean_clean": mc, "separation": mp - mc}
