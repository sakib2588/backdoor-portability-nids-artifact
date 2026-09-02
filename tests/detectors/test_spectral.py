import numpy as np

from src.detectors import poison_recall
from src.detectors.spectral import spectral_scores, flag_by_scores, flag_by_mad_threshold


def test_spectral_flags_planted_outliers():
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 1, size=(190, 20))
    poison = rng.normal(0, 1, size=(10, 20)) + 8.0  # shifted sub-population
    feats = np.vstack([clean, poison])
    is_poison = np.array([False] * 190 + [True] * 10)

    scores = spectral_scores(feats)
    flagged = flag_by_scores(scores, expected_frac=0.05)
    assert poison_recall(flagged, is_poison) >= 0.9


def test_topk_recovers_backdoor_spread_across_components():
    """Regression test for the M1 Spectral fix (2026-07-14).

    Vanilla Spectral Signatures scores on the single top singular vector. When the
    backdoor signal is not confined to v1 -- as on the MNIST vision control, where a
    ~13% tail of poison projected weakly onto v1 and carried its signature on v2-v5
    -- single-vector scoring misses that portion of the poison. Scoring over the
    top-k subspace (n_components>1) recovers it.

    This fixture reproduces the principle cleanly with a HETEROGENEOUS trigger
    response: one half of the poison responds on axis 0, the other half on axis 1,
    so the backdoor occupies two orthogonal singular vectors. top-1 (v1) isolates
    only the half on that axis; top-k spanning both recovers all of it. The test
    pins the behaviour so the detector cannot silently regress to top-1-only.
    See notes/20260714-decision-spectral-ceiling.md for the full diagnostic.
    """
    rng = np.random.default_rng(0)
    n_clean, n_poison, d = 190, 10, 20
    feats = rng.normal(0, 1, size=(n_clean + n_poison, d))
    is_poison = np.array([False] * n_clean + [True] * n_poison)
    half = n_poison // 2
    feats[n_clean:n_clean + half, 0] += 8.0       # half the poison respond on axis 0 -> v1
    feats[n_clean + half:, 1] += 8.0              # the other half on axis 1 -> v2

    top1 = poison_recall(flag_by_scores(spectral_scores(feats, n_components=1),
                                        expected_frac=0.05), is_poison)
    topk = poison_recall(flag_by_scores(spectral_scores(feats, n_components=2),
                                        expected_frac=0.05), is_poison)
    assert top1 < 0.9     # v1 alone does not fully isolate the spread-out poison
    assert topk >= 0.9    # spanning v1+v2 recovers the rest
    assert topk > top1    # top-k strictly improves on single-vector scoring


def test_mad_threshold_flags_planted_outliers_without_a_budget():
    """flag_by_mad_threshold takes no expected_frac -- unlike flag_by_scores, it must find the
    same well-separated outliers from the score distribution alone."""
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 1, size=(190, 20))
    poison = rng.normal(0, 1, size=(10, 20)) + 8.0
    feats = np.vstack([clean, poison])
    is_poison = np.array([False] * 190 + [True] * 10)

    scores = spectral_scores(feats)
    flagged = flag_by_mad_threshold(scores, z_thresh=3.0)
    assert poison_recall(flagged, is_poison) >= 0.9


def test_mad_threshold_at_a_starved_fixed_budget_still_recovers_recall():
    """The scenario this function exists for: a fixed top-k budget (flag_by_scores) sized for a much
    lower expected_frac than the true poison fraction starves and misses most of the poison, even
    though the score distribution separates poison from clean almost perfectly. The MAD threshold,
    having no budget, is not starved by an under-estimated expected_frac."""
    rng = np.random.default_rng(1)
    n_clean, n_poison, d = 990, 10, 20   # true poison frac = 1%
    feats = rng.normal(0, 1, size=(n_clean + n_poison, d))
    is_poison = np.array([False] * n_clean + [True] * n_poison)
    feats[n_clean:] += 8.0

    scores = spectral_scores(feats)
    starved = flag_by_scores(scores, expected_frac=0.001)   # budget sized for 0.1%, true is 1%
    adaptive = flag_by_mad_threshold(scores, z_thresh=3.0)

    starved_recall = poison_recall(starved, is_poison)
    adaptive_recall = poison_recall(adaptive, is_poison)
    assert starved_recall < 0.5      # the starved fixed budget misses most of the poison
    assert adaptive_recall >= 0.9    # the adaptive threshold is not budget-limited
    assert adaptive_recall > starved_recall


def test_mad_threshold_degenerate_zero_mad_falls_back_to_std():
    """When >=half the scores are identical, MAD is 0; must fall back to a std-based z-score rather
    than dividing by zero (which would flag everything as z=inf)."""
    scores = np.array([1.0] * 60 + [1.0] * 30 + [50.0] * 10)   # 90/100 identical -> MAD=0
    flagged = flag_by_mad_threshold(scores, z_thresh=3.0)
    assert not np.any(np.isnan(flagged.astype(float)))
    assert flagged[-10:].all()          # the shifted outliers are still flagged
    assert not flagged[:90].any()       # the identical bulk is not flagged
