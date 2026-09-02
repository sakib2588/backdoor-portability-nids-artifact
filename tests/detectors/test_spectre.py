import numpy as np

from src.detectors import poison_recall
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores, spectral_scores
from src.detectors.spectre import spectre_scores


def _planted(n_clean=400, n_poison=20, dim=24, shift=6.0, seed=0):
    """Anisotropic clean bulk plus a compact shifted poison cluster.

    The clean covariance is deliberately stretched along one axis, which is the case SPECTRE exists
    to handle: a raw top-singular-vector score mostly reads that stretch, while whitening removes it.
    """
    rng = np.random.default_rng(seed)
    scale = np.ones(dim)
    scale[0] = 12.0                      # dominant clean direction, unrelated to the poison
    clean = rng.normal(0, 1, size=(n_clean, dim)) * scale
    direction = np.zeros(dim)
    direction[3] = 1.0                   # poison displaces along a different, minor axis
    poison = rng.normal(0, 1, size=(n_poison, dim)) * scale + shift * direction
    feats = np.vstack([clean, poison])
    is_poison = np.array([False] * n_clean + [True] * n_poison)
    return feats, is_poison


def test_spectre_ranks_poison_above_clean():
    feats, is_poison = _planted()
    scores = spectre_scores(feats)
    assert scores.shape == (feats.shape[0],)
    assert np.isfinite(scores).all()
    assert scores[is_poison].mean() > scores[~is_poison].mean()


def test_spectre_recovers_poison_under_both_decision_rules():
    feats, is_poison = _planted()
    scores = spectre_scores(feats, expected_frac=float(is_poison.mean()))
    budget_flag = flag_by_scores(scores, expected_frac=float(is_poison.mean()))
    mad_flag = flag_by_mad_threshold(scores, z_thresh=3.0)
    assert poison_recall(budget_flag, is_poison) >= 0.5
    assert poison_recall(mad_flag, is_poison) >= 0.5


def test_spectre_beats_raw_spectral_when_clean_variance_is_anisotropic():
    """The case the whitening step is for. Not a claim about tabular NIDS, only that this module
    implements the geometry it says it does."""
    feats, is_poison = _planted()
    frac = float(is_poison.mean())
    spec = poison_recall(flag_by_scores(spectral_scores(feats, n_components=5), frac), is_poison)
    spect = poison_recall(flag_by_scores(spectre_scores(feats, expected_frac=frac), frac), is_poison)
    assert spect >= spec


def test_spectre_is_deterministic():
    feats, _ = _planted()
    a = spectre_scores(feats)
    b = spectre_scores(feats)
    assert np.array_equal(a, b)


def test_spectre_handles_degenerate_input():
    assert spectre_scores(np.zeros((0, 4))).shape == (0,)
    identical = np.ones((50, 6))
    out = spectre_scores(identical)
    assert out.shape == (50,)
    assert np.isfinite(out).all()
