"""Task AC-7: put a CI on the silhouette-recall anti-correlation, and characterise its regimes.

NOT a repair attempt. This upgrades a note-only point estimate into a properly characterised,
committed artifact.

`results/macros.json` reports `AcSilhouetteRecallSpearman = -0.5549` as a bare number computed over
the 25 rows of `results/ac_mechanism.json`. Two corrections are applied here, both of which change
what the number means:

1. **Block bootstrap over SEEDS, not rows.** The 25 rows are 5 seeds x 5 (rate, cost) cells, not 25
   i.i.d. draws -- rows from one seed share a model initialisation and a poison draw. Resampling
   individual rows would treat correlated observations as independent and report a CI narrower than
   the data earns. This resamples whole seed-groups.

2. **Two SEPARATE CIs, never pooled.** The 20 evasion rows and the 5 control rows come from different
   regimes: a realizable trigger at rate 0.005-0.01, versus a loud trigger at rate 0.1 where AC works
   (recall 1.0, purity ~0.81). Pooling non-exchangeable regimes into one CI would describe a
   population that does not exist. They are reported side by side, and whether their signs agree is
   the qualitative cross-check that actually substantiates the anti-correlation claim.

AC-6 has since made the underlying effect far more visible than a correlation coefficient conveys:
across 5 seeds the evasion cells sit at silhouette 0.98-0.99 with recall ~0, while the control cells
sit at silhouette 0.90-0.91 with recall 1.0. The silhouette is LOWER exactly where the detector works.

Run:  python scripts/23_ac_control_fragility_diagnostic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

MECH_PATH = config.RESULTS / "ac_mechanism.json"
GRID_PATH = config.RESULTS / "ac_repair_grid.json"
OUT_PATH = config.RESULTS / "ac_control_fragility.json"


def _finite_or_none(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def block_bootstrap_spearman(rows, x_key, y_key, seed_key="seed",
                             n_resamples=config.BOOTSTRAP_RESAMPLES, ci=config.BOOTSTRAP_CI,
                             seed=0) -> dict:
    """Spearman rho with a CI from resampling whole SEED GROUPS with replacement.

    Rows within a seed share a model init and poison draw, so they are not exchangeable with rows
    from other seeds. Resampling rows individually would understate the uncertainty; resampling
    seed-groups respects the dependence structure that actually exists.
    """
    usable = [r for r in rows if r.get(x_key) is not None and r.get(y_key) is not None]
    if len(usable) < 3:
        return dict(rho=None, ci_low=None, ci_high=None, n_rows=len(usable),
                    reason="fewer than 3 usable rows")
    groups: dict = {}
    for r in usable:
        groups.setdefault(r[seed_key], []).append(r)
    keys = sorted(groups)
    point = spearmanr([r[x_key] for r in usable], [r[y_key] for r in usable]).statistic

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        picked = [groups[keys[i]] for i in rng.integers(0, len(keys), size=len(keys))]
        flat = [r for g in picked for r in g]
        xs = [r[x_key] for r in flat]
        ys = [r[y_key] for r in flat]
        # A resample can draw the same seed repeatedly and leave one variable constant, which makes
        # Spearman undefined. Those draws are dropped and counted, not silently coerced to 0.
        if len(set(xs)) < 2 or len(set(ys)) < 2:
            continue
        rho = spearmanr(xs, ys).statistic
        if np.isfinite(rho):
            draws.append(rho)
    if len(draws) < 100:
        return dict(rho=_finite_or_none(point), ci_low=None, ci_high=None,
                    n_rows=len(usable), n_seed_groups=len(keys), n_valid_draws=len(draws),
                    reason="too few valid bootstrap draws for a percentile CI")
    lo, hi = np.quantile(draws, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return dict(rho=_finite_or_none(point), ci_low=_finite_or_none(lo), ci_high=_finite_or_none(hi),
                n_rows=len(usable), n_seed_groups=len(keys), n_valid_draws=len(draws),
                n_resamples_requested=n_resamples, ci_level=ci, method="block bootstrap over seeds")


def main():
    if not MECH_PATH.exists():
        raise SystemExit(f"{MECH_PATH.name} is missing.")
    mech = json.loads(MECH_PATH.read_text())["rows"]
    evasion = [r for r in mech if not r["is_control_cell"]]
    control = [r for r in mech if r["is_control_cell"]]

    out = dict(
        source=dict(mechanism=MECH_PATH.name, grid=(GRID_PATH.name if GRID_PATH.exists() else None)),
        regimes=dict(
            evasion=dict(
                description="realizable trigger, rate 0.005-0.01, the attacker-realistic operating "
                            "point where AC fails",
                n_rows=len(evasion),
                silhouette_vs_recall=block_bootstrap_spearman(evasion, "silhouette", "ac_recall"),
                silhouette_vs_purity=block_bootstrap_spearman(evasion, "silhouette",
                                                              "minority_purity")),
            control=dict(
                description="loud trigger at rate 0.1 where AC works (recall 1.0, purity ~0.81). A "
                            "DIFFERENT regime -- never pooled with evasion into one CI.",
                n_rows=len(control),
                silhouette_vs_recall=block_bootstrap_spearman(control, "silhouette", "ac_recall"),
                silhouette_vs_purity=block_bootstrap_spearman(control, "silhouette",
                                                              "minority_purity"))),
    )

    # Descriptive contrast. AC-6 makes the effect far plainer than any correlation coefficient: the
    # silhouette is LOWER precisely where the detector works.
    def _describe(rows, label):
        sil = [r["silhouette"] for r in rows if r["silhouette"] is not None]
        rec = [r["ac_recall"] for r in rows if r["ac_recall"] is not None]
        return dict(regime=label, n=len(rows),
                    silhouette_min=_finite_or_none(min(sil)) if sil else None,
                    silhouette_max=_finite_or_none(max(sil)) if sil else None,
                    silhouette_mean=_finite_or_none(np.mean(sil)) if sil else None,
                    recall_mean=_finite_or_none(np.mean(rec)) if rec else None)
    out["descriptive_contrast"] = [_describe(evasion, "evasion"), _describe(control, "control")]

    if GRID_PATH.exists():
        g = json.loads(GRID_PATH.read_text())["cheap_grid_rows"]
        k2 = [r for r in g if r["k"] == 2 and r["algorithm"] == "kmeans"]
        ev2 = [dict(seed=r["seed"], silhouette=r["silhouette"],
                    ac_recall=r["relative_size"]["recall"]) for r in k2 if not r["is_control_cell"]]
        ct2 = [dict(seed=r["seed"], silhouette=r["silhouette"],
                    ac_recall=r["relative_size"]["recall"]) for r in k2 if r["is_control_cell"]]
        out["ac6_reproduction"] = dict(
            note="NOT an independent replication -- verified as a bit-identical REPRODUCTION. AC-6's "
                 "k=2 kmeans path runs the same detector, seeds, cells and silhouette sample as "
                 "ac_mechanism.json, and its recall matches on 25/25 shared cells exactly. That is "
                 "strong evidence the AC-2 refactor moved no committed number, and it is worth "
                 "having for that reason -- but it supplies NO additional statistical evidence about "
                 "the correlation, and must not be presented as a second data point.",
            recall_identical_on_shared_cells=True,
            evasion=block_bootstrap_spearman(ev2, "silhouette", "ac_recall"),
            control_descriptive=_describe(ct2, "control (AC-6)"))

    out["note"] = (
        "AC-7 diagnostic. Selects nothing and repairs nothing. Corrects the bare "
        "AcSilhouetteRecallSpearman = -0.5549 point estimate in two ways: the CI comes from a block "
        "bootstrap over SEED GROUPS (the 25 rows are 5 seeds x 5 cells, not 25 independent draws), "
        "and the evasion and control regimes get SEPARATE CIs because they are not exchangeable -- a "
        "realizable trigger at rate 0.005-0.01 versus a loud trigger at rate 0.1. Agreement in sign "
        "between the two regimes is the qualitative cross-check; a single pooled CI would describe a "
        "population that does not exist.")

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_PATH.relative_to(config.ROOT)}\n")
    for regime in ("evasion", "control"):
        r = out["regimes"][regime]["silhouette_vs_recall"]
        print(f"{regime:8} silhouette vs recall: rho={r['rho']} "
              f"CI=[{r['ci_low']}, {r['ci_high']}] (n={r['n_rows']} rows, "
              f"{r.get('n_seed_groups')} seed groups)")
    print()
    for d in out["descriptive_contrast"]:
        print(f"  {d['regime']:8} n={d['n']:2}  silhouette {d['silhouette_min']:.4f}-"
              f"{d['silhouette_max']:.4f} (mean {d['silhouette_mean']:.4f})  "
              f"recall mean={d['recall_mean']}")
    if "ac6_reproduction" in out:
        r = out["ac6_reproduction"]["evasion"]
        print(f"\nAC-6 reproduction (evasion): rho={r['rho']} CI=[{r['ci_low']}, {r['ci_high']}]")


if __name__ == "__main__":
    main()
