#!/usr/bin/env python
"""Post-hoc diagnostic for the NetFlow study: does projection clipping decide the realization mode?

NOT PRE-REGISTERED. Written 2026-09-03 after the primary corpus completed, when the per-seed table
showed the clean-stamped ASR to be bimodal within every benign-share level rather than moving with
the share. It is reported as exploratory wherever it is quoted, alongside the pre-registered
INCONCLUSIVE/REFUTED verdict from scripts/82, never in place of it.

The observation it tests: a cell's clean model either already accepts the watermarked flow (clean
stamped ASR near 1, the recipe realizes as evasion) or does not (near 0, the recipe must plant a
backdoor). The constraint projection clips some of the 16 stamped features on every trigger, and the
number it clips varies by cell. If heavily clipped triggers are the ones the clean model still calls
attack, then realizability is what decides the mode, which is this paper's own thread.

Three statistics, all rank-based with permutation nulls, no parametric test:
  clip count vs clean-stamped ASR (Spearman, permutation p)
  clip count in saturated vs unsaturated cells (Mann-Whitney, exact for these sizes)
  ranking AUC of clip count as a predictor of the backdoor mode
plus clip count vs benign share, to expose any indirect share pathway the primary test could miss.

Cells are not fully exchangeable (two corpora, six levels, five seeds), so the permutation p-values
are indicative. The per-corpus breakdown is printed for that reason.

Run:  .venv/bin/python scripts/83_netflow_clipping_diagnostic.py
      .venv/bin/python scripts/83_netflow_clipping_diagnostic.py --from-checkpoint
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

SWEEP = config.RESULTS / "netflow_poison_sweep.json"
CKPT = config.RESULTS / "netflow_poison_sweep.checkpoint.json"
OUT = config.RESULTS / "netflow_clipping_diagnostic.json"
N_PERM = 20_000
SEED = 20260903


def perm_spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    rho = float(spearmanr(x, y).statistic)
    null = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(N_PERM)])
    return dict(rho=rho, p_permutation=float((np.abs(null) >= abs(rho) - 1e-12).mean()),
                n=int(len(x)), n_permutations=N_PERM)


def analyze(cells: list, label: str, rng: np.random.Generator) -> dict:
    clip = np.array([c["trigger_features_clipped_by_projection"] for c in cells], float)
    cs = np.array([c["clean_model_stamped_asr"] for c in cells], float)
    sat = np.array([c["status"] == "saturated" for c in cells])
    share = np.array([c["benign_share"] for c in cells], float)
    out = dict(label=label, n_cells=int(len(cells)), n_saturated=int(sat.sum()),
               clip_vs_clean_stamped_asr=perm_spearman(clip, cs, rng),
               clip_vs_benign_share=perm_spearman(share, clip, rng))
    if sat.any() and (~sat).any():
        u = mannwhitneyu(clip[sat], clip[~sat], alternative="two-sided")
        out["clip_saturated_vs_backdoor"] = dict(
            median_saturated=float(np.median(clip[sat])),
            median_backdoor=float(np.median(clip[~sat])),
            mannwhitney_p=float(u.pvalue))
        out["auc_clip_predicts_backdoor"] = float(roc_auc_score(~sat, clip))
    out["mode_by_clip_count"] = {
        int(k): dict(saturated=int((sat & (clip == k)).sum()), backdoor=int((~sat & (clip == k)).sum()))
        for k in np.unique(clip)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-checkpoint", action="store_true")
    args = ap.parse_args()
    src = CKPT if args.from_checkpoint else SWEEP
    if not src.exists():
        print(f"missing {src}")
        return 1
    blob = json.loads(src.read_text())
    cells = [c for c in blob["cells"].values()
             if c["arm"] == "Q1" and c["cost"] == config.NETFLOW_PRIMARY_COST and c["asr"] is not None]
    rng = np.random.default_rng(SEED)

    result = dict(
        preregistered=False,
        note="Post-hoc diagnostic. Report beside the pre-registered verdict in "
             "results/netflow_property_analysis.json, never instead of it.",
        source=src.name, primary_cost=config.NETFLOW_PRIMARY_COST,
        pooled=analyze(cells, "both corpora", rng),
        per_corpus={c: analyze([x for x in cells if x["corpus"] == c], c, rng)
                    for c in (config.NETFLOW_MANIPULATION_CORPUS, config.NETFLOW_REPLICATION_CORPUS)
                    if any(x["corpus"] == c for x in cells)},
    )
    OUT.write_text(json.dumps(result, indent=2))

    for block in [result["pooled"], *result["per_corpus"].values()]:
        a = block["clip_vs_clean_stamped_asr"]; b = block["clip_vs_benign_share"]
        print(f"\n{block['label']}  (n={block['n_cells']}, saturated {block['n_saturated']})")
        print(f"  clip count vs clean-stamped ASR   rho {a['rho']:+.3f}  p {a['p_permutation']:.4f}")
        print(f"  clip count vs benign share        rho {b['rho']:+.3f}  p {b['p_permutation']:.4f}")
        if "auc_clip_predicts_backdoor" in block:
            m = block["clip_saturated_vs_backdoor"]
            print(f"  median clipped: saturated {m['median_saturated']:.0f}, backdoor "
                  f"{m['median_backdoor']:.0f}  Mann-Whitney p {m['mannwhitney_p']:.4f}")
            print(f"  AUC, clip count predicting backdoor mode: {block['auc_clip_predicts_backdoor']:.3f}")
    print(f"\nwrote {OUT.relative_to(config.ROOT)}  [post-hoc, not pre-registered]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
