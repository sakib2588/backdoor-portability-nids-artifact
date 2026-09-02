#!/usr/bin/env python3
"""Consolidate every number the reframe will quote into one traceable manifest.

This manuscript has no LaTeX macro layer, so each number in the .tex is a hand-typed
literal. Cycle 2 found 22 result JSONs that had never been folded into the advisory
macro file, and the project's standing rule is that every literal traces to disk.
The reframe adds a whole contribution's worth of new numbers at once, so they get
collected and named HERE, before any of them is typed into a section file, and the
prose is written from this manifest rather than from a planning note.

Reads only committed experiment manifests. Computes nothing that those files do not
already contain, except the two rounding forms the prose needs (percentages and the
uniform-baseline multiple), so there is no second code path that could disagree.

Run:  python scripts/45_reframe_manifest.py [--write-manifest]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config

SOURCES = {
    "primary_decomposition": "clean_model_stamped_asr.json",
    "secondary_decomposition": "secondary_clean_model_stamped_asr.json",
    "random_control": "random_feature_evasion_control.json",
    "displacement": "projection_displacement_diagnostic.json",
    "mechanism": "dissociation_mechanism_probe.json",
}


def load(name):
    path = config.RESULTS / name
    if not path.exists():
        raise SystemExit(f"missing required manifest: {path}")
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()

    src = {k: load(v) for k, v in SOURCES.items()}
    p16 = src["primary_decomposition"]["summary"]["16"]
    p8 = src["primary_decomposition"]["summary"]["8"]
    s16 = src["secondary_decomposition"]["summary"]["16"]
    s8 = src["secondary_decomposition"]["summary"]["8"]
    rnd = src["random_control"]
    disp = src["displacement"]
    mech = src["mechanism"]

    ctu_share = mech["ctu_mean_attribution_share"]
    unsw_share = mech["unsw_mean_attribution_share"]
    ctu_D = mech["per_seed"]["ctu_13"][0]["n_features"]
    unsw_D = mech["per_seed"]["unsw_nb15"][0]["n_features"]
    ctu_uniform = mech["per_seed"]["ctu_13"][0]["uniform_baseline"]
    unsw_uniform = mech["per_seed"]["unsw_nb15"][0]["uniform_baseline"]

    out = {
        "provenance": {k: f"results/{v}" for k, v in SOURCES.items()},
        "verdicts": {
            "primary": src["primary_decomposition"]["verdict"],
            "secondary": src["secondary_decomposition"]["verdict"],
            "random_control": rnd["verdict"],
            "displacement": disp["verdict"],
            "mechanism": mech["verdict"],
        },
        "dissociation_table_cost16": {
            "ctu_clean_plain": p16["a_clean_plain_mean"],
            "ctu_clean_stamped": p16["a_clean_stamped_mean"],
            "ctu_poisoned_stamped": p16["a_poisoned_stamped_mean"],
            "ctu_evasion_E": p16["evasion_component_E"],
            "ctu_backdoor_B": p16["backdoor_component_B"],
            "unsw_clean_plain": s16["a_clean_plain_mean"],
            "unsw_clean_stamped": s16["a_clean_stamped_mean"],
            "unsw_poisoned_stamped": s16["a_poisoned_stamped_mean"],
            "unsw_evasion_E": s16["evasion_component_E"],
            "unsw_backdoor_B": s16["backdoor_component_B"],
        },
        # Cost 8 carries the same five quantities as cost 16, because it is the operating
        # point where neither CTU-13 arm saturates. The cost-16 column has both CTU arms at
        # 1.0, so its backdoor component of zero is bounded by construction; at cost 8 both
        # components are measured on both datasets. The spread matters as much as the mean
        # here: CTU-13's clean-stamped rate ranges from 0.113 to 1.0 across the five seeds,
        # which cost 16 does not.
        "dissociation_table_cost8": {
            "ctu_clean_plain": p8["a_clean_plain_mean"],
            "ctu_clean_stamped": p8["a_clean_stamped_mean"],
            "ctu_poisoned_stamped": p8["a_poisoned_stamped_mean"],
            "ctu_evasion_E": p8["evasion_component_E"],
            "ctu_backdoor_B": p8["backdoor_component_B"],
            "ctu_clean_stamped_per_seed": p8["per_seed_stamped"],
            "ctu_clean_stamped_min": min(p8["per_seed_stamped"]),
            "ctu_clean_stamped_max": max(p8["per_seed_stamped"]),
            "unsw_clean_plain": s8["a_clean_plain_mean"],
            "unsw_clean_stamped": s8["a_clean_stamped_mean"],
            "unsw_poisoned_stamped": s8["a_poisoned_stamped_mean"],
            "unsw_evasion_E": s8["evasion_component_E"],
            "unsw_backdoor_B": s8["backdoor_component_B"],
        },
        "cost8_context": {
            "ctu_clean_stamped": p8["a_clean_stamped_mean"],
            "ctu_poisoned_stamped": p8["a_poisoned_stamped_mean"],
            "ctu_backdoor_B": p8["backdoor_component_B"],
            "unsw_backdoor_B": s8["backdoor_component_B"],
        },
        "random_control": {
            "shap_trigger_clean_asr": rnd["shap_trigger_clean_asr_mean"],
            "random_trigger_clean_asr": rnd["random_trigger_clean_asr_mean"],
            "random_trigger_sd": rnd["random_trigger_clean_asr_sd"],
            "n_draws": rnd["n_draws_per_seed"] * 5,
        },
        "displacement": {
            "ctu_sigma": disp["ctu_mean_retention"] * disp["intended_sigma"],
            "unsw_sigma": disp["unsw_mean_retention"] * disp["intended_sigma"],
            "ctu_retention": disp["ctu_mean_retention"],
            "unsw_retention": disp["unsw_mean_retention"],
            "intended_sigma": disp["intended_sigma"],
        },
        "mechanism": {
            "ctu_delta_p": mech["ctu_mean_delta_p"],
            "unsw_delta_p": mech["unsw_mean_delta_p"],
            "ctu_attribution_share": ctu_share,
            "unsw_attribution_share": unsw_share,
            "ctu_n_features": ctu_D,
            "unsw_n_features": unsw_D,
            "ctu_multiple_of_uniform": ctu_share / ctu_uniform,
            "unsw_multiple_of_uniform": unsw_share / unsw_uniform,
        },
        "caveats": {
            "ceiling": ("At cost 16 both CTU arms saturate at 1.0, so B = 0 there is bounded "
                        "by construction and does not evidence the absence of a backdoor. B is "
                        "interpretable only where evasion leaves headroom, which is cost 8."),
            "attribution_descriptive": mech["note"],
            "random_pool": ("The committed random control drew from the equality-eligible set "
                            "without excluding zero-variance columns, so each 16-feature draw "
                            "carried about 3.4 constant features in expectation. The verdict is "
                            "unaffected at 0.058 against 1.000, and the pool is cleaned in "
                            "Experiment E."),
        },
    }

    print("Reframe manifest, every number the new prose may quote:\n")
    print(f"  verdicts: {out['verdicts']}\n")
    d = out["dissociation_table_cost16"]
    print(f"  {'quantity':<22}{'CTU-13':>12}{'UNSW-NB15':>12}")
    for label, a, b in [("clean, unstamped", "ctu_clean_plain", "unsw_clean_plain"),
                        ("clean, stamped", "ctu_clean_stamped", "unsw_clean_stamped"),
                        ("poisoned, stamped", "ctu_poisoned_stamped", "unsw_poisoned_stamped"),
                        ("evasion E", "ctu_evasion_E", "unsw_evasion_E"),
                        ("backdoor B", "ctu_backdoor_B", "unsw_backdoor_B")]:
        print(f"  {label:<22}{d[a]:>12.4f}{d[b]:>12.4f}")
    m = out["mechanism"]
    print(f"\n  dP: CTU {m['ctu_delta_p']:+.4f}   UNSW {m['unsw_delta_p']:+.4f}")
    print(f"  attribution share: CTU {m['ctu_attribution_share']:.4f} "
          f"({m['ctu_multiple_of_uniform']:.1f}x uniform)   "
          f"UNSW {m['unsw_attribution_share']:.4f} ({m['unsw_multiple_of_uniform']:.1f}x)")
    dd = out["displacement"]
    print(f"  surviving displacement: CTU {dd['ctu_sigma']:.3f} sigma   UNSW {dd['unsw_sigma']:.3f} sigma")
    r = out["random_control"]
    print(f"  random control: {r['random_trigger_clean_asr']:.4f} against SHAP "
          f"{r['shap_trigger_clean_asr']:.4f}")

    if args.write_manifest:
        path = config.RESULTS / "reframe_manifest.json"
        path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {path.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
