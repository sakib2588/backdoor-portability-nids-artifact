#!/usr/bin/env python3
"""One-shot migration: stamp the constraint-manifest fingerprint into existing checkpoints.

WHY THIS EXISTS (2026-09-03). Every `config_key` in the UNSW-NB15 pipeline captured rates, costs,
epochs, sigma and the like, but nothing about the constraint manifest. Tightening or re-tagging a
constraint therefore left every stored key still matching, and `load_checkpoint` would resume from
cells computed under the OLD manifest without a word -- the silent-stale-reuse failure the project's
checkpoint rule exists to prevent. `src.constraints_secondary.manifest_fingerprint()` now enters the
key of every script that projects onto the feasible set.

Adding a field to a key invalidates every checkpoint that predates it. The cells in those files were
computed under the manifest that is on disk right now, unchanged, so stamping the CURRENT fingerprint
into them is a statement of fact rather than a rewrite of history -- and it preserves hours of
completed compute that would otherwise be discarded on the next run.

Safe to re-run: a checkpoint that already carries a fingerprint is left alone, and a file whose
stored fingerprint DISAGREES with the current manifest is reported and NOT touched, because that is
exactly the stale-reuse case the guard is for.

Run:  python scripts/76_migrate_checkpoint_constraint_fingerprint.py [--apply]
Without --apply it is a dry run and writes nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.constraints_secondary import manifest_fingerprint

# Checkpoints written by scripts that project onto the UNSW-NB15 feasible set. A checkpoint whose
# script never touches the manifest is deliberately absent: stamping it would imply a dependency
# that does not exist.
CONSTRAINT_DEPENDENT = (
    "secondary_poison_sweep",
    "secondary_detectors",
    "secondary_adaptive_threshold",
    "secondary_adaptive_threshold_replication",
    "secondary_adaptive_threshold_batch3",
    "secondary_threshold_margin",
    "secondary_threshold_margin_pilot",
    "secondary_margin_decomposition",
    "secondary_random_feature_evasion_control",
    "secondary_post_removal_asr",
    "strip_detector",
    "strip_detector_smoke",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the files (default is a dry run)")
    args = ap.parse_args()

    fp = manifest_fingerprint()
    print(f"current manifest fingerprint: {fp}")
    print(f"mode: {'APPLY' if args.apply else 'DRY RUN (no writes)'}\n")

    stamped = skipped = absent = conflict = 0
    for name in CONSTRAINT_DEPENDENT:
        path = config.RESULTS / f"{name}.checkpoint.json"
        if not path.exists():
            print(f"  absent    {name}")
            absent += 1
            continue
        try:
            blob = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  UNREADABLE {name}: {exc}")
            continue
        key = blob.get("config_key")
        if not isinstance(key, dict):
            print(f"  no key    {name} (nothing to stamp)")
            skipped += 1
            continue
        existing = key.get("constraints")
        if existing == fp:
            print(f"  current   {name}")
            skipped += 1
            continue
        if existing is not None:
            print(f"  CONFLICT  {name}: stored {existing} != current {fp} -- left alone, "
                  f"this checkpoint predates a manifest change and must be recomputed")
            conflict += 1
            continue
        key["constraints"] = fp
        blob["config_key"] = key
        if args.apply:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(blob))
            tmp.replace(path)
        print(f"  stamped   {name}")
        stamped += 1

    print(f"\nstamped {stamped}, already current {skipped}, absent {absent}, conflicts {conflict}")
    if not args.apply and stamped:
        print("dry run only -- re-run with --apply to write")
    return 1 if conflict else 0


if __name__ == "__main__":
    raise SystemExit(main())
