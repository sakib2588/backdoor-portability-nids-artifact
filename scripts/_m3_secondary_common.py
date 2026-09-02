"""Shared setup for the secondary-dataset (UNSW-NB15) extension orchestration scripts -- Task 4's
`14_secondary_poison_sweep.py` and Task 5's detector-sweep successor. Mirrors `scripts/_m2_common.py`'s
scope and naming convention for the primary (CTU-13) pipeline's 02/03/04/05 scripts.

Trigger-eligibility policy (`EXCLUDED_TRIGGER_FEATURES`/`eligible_indices_for`), the smoke/full config
shape (`get_config`/`apply_overrides`), and the checkpoint-compatibility key convention (`cell_key`/
`config_key`) all live here so Task 5 -- which must retrain against this task's exact seed/trigger/
poison construction and self-check against its stored per-cell ASR/clean-accuracy numbers (the same
`determinism_ok` self-check pattern `scripts/05_run_detectors.py` uses against `scripts/04`'s numbers)
-- can import the identical policy directly, rather than reconstructing it by copy-paste (risking
silent drift) or reaching into `14_secondary_poison_sweep.py` via the `importlib` script-loading
trick. That trick is a TESTING convention in this codebase (see `tests/test_nc_repair_gate.py`,
`tests/test_secondary_poison_sweep.py`), not a documented production-reuse path -- this module is.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.constraints_secondary import manifest_fingerprint

# tcprtt's own projectable repair recomputes it (tcprtt := synack + ackdat) -- a trigger stamped onto
# tcprtt would be silently erased by projection. Excluded from trigger eligibility, the secondary
# analogue of CTU strategy A's equality-family exclusion (src/trigger.py's `equality_features`), but
# re-derived against src.constraints_secondary's own manifest per Task 4 Step 1's explicit requirement
# not to reuse CTU's EqualConstraint/equality_features() machinery (which does not even type-check
# against SecondaryConstraint objects).
EXCLUDED_TRIGGER_FEATURES = {"tcprtt"}


def eligible_indices_for(feature_names) -> list:
    return [i for i, n in enumerate(feature_names) if n not in EXCLUDED_TRIGGER_FEATURES]


def get_config(smoke: bool) -> dict:
    if smoke:
        return dict(smoke=True, seeds=[42], rates=(0.01,), costs=(16,),
                   mlp_epochs=10, control_rate=0.05, control_cost=8, n_sigma=6.0,
                   nc_sample=200, nc_steps=100)
    return dict(smoke=False, seeds=list(config.SEEDS), rates=config.SECONDARY_RATES,
               costs=config.SECONDARY_COSTS, mlp_epochs=20, control_rate=0.05, control_cost=8,
               n_sigma=6.0, nc_sample=400, nc_steps=300)


def apply_overrides(cfg: dict, argv) -> dict:
    """CLI overrides shared by every secondary-dataset orchestration script -- `--seeds a,b,c`,
    `--rates x,y`, `--costs m,n` all restrict to those values at the SAME (full or smoke) config."""
    if "--seeds" in argv:
        cfg["seeds"] = [int(s) for s in argv[argv.index("--seeds") + 1].split(",")]
    if "--rates" in argv:
        cfg["rates"] = tuple(float(s) for s in argv[argv.index("--rates") + 1].split(","))
    if "--costs" in argv:
        cfg["costs"] = tuple(int(s) for s in argv[argv.index("--costs") + 1].split(","))
    return cfg


def cell_key(seed, rate, cost) -> str:
    return f"{seed}|{rate}|{cost}"


def config_key(cfg: dict) -> dict:
    """Checkpoint compatibility key -- excludes the seed list (same convention as the primary
    poison_sweep script) so a single-seed smoke/calibration run's cells are reused by the full run.

    Renamed from `14_secondary_poison_sweep.py`'s original private `_config_key` when this function
    moved here (2026-07-31 module-boundary fix): a leading underscore signals "not for import
    elsewhere", which stopped being true the moment this became the shared cross-script convention
    Task 5 is expected to import directly."""
    return dict(smoke=cfg["smoke"], rates=list(cfg["rates"]), costs=list(cfg["costs"]),
               mlp_epochs=cfg["mlp_epochs"], control_rate=cfg["control_rate"],
               control_cost=cfg["control_cost"], n_sigma=cfg["n_sigma"],
               nc_sample=cfg["nc_sample"], nc_steps=cfg["nc_steps"],
               constraints=manifest_fingerprint())
