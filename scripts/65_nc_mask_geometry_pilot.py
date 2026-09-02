"""N1/N2/N3 of the Rung-2 repair pilots: does the SHAPE of the Neural Cleanse mask separate poisoned
victims from clean models, where the two-class norm RATIO does not?

WHY THIS IS NOT NC-6 AGAIN. NC-6 refuted one statistic, the mask ratio, and scripts/64 has since
refuted the whole scalar residue (best two-sided AUC 0.6291 against a permutation bar of 0.8186). But
the inversion still localises the planted trigger at 5.4-6.9x chance (results/nc_masks.json), so a
signal demonstrably survives -- it is the DECISION LAYER built on it that fails. That is the same
decomposition the Spectral repair used: keep the detector's computation, rebuild the statistic.

Binary NIDS is what starves the published rule. MAD-over-classes needs many classes to have a null;
with 2 classes it has 2 numbers. This pilot replaces the class dimension with a CLEAN-REPLICA
POPULATION: 34 clean models per seed, inverted identically, giving a 102-model null against which one
suspect model's mask geometry is scored. The replicas differ from the victim only in poisoning, so
the null is exchangeable under the hypothesis of no backdoor.

THREE STATISTICS, ONE RUN, ONE NULL (pre-registered in
notes/20260901-decision-rung2-repair-pilots-preregistration.md):

  N1 concentration     top-16 mass share, entropy, Gini of |mask|. A trigger-carrying inversion should
                       concentrate; a clean one should smear.
  N2 SHAP alignment    overlap of the mask's top-16 with the SUSPECT MODEL'S OWN SHAP top-16.
                       Defender-computable, no oracle knowledge of the trigger. Note this is a
                       deliberate test of Tabdoor's feature-importance confound, not a way around it:
                       the attack CHOSE the trigger by SHAP rank, so clean models may score high for
                       the same reason victims do. The pre-registration locks that reading in advance
                       ("confound-saturated"), so a null here is a mechanism finding, not a dud run.
  N3 restart stability std of per-start mask_l1 and mean pairwise cosine across starts. A real trigger
                       should pull every start to the same place; a clean model's starts should wander.

CONFIG. The MULTISTART candidate (n_starts=3), not baseline: N3 is undefined at one start, and NC-7
verified multistart as a clean test (unlike constraint_aware, whose penalty scale swamps the
objective). Single cell rate=0.005 cost=16 -- clean replicas are cell-independent, and it is the only
cell with stored victims to cross-check against.

ONE CODE PATH. The 3 calibration victims are re-inverted HERE, through the same
reverse_engineer_tabular_multistart call the null uses. results/nc_masks.json's masks came from
scripts/10_nc_masks.py (steps=300, nc_sample=400, different path) and are used only as a cosine
cross-check, never as the compared quantity -- comparing a victim from one path against a null from
another is exactly the class of error that produced two different numbers for one quantity earlier in
this project.

GATE IS DESCRIPTIVE, BY CONSTRUCTION. With 3 calibration victims, "flags >= 90% of backdoored models"
cannot be established: 3 of 3 gives a binomial 95% lower bound of 0.368. The pre-registered gate is
therefore a rank statement -- every victim above the clean 90th percentile -- reported with exact
ranks, the binomial CI and a permutation p. Read it as "consistent with separation", never as a
detection rate.

Run:  python scripts/65_nc_mask_geometry_pilot.py --probe --seeds 42   # timing probe FIRST
      python scripts/65_nc_mask_geometry_pilot.py --seeds 42,123,456   # calibration (resumable)
      python scripts/65_nc_mask_geometry_pilot.py --seeds 789,1337 --stage evaluation
"""
from __future__ import annotations

import os

# MUST precede `import torch`: cuBLAS reads this when CUDA is first initialised and ignores it
# afterwards, so assigning it later would silently do nothing. train_ft_transformer calls
# require_deterministic_cuda(), which refuses to run without it, because
# scaled_dot_product_attention's CUDA backward is non-deterministic by default (two same-seed runs
# measured 1.014e-04 apart). Setting it here keeps the script self-sufficient rather than relying on
# a shell prefix that is easy to forget on a thirty-hour run. The mlp arm never calls that function,
# so its results are unaffected and stay byte-reproducible.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup, apply_overrides
from src import config
from src.constraints_torch import make_constraint_penalty
from src.data import apply_standardiser
from src.detectors.neural_cleanse import (
    TabularInversionConfig,
    balanced_inversion_sample,
    reverse_engineer_tabular_multistart,
)
from src.models import attack_success_rate, clean_accuracy, train_ft_transformer, train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import apply_trigger, build_trigger, raw_trigger_stats, shap_rank_features

TARGET = config.ATTACK_TARGET

# Pinned by the pre-registration. multistart == 18a's CANDIDATES["multistart"], reproduced here as a
# literal so a future edit to 18a's grid cannot silently change what this pilot ran.
CANDIDATE = "multistart"
INVERSION_SPEC = dict(l1_weight=0.001, n_starts=3, constraint_weight=0.0)
CELL = (0.005, 16)          # rate, cost -- the only cell with stored victims in nc_masks.json
TOP_K = 16                  # trigger size at cost 16; the top-k window for N1/N2
CLEAN_PERCENTILE = 90.0     # the pre-registered rank bar

ARCH_CHOICES = ("mlp", "ft_transformer")


def output_paths(arch: str) -> dict:
    """Artifact paths for one victim family.

    Architecture-suffixed so a second family cannot overwrite the committed MLP artifacts -- the
    102-row null in nc_mask_geometry_pilot.json is finished work and a transformer run must not
    land on top of it. "mlp" keeps the ORIGINAL names exactly, so that committed run stays
    reproducible and resumable from its own checkpoint."""
    suffix = "" if arch == "mlp" else f"_{arch}"
    return {
        "out": config.RESULTS / f"nc_mask_geometry_pilot{suffix}.json",
        "ckpt": config.RESULTS / f"nc_mask_geometry_pilot{suffix}.checkpoint.json",
        "masks": config.RESULTS / f"nc_pilot_masks{suffix}",
    }


NC_MASKS_PATH = config.RESULTS / "nc_masks.json"


def train_victim(arch: str, x_std, y, seed: int, epochs: int, device: str, arch_kwargs: dict):
    """The single seam between this pilot and the victim family.

    Everything else in this script -- the inversion, the eight statistics, the clean-replica null,
    the rank gate -- is architecture-blind and stays untouched. That is the point: the transformer
    arm must differ from the MLP arm in the victim and in nothing else, or a difference in the
    result cannot be attributed to architecture."""
    if arch == "mlp":
        return train_mlp(x_std, y, seed, epochs=epochs, device=device)
    if arch == "ft_transformer":
        return train_ft_transformer(x_std, y, seed, epochs=epochs, device=device, **arch_kwargs)
    raise ValueError(f"unknown arch {arch!r}; expected one of {ARCH_CHOICES}")


# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------

def n1_concentration(mask: np.ndarray, top_k: int = TOP_K) -> dict:
    """Concentration of mask mass. All three are scale-free, so a victim with a globally larger mask
    is not flagged for its size -- which is what scripts/64 already showed carries nothing."""
    a = np.abs(np.asarray(mask, dtype=np.float64))
    total = a.sum()
    if total <= 0:
        return {"top_k_share": 0.0, "entropy": 0.0, "gini": 0.0}
    p = a / total
    top_share = float(np.sort(p)[::-1][:top_k].sum())
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    srt = np.sort(p)
    n = len(srt)
    gini = float((2.0 * np.arange(1, n + 1) - n - 1).dot(srt) / n)
    return {"top_k_share": top_share, "entropy": entropy, "gini": gini}


def n2_shap_alignment(mask: np.ndarray, shap_ranking, top_k: int = TOP_K) -> dict:
    """Overlap between the mask's top-k features and the model's OWN SHAP top-k.

    `shap_ranking` is whatever shap_rank_features returns for this model -- the DEFENDER computes it
    on the model under test, so this uses no knowledge of the planted trigger.
    """
    a = np.abs(np.asarray(mask, dtype=np.float64))
    mask_top = set(np.argsort(a)[::-1][:top_k].tolist())
    shap_top = set(np.asarray(shap_ranking, dtype=int)[:top_k].tolist())
    inter = mask_top & shap_top
    union = mask_top | shap_top
    total = a.sum()
    return {
        "jaccard": float(len(inter) / len(union)) if union else 0.0,
        "overlap_count": int(len(inter)),
        # How much of the mask's mass sits on SHAP-important features, not just how many.
        "shap_mass_share": float(a[list(shap_top)].sum() / total) if total > 0 else 0.0,
    }


def n3_restart_stability(start_masks: list, target_class: int, converged_flags: dict) -> dict:
    """Dispersion across starts for one class.

    Only CONVERGED starts count: a diverged start's mask is an artefact of the failure, not evidence
    the model's basin is wide. Fewer than 2 converged starts leaves this undefined rather than
    imputed -- an undefined value is honest, a zero would read as perfect stability.
    """
    entries = [e for e in start_masks if int(e["target_class"]) == target_class
               and converged_flags.get((target_class, int(e["start"])), False)]
    if len(entries) < 2:
        return {"n_converged_starts": len(entries), "mask_l1_std": None, "mean_pairwise_cosine": None}

    mats = np.asarray([np.abs(np.asarray(e["mask"], dtype=np.float64)) for e in entries])
    l1 = mats.sum(axis=1)
    norms = np.linalg.norm(mats, axis=1)
    cosines = []
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            denom = norms[i] * norms[j]
            cosines.append(float(mats[i].dot(mats[j]) / denom) if denom > 0 else 0.0)
    return {
        "n_converged_starts": len(entries),
        "mask_l1_std": float(l1.std(ddof=0)),
        "mean_pairwise_cosine": float(np.mean(cosines)),
    }


def statistics_for(mask, start_masks, ledger, shap_ranking, flagged_class) -> dict:
    converged = {(int(s["target_class"]), int(s["start"])): bool(s["converged"])
                 for s in ledger["starts"]}
    out = {}
    for k, v in n1_concentration(mask).items():
        out[f"n1_{k}"] = v
    for k, v in n2_shap_alignment(mask, shap_ranking).items():
        out[f"n2_{k}"] = v
    for k, v in n3_restart_stability(start_masks, flagged_class, converged).items():
        out[f"n3_{k}"] = v
    return out


STATISTIC_FIELDS = [
    "n1_top_k_share", "n1_entropy", "n1_gini",
    "n2_jaccard", "n2_overlap_count", "n2_shap_mass_share",
    "n3_mask_l1_std", "n3_mean_pairwise_cosine",
]


# ---------------------------------------------------------------------------------------------
# Inversion plumbing (imported machinery only; no re-implemented maths)
# ---------------------------------------------------------------------------------------------

def _inversion_config(steps: int) -> TabularInversionConfig:
    return TabularInversionConfig(steps=steps, lr=0.1, l1_weight=INVERSION_SPEC["l1_weight"],
                                  n_starts=INVERSION_SPEC["n_starts"],
                                  constraint_weight=INVERSION_SPEC["constraint_weight"])


def invert_and_score(model, sample, bnd, inv_cfg, seed, penalty, device, shap_ranking) -> tuple[dict, np.ndarray]:
    """One inversion plus its three statistics. Returns (row-without-identity, selected mask)."""
    norms, masks, ledger = reverse_engineer_tabular_multistart(
        model, sample, bnd, inv_cfg, seed=seed,
        constraint_penalty=(penalty if inv_cfg.constraint_weight else None),
        num_classes=2, device=device, retain_start_masks=True)

    # The published rule flags argmin-norm; the mask analysed is that class's, so victim and null are
    # scored on the same object the detector itself would inspect.
    flagged_class = int(np.argmin(norms))
    mask = np.asarray(masks[flagged_class], dtype=np.float64)
    stats = statistics_for(mask, ledger.get("start_masks", []), ledger, shap_ranking, flagged_class)
    row = dict(
        flagged_class=flagged_class,
        mask_norms=[float(v) for v in norms],
        n_convergence_failures=int(ledger["n_convergence_failures"]),
        n_no_converged_start=int(ledger["n_no_converged_start"]),
        **stats,
    )
    return row, mask


# ---------------------------------------------------------------------------------------------
# Checkpointing (scripts/05 pattern; JSON-projected key -- the 604e88a resume bug)
# ---------------------------------------------------------------------------------------------

def _config_key(seeds, cfg, stage, arch="mlp", arch_kwargs=None) -> dict:
    key = dict(stage=stage, seeds=list(seeds), cell=list(CELL), candidate=CANDIDATE,
               inversion_spec=INVERSION_SPEC, top_k=TOP_K,
               nc_steps=cfg["nc_steps"], nc_sample=cfg["nc_sample"], mlp_epochs=cfg["mlp_epochs"],
               n_replicas=cfg["n_replicas"], replica_stride=config.NC_REPLICA_SEED_STRIDE)
    if arch != "mlp":
        # "mlp" keeps the ORIGINAL key byte-identical, so the committed checkpoint still matches and
        # the finished MLP run stays resumable from its own artifact. Any other architecture adds
        # discriminating fields, so it can never match those rows -- two victim families must never
        # pool into one null, and arch_kwargs is included because the one authorised gate-0 retry
        # changes the model shape and a d=32 run must not resume from d=16 rows.
        key["arch"] = arch
        key["arch_kwargs"] = dict(arch_kwargs or {})
    # Compare against the key's own JSON projection: tuples become lists on a round trip, so an
    # in-memory key never equals the one read back and every resume would silently start fresh.
    return json.loads(json.dumps(key))


def load_checkpoint(path, key) -> dict:
    if not path.exists():
        return {}
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if blob.get("config_key") != key:
        print("checkpoint config mismatch -- starting fresh (old checkpoint ignored)")
        return {}
    print(f"resuming from checkpoint: {len(blob.get('rows', {}))} rows already done")
    return blob.get("rows", {})


def save_checkpoint(path, key, rows) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(path)


def save_masks(mask_dir, seed, kind, index, mask) -> None:
    mask_dir.mkdir(parents=True, exist_ok=True)
    path = mask_dir / f"{kind}_seed{seed}_{index}.npz"
    tmp = path.with_suffix(".npz.tmp")
    # Write through an open handle: np.savez_compressed APPENDS '.npz' to a path that does not
    # already end in it, so passing the temp path by name silently produces '<name>.npz.tmp.npz'
    # and the atomic rename then fails on a file that was never created.
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, mask=np.asarray(mask, dtype=np.float32))
    tmp.replace(path)


# ---------------------------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------------------------

def _binomial_lower_bound(k: int, n: int, alpha: float = 0.05) -> float:
    """ONE-SIDED Clopper-Pearson 95% lower confidence bound on the flag rate.

    One-sided, not the lower limit of a two-sided interval, because the claim being bounded is
    directional -- "the detection rate is at least X". At k=n=3 this is 0.05^(1/3) = 0.368; the
    two-sided lower limit would be 0.025^(1/3) = 0.292. Both refute a 0.9 gate, but the artifact must
    report the same convention the pre-registration quotes, so the number is pinned here and in
    tests/test_rung2_pilot_statistics.py rather than left to whichever the caller assumed.
    """
    if n == 0 or k == 0:
        return 0.0
    from scipy.stats import beta
    if k == n:
        return float(alpha ** (1.0 / n))
    return float(beta.ppf(alpha, k, n - k + 1))


def evaluate_statistic(field: str, clean_vals: list, victim_vals: list, rng) -> dict:
    clean = np.asarray([v for v in clean_vals if v is not None], dtype=float)
    victims = np.asarray([v for v in victim_vals if v is not None], dtype=float)
    if len(clean) < 2 or len(victims) == 0:
        return {"field": field, "verdict": "undefined", "n_clean": len(clean), "n_victims": len(victims)}

    hi = float(np.percentile(clean, CLEAN_PERCENTILE))
    lo = float(np.percentile(clean, 100.0 - CLEAN_PERCENTILE))
    above = int((victims > hi).sum())
    below = int((victims < lo).sum())
    # Direction-free, like scripts/64: a statistic that puts victims consistently LOW is as useful as
    # one that puts them high. The direction is recorded, not assumed.
    direction = "above" if above >= below else "below"
    n_flagged = max(above, below)

    # Exact ranks of each victim within the pooled clean null.
    ranks = [float((clean < v).mean()) for v in victims]

    # Permutation p on the mean-rank statistic: how often does a random relabelling of
    # (clean + victims) put a victim group this far into the tail?
    pooled = np.concatenate([clean, victims])
    obs = float(np.mean([(clean < v).mean() for v in victims]))
    obs_stat = obs if direction == "above" else 1.0 - obs
    n_perm = 10_000
    hits = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        fake_v = pooled[idx[:len(victims)]]
        fake_c = pooled[idx[len(victims):]]
        st = float(np.mean([(fake_c < v).mean() for v in fake_v]))
        st = st if direction == "above" else 1.0 - st
        if st >= obs_stat:
            hits += 1
    p_perm = float((hits + 1) / (n_perm + 1))

    passed = n_flagged == len(victims)
    return {
        "field": field,
        "n_clean": int(len(clean)),
        "n_victims": int(len(victims)),
        "clean_median": float(np.median(clean)),
        "clean_p90": hi,
        "clean_p10": lo,
        "victim_values": [float(v) for v in victims],
        "victim_ranks_in_clean_null": ranks,
        "direction": direction,
        "n_victims_beyond_bar": int(n_flagged),
        "binomial_lower_bound_95": _binomial_lower_bound(n_flagged, len(victims)),
        "permutation_p": p_perm,
        "passes_rank_gate": bool(passed),
        "verdict": "consistent_with_separation" if passed else "victims_inside_clean_null",
    }


def main() -> int:
    argv = sys.argv[1:]
    probe = "--probe" in argv
    stage = argv[argv.index("--stage") + 1] if "--stage" in argv else "calibration"
    seeds = ([int(s) for s in argv[argv.index("--seeds") + 1].split(",")]
             if "--seeds" in argv else [42, 123, 456])

    arch = argv[argv.index("--arch") + 1] if "--arch" in argv else "mlp"
    if arch not in ARCH_CHOICES:
        raise SystemExit(f"unknown --arch {arch!r}; expected one of {ARCH_CHOICES}")
    getf = lambda flag, default: (int(argv[argv.index(flag) + 1]) if flag in argv else default)
    arch_kwargs = ({} if arch == "mlp" else
                   dict(d_token=getf("--d-token", 16), n_layers=getf("--n-layers", 2),
                        n_heads=getf("--n-heads", 4)))
    paths = output_paths(arch)
    out_path, ckpt_path, mask_dir = paths["out"], paths["ckpt"], paths["masks"]

    # --probe runs the FULL config with fewer replicas; --smoke runs a tiny config (100 inversion
    # steps, a 15k subsample) and exists only to prove the plumbing executes end to end. Never
    # project a run from --smoke, and prefer reading the marginal rate off the real run's own
    # checkpoint over spending a --probe whose rows a different n_replicas key would discard.
    cfg = apply_overrides(get_config(smoke="--smoke" in argv), argv)
    cfg["n_replicas"] = 3 if probe else config.NC_CLEAN_REPLICAS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} arch={arch} {arch_kwargs} stage={stage} seeds={seeds} "
          f"replicas={cfg['n_replicas']} candidate={CANDIDATE} cell={CELL}")

    t0 = time.time()
    S = load_setup(cfg)
    # Derived exactly as scripts/18a_nc_repair.py does after load_setup, including using its
    # std_bounds for the inversion box rather than recomputing min/max here -- the null and the
    # victims must be inverted inside the same box the published pipeline uses.
    S["x_te_std"] = apply_standardiser(S["scaler"], S["x_te_raw"])
    S["x_bot_raw"] = S["x_te_raw"][S["y_te"] != TARGET]
    x_tr_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
    std_lo, std_hi = S["std_bounds"]
    bnd = (torch.tensor(std_lo, dtype=torch.float32), torch.tensor(std_hi, dtype=torch.float32))
    penalty = make_constraint_penalty(S["constraints"], S["features"], S["scaler"], device=device)
    inv_cfg = _inversion_config(cfg["nc_steps"])
    print(f"setup loaded ({time.time() - t0:.0f}s)")

    key = _config_key(seeds, cfg, stage, arch, arch_kwargs)
    ckpt = load_checkpoint(ckpt_path, key)
    save_cb = lambda: save_checkpoint(ckpt_path, key, ckpt)

    rate, cost = CELL
    clean_rows, victim_rows = [], []
    timings = []

    for seed in seeds:
        print(f"--- seed {seed} ({time.time() - t0:.0f}s) ---")
        clean_mlp = train_victim(arch, x_tr_std, S["y_tr"], seed, cfg["mlp_epochs"], device,
                                 arch_kwargs)
        ranking = shap_rank_features(clean_mlp, S["x_tr_raw"], S["scaler"], target=TARGET,
                                     kind="mlp", device=device)
        stats_raw = raw_trigger_stats(S["x_tr_raw"], S["constraints"])

        # ---- the victim for this seed (one per seed at the pinned cell) ----
        vkey = f"victim|{seed}|{rate}|{cost}"
        if vkey in ckpt:
            victim_rows.append(ckpt[vkey])
        else:
            ts = time.time()
            realiz, _ = build_trigger(ranking, cost, S["x_tr_raw"], S["constraints"], S["features"],
                                      S["bounds"], stats=stats_raw)
            x_p_std, y_p, _ = poison_trainset_cleanlabel(
                S["x_tr_raw"], S["y_tr"], realiz, rate, S["scaler"], target=TARGET, seed=seed,
                constraints=S["constraints"], feature_names=S["features"], bounds=S["bounds"])
            victim_mlp = train_victim(arch, x_p_std, y_p, seed, cfg["mlp_epochs"], device,
                                      arch_kwargs)
            x_bot_trig = apply_trigger(S["x_bot_raw"], realiz, S["constraints"], S["features"],
                                       S["bounds"])
            asr = float(attack_success_rate(victim_mlp,
                                            apply_standardiser(S["scaler"], x_bot_trig), TARGET, device))
            cacc = float(clean_accuracy(victim_mlp, S["x_te_std"], S["y_te"], device))
            # The victim's own SHAP ranking -- N2 must be computed on the model under test, not on the
            # clean model whose ranking chose the trigger. Using the latter would hand the defender
            # the attacker's own key.
            victim_ranking = shap_rank_features(victim_mlp, S["x_tr_raw"], S["scaler"], target=TARGET,
                                                kind="mlp", device=device)
            sample = balanced_inversion_sample(x_p_std, y_p, cfg["nc_sample"], seed)
            row, mask = invert_and_score(victim_mlp, sample, bnd, inv_cfg, seed, penalty, device,
                                         victim_ranking)
            row.update(kind="victim", seed=seed, rate=rate, cost=cost, asr=asr, clean_acc=cacc,
                       trigger_indices=[int(i) for i in realiz["indices"]],
                       seconds=float(time.time() - ts))
            save_masks(mask_dir, seed, "victim", f"{rate}_{cost}", mask)
            ckpt[vkey] = row
            victim_rows.append(row)
            save_cb()
            print(f"  victim asr={asr:.4f} clean_acc={cacc:.4f} "
                  f"top_k_share={row['n1_top_k_share']:.4f} jaccard={row['n2_jaccard']:.4f} "
                  f"({row['seconds']:.0f}s)")

        # ---- clean replicas: the null ----
        for replica in range(cfg["n_replicas"]):
            rkey = f"clean|{seed}|{replica}"
            if rkey in ckpt:
                clean_rows.append(ckpt[rkey])
                continue
            ts = time.time()
            init_seed = seed if replica == 0 else seed + replica * config.NC_REPLICA_SEED_STRIDE
            model = clean_mlp if replica == 0 else train_victim(
                arch, x_tr_std, S["y_tr"], init_seed, cfg["mlp_epochs"], device, arch_kwargs)
            rep_ranking = ranking if replica == 0 else shap_rank_features(
                model, S["x_tr_raw"], S["scaler"], target=TARGET, kind="mlp", device=device)
            sample = balanced_inversion_sample(x_tr_std, S["y_tr"], cfg["nc_sample"], init_seed)
            row, mask = invert_and_score(model, sample, bnd, inv_cfg, init_seed, penalty, device,
                                         rep_ranking)
            row.update(kind="clean", seed=seed, replica=replica, init_seed=init_seed,
                       seconds=float(time.time() - ts))
            save_masks(mask_dir, seed, "clean", replica, mask)
            ckpt[rkey] = row
            clean_rows.append(row)
            save_cb()
            timings.append(row["seconds"])
            print(f"  clean r{replica} top_k_share={row['n1_top_k_share']:.4f} "
                  f"jaccard={row['n2_jaccard']:.4f} cos={row['n3_mean_pairwise_cosine']} "
                  f"({row['seconds']:.0f}s)")

    elapsed = time.time() - t0
    # Marginal rate from replicas 2+ only: the first eats CUDA warmup and would inflate any
    # projection built on it (the standing timing lesson).
    marginal = float(np.mean(timings[1:])) if len(timings) > 2 else (timings[-1] if timings else None)

    rng = np.random.default_rng(42)
    gate = {f: evaluate_statistic(f, [r.get(f) for r in clean_rows], [r.get(f) for r in victim_rows], rng)
            for f in STATISTIC_FIELDS}
    passing = sorted([f for f, g in gate.items() if g.get("passes_rank_gate")])

    # N2's pre-registered third outcome: both sides high means the confound is confirmed, which is a
    # mechanism finding, not a failed run. Decided here rather than by eye later.
    n2_clean_median = gate["n2_jaccard"].get("clean_median")
    n2_confound = bool(n2_clean_median is not None and n2_clean_median >= 0.5)

    payload = {
        "preregistration": "notes/20260901-decision-rung2-repair-pilots-preregistration.md",
        "stage": stage, "arch": arch, "arch_kwargs": arch_kwargs,
        "probe": probe, "seeds": seeds, "cell": list(CELL),
        "candidate": CANDIDATE, "inversion_spec": INVERSION_SPEC, "top_k": TOP_K,
        "n_clean_rows": len(clean_rows), "n_victim_rows": len(victim_rows),
        "clean_percentile_bar": CLEAN_PERCENTILE,
        "elapsed_seconds": elapsed,
        "marginal_seconds_per_clean_replica": marginal,
        "clean_rows": clean_rows, "victim_rows": victim_rows,
        "gate": gate,
        "passing_statistics": passing,
        "n2_confound_saturated": n2_confound,
        "verdict": ("separation_found" if passing
                    else ("n2_confound_saturated" if n2_confound else "no_statistic_separates")),
    }
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.rename(out_path)

    print(f"\n=== {elapsed:.0f}s total, marginal {marginal:.1f}s/replica ===" if marginal
          else f"\n=== {elapsed:.0f}s total ===")
    for f in STATISTIC_FIELDS:
        g = gate[f]
        if g.get("verdict") == "undefined":
            print(f"  {f:26s} undefined")
            continue
        print(f"  {f:26s} victims {g['n_victims_beyond_bar']}/{g['n_victims']} beyond p{int(CLEAN_PERCENTILE)} "
              f"({g['direction']}) p_perm={g['permutation_p']:.4f} -> {g['verdict']}")
    print(f"\nverdict: {payload['verdict']}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
