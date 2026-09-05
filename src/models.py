"""Tabular NIDS victims: LightGBM/GBDT (no activations -> H4) and an MLP (has activations for
Activation Clustering / Spectral). Both consume STANDARDISED features (the scaler is applied at this
boundary; trigger + constraints live in raw space, see src/trigger.py and src/tb_vendor).

Imbalance handling (decision 8): LightGBM uses scale_pos_weight = n_neg/n_pos; the MLP uses a
class-weighted cross-entropy, so the clean victim is a competent NIDS rather than an all-benign
degenerate. H3's rebalancing is a separate detector-side manipulation in M3, not a change here.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.torch_utils import batched_apply
from src.vision_control import set_seed

_LGB_PARAMS = dict(n_estimators=200, num_leaves=64, learning_rate=0.05, subsample=0.8,
                   colsample_bytree=0.8, min_child_samples=20, n_jobs=-1)


def train_lightgbm(x_tr_std, y_tr, seed: int):
    """LightGBM victim with scale_pos_weight for the ~98/2 imbalance. Params are fixed (logged by the
    caller), never tuned on test."""
    import lightgbm as lgb

    y = np.asarray(y_tr)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    spw = (n_neg / n_pos) if n_pos else 1.0
    model = lgb.LGBMClassifier(random_state=seed, scale_pos_weight=spw, verbosity=-1, **_LGB_PARAMS)
    model.fit(np.asarray(x_tr_std), y)
    return model


class MLPVictim(nn.Module):
    def __init__(self, in_dim: int, feat_dim: int = 128, n_classes: int = 2) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, feat_dim), nn.ReLU())
        self.head = nn.Linear(feat_dim, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)                      # penultimate (post-ReLU) -- AC/Spectral source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def _class_weights(y: np.ndarray, n_classes: int = 2) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = len(y) / (n_classes * counts)            # inverse-frequency weighting
    return torch.tensor(w, dtype=torch.float32)


def train_mlp(x_tr_std, y_tr, seed: int, epochs: int = 20, device: str = "cpu",
              batch_size: int = 2048, log_every: int = 0,
              n_classes: int | None = None) -> MLPVictim:
    """`in_dim`/class-weighting are already inferred from `x_tr_std`/`y_tr` at call time -- verified
    generic across dataset scale (timed at ~4s/epoch on UNSW-NB15's 1.267M-row/38-feature secondary
    train set on GPU, vs. CTU-13's 143k-row/757-feature primary set; no NaN/Inf, no dtype overflow, no
    hardcoded feature/class count found on inspection). `log_every` is the one addition this task
    (extension Task 4's secondary-dataset sweep) needed: this project is two-machine (RTX 3060 Ti main
    PC, CPU-only P520 laptop -- see this project's CLAUDE.md), and CPU-only training of the SAME
    1.267M-row set is far slower per epoch than the GPU timing above, with no way to tell a slow-but-
    progressing training from a stalled one across a checkpointed multi-call foreground run. `log_every=0`
    (default) is a silent no-op, so every existing call site/test is unaffected."""
    set_seed(seed)
    torch.backends.cudnn.deterministic = True    # closes M1 Live Risk 4 for the MLP
    torch.backends.cudnn.benchmark = False
    # Keep the whole training set resident on the device and index it with a shuffled permutation --
    # avoids per-batch CPU->GPU transfer and DataLoader overhead, which dominate wall-clock for this
    # small model (the compute per step is tiny; the transfers are not). Big speedup on weak GPUs.
    x = torch.as_tensor(np.asarray(x_tr_std), dtype=torch.float32, device=device)
    y = torch.as_tensor(np.asarray(y_tr), dtype=torch.long, device=device)
    # n_classes is inferred from the labels unless given. Binary y yields max(2, 1+1) = 2,
    # so every existing call site is bit-identical to before this parameter existed. It is
    # here because MLPVictim and _class_weights already accepted n_classes while train_mlp
    # never forwarded it, so a ten-class label vector raised inside cross_entropy rather
    # than training a ten-class head. See notes/20260904-decision-nc-multiclass-prep.md.
    n_cls = int(n_classes) if n_classes is not None else max(2, int(np.asarray(y_tr).max()) + 1)
    model = MLPVictim(in_dim=x.shape[1], n_classes=n_cls).to(device)
    weight = _class_weights(np.asarray(y_tr), n_classes=n_cls).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(x)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            F.cross_entropy(model(x[idx]), y[idx], weight=weight).backward()
            opt.step()
        if log_every and (epoch + 1) % log_every == 0:
            print(f"    train_mlp seed={seed} epoch={epoch + 1}/{epochs} n={n}")
    return model


class _FTBlock(nn.Module):
    """Pre-norm encoder block. Uses F.scaled_dot_product_attention rather than
    nn.TransformerEncoderLayer so the attention path is explicit and every timing measured for this
    plan applies to exactly this code. At 758 tokens with a small d_token the workload is
    launch-bound, so SDPA is chosen for memory headroom and clarity, not for speed."""

    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        self.h = h
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        b, n, d = t.shape
        q, k, v = self.qkv(self.n1(t)).chunk(3, dim=-1)
        split = lambda z: z.view(b, n, self.h, d // self.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(split(q), split(k), split(v))
        t = t + self.proj(a.transpose(1, 2).reshape(b, n, d))
        return t + self.ff(self.n2(t))


class FTTransformerVictim(nn.Module):
    """Feature-tokenizer transformer (FT-Transformer / TabTransformer family). Each of the `in_dim`
    scalar features becomes its own token, so the model is permutation-equivariant over features and
    assumes nothing about NetFlow column order -- the property that makes a null on this victim
    attributable to architecture rather than to an injected locality bias (a 1D-CNN's locality would
    be misaligned with a trigger scattered across 757 columns).

    Interface matches MLPVictim exactly: `features()` returns the penultimate representation (here
    the CLS token, width d_token) and `forward()` returns logits. That parity is what lets
    scripts/65 swap victims without touching the detector code, and it makes Activation Clustering
    and Spectral generality tests available later at no extra cost."""

    def __init__(self, in_dim: int, d_token: int = 16, n_layers: int = 2, n_heads: int = 4,
                 n_classes: int = 2) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_dim, d_token) * 0.02)   # per-feature scale
        self.b = nn.Parameter(torch.zeros(in_dim, d_token))          # per-feature bias
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        self.blocks = nn.ModuleList([_FTBlock(d_token, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        t = x.unsqueeze(-1) * self.w + self.b                        # (B, D, d_token)
        t = torch.cat([self.cls.expand(x.shape[0], -1, -1), t], dim=1)
        for blk in self.blocks:
            t = blk(t)
        return self.norm(t[:, 0])                                    # CLS -- AC/Spectral source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


_CUBLAS_ENV = "CUBLAS_WORKSPACE_CONFIG"
_CUBLAS_OK = (":4096:8", ":16:8")


def require_deterministic_cuda() -> None:
    """Make CUDA training bitwise reproducible, or refuse to run.

    `F.scaled_dot_product_attention`'s CUDA backward is non-deterministic by default: two same-seed
    `train_ft_transformer` runs were measured to diverge by 1.0e-04 in their final logits, which
    breaks this project's standing rule that every run is seeded and reproducible. Determinism was
    measured to cost about 7% on the bf16 training step and nothing on the fp32 inversion step, so
    it is bought cheaply.

    `CUBLAS_WORKSPACE_CONFIG` must be set BEFORE the process makes its first CUDA call, and a
    library function called partway through a script cannot guarantee that. Setting it here would
    silently do nothing on a process that has already touched CUDA, so this raises instead -- a
    loud failure at the start of a 30-hour run beats an unreproducible result at the end of one."""
    actual = os.environ.get(_CUBLAS_ENV)
    if actual not in _CUBLAS_OK:
        raise RuntimeError(
            f"{_CUBLAS_ENV} must be one of {_CUBLAS_OK} before the first CUDA call for "
            f"deterministic training (got {actual!r}). Set it in the launcher, for example: "
            f"{_CUBLAS_ENV}=:4096:8 .venv/bin/python scripts/67_arch_gate0.py")
    torch.use_deterministic_algorithms(True, warn_only=False)


def train_ft_transformer(x_tr_std, y_tr, seed: int, epochs: int = 20, device: str = "cpu",
                         batch_size: int = 1024, d_token: int = 16, n_layers: int = 2,
                         n_heads: int = 4, log_every: int = 0) -> FTTransformerVictim:
    """Same contract as train_mlp: standardised inputs, inverse-frequency class weighting, seeded.

    Trains under bf16 autocast on CUDA (measured 6.8x faster at 757 tokens, and the run is otherwise
    28-32 hours). Autocast is a COMPUTATION context, not a parameter dtype -- the returned model's
    weights are fp32, so the Neural Cleanse inversion, which runs outside any autocast context, sees
    the same precision the committed MLP arm did. That split is deliberate: training precision does
    not enter the comparison as long as the gate-0 controls pass, but the inverted mask IS the
    measured quantity and must not be computed at reduced precision.

    batch_size defaults to 1024 rather than train_mlp's 2048: measured peak is 1499 MiB against 2982
    MiB for identical total wall-clock (138 ms/step at 1024 against 274 ms/step at 2048), and this
    GPU is shared with a coursework job."""
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    use_amp = str(device).startswith("cuda")
    if use_amp:
        require_deterministic_cuda()
    x = torch.as_tensor(np.asarray(x_tr_std), dtype=torch.float32, device=device)
    y = torch.as_tensor(np.asarray(y_tr), dtype=torch.long, device=device)
    model = FTTransformerVictim(in_dim=x.shape[1], d_token=d_token, n_layers=n_layers,
                                n_heads=n_heads).to(device)
    weight = _class_weights(np.asarray(y_tr)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(x)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(model(x[idx]), y[idx], weight=weight)
            else:
                loss = F.cross_entropy(model(x[idx]), y[idx], weight=weight)
            loss.backward()
            opt.step()
        if log_every and (epoch + 1) % log_every == 0:
            print(f"    train_ft seed={seed} epoch={epoch + 1}/{epochs} n={n}")
    return model


@torch.no_grad()
def mlp_penultimate_features(model: MLPVictim, x_std, device: str = "cpu") -> np.ndarray:
    """Batched penultimate activations (N, feat_dim) -- the detector input, sharing
    src.torch_utils.batched_apply with vision_control.penultimate_features."""
    model = model.to(device).eval()
    x = torch.as_tensor(np.asarray(x_std), dtype=torch.float32)
    return np.concatenate(
        batched_apply(lambda xb: model.features(xb).cpu().numpy(), x, device), axis=0)


def mlp_input_gradients(model: MLPVictim, x_std, target: int, device: str = "cpu") -> np.ndarray:
    """Per-row gradient of the target-class logit with respect to the standardized input, (N, D).

    The local linear model of a piecewise-linear network. For a ReLU victim the value of this
    gradient is fixed by which units are active at that row, so it is the row's active path
    expressed in input coordinates. Used by `src.detectors.active_paths`.

    This is the gradient-carrying counterpart to `mlp_penultimate_features`, which is decorated
    `@torch.no_grad()` and therefore cannot supply it. Model parameters are not updated and no
    optimizer is involved; the graph is built only to differentiate the output with respect to the
    input, and the model is left in eval mode so dropout and batch statistics do not move.

    Batching follows `batched_apply`'s contract, but the gradient must be taken per chunk rather
    than under a global no-grad, so the loop is written out here.
    """
    model = model.to(device).eval()
    x = torch.as_tensor(np.asarray(x_std), dtype=torch.float32)
    out = []
    for i in range(0, len(x), 1024):
        xb = x[i:i + 1024].to(device).clone().requires_grad_(True)
        logit = model(xb)[:, target].sum()
        (grad,) = torch.autograd.grad(logit, xb)
        out.append(grad.detach().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, x.shape[1]), dtype=np.float32)


@torch.no_grad()
def predict_labels(model, x_std, device: str = "cpu") -> np.ndarray:
    """Predicted class labels, dispatching on victim type (LightGBM .predict vs MLP argmax)."""
    if isinstance(model, torch.nn.Module):
        model = model.to(device).eval()
        x = torch.as_tensor(np.asarray(x_std), dtype=torch.float32)
        return np.concatenate(
            batched_apply(lambda xb: model(xb).argmax(1).cpu().numpy(), x, device), axis=0)
    return np.asarray(model.predict(np.asarray(x_std)))


def clean_accuracy(model, x_std, y, device: str = "cpu") -> float:
    return float((predict_labels(model, x_std, device) == np.asarray(y)).mean())


def attack_success_rate(model, x_triggered_std, target: int, device: str = "cpu") -> float:
    """Fraction of already-triggered, already-standardised inputs predicted as `target`. The caller
    stamps the trigger in RAW space and standardises before calling (space discipline)."""
    pred = predict_labels(model, x_triggered_std, device)
    return float((pred == target).mean())
