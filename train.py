import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import numpy as np
import os
from torch.utils.data import DataLoader
import time
from sklearn import metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import random
import argparse
import sys

# Weights & Biases logging is optional. Without the package (or with
# WANDB_MODE unset) every wandb call below is a no-op; set WANDB_MODE=online
# to log runs.
try:
    import wandb
except ImportError:                                   # pragma: no cover
    class _NoWandb:
        run = None
        def __getattr__(self, name):
            return lambda *a, **k: None
    wandb = _NoWandb()

# ==================== FIBER TAXONOMY (for per-class metrics / diagnostics) ====
FIBER_NAMES = ['SP', 'CA', 'L', 'R', 'M', 'T', 'P', 'W', 'N', 'A', 'S', 'C']
FIBER_FULL_NAMES = {
    'SP': 'Spandex', 'CA': 'Cashmere', 'L': 'Linen', 'R': 'Rayon',
    'M': 'Modal', 'T': 'Tencel', 'P': 'Polyester', 'W': 'Wool',
    'N': 'Nylon', 'A': 'Acrylic', 'S': 'Silk', 'C': 'Cotton',
}
# 5 parent material families (indices into FIBER_NAMES); used by the
# confusion/substitution diagnostic to flag same-family swaps.
_PARENT_GROUPS = [
    [0],          # Spandex
    [1, 7, 10],   # Animal     (CA, W, S)
    [2, 11],      # Vegetable  (L, C)
    [3, 4, 5],    # Regenerated (R, M, T)
    [6, 8, 9],    # Polymer    (P, N, A)
]

# ==================== ARCHITECTURE ====================

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer for 1D data """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        # x: (B, L, C)
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ ConvNeXtV2 Block for 1D data """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        x = input + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """
    ConvNeXt V2 for 1D time series data with dual outputs (classification + regression).

    Args:
        in_chans (int): Number of input channels. Default: 2
        num_classes (int): Number of output classes for both heads. Default: 12
        depths (list): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (list): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier/regressor weights. Default: 1.
    """
    def __init__(self, in_chans=2, num_classes=12,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1., use_multiscale=False):
        super().__init__()
        self.depths = depths
        self.use_multiscale = use_multiscale
        self.downsample_layers = nn.ModuleList()

        # Stem
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        # Intermediate downsample layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # Multi-scale aggregation with separate pooling for each stage
        if use_multiscale:
            self.ms_pools = nn.ModuleList([nn.AdaptiveAvgPool1d(1) for _ in range(4)])
            self.ms_proj = nn.ModuleList([nn.Linear(dims[i], dims[-1]) for i in range(4)])

        self.mlc_proj = nn.Sequential(
            nn.Linear(dims[-1], dims[-1] // 2),
            nn.GELU(),
            nn.LayerNorm(dims[-1] // 2)
        )
        self.reg_proj = nn.Sequential(
            nn.Linear(dims[-1], dims[-1] // 2),
            nn.GELU(),
            nn.LayerNorm(dims[-1] // 2)
        )

        self.head  = nn.Linear(dims[-1] // 2, num_classes) # Classification head (mlc)
        self.head2 = nn.Linear(dims[-1] // 2, num_classes) # Regression head (reg)

        self.apply(self._init_weights)
        # ═══════════════════════════════════════════════════════════════════
        # REMOVED: The old zero-init / identity-init block.
        # ms_proj layers are already initialized by self.apply() above
        # with trunc_normal_(std=.02), same as all other Linear layers.
        # ═══════════════════════════════════════════════════════════════════
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        self.head2.weight.data.mul_(head_init_scale)
        self.head2.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        feats = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if self.use_multiscale:
                feats.append(self.ms_pools[i](x).flatten(1))   # (B, dims[i])
        if self.use_multiscale:
            agg = sum(proj(f) for proj, f in zip(self.ms_proj, feats))
            x = self.norm(agg)
        else:
            x = self.pooling(x)
            x = x.view(x.size(0), -1)
            x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        mlc_features = self.mlc_proj(x)
        reg_features = self.reg_proj(x)
        mlc = self.head(mlc_features)
        reg = self.head2(reg_features)
        return mlc, reg


# ==================== MODEL VARIANTS ====================

MODEL_REGISTRY = {
    "yocto":  dict(depths=[1, 1, 3, 1],  dims=[24, 48, 96, 192]),
    "zepto":  dict(depths=[2, 2, 4, 2],  dims=[32, 64, 128, 256]),
    "deep_zepto": dict(
        depths=[2, 2, 6, 2],         # Uses Atto's deeper block layout
        dims=[32, 64, 128, 256]      # Uses Zepto's narrower channel width
    ),
    "shallow_atto": dict(
        depths=[2, 2, 4, 2],         # Cuts Atto's Stage 3 depth down to 4
        dims=[40, 80, 160, 320]      # Uses Atto's wider channel width
    ),
    "atto":   dict(depths=[2, 2, 6, 2],  dims=[40, 80, 160, 320]),
    "femto":  dict(depths=[2, 2, 6, 2],  dims=[48, 96, 192, 384]),
    "pico":   dict(depths=[2, 2, 6, 2],  dims=[64, 128, 256, 512]),
    "nano":   dict(depths=[2, 2, 8, 2],  dims=[80, 160, 320, 640]),
    "tiny":   dict(depths=[3, 3, 9, 3],  dims=[96, 192, 384, 768]),
    "base":   dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]),
    "large":  dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]),
    "huge":   dict(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816]),
}

def build_model(model_size, in_chans, num_classes, drop_path_rate, use_multiscale=False):
    if model_size not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model size '{model_size}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    cfg = MODEL_REGISTRY[model_size]
    return ConvNeXtV2(
        in_chans=in_chans,
        num_classes=num_classes,
        depths=cfg["depths"],
        dims=cfg["dims"],
        drop_path_rate=drop_path_rate,
        use_multiscale=use_multiscale,
    )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==================== UTILS ====================

def cal_error_acc(output, label, threshold):
    """Fraction of samples where ALL predicted values are within `threshold` of the label."""
    within_threshold = (np.abs(output - label) <= threshold).all(axis=1)
    return np.mean(within_threshold)

def threshold_predictions(logits, threshold=0.5):
    return (torch.sigmoid(logits) > threshold).int()


def load_checkpoint_for_inference(path, model, which="full_val"):
    """Load a checkpoint saved by this script and return (model, thresholds).

    `thresholds` is a plain (12,) numpy array of per-class decision cutoffs —
    it is NOT part of model.state_dict() (it's a post-hoc grid-searched value,
    not a learned weight), so it always has to be loaded and applied
    separately from the model weights.

    which:
      "full_val" -> thresholds fit on the ENTIRE val set (more data -> more
                    stable; use this for actual deployment).
      "honest"   -> thresholds fit on only half the val set (what the reported
                    acc_tuned number corresponds to; use if you need the
                    threshold that matches a specific logged metric exactly).
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    key = "thresholds_full_val" if which == "full_val" else "thresholds"
    thresholds = ckpt.get(key, ckpt.get("thresholds"))  # fall back for old ckpts
    return model, thresholds


def predict_with_thresholds(model, x, thresholds):
    """Run the model and apply saved per-class thresholds instead of 0.5.
    x: input tensor on the right device. thresholds: (12,) numpy array."""
    model.eval()
    with torch.no_grad():
        logits, reg, _ = fwd(model, x)
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs > thresholds[None, :]).astype(int)
    return preds, reg.cpu().numpy()

def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ==================== AUGMENTATION / CONSTRAINT / METRIC HELPERS ============

def mixup_1d(x, y, comp, alpha=0.2):
    """Mixup for 1D spectra. Returns mixed x and SOFT targets (y, comp).
    Physically reasonable here: in absorbance space components add ~linearly,
    so a convex blend of two spectra ~ a blend of their compositions."""
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    x = lam * x + (1.0 - lam) * x[perm]
    y_soft = lam * y.float() + (1.0 - lam) * y[perm].float()
    comp = lam * comp + (1.0 - lam) * comp[perm]
    return x, y_soft, comp, lam


def reg_loss_fn(pred, target, kind="l1", beta=1.0):
    """Regression loss: 'l1' (MAE), 'l2' (MSE), or 'smooth_l1' (Huber)."""
    if kind == "smooth_l1":
        return F.smooth_l1_loss(pred, target, beta=beta)
    elif kind == "l2":
        return F.mse_loss(pred, target, reduction="mean")
    elif kind == "l1":
        return F.l1_loss(pred, target)
    else:
        raise ValueError(f"Unknown loss kind: {kind}. Choose from 'l1', 'l2', or 'smooth_l1'.")


def sum_constraint_loss(reg_out, comp, mlc_out, y):
    """Physical sum-to-total constraints (H-MT-FCN style).
    reg: total predicted composition should match the true total per sample.
    cls: sum of class probabilities should match the true number of fibers.
    Returns (loss_sum_reg, loss_sum_cls)."""
    loss_sum_reg = torch.abs(reg_out.sum(dim=1) - comp.sum(dim=1)).mean()
    pred_cls_sum = torch.sigmoid(mlc_out).sum(dim=1)
    true_cls_sum = y.float().sum(dim=1)
    loss_sum_cls = torch.abs(pred_cls_sum - true_cls_sum).mean()
    return loss_sum_reg, loss_sum_cls


# ═══════════════════════════════════════════════════════════════════════════════
#  PHYSICS-INFORMED MODULE
# ═══════════════════════════════════════════════════════════════════════════════
#  Everything below is OFF by default. Each piece is an independent flag so the
#  ablation table can attribute gains. The scientific claim these support:
#
#  NIR absorbance of a fibre blend is, to first order, a composition-weighted
#  LINEAR MIXTURE of the pure-fibre spectra (Beer-Lambert). The network already
#  predicts that composition — so it can be forced to also EXPLAIN the measured
#  spectrum with it (analysis-by-synthesis). That converts an unconstrained
#  regression head into a physically identifiable spectral-unmixing model,
#  yields interpretable learned endmembers, and gives a reconstruction residual
#  that doubles as an out-of-distribution / novel-material score.
#
#  Design note: these are implemented as a WRAPPER around the backbone (see
#  PhysicsWrapper) rather than inside ConvNeXtV2, so the same physics can be
#  attached to any network that returns (logits, composition).
# ═══════════════════════════════════════════════════════════════════════════════

def reflectance_channel_indices(channels, channel_idx):
    """Which input channel indices hold REFLECTANCE (not absorbance).
    Dataset convention in this repo: ch0=absorbance, ch1=reflectance."""
    if channels == 2:
        return (1,)
    return (0,) if channel_idx == 1 else ()


def to_absorbance(x, refl_idx, eps=1e-4):
    """Map the reflectance channels of x to absorbance A = -log10(R).
    Blends mix ~linearly in ABSORBANCE, not reflectance, so every physics
    operation (unmixing, synthetic blending) must happen in this space."""
    if not refl_idx:
        return x
    x = x.clone()
    for c in refl_idx:
        x[:, c] = -torch.log10(x[:, c].clamp(min=eps))
    return x


def kubelka_munk(x, refl_idx, eps=1e-4):
    """Kubelka-Munk remission function F(R) = (1-R)^2 / (2R).

    For diffuse reflectance from a thick SCATTERING medium (exactly what a
    fabric is) K-M is the physically correct linearisation of concentration,
    where plain -log10(R) is only an approximation. Rarely used in textile
    NIR deep learning -> cheap, defensible novelty."""
    if not refl_idx:
        return x
    x = x.clone()
    for c in refl_idx:
        R = x[:, c].clamp(min=eps, max=1.0)
        x[:, c] = (1.0 - R) ** 2 / (2.0 * R)
    return x


# ── Scatter correction (differentiable, optionally learnable) ─────────────────

def snv(x, eps=1e-6):
    """Standard Normal Variate: per-spectrum, per-channel standardisation.
    Removes multiplicative scatter + additive baseline offset."""
    m = x.mean(dim=-1, keepdim=True)
    s = x.std(dim=-1, keepdim=True)
    return (x - m) / (s + eps)


def msc(x, ref=None, eps=1e-6):
    """Multiplicative Scatter Correction against a reference spectrum
    (batch mean if none): fit x ~ a + b*ref per spectrum, return (x-a)/b."""
    if ref is None:
        ref = x.mean(dim=0, keepdim=True)             # (1, C, L)
    r = ref.expand_as(x)
    rm = r.mean(dim=-1, keepdim=True)
    xm = x.mean(dim=-1, keepdim=True)
    b = ((x - xm) * (r - rm)).sum(-1, keepdim=True) / \
        (((r - rm) ** 2).sum(-1, keepdim=True) + eps)
    a = xm - b * rm
    return (x - a) / (b + eps)


class LearnableEMSC(nn.Module):
    """Extended MSC as a differentiable layer with a LEARNABLE reference.

    Classical EMSC models each spectrum as
        x(l) ~ m * ref(l) + b0 + b1*l + b2*l^2 + ...
    and divides out the physical (non-chemical) scatter/baseline terms. Here
    `ref` is a free parameter trained end-to-end, so the network learns the
    optimal reference rather than inheriting a fixed chemometric choice.
    Polynomial coefficients are solved per-sample in closed form (least
    squares), which keeps it exact and differentiable."""

    def __init__(self, n_channels, length, poly_degree=2):
        super().__init__()
        self.ref = nn.Parameter(torch.zeros(n_channels, length))
        nn.init.normal_(self.ref, mean=1.0, std=0.01)
        lam = torch.linspace(-1.0, 1.0, length)
        basis = torch.stack([lam ** d for d in range(poly_degree + 1)], 0)  # (P,L)
        self.register_buffer("poly", basis, persistent=False)

    def forward(self, x, eps=1e-5):
        B, C, L = x.shape
        out = []
        for c in range(C):
            D = torch.cat([self.ref[c].unsqueeze(0), self.poly], 0)     # (nb, L)
            G = D @ D.t()
            G = G + eps * torch.eye(G.size(0), device=x.device, dtype=G.dtype)
            coef = (x[:, c] @ D.t()) @ torch.linalg.inv(G)              # (B, nb)
            m = coef[:, 0:1]
            base = coef[:, 1:] @ self.poly                              # (B, L)
            out.append((x[:, c] - base) / (m + eps))
        return torch.stack(out, dim=1)


def apply_scatter_correction(x, mode, module=None):
    if mode in (None, "none"):
        return x
    if mode == "snv":
        return snv(x)
    if mode == "msc":
        return msc(x)
    if mode == "emsc":
        return module(x) if module is not None else snv(x)
    raise ValueError(f"unknown scatter_correct '{mode}'")


# ── Backbone input representation (--input_mode) ─────────────────────────────
#
#   raw      : the spectra as stored (ch0 absorbance, ch1 reflectance). DEFAULT,
#              bit-identical to every earlier run.
#   snv      : SNV of each channel. Same (C, L) shape, so the backbone is
#              unchanged; it just never sees per-spectrum level or scale.
#   raw+snv  : raw channels followed by their SNV, concatenated -> (2C, L).
#              Only the backbone STEM widens (C -> 2C input channels).
#
# In every mode the physics decoder still reconstructs the ORIGINAL raw
# spectrum in absorbance, so the endmembers stay (K, C, L) raw absorbance and
# every endmember analysis (SAM raw / SNV / detrended, Gram, cross-seed) is
# directly comparable with the raw-input runs. The input mode changes what the
# ENCODER sees, not what the decoder has to explain.

INPUT_MODES = ("raw", "snv", "raw+snv")


def input_mode_of(args):
    m = getattr(args, "input_mode", "raw") or "raw"
    if m not in INPUT_MODES:
        raise ValueError(f"unknown input_mode '{m}' (choose from {INPUT_MODES})")
    return m


def backbone_in_chans(args):
    """Number of channels the backbone stem receives (data channels x 1 or 2)."""
    return int(args.channels) * (2 if input_mode_of(args) == "raw+snv" else 1)


def apply_input_mode(x, mode):
    if mode in (None, "raw"):
        return x
    if mode == "snv":
        return snv(x)
    if mode == "raw+snv":
        return torch.cat([x, snv(x)], dim=1)
    raise ValueError(f"unknown input_mode '{mode}'")


def input_mode_tag(mode):
    """Run-name token. Also what resolve_input_mode() looks for in filenames."""
    return {"raw": None, "snv": "in-snv", "raw+snv": "in-rawsnv"}[mode]


def _ckpt_meta(args):
    """Small, loader-agnostic metadata stored next to the weights so analysis
    scripts can rebuild the exact preprocessing (SNV has no parameters, so it
    cannot be inferred from the state dict)."""
    return {"input_mode":      input_mode_of(args),
            "data_channels":   int(args.channels),
            "channel_idx":     int(getattr(args, "channel_idx", 0) or 0),
            "scatter_correct": getattr(args, "scatter_correct", "none"),
            "km_transform":    bool(getattr(args, "km_transform", False))}


def resolve_input_mode(ckpt=None, ckpt_path=None, data_channels=2,
                       stem_in_chans=None, override=None, verbose=True):
    """Work out which input representation a checkpoint was trained with.

    SNV has no parameters, so a raw-input and an SNV-input checkpoint contain
    tensors of IDENTICAL shape and load without a single missing key. An
    analysis script that guesses wrong would silently feed raw spectra to an
    SNV-trained network. So we decide explicitly, in priority order:
      1. `override` (a script's --input_mode flag)
      2. 'input_mode' stored in the checkpoint (runs from this version on)
      3. the run-name token in the filename (in-rawsnv / in-snv; a legacy
         '_snv_' token from --scatter_correct snv also means SNV input)
      4. stem width: 2 x data_channels input channels => raw+snv
      5. 'raw'
    Returns (mode, source)."""
    def _say(mode, src):
        if verbose:
            print(f"  [input_mode] {mode}  (from {src})")
        return mode, src

    if override not in (None, "", "auto"):
        return _say(override, "command line")
    if isinstance(ckpt, dict) and ckpt.get("input_mode") in INPUT_MODES:
        return _say(ckpt["input_mode"], "checkpoint metadata")
    if ckpt_path:
        toks = os.path.basename(str(ckpt_path)).replace(".pt", "").split("_")
        if "in-rawsnv" in toks:
            return _say("raw+snv", "filename tag")
        if "in-snv" in toks or "snv" in toks:
            return _say("snv", "filename tag")
    if stem_in_chans is not None and int(stem_in_chans) == 2 * int(data_channels):
        return _say("raw+snv", f"stem width {stem_in_chans} = 2 x {data_channels}")
    return _say("raw", "default")


def infer_arch_from_ckpt(sd):
    """Architecture read back out of a state dict (wrapped or bare):
    model size from the stem width, multiscale from ms_proj keys, and the
    stem's input-channel count (= 2 x data channels under raw+snv)."""
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    strip = lambda k: k[len("base."):] if k.startswith("base.") else k
    multiscale = any(strip(k).startswith("ms_proj") for k in sd)
    in_chans = dims0 = None
    for k, w in sd.items():
        if strip(k) == "downsample_layers.0.0.weight":
            dims0, in_chans = int(w.shape[0]), int(w.shape[1])
            break
    size = {40: "atto", 48: "femto", 64: "pico", 80: "nano",
            96: "tiny", 128: "base"}.get(dims0)
    return {"multiscale": multiscale, "in_chans": in_chans,
            "dims0": dims0, "model": size}


def configure_input_from_checkpoint(args, ckpt, ckpt_path=None, override=None):
    """For analysis scripts: set args.input_mode to what the checkpoint was
    trained with, and keep args.channels = the DATA channel count (the stem is
    2x wider under raw+snv). Call right before build_student(args)."""
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    stem = infer_arch_from_ckpt(sd)["in_chans"]
    if override in (None, "", "auto") and isinstance(ckpt, dict) \
            and ckpt.get("data_channels") is not None:
        args.channels = int(ckpt["data_channels"])
    mode, src = resolve_input_mode(ckpt, ckpt_path, data_channels=args.channels,
                                   stem_in_chans=stem, override=override)
    args.input_mode = mode
    # The other parameter-free preprocessing must be restored too: a run made
    # with --scatter_correct snv stores input_mode='raw' + scatter_correct='snv'
    # and would otherwise be fed raw spectra just as silently.
    if override in (None, "", "auto") and isinstance(ckpt, dict):
        for k in ("scatter_correct", "km_transform"):
            if k in ckpt and ckpt[k] != getattr(args, k, None):
                print(f"  [preprocess] {k} = {ckpt[k]}  (from checkpoint metadata)")
                setattr(args, k, ckpt[k])
    if stem is not None:
        factor = 2 if mode == "raw+snv" else 1
        if stem % factor:
            raise SystemExit(f"!! input_mode={mode} ({src}) needs an even stem "
                             f"width, but the checkpoint stem has {stem} channels")
        data_ch = stem // factor
        if data_ch != args.channels:
            print(f"  [input_mode] data channels {args.channels} -> {data_ch} "
                  f"(checkpoint stem {stem}, mode {mode})")
            args.channels = data_ch
    return mode


# ── Simplex / compositional output heads ─────────────────────────────────────

def sparsemax(z, dim=-1):
    """Sparsemax (Martins & Astudillo 2016): Euclidean projection onto the
    simplex. Unlike softmax it produces EXACT zeros, which matches a fibre
    composition vector (mostly structural zeros) far better."""
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    cssv = z_sorted.cumsum(dim) - 1.0
    rng = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim(); shape[dim] = -1
    rng = rng.view(shape)
    support = (z_sorted - cssv / rng) > 0
    k = support.to(z.dtype).sum(dim=dim, keepdim=True).clamp(min=1)
    tau = cssv.gather(dim, (k.long() - 1)) / k
    return torch.clamp(z - tau, min=0.0)


def project_simplex_torch(v, z=100.0, dim=-1):
    """Euclidean projection onto {x >= 0, sum x = z} (differentiable).
    Projection onto a convex set is non-expansive, so applying it can only
    reduce squared error against a target that already lies on the set."""
    n = v.size(dim)
    u, _ = torch.sort(v, dim=dim, descending=True)
    cssv = u.cumsum(dim) - z
    rng = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    shape = [1] * v.dim(); shape[dim] = -1
    rng = rng.view(shape)
    support = (u - cssv / rng) > 0
    k = support.to(v.dtype).sum(dim=dim, keepdim=True).clamp(min=1)
    tau = cssv.gather(dim, (k.long() - 1)) / k
    return torch.clamp(v - tau, min=0.0)


def dirichlet_composition_loss(alpha, target_frac):
    """Evidential (Dirichlet) regression on the simplex, Sensoy et al. NeurIPS'18
    expected-sum-of-squares form:
        L = sum_k [ (y_k - p_k)^2 + p_k(1-p_k)/(S+1) ],  p = alpha/S.
    Gives a calibrated composition AND an uncertainty (vacuity = K/S) for free,
    which is exactly the quantity an open-set / OOD textile sorter needs."""
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    err = (target_frac - p) ** 2
    var = p * (1.0 - p) / (S + 1.0)
    return (err + var).sum(dim=1).mean()


# ── Beer-Lambert unmixing decoder (analysis-by-synthesis) ────────────────────

class UnmixingDecoder(nn.Module):
    """Physics decoder: reconstruct the input spectrum from the PREDICTED
    composition and a set of learned pure-fibre endmember spectra.

        linear (LMM) :  x_hat = sum_k a_k * E_k
        bilinear(Fan):  + sum_{k<j} a_k a_j E_k*E_j   (2nd-order scattering)
        ppnm         :  + xi * (sum_k a_k E_k)^2      (polynomial post-nonlinear)

    Because a fibre can only receive abundance if its endmember helps
    reconstruct the measurement, this structurally suppresses the "mass leaked
    onto absent fibres" failure mode, and ||x - x_hat|| is a physically
    meaningful novelty score for materials outside the known 12 classes."""

    def __init__(self, num_classes, n_channels, length, mixing="linear",
                 init_endmembers=None, nonneg=True):
        super().__init__()
        self.mixing = mixing
        self.nonneg = nonneg
        self.n_channels = n_channels
        self.length = length

        # "mlp" is the ABLATION CONTROL, not a physics model: same input
        # (the composition), same reconstruction target, same loss weight --
        # but a free nonlinear decoder instead of Beer-Lambert mixing. It
        # isolates "does the reconstruction loss help?" from "does the LINEAR
        # MIXING STRUCTURE help?". It has no endmembers, so nothing about it
        # is interpretable; that contrast is the point.
        if mixing == "mlp":
            hidden = 256
            self.net = nn.Sequential(
                nn.Linear(num_classes, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, n_channels * length))
            self.endmembers = None
            return

        E = torch.zeros(num_classes, n_channels, length)
        if init_endmembers is not None:
            E.copy_(torch.as_tensor(init_endmembers, dtype=torch.float32))
        else:
            nn.init.uniform_(E, 0.05, 0.35)
        # stored pre-activation; softplus at use time keeps endmembers >= 0
        self.endmembers = nn.Parameter(torch.log(torch.expm1(E.clamp(min=1e-3))))
        if mixing == "ppnm":
            self.xi = nn.Parameter(torch.zeros(1))

    def get_endmembers(self):
        if self.endmembers is None:
            return None                      # mlp control has no endmembers
        return F.softplus(self.endmembers) if self.nonneg else self.endmembers

    def forward(self, abundance):
        """abundance: (B, K) non-negative, ~sums to 1. Returns (B, C, L)."""
        if self.mixing == "mlp":
            return self.net(abundance).view(-1, self.n_channels, self.length)
        E = self.get_endmembers()                       # (K, C, L)
        x_hat = torch.einsum("bk,kcl->bcl", abundance, E)
        if self.mixing == "bilinear":
            # Fan model: pairwise products approximated by (sum a E)^2 - sum a^2 E^2
            sq = x_hat ** 2
            pure_sq = torch.einsum("bk,kcl->bcl", abundance ** 2, E ** 2)
            x_hat = x_hat + 0.5 * (sq - pure_sq)
        elif self.mixing == "ppnm":
            x_hat = x_hat + self.xi * x_hat ** 2
        return x_hat


def spectral_angle_loss(x_hat, x, eps=1e-8):
    """Spectral Angle Distance: scale-invariant reconstruction loss standard in
    hyperspectral unmixing. Insensitive to the multiplicative scatter that
    dominates raw fabric reflectance, so it complements plain MSE."""
    a = x_hat.flatten(1); b = x.flatten(1)
    cos = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)
    return torch.arccos(cos.clamp(-1 + 1e-6, 1 - 1e-6)).mean()


def endmember_smoothness(E):
    """Second-difference penalty: real NIR endmembers are SMOOTH curves, not
    high-frequency noise. Standard minimum-roughness prior in unmixing."""
    return (E[..., 2:] - 2 * E[..., 1:-1] + E[..., :-2]).pow(2).mean()


def endmember_volume(E):
    """Minimum-volume-style penalty: pull endmembers toward their centroid so
    the simplex does not inflate into unoccupied spectral space."""
    mu = E.mean(dim=0, keepdim=True)
    return -((E - mu) ** 2).mean()


@torch.no_grad()
def estimate_endmembers_from_data(dataset, indices, num_classes, n_channels,
                                  refl_idx, purity=99.0, max_per_class=400,
                                  km=False):
    """Initialise endmembers from observed near-PURE samples (composition of a
    single fibre >= `purity`%). Physically grounded starting point; classes
    with no pure sample fall back to the global mean spectrum."""
    sums = np.zeros((num_classes, n_channels, dataset.X.shape[-1]), dtype=np.float64)
    cnt = np.zeros(num_classes, dtype=np.int64)      # used for the AVERAGE (capped)
    total = np.zeros(num_classes, dtype=np.int64)    # TRUE count (uncapped)
    glob = np.zeros((n_channels, dataset.X.shape[-1]), dtype=np.float64); gn = 0
    for i in indices:
        comp = np.asarray(dataset.comps[i], dtype=np.float64)
        xi = np.asarray(dataset.X[i], dtype=np.float64)
        t = torch.from_numpy(xi).float().unsqueeze(0)
        t = kubelka_munk(t, refl_idx) if km else to_absorbance(t, refl_idx)
        xa = t.squeeze(0).numpy()
        glob += xa; gn += 1
        k = int(np.argmax(comp))
        if comp[k] >= purity:
            total[k] += 1
            if cnt[k] < max_per_class:
                sums[k] += xa; cnt[k] += 1
    glob = glob / max(gn, 1)
    out = np.zeros_like(sums)
    for k in range(num_classes):
        out[k] = sums[k] / cnt[k] if cnt[k] > 0 else glob
    # `total` is the honest count; `cnt` only says how many were averaged.
    # Reporting `cnt` made this function disagree with the plotting script,
    # which counts every pure sample it sees.
    return out.astype(np.float32), total


# ── Beer-Lambert synthetic blend augmentation ────────────────────────────────

def synth_blend_batch(x, y, comp, refl_idx, max_fibers=4, alpha=1.0,
                      hard_labels=True, min_frac=0.0, eps=1e-4):
    """Synthesise NEW fibre blends by mixing spectra in ABSORBANCE space with
    Dirichlet weights, and derive their labels/compositions exactly.

    `min_frac` (percent) drops trace components below the threshold and
    renormalises, then relabels. Without it, Dirichlet(alpha=1) over 4 sources
    routinely produces 1-2% components that get labelled PRESENT -- teaching
    the model to fire on trace amounts, which inflates false positives and
    degrades composition MSE. Real compositions rarely go below ~2-5%, so set
    min_frac to match your dataset's smallest genuine fraction.

    Rationale for the whole function: the test split contains composition
    combinations never seen in training, and accuracy collapses on 3+ fibre
    blends. Beer-Lambert lets us manufacture physically-consistent high-order
    blends -- unlike generic mixup the (spectrum, composition) pair is not an
    approximation, it is what the mixture would actually measure under the
    linear model.

    NOTE ON DISTRIBUTION MATCH: mixing k sources shifts the blend-size
    histogram upward. If your test set is dominated by 1-3 fibre samples,
    large `max_fibers` with `alpha=1.0` trains on a distribution the test set
    does not contain. Prefer max_fibers<=3 and alpha<=0.5 (concentrates mass on
    few components) unless you specifically want the high-order regime.
    """
    B, C, L = x.shape
    k = int(np.random.randint(2, max_fibers + 1))
    w = np.random.dirichlet([alpha] * k).astype(np.float32)
    perms = [torch.arange(B, device=x.device)] + \
            [torch.randperm(B, device=x.device) for _ in range(k - 1)]

    xa = x.clone()
    for c in refl_idx:                                  # -> absorbance
        xa[:, c] = -torch.log10(xa[:, c].clamp(min=eps))
    mixed = sum(float(wi) * xa[p] for wi, p in zip(w, perms))
    for c in refl_idx:                                  # -> back to reflectance
        mixed[:, c] = torch.pow(10.0, -mixed[:, c]).clamp(min=eps, max=1.0)

    comp_mix = sum(float(wi) * comp[p] for wi, p in zip(w, perms))
    if min_frac > 0:
        comp_mix = torch.where(comp_mix < min_frac,
                               torch.zeros_like(comp_mix), comp_mix)
        comp_mix = 100.0 * comp_mix / (comp_mix.sum(1, keepdim=True) + 1e-6)
    if hard_labels:
        y_mix = (comp_mix > 0.5).float()
    else:
        y_mix = sum(float(wi) * y[p].float() for wi, p in zip(w, perms))
        if min_frac > 0:
            y_mix = y_mix * (comp_mix > 0.5).float()
    return mixed, y_mix, comp_mix


# ── The wrapper that ties it together ────────────────────────────────────────

class PhysicsWrapper(nn.Module):
    """Wraps ANY (logits, reg) backbone and adds the physics machinery.

    Returns (mlc, reg, aux) where aux carries everything the losses and the
    OOD/diagnostic reporting need. Backbones stay untouched, so the identical
    physics can be evaluated on ConvNeXtV2 / MaxViT / OverLock."""

    def __init__(self, base, args, n_channels, length, num_classes=12,
                 init_endmembers=None):
        super().__init__()
        self.base = base
        self.comp_head = getattr(args, "comp_head", "linear")
        self.scatter_mode = getattr(args, "scatter_correct", "none")
        self.input_mode = input_mode_of(args)
        if self.input_mode != "raw" and self.scatter_mode == "snv":
            raise ValueError("--input_mode and --scatter_correct snv both apply "
                             "SNV; use --input_mode alone")
        self.km = bool(getattr(args, "km_transform", False))
        self.refl_idx = reflectance_channel_indices(args.channels, args.channel_idx)
        self.use_decoder = bool(getattr(args, "physics_decoder", False))
        self.physics_km_target = bool(getattr(args, "physics_km_target", False))
        # How the decoder's abundance is derived from the composition head.
        #   "normalize" : clamp>=0 then divide by the sum -> simplex (DEFAULT).
        #                 Scale-INVARIANT: isolates the relative proportions,
        #                 which is what the endmembers actually identify, and
        #                 lets comp_head stay "linear".
        #   "as_is"     : reg/100 clamped. Also penalises leakage, but is NOT
        #                 scale-invariant, so during joint training the
        #                 endmembers can absorb a systematic scale bias.
        #   "softmax"   : NOT RECOMMENDED -- softmax(reg/100) with reg in
        #                 [0,100] is nearly uniform, so it barely distinguishes
        #                 compositions at all (measured penalty ratio ~1x).
        self.abundance_mode = getattr(args, "physics_abundance", "normalize")
        self.abundance_softplus = bool(getattr(args, "physics_abundance_softplus", False))

        self.emsc = (LearnableEMSC(n_channels, length,
                                   int(getattr(args, "emsc_degree", 2)))
                     if self.scatter_mode == "emsc" else None)
        self.decoder = (UnmixingDecoder(
                            num_classes, n_channels, length,
                            mixing=getattr(args, "physics_mixing", "linear"),
                            init_endmembers=init_endmembers,
                            nonneg=True)
                        if self.use_decoder else None)

    def _preprocess(self, x):
        if self.km:
            x = kubelka_munk(x, self.refl_idx)
        x = apply_scatter_correction(x, self.scatter_mode, self.emsc)
        # Backbone input representation. Applied LAST and only to what the
        # encoder sees; forward() still builds the decoder target from the
        # original x, so the physics is untouched by this choice.
        return apply_input_mode(x, self.input_mode)

    def _composition(self, mlc, reg_raw):
        """Map the raw regression output to a composition in [0,100]."""
        aux = {}
        if self.comp_head == "linear":
            return reg_raw, aux
        if self.comp_head == "gated":
            # presence-gated, non-negative, renormalised to sum 100: the two
            # heads can no longer contradict each other, and regression
            # gradients also sharpen the classifier logits.
            g = torch.sigmoid(mlc) * F.softplus(reg_raw)
            return 100.0 * g / (g.sum(1, keepdim=True) + 1e-6), aux
        if self.comp_head == "softmax":
            return 100.0 * torch.softmax(reg_raw, dim=1), aux
        if self.comp_head == "sparsemax":
            return 100.0 * sparsemax(reg_raw, dim=1), aux
        if self.comp_head == "simplex":
            return project_simplex_torch(reg_raw, z=100.0, dim=1), aux
        if self.comp_head == "dirichlet":
            alpha = F.softplus(reg_raw) + 1.0
            S = alpha.sum(1, keepdim=True)
            aux["alpha"] = alpha
            aux["vacuity"] = (alpha.size(1) / S).squeeze(1)   # evidential uncertainty
            return 100.0 * alpha / S, aux
        raise ValueError(f"unknown comp_head '{self.comp_head}'")

    def forward(self, x):
        x_in = self._preprocess(x)
        mlc, reg_raw = self.base(x_in)
        reg, aux = self._composition(mlc, reg_raw)
        aux["reg_raw"] = reg_raw
        aux["composition"] = reg

        if self.decoder is not None:
            # Reconstruct in the space where mixing is linear.
            target = (kubelka_munk(x, self.refl_idx) if self.physics_km_target
                      else to_absorbance(x, self.refl_idx))
            # Abundance fed to the decoder. This is deliberately DECOUPLED from
            # the reported composition: the reconstruction loss is invariant to
            # the scale of `abundance` (endmembers simply rescale to absorb it),
            # so unless it is projected onto the simplex the physics term
            # carries no information about the composition at all. Normalising
            # here lets comp_head stay "linear" -- keeping whatever accuracy
            # that head gives you -- while the physics still constrains the
            # RELATIVE proportions, which is what the endmembers identify.
            if self.abundance_mode == "as_is":
                abundance = (reg / 100.0).clamp(min=0.0)
            elif self.abundance_mode == "softmax":
                abundance = torch.softmax(reg / 100.0, dim=1)
            else:                                    # "normalize" (default)
                a = F.softplus(reg / 100.0) if self.abundance_softplus \
                    else reg.clamp(min=0.0)
                abundance = a / (a.sum(dim=1, keepdim=True) + 1e-6)
            aux["abundance"] = abundance
            x_hat = self.decoder(abundance)
            aux["recon"] = x_hat
            aux["recon_target"] = target
            # per-sample residual -> physics-grounded OOD / novelty score
            aux["residual"] = ((x_hat - target) ** 2).flatten(1).mean(1)
        return mlc, reg, aux


def fwd(model, x):
    """Call a model that may return (mlc, reg) or (mlc, reg, aux).
    Keeps external student scripts (maxvit/overlock/accvit) working unchanged."""
    out = model(x)
    if isinstance(out, (tuple, list)) and len(out) == 3:
        return out[0], out[1], out[2]
    return out[0], out[1], {}


def physics_losses(aux, comp_target, mlc_out, y_target, args):
    """All physics-informed loss terms. Returns (total, dict_for_logging)."""
    logs, total = {}, torch.zeros((), device=mlc_out.device)

    # 1. Analysis-by-synthesis reconstruction
    if "recon" in aux and getattr(args, "physics_weight", 0.0) > 0:
        xh, xt = aux["recon"], aux["recon_target"]
        kind = getattr(args, "physics_recon_loss", "mse")
        if kind == "mse":
            l_rec = F.mse_loss(xh, xt)
        elif kind == "sad":
            l_rec = spectral_angle_loss(xh, xt)
        else:                                   # "both"
            l_rec = F.mse_loss(xh, xt) + spectral_angle_loss(xh, xt)
        total = total + args.physics_weight * l_rec
        logs["phys/recon"] = float(l_rec.detach())

    # 2. Endmember priors (smoothness / min-volume / anchoring to pure spectra)
    dec = getattr(args, "_decoder_ref", None)
    if dec is not None and dec.get_endmembers() is not None:
        E = dec.get_endmembers()
        if getattr(args, "endmember_smooth_weight", 0.0) > 0:
            l_sm = endmember_smoothness(E)
            total = total + args.endmember_smooth_weight * l_sm
            logs["phys/em_smooth"] = float(l_sm.detach())
        if getattr(args, "endmember_volume_weight", 0.0) > 0:
            l_v = endmember_volume(E)
            total = total + args.endmember_volume_weight * l_v
            logs["phys/em_volume"] = float(l_v.detach())
        anchor = getattr(args, "_endmember_anchor", None)
        if anchor is not None and getattr(args, "endmember_anchor_weight", 0.0) > 0:
            l_a = F.mse_loss(E, anchor.to(E.device))
            total = total + args.endmember_anchor_weight * l_a
            logs["phys/em_anchor"] = float(l_a.detach())

    # 3. Evidential (Dirichlet) composition likelihood
    if "alpha" in aux and getattr(args, "evidential_weight", 0.0) > 0:
        l_ev = dirichlet_composition_loss(aux["alpha"], comp_target / 100.0)
        total = total + args.evidential_weight * l_ev
        logs["phys/evidential"] = float(l_ev.detach())

    # 4. Head-consistency: predicted PRESENCE must agree with predicted MASS.
    #    Targets exactly the 20k+ "regressor-present but classifier-empty"
    #    disagreement cells in the diagnostics. Symmetric hinge, fully
    #    differentiable, and uses NO labels -- it is a structural constraint,
    #    so it can also be applied to unlabelled / test-time data later.
    if getattr(args, "consistency_weight", 0.0) > 0 and "composition" in aux:
        p = torch.sigmoid(mlc_out)                       # (B,K) presence prob
        a = aux["composition"] / 100.0                   # (B,K) predicted mass
        tau = float(getattr(args, "consistency_tau", 0.01))
        # mass present but classifier says absent  ->  (1-p) * relu(a - tau)
        # classifier says present but no mass      ->  p     * relu(tau - a)
        l_c = ((1.0 - p) * F.relu(a - tau) + p * F.relu(tau - a)).mean()
        total = total + args.consistency_weight * l_c
        logs["phys/consistency"] = float(l_c.detach())

    return total, logs


@torch.no_grad()
def leakage_metric(reg, comps):
    """Mean predicted composition mass placed on truly-ABSENT fibres.
    The single most diagnostic number for the physics decoder's core claim;
    report it alongside MSE."""
    absent = (comps <= 0.5)
    if absent.sum() == 0:
        return 0.0
    return float(np.abs(reg[absent]).mean())


# ═══════════════════════════════════════════════════════════════════════════════
#  HELD-OUT-CLASS (OPEN-SET) PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════════
#  Remove one or more fibres from TRAINING entirely, then ask whether the
#  physics decoder flags test samples that contain them as novel.
#
#  The prediction under test (made BEFORE running, from the endmember geometry):
#  a held-out fibre is detectable only if the REMAINING endmembers cannot
#  synthesise it. Spandex is 26.71 deg from its nearest neighbour -> predict
#  DETECTED. Linen is 1.24 deg from Cotton -> predict MISSED, because Cotton's
#  endmember can impersonate it almost exactly.
#
#  Note the classes are genuinely REMOVED (K -> K - h), not merely masked. A
#  masked-but-present endmember would be an unconstrained slack channel free to
#  absorb the novel material, which would destroy the very signal we measure.
# ═══════════════════════════════════════════════════════════════════════════════

def parse_holdout(spec):
    """'CA,M' or '1,4' -> sorted list of class indices. Empty -> []."""
    if not spec:
        return []
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if tok == "":
            continue
        if tok.isdigit():
            out.append(int(tok))
        else:
            up = tok.upper()
            match = [i for i, n in enumerate(FIBER_NAMES) if n.upper() == up]
            if not match:
                raise ValueError(f"unknown fibre '{tok}'; choose from {FIBER_NAMES}")
            out.append(match[0])
    return sorted(set(out))


def apply_holdout(ds, holdout, keep, drop_samples):
    """Restrict a dataset to the kept classes.
    drop_samples=True  -> also delete every sample containing a held-out fibre
                          (use for TRAIN and for the 'known' evaluation split).
    drop_samples=False -> keep all rows, only drop the columns (use when the
                          held-out rows are themselves the OOD positives)."""
    contains = (np.asarray(ds.comps)[:, holdout] > 0.5).any(axis=1)
    if drop_samples:
        m = ~contains
        ds.X = ds.X[m]; ds.Y = np.asarray(ds.Y)[m]; ds.comps = np.asarray(ds.comps)[m]
        contains = contains[m]
    ds.Y = np.asarray(ds.Y)[:, keep]
    ds.comps = np.asarray(ds.comps)[:, keep]
    return contains


@torch.no_grad()
def evaluate_open_set(model, ds_all, is_unknown, batch=512, prefix="openset"):
    """Score every test sample with the physics reconstruction residual and ask
    how well it separates samples CONTAINING a held-out fibre from the rest."""
    model.eval()
    res, probs = [], []
    for s in range(0, len(ds_all.X), batch):
        x = torch.from_numpy(ds_all.X[s:s + batch]).float().cuda()
        mlc, reg, aux = fwd(model, x)
        if "residual" not in aux:
            print("  [open-set] no physics decoder -> residual unavailable; skipped.")
            return None
        res.append(aux["residual"].cpu().numpy())
        probs.append(torch.sigmoid(mlc).cpu().numpy())
    res = np.concatenate(res); probs = np.concatenate(probs)
    y = is_unknown.astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        print("  [open-set] test split has no contrast; skipped.")
        return None

    def rates(score):
        auroc = float(metrics.roc_auc_score(y, score))
        aupr = float(metrics.average_precision_score(y, score))
        thr = np.percentile(score[y == 1], 5)          # 95% of unknowns caught
        fpr95 = float((score[y == 0] >= thr).mean())
        return auroc, aupr, fpr95

    a_r, p_r, f_r = rates(res)
    # baseline OOD score from the classifier alone: low max-probability
    a_m, p_m, f_m = rates(-probs.max(axis=1))
    print("\n" + "=" * 70)
    print(f"  OPEN-SET DETECTION  ({int(y.sum()):,} unknown / {int((1-y).sum()):,} known)")
    print(f"{'='*70}")
    print(f"  {'score':<28}{'AUROC':>9}{'AUPR':>9}{'FPR@95TPR':>12}")
    print(f"  {'physics recon residual':<28}{a_r:>9.4f}{p_r:>9.4f}{f_r:>12.4f}")
    print(f"  {'classifier max-prob (base)':<28}{a_m:>9.4f}{p_m:>9.4f}{f_m:>12.4f}")
    print(f"  residual: known p50={np.percentile(res[y==0],50):.5f}  "
          f"p99={np.percentile(res[y==0],99):.5f}   "
          f"unknown p50={np.percentile(res[y==1],50):.5f}")
    wandb.log({f"{prefix}/auroc_residual": a_r, f"{prefix}/aupr_residual": p_r,
               f"{prefix}/fpr95_residual": f_r, f"{prefix}/auroc_maxprob": a_m})
    return {"auroc": a_r, "aupr": p_r, "fpr95": f_r, "auroc_maxprob": a_m}


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASS-IMBALANCE CORRECTIONS  (Stage 1)
# ═══════════════════════════════════════════════════════════════════════════════
#  Motivation from the diagnostics: rare fibres are substituted by 4-17x more
#  COMMON fibres (CA->W, M->C/R, T->P), corr(frequency, F1) = +0.61, and the
#  substitutions persist in 1-FIBRE samples where co-occurrence is impossible.
#  That is the signature of a class-PRIOR effect, not a spectral resolution
#  limit. These two corrections target it directly.
# ═══════════════════════════════════════════════════════════════════════════════

class _LAState:
    """Module-level holder so validate()/evaluate_on_test() can reach the
    training-set class prior without threading it through every signature."""
    prior = None                                    # (K,) P(y_j = 1) on TRAIN
    taus = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)         # post-hoc sweep grid


_LA = _LAState()


def compute_class_prior(labels, eps=1e-6):
    """P(y_j = 1) per fibre, from the TRAINING labels only. (N, K) -> (K,)"""
    p = np.asarray(labels, dtype=np.float64).mean(axis=0)
    return np.clip(p, eps, 1.0 - eps)


def logit_offset(prior, tau=1.0):
    """tau * log(pi / (1-pi)) -- the Bayes-optimal threshold shift for a binary
    problem with positive rate pi (Menon et al., ICLR 2021, adapted per-label
    because our head is K independent sigmoids, not a softmax)."""
    pr = np.asarray(prior, dtype=np.float64)
    return tau * np.log(pr / (1.0 - pr))


def adjust_logits(logits, prior, tau=1.0):
    """POST-HOC logit adjustment: subtract the prior offset at inference.
    Requires no retraining -- it only moves each class's decision threshold to
    where it would be under a balanced prior. tau=0 reproduces the raw model."""
    if tau == 0:
        return logits
    return logits - logit_offset(prior, tau)[None, :]


class DistributionBalancedLoss(nn.Module):
    """Distribution-Balanced Loss for multi-label classification
    (Wu, Huang, Liu, Wang & Lin, ECCV 2020 -- arXiv:2007.09654).

    Two modifications to plain BCE, both aimed at long-tailed MULTI-label data:

      1. Re-balanced weighting that accounts for LABEL CO-OCCURRENCE. Naive
         inverse-frequency weighting over-counts a rare fibre every time it
         co-occurs with a common one; the rebalance term divides the expected
         class-level sampling rate by the instance-level one to undo that.

      2. Negative-tolerant regularisation. With 12 labels and ~2 present per
         sample, negatives dominate ~5:1 and over-suppress rare positives. A
         class-specific bias `nu` plus a scale on the negative term relieves it.

    Defaults follow the official ResampleLoss configuration.
    """

    def __init__(self, class_freq, train_num, alpha=0.1, beta=10.0, mu=0.3,
                 neg_scale=5.0, init_bias=0.05, focal_gamma=2.0, focal_alpha=0.5,
                 use_focal=True):
        super().__init__()
        cf = torch.as_tensor(class_freq, dtype=torch.float32).clamp(min=1.0)
        self.register_buffer("class_freq", cf)
        self.register_buffer("freq_inv", 1.0 / cf)
        self.train_num = float(train_num)
        self.alpha, self.beta, self.mu = alpha, beta, mu
        self.neg_scale, self.use_focal = neg_scale, use_focal
        self.focal_gamma, self.focal_alpha = focal_gamma, focal_alpha
        # class-specific bias: pushes the decision point of rare classes
        nu = -torch.log(self.train_num / cf - 1.0) * init_bias / neg_scale
        self.register_buffer("nu", nu)

    def rebalance_weight(self, y):
        # expected sampling rate if we sampled by CLASS vs by INSTANCE
        repeat_rate = (self.freq_inv[None, :] * y).sum(dim=1, keepdim=True) \
                      / y.sum(dim=1, keepdim=True).clamp(min=1.0)
        pos_w = self.freq_inv[None, :] / repeat_rate.clamp(min=1e-8)
        return torch.sigmoid(self.beta * (pos_w - self.mu)) + self.alpha

    def forward(self, logits, y):
        y = y.float()
        w = self.rebalance_weight(y)                       # (B, K)
        z = logits + self.nu[None, :]                      # class-specific bias
        # scale down the NEGATIVE branch so rare positives are not over-suppressed
        z = z * y + z * (1.0 - y) * self.neg_scale
        w = w * y + w * (1.0 - y) / self.neg_scale
        loss = F.binary_cross_entropy_with_logits(z, y, reduction="none")
        if self.use_focal:
            p = torch.sigmoid(z)
            pt = p * y + (1 - p) * (1 - y)
            fw = (self.focal_alpha * y + (1 - self.focal_alpha) * (1 - y)) \
                 * (1 - pt).pow(self.focal_gamma)
            loss = fw * loss
        return (loss * w).mean()


def classification_loss(mlc_out, y_target, args, db_module=None):
    """Dispatch the multi-label classification loss, optionally with TRAIN-TIME
    logit adjustment (which enforces a larger margin for rare classes rather
    than only shifting the threshold afterwards)."""
    tau = float(getattr(args, "logit_adjust_train", 0.0)) if args is not None else 0.0
    if tau > 0 and getattr(args, "_prior_t", None) is not None:
        mlc_out = mlc_out + tau * args._prior_t.to(mlc_out.device)[None, :]
    kind = getattr(args, "cls_loss", "soft_margin") if args is not None else "soft_margin"
    if kind == "db" and db_module is not None:
        return db_module(mlc_out, y_target)
    return F.multilabel_soft_margin_loss(mlc_out, y_target)


def logit_adjust_sweep(probs_or_logits, labels, prior, taus, is_logits=True):
    """Exact-match accuracy and macro-F1 across a sweep of tau.
    Returns {tau: (acc, macro_f1)} plus the best tau by accuracy."""
    out = {}
    z = probs_or_logits if is_logits else np.log(
        np.clip(probs_or_logits, 1e-6, 1 - 1e-6) /
        (1 - np.clip(probs_or_logits, 1e-6, 1 - 1e-6)))
    for t in taus:
        preds = (adjust_logits(z, prior, t) > 0).astype(int)
        acc = float(metrics.accuracy_score(labels, preds))
        f1 = float(metrics.f1_score(labels, preds, average="macro", zero_division=0))
        out[float(t)] = (acc, f1)
    best = max(out, key=lambda k: out[k][0])
    return out, best


# ── Post-hoc per-class threshold tuning (reported only; never touches training)
def subset_acc_with_thresholds(probs, labels, thr):
    preds = (probs > thr[None, :]).astype(int)
    return float((preds == labels).all(axis=1).mean())


def hamming_with_thresholds(probs, labels, thr):
    preds = (probs > thr[None, :]).astype(int)
    return float((preds != labels).mean())


def tune_thresholds_subset_acc(probs, labels, grid=None, n_rounds=3):
    """Per-class thresholds maximising exact-match accuracy via coordinate
    ascent (subset accuracy isn't class-separable). Returns (thr[C], acc)."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    probs = np.asarray(probs); labels = np.asarray(labels)
    C = probs.shape[1]
    thr = np.full(C, 0.5, dtype=float)
    best = subset_acc_with_thresholds(probs, labels, thr)
    for _ in range(n_rounds):
        improved = False
        for c in range(C):
            best_t, best_c = thr[c], best
            for t in grid:
                thr[c] = t
                a = subset_acc_with_thresholds(probs, labels, thr)
                if a > best_c:
                    best_c, best_t = a, t
            thr[c] = best_t
            if best_c > best:
                best, improved = best_c, True
        if not improved:
            break
    return thr, best


def tune_thresholds_hamming(probs, labels, grid=None):
    """Per-class thresholds minimising Hamming loss (separable, one pass)."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    probs = np.asarray(probs); labels = np.asarray(labels)
    C = probs.shape[1]
    thr = np.full(C, 0.5, dtype=float)
    for c in range(C):
        yc = labels[:, c]
        best_err, best_t = 1.0, 0.5
        for t in grid:
            err = np.mean((probs[:, c] > t).astype(int) != yc)
            if err < best_err:
                best_err, best_t = err, t
        thr[c] = best_t
    return thr, hamming_with_thresholds(probs, labels, thr)


def tuned_metrics(probs, labels, reg, comps, seed=0):
    """
    Tuned (honest) and oracle (optimistic) versions of exact-match accuracy,
    Hamming loss, and MSE.

      HONEST  : split val in half, FIT thresholds on one half, REPORT on the
                other -> generalisable estimate (this is what save_best_acc uses).
      ORACLE  : acc/hamming fit+report on the FULL set (peeking) = upper bound;
                MSE uses the TRUE presence mask to gate the regression = the MSE
                you'd get if presence detection were perfect.

    Tuned MSE gates the regression by PREDICTED presence (zeroing fibers judged
    absent, which is where most regression error on absent fibers comes from).
    Returns (metrics_dict, honest_acc_thresholds or None, full_val_acc_thresholds,
             full_val_hamming_thresholds).
    """
    out = {}
    thr_honest = None
    N = probs.shape[0]
    idx = np.arange(N)
    np.random.default_rng(seed).shuffle(idx)
    fit, ev = idx[:N // 2], idx[N // 2:]

    if len(fit) > 0 and len(ev) > 0:
        thr_a, _ = tune_thresholds_subset_acc(probs[fit], labels[fit])
        thr_honest = thr_a
        out["val/acc_tuned"] = subset_acc_with_thresholds(probs[ev], labels[ev], thr_a)
        thr_h, _ = tune_thresholds_hamming(probs[fit], labels[fit])
        out["val/hamming_tuned"] = hamming_with_thresholds(probs[ev], labels[ev], thr_h)
        mask_ev = (probs[ev] > thr_a[None, :]).astype(float)
        out["val/mse_tuned"] = float(((reg[ev] * mask_ev - comps[ev]) ** 2).mean())

    # "Oracle" thresholds are fit+reported on the FULL val set (peeking, so the
    # accuracy number is optimistic) -- but the thresholds themselves are the
    # best estimate we have for deployment, since they use all available data.
    thr_oracle, acc_or = tune_thresholds_subset_acc(probs, labels)
    out["val/acc_tuned_oracle"] = acc_or
    thr_ham_oracle, ham_or = tune_thresholds_hamming(probs, labels)
    out["val/hamming_tuned_oracle"] = ham_or
    mask_true = labels.astype(float)
    out["val/mse_tuned_oracle"] = float(((reg * mask_true - comps) ** 2).mean())
    return out, thr_honest, thr_oracle, thr_ham_oracle


def compute_per_class_metrics(labels, logits, threshold=0.5):
    """Per-fiber precision / recall / F1 at a fixed threshold."""
    preds = (torch.sigmoid(torch.as_tensor(logits)) > threshold).int().numpy()
    labels = np.asarray(labels)
    out = {}
    for i, name in enumerate(FIBER_NAMES):
        tp = int(((preds[:, i] == 1) & (labels[:, i] == 1)).sum())
        fp = int(((preds[:, i] == 1) & (labels[:, i] == 0)).sum())
        fn = int(((preds[:, i] == 0) & (labels[:, i] == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[f"{name}/precision"] = prec
        out[f"{name}/recall"] = rec
        out[f"{name}/f1"] = f1
    return out


def compute_per_class_mae(true_comps, pred_comps):
    """Per-fiber mean absolute composition error."""
    return {f"{name}/mae": float(np.abs(true_comps[:, i] - pred_comps[:, i]).mean())
            for i, name in enumerate(FIBER_NAMES)}


# ==================== DIAGNOSTICS ====================

def run_diagnostics(logits, reg, y_labels, comps, epoch=0, step=None,
                    cls_thresh=0.5, reg_thresh=0.5, log_wandb=True):
    """Error analysis: exact-match by #fibers, per-class exact-match by blend
    size, per-class P/R/F1, head disagreement, and exact-match from the
    regression head alone. Prints, and optionally logs to wandb."""
    logits = np.asarray(logits); reg = np.asarray(reg)
    y_labels = np.asarray(y_labels); comps = np.asarray(comps)
    probs = 1.0 / (1.0 + np.exp(-logits))
    cls_pred = (probs > cls_thresh).astype(int)
    reg_pred_present = (reg > reg_thresh).astype(int)
    n_fibers = y_labels.sum(axis=1).astype(int)
    exact = (cls_pred == y_labels).all(axis=1)

    print(f"\n{'='*60}\n  DIAGNOSTICS (epoch {epoch})\n{'='*60}")
    print("  Exact-match accuracy by #fibers in sample:")
    strat = {}
    for k in sorted(set(n_fibers.tolist())):
        mask = n_fibers == k
        a = float(exact[mask].mean()) if mask.any() else 0.0
        strat[int(k)] = (a, int(mask.sum()))
        print(f"    {k} fiber(s): acc={a:.4f}  (n={int(mask.sum())})")

    ks = sorted(set(n_fibers.tolist()))
    print("\n  Per-class exact-match accuracy by blend size  [acc (n); NaN = none]:")
    header = "    " + f"{'class':>11} | " + " ".join(f"{f'{k}f':>13}" for k in ks)
    print(header)
    print("    " + "-" * (len(header) - 4))
    per_class_by_k = {}
    for i, name in enumerate(FIBER_NAMES):
        cells, row = {}, []
        for k in ks:
            mask = (y_labels[:, i] == 1) & (n_fibers == k)
            n_k = int(mask.sum())
            if n_k > 0:
                a = float(exact[mask].mean()); cells[k] = (a, n_k)
                row.append(f"{f'{a:.3f}({n_k})':>13}")
            else:
                cells[k] = (float('nan'), 0); row.append(f"{'NaN':>13}")
        per_class_by_k[name] = cells
        print(f"    {name+'/'+FIBER_FULL_NAMES.get(name, name):>11} | " + " ".join(row))

    print("\n  Per-class precision / recall / F1:")
    per_class = {}
    for i, name in enumerate(FIBER_NAMES):
        tp = int(((cls_pred[:, i] == 1) & (y_labels[:, i] == 1)).sum())
        fp = int(((cls_pred[:, i] == 1) & (y_labels[:, i] == 0)).sum())
        fn = int(((cls_pred[:, i] == 0) & (y_labels[:, i] == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[name] = (prec, rec, f1)
        print(f"    {name:>3}: P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

    cls_not_reg = int(((cls_pred == 1) & (reg_pred_present == 0)).sum())
    reg_not_cls = int(((cls_pred == 0) & (reg_pred_present == 1)).sum())
    print("  Head disagreement (label-cells):")
    print(f"    classifier-present but regressor-empty: {cls_not_reg}")
    print(f"    regressor-present but classifier-empty: {reg_not_cls}")

    exact_reg = (reg_pred_present == y_labels).all(axis=1).mean()
    print(f"  Exact-match from regression presence alone: {exact_reg:.4f}")
    print(f"  Exact-match from classifier (ref):          {exact.mean():.4f}")
    print(f"{'='*60}\n")

    if log_wandb:
        logs = {f"diag/acc_by_{k}fiber": v[0] for k, v in strat.items()}
        logs["diag/exact_from_regression"] = float(exact_reg)
        logs["diag/disagree_cls_not_reg"] = cls_not_reg
        logs["diag/disagree_reg_not_cls"] = reg_not_cls
        for name, (p, r, f1) in per_class.items():
            logs[f"diag/{name}/f1"] = f1
        for name, cells in per_class_by_k.items():
            for k, (a, n_k) in cells.items():
                if n_k > 0:
                    logs[f"diag/{name}/acc_{k}fiber"] = a
        wandb.log(logs, step=step)

    confusion_analysis(logits, y_labels, cls_thresh=cls_thresh, step=step, log_wandb=log_wandb)
    return {"strat": strat, "per_class": per_class, "per_class_by_k": per_class_by_k}


def confusion_analysis(logits, y_labels, cls_thresh=0.5, focus_classes=None,
                       weak_f1=0.80, step=None, log_wandb=True):
    """For each weak class, where its mass goes when missed: top spurious
    substitutes, same-family swap rates, and whether misses are borderline
    (recall/threshold fixable) or confident (resolution limit)."""
    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    y = np.asarray(y_labels)
    pred = (probs > cls_thresh).astype(int)
    n = len(FIBER_NAMES)
    name_to_idx = {nm: i for i, nm in enumerate(FIBER_NAMES)}

    sub = np.zeros((n, n)); fn_counts = np.zeros(n)
    for i in range(y.shape[0]):
        fn = np.where((y[i] == 1) & (pred[i] == 0))[0]
        fp = np.where((y[i] == 0) & (pred[i] == 1))[0]
        for c in fn:
            fn_counts[c] += 1
            for k in fp:
                sub[c, k] += 1

    group_of = {}
    for g_idx, members in enumerate(_PARENT_GROUPS):
        for m in members:
            group_of[m] = g_idx

    if focus_classes is None:
        focus = []
        for c in range(n):
            tp = int(((pred[:, c] == 1) & (y[:, c] == 1)).sum())
            fp_ = int(((pred[:, c] == 1) & (y[:, c] == 0)).sum())
            fn_ = int(((pred[:, c] == 0) & (y[:, c] == 1)).sum())
            prec = tp / (tp + fp_) if (tp + fp_) else 0.0
            rec = tp / (tp + fn_) if (tp + fn_) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if f1 < weak_f1:
                focus.append(c)
    else:
        focus = [name_to_idx[nm] for nm in focus_classes]

    print(f"\n{'='*60}\n  CONFUSION / SUBSTITUTION ANALYSIS\n{'='*60}")
    result = {}
    for c in focus:
        cname = FIBER_NAMES[c]; nfn = fn_counts[c]
        if nfn == 0:
            continue
        row = sub[c] / nfn
        subs = [(FIBER_NAMES[k], float(row[k])) for k in np.argsort(-row)[:3] if row[k] > 0]
        fn_mask = (y[:, c] == 1) & (pred[:, c] == 0)
        fn_probs = probs[fn_mask, c]
        borderline = float((fn_probs >= 0.30).mean()) if fn_probs.size else 0.0
        confident = float((fn_probs < 0.10).mean()) if fn_probs.size else 0.0
        mean_fn_prob = float(fn_probs.mean()) if fn_probs.size else 0.0
        sibs = [m for m in _PARENT_GROUPS[group_of[c]] if m != c]
        sib_swaps = [(FIBER_NAMES[s], float(sub[c, s] / nfn)) for s in sibs]
        result[cname] = {"misses": int(nfn), "mean_fn_prob": mean_fn_prob,
                         "borderline_frac": borderline, "sibling_swaps": sib_swaps}
        sub_str = ", ".join(f"{nm}:{v:.0%}" for nm, v in subs) or "(none)"
        sib_str = ", ".join(f"{nm}:{v:.0%}" for nm, v in sib_swaps) or "(none)"
        verdict = ("BORDERLINE -> recall/threshold can help" if mean_fn_prob >= 0.25
                   else "CONFIDENT miss -> likely resolution limit")
        print(f"  {cname:>3} (missed {int(nfn)}x): substitutes [{sub_str}]")
        print(f"        same-group swaps [{sib_str}]")
        print(f"        mean prob on misses={mean_fn_prob:.3f}  -> {verdict}")
    print(f"{'='*60}\n")

    if log_wandb:
        logs = {}
        for cname, d in result.items():
            logs[f"confusion/{cname}/mean_fn_prob"] = d["mean_fn_prob"]
            logs[f"confusion/{cname}/borderline_frac"] = d["borderline_frac"]
            for nm, v in d["sibling_swaps"]:
                logs[f"confusion/{cname}/swap_{nm}"] = v
        if logs:
            wandb.log(logs, step=step)
    return result


def regression_diagnostics(reg, comps, y_labels, step=None, log_wandb=True):
    """Regression-error analysis mirroring the classification diagnostics:
    MSE/MAE by #fibers in a sample, per-class composition MAE (present-only and
    overall), and a per-class MAE-by-blend-size table."""
    reg = np.asarray(reg); comps = np.asarray(comps); y = np.asarray(y_labels)
    n_fibers = y.sum(axis=1).astype(int)
    ks = sorted(set(n_fibers.tolist()))
    ae = np.abs(reg - comps)          # (N, 12)
    se = (reg - comps) ** 2

    print(f"\n{'='*60}\n  REGRESSION DIAGNOSTICS\n{'='*60}")

    # 1. MSE / MAE by #fibers in the sample
    print("  MSE / MAE by #fibers in sample:")
    by_k = {}
    for k in ks:
        m = n_fibers == k
        mse = float(se[m].mean()); mae = float(ae[m].mean())
        by_k[k] = (mse, mae, int(m.sum()))
        print(f"    {k} fiber(s): MSE={mse:.4f}  MAE={mae:.4f}  (n={int(m.sum())})")

    # 2. Per-class composition MAE (on samples where the fiber is present, and overall)
    print("\n  Per-class composition MAE  [present-only (n) | all]:")
    per_class_mae = {}
    for i, name in enumerate(FIBER_NAMES):
        present = y[:, i] == 1
        mae_p = float(ae[present, i].mean()) if present.any() else float('nan')
        mae_all = float(ae[:, i].mean())
        per_class_mae[name] = (mae_p, mae_all, int(present.sum()))
        lab = f"{name}/{FIBER_FULL_NAMES.get(name, name)}"
        pstr = "NaN" if present.sum() == 0 else f"{mae_p:.3f} (n={int(present.sum())})"
        print(f"    {lab:>11}: present={pstr:>16}   all={mae_all:.3f}")

    # 3. Per-class MAE by blend size (MAE of fiber c on samples containing c with k fibers)
    print("\n  Per-class composition MAE by blend size  [MAE (n); NaN = none]:")
    header = "    " + f"{'class':>11} | " + " ".join(f"{f'{k}f':>13}" for k in ks)
    print(header)
    print("    " + "-" * (len(header) - 4))
    per_class_by_k = {}
    for i, name in enumerate(FIBER_NAMES):
        cells, row = {}, []
        for k in ks:
            mask = (y[:, i] == 1) & (n_fibers == k)
            n_k = int(mask.sum())
            if n_k > 0:
                v = float(ae[mask, i].mean()); cells[k] = (v, n_k)
                row.append(f"{f'{v:.2f}({n_k})':>13}")
            else:
                cells[k] = (float('nan'), 0); row.append(f"{'NaN':>13}")
        per_class_by_k[name] = cells
        print(f"    {name+'/'+FIBER_FULL_NAMES.get(name, name):>11} | " + " ".join(row))
    print(f"{'='*60}\n")

    if log_wandb:
        logs = {}
        for k, (mse, mae, _) in by_k.items():
            logs[f"reg_diag/mse_by_{k}fiber"] = mse
            logs[f"reg_diag/mae_by_{k}fiber"] = mae
        for name, (mae_p, mae_all, npr) in per_class_mae.items():
            if npr > 0:
                logs[f"reg_diag/{name}/mae_present"] = mae_p
            logs[f"reg_diag/{name}/mae_all"] = mae_all
        for name, cells in per_class_by_k.items():
            for k, (v, n_k) in cells.items():
                if n_k > 0:
                    logs[f"reg_diag/{name}/mae_{k}fiber"] = v
        wandb.log(logs, step=step)
    return {"by_k": by_k, "per_class_mae": per_class_mae, "per_class_by_k": per_class_by_k}


def confusion_by_blend_size(logits, y_labels, focus=("CA", "M", "T"),
                            cls_thresh=0.5, top=3, step=None, log_wandb=True):
    """
    For each focus fiber (default the hard ones CA/Modal/Tencel), broken down by
    blend size k (# fibers in the sample): when that fiber is *missed* in a
    k-fiber blend, which other classes fired spuriously most often -> i.e. what
    it gets confused with, per blend size. Also reports recall at that blend
    size so you can see where the misses concentrate.
    """
    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    y = np.asarray(y_labels)
    pred = (probs > cls_thresh).astype(int)
    n_fibers = y.sum(axis=1).astype(int)
    ks = sorted(set(n_fibers.tolist()))
    fp_mat = ((y == 0) & (pred == 1)).astype(int)        # (N, 12) spurious fires

    print(f"\n{'='*60}\n  CA / M / T CONFUSION BY BLEND SIZE\n{'='*60}")
    print("  (when the fiber is missed in a k-fiber blend, what fired instead)\n")
    result = {}
    for cname in focus:
        if cname not in FIBER_NAMES:
            continue
        c = FIBER_NAMES.index(cname)
        print(f"  {cname}/{FIBER_FULL_NAMES.get(cname, cname)}:")
        result[cname] = {}
        for k in ks:
            sel = (y[:, c] == 1) & (n_fibers == k)        # k-fiber blends containing c
            nsel = int(sel.sum())
            if nsel == 0:
                print(f"    {k}f: NotExist")
                continue
            fn_mask = sel & (pred[:, c] == 0)             # ... where c was missed
            nfn = int(fn_mask.sum())
            recall = (nsel - nfn) / nsel
            if nfn == 0:
                print(f"    {k}f: recall={recall:.3f}  missed=0/{nsel}  (no confusions)")
                result[cname][k] = {"recall": recall, "missed": 0, "n": nsel, "subs": []}
                continue
            sub = fp_mat[fn_mask].sum(axis=0).astype(float)   # (12,)
            sub[c] = 0
            order = np.argsort(-sub)[:top]
            subs = [(FIBER_NAMES[j], float(sub[j] / nfn)) for j in order if sub[j] > 0]
            sub_str = ", ".join(f"{nm}:{v:.0%}" for nm, v in subs) or "(none)"
            print(f"    {k}f: recall={recall:.3f}  missed={nfn}/{nsel}  confused_with [{sub_str}]")
            result[cname][k] = {"recall": recall, "missed": nfn, "n": nsel, "subs": subs}
    print(f"{'='*60}\n")

    if log_wandb:
        logs = {}
        for cname, perk in result.items():
            for k, d in perk.items():
                logs[f"cmt_confusion/{cname}/recall_{k}fiber"] = d["recall"]
                for nm, v in d["subs"]:
                    logs[f"cmt_confusion/{cname}/{k}fiber/swap_{nm}"] = v
        if logs:
            wandb.log(logs, step=step)
    return result


@torch.no_grad()
def collect_val_predictions(model, val_loader):
    """Run the model over the loader (dataset order) and return numpy
    logits (N,12), reg (N,12), y (N,12 binary), comps (N,12).
    no_grad: inference only — without it this builds an autograd graph over the
    whole split (wasteful, and OOM-prone when called every epoch on test)."""
    model.eval()
    L, R, Y, Cm = [], [], [], []
    with torch.no_grad():
        for x, y, comp in val_loader:
            mlc, reg, _ = fwd(model, x.cuda())
            L.append(mlc.cpu().numpy()); R.append(reg.cpu().numpy())
            Y.append(y.numpy()); Cm.append(comp.numpy())
    return np.vstack(L), np.vstack(R), np.vstack(Y), np.vstack(Cm)


def final_report(model, val_loader, step=None, log_wandb=True):
    """Full diagnostics on (the best) model: classification diagnostics,
    regression diagnostics, and the CA/M/T-by-blend-size confusion breakdown."""
    logits, reg, y, comps = collect_val_predictions(model, val_loader)
    run_diagnostics(logits, reg, y, comps, epoch=-1, step=step, log_wandb=log_wandb)
    regression_diagnostics(reg, comps, y, step=step, log_wandb=log_wandb)
    confusion_by_blend_size(logits, y, focus=("CA", "M", "T"), step=step, log_wandb=log_wandb)


def evaluate_on_test(model, test_loader, thr_acc=None, thr_ham=None, step=None,
                     prefix="test", reg_loss="l1", reg_smooth_beta=1.0, verbose=True,
                     la_tau=None):
    """Evaluate the CURRENT model weights on the official test set.

    Logged under `{prefix}/*` only — it never writes val/* keys, so the val
    namespace stays clean and val/test curves can be overlaid in wandb.

    thr_acc / thr_ham are threshold vectors TUNED ON VAL (full-val fit). Applying
    val-fit thresholds to test is the deployment-correct recipe: nothing is fit on
    test, so `{prefix}/acc_tuned_by_val` is an honest number.

    ⚠️  Per-epoch test curves are for ANALYSIS ONLY. Never early-stop, pick a
    checkpoint, or report max-over-epochs from them — that is selecting on test.
    The headline number is final/test_* below, taken at the best VAL epoch.
    """
    logits, reg, y, comps = collect_val_predictions(model, test_loader)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    acc     = float(metrics.accuracy_score(y, preds))
    hamming = float(metrics.hamming_loss(y, preds))
    mse     = float(metrics.mean_squared_error(comps, reg))
    within  = {f"{prefix}/acc_within_{t}": cal_error_acc(reg, comps, t)
               for t in [0, 1, 2, 3, 5, 10]}

    # Test loss (same components as val loss) so train/val/test curves compare.
    lt = torch.from_numpy(logits); yt = torch.from_numpy(y).float()
    loss_mlc = float(F.multilabel_soft_margin_loss(lt, yt).item())
    loss_reg = float(reg_loss_fn(torch.from_numpy(reg), torch.from_numpy(comps),
                                 reg_loss, reg_smooth_beta).item())

    logs = {f"{prefix}/acc": acc, f"{prefix}/hamming": hamming, f"{prefix}/mse": mse,
            f"{prefix}/loss_mlc": loss_mlc, f"{prefix}/loss_reg": loss_reg,
            f"{prefix}/loss_total": loss_mlc + loss_reg, **within}

    # POST-HOC logit adjustment using the tau CHOSEN ON VAL (never fit on test)
    acc_la = f1_la = None
    _prior = getattr(_LA, "prior", None)
    if la_tau is not None and _prior is not None:
        preds_la = (adjust_logits(logits, _prior, la_tau) > 0).astype(int)
        acc_la = float(metrics.accuracy_score(y, preds_la))
        f1_la = float(metrics.f1_score(y, preds_la, average="macro", zero_division=0))
        logs[f"{prefix}/acc_la_by_val"] = acc_la
        logs[f"{prefix}/macroF1_la_by_val"] = f1_la
        logs[f"{prefix}/la_tau_from_val"] = la_tau

    acc_t = ham_t = mse_t = None
    if thr_acc is not None:
        acc_t = subset_acc_with_thresholds(probs, y, thr_acc)
        logs[f"{prefix}/acc_tuned_by_val"] = acc_t
        # MSE gated by PREDICTED presence, using the val-tuned acc thresholds.
        mask = (probs > thr_acc[None, :]).astype(float)
        mse_t = float(((reg * mask - comps) ** 2).mean())
        logs[f"{prefix}/mse_tuned_by_val"] = mse_t
    if thr_ham is not None:
        ham_t = hamming_with_thresholds(probs, y, thr_ham)
        logs[f"{prefix}/hamming_tuned_by_val"] = ham_t

    if step is not None:
        wandb.log(logs, step=step)
    else:
        wandb.log(logs)

    if verbose:
        tt = f"   tuned_by_val: {acc_t:.4f}" if acc_t is not None else ""
        ht = f"   tuned_by_val: {ham_t:.4f}" if ham_t is not None else ""
        mt = f"   tuned_by_val: {mse_t:.4f}" if mse_t is not None else ""
        print(f"  Test Acc:     {acc:.4f}{tt}")
        print(f"  Test Hamming: {hamming:.4f}{ht}")
        print(f"  Test MSE:     {mse:.4f}{mt}")
        if acc_la is not None:
            print(f"  Test Acc (logit-adj, tau={la_tau:g} from val): {acc_la:.4f}"
                  f"   macro-F1: {f1_la:.4f}")
        print("  Within ±0/1/2/3/5/10: "
              + " / ".join(f"{within[f'{prefix}/acc_within_{t}']:.3f}" for t in [0, 1, 2, 3, 5, 10]))
    return {"acc": acc, "hamming": hamming, "mse": mse,
            "acc_tuned_by_val": acc_t, "hamming_tuned_by_val": ham_t,
            "mse_tuned_by_val": mse_t, "acc_la_by_val": acc_la,
            "macroF1_la_by_val": f1_la}


# ==================== TRAIN / VALIDATE ====================

def train(model, optimizer, train_loader, epoch,
          use_mixup=False, mixup_alpha=0.2, mixup_prob=0.5,
          use_sum_constraint=False, sum_constraint_weight=0.1,
          reg_loss="l1", reg_smooth_beta=1.0, args=None):
    model.train()
    phys_log_acc, phys_log_n = {}, 0

    total_loss_mlc = 0.0
    total_loss_reg = 0.0
    total_loss     = 0.0
    n_batches      = len(train_loader)
    log_interval   = max(1, n_batches // 5)  # ~5 prints per epoch

    all_logits, all_labels, all_reg, all_comps = [], [], [], []
    epoch_start = time.time()

    _refl_idx = (reflectance_channel_indices(args.channels, args.channel_idx)
                 if args is not None else ())

    for batch_idx, (x, y, comp) in enumerate(train_loader):
        x, y, comp = x.cuda(), y.cuda(), comp.cuda()

        # Beer-Lambert synthetic blends: manufacture physically-consistent
        # HIGH-ORDER compositions (the regime where accuracy collapses and where
        # the test split contains unseen combinations). Applied before mixup.
        if (args is not None and getattr(args, "synth_blend", False)
                and np.random.rand() < getattr(args, "synth_blend_prob", 0.3)):
            x, y_s, comp_s = synth_blend_batch(
                x, y, comp, _refl_idx,
                max_fibers=int(getattr(args, "synth_blend_max_fibers", 4)),
                alpha=float(getattr(args, "synth_blend_alpha", 1.0)),
                hard_labels=bool(getattr(args, "synth_blend_hard_labels", True)),
                min_frac=float(getattr(args, "synth_blend_min_frac", 0.0)))
            y_target, comp_target = y_s, comp_s
        # Mixup (soft targets). Keep the original hard y for metric logging.
        elif use_mixup and np.random.rand() < mixup_prob:
            x, y_target, comp_target, _ = mixup_1d(x, y, comp, alpha=mixup_alpha)
        else:
            y_target, comp_target = y.float(), comp

        optimizer.zero_grad()
        mlc_out, reg_out, aux = fwd(model, x)

        loss_mlc = classification_loss(mlc_out, y_target, args,
                                       getattr(args, "_db_loss", None))
        loss_reg = reg_loss_fn(reg_out, comp_target, reg_loss, reg_smooth_beta)
        mlc_w = float(getattr(args, "mlc_weight", 1.0)) if args is not None else 1.0
        loss     = mlc_w * loss_mlc + loss_reg

        if use_sum_constraint:
            loss_sum_reg, loss_sum_cls = sum_constraint_loss(reg_out, comp_target, mlc_out, y_target)
            loss = loss + sum_constraint_weight * (loss_sum_reg + loss_sum_cls)

        if args is not None and aux:
            l_phys, phys_logs = physics_losses(aux, comp_target, mlc_out, y_target, args)
            loss = loss + l_phys
            for k, v in phys_logs.items():
                phys_log_acc[k] = phys_log_acc.get(k, 0.0) + v
            phys_log_n += 1

        loss.backward()
        optimizer.step()

        total_loss_mlc += loss_mlc.item()
        total_loss_reg += loss_reg.item()
        total_loss     += loss.item()

        all_logits.append(mlc_out.detach().cpu().numpy())
        all_labels.append(y.cpu().numpy())
        all_reg.append(reg_out.detach().cpu().numpy())
        all_comps.append(comp.cpu().numpy())

        if batch_idx % log_interval == 0:
            preds   = threshold_predictions(mlc_out).cpu().numpy()
            acc     = metrics.accuracy_score(y.cpu().numpy(), preds)
            mse     = metrics.mean_squared_error(comp.cpu().numpy(), reg_out.cpu().detach().numpy())
            elapsed = time.time() - epoch_start
            print(
                f"  [Batch {batch_idx+1:04d}/{n_batches}] "
                f"loss={loss.item():.4f}  mlc={loss_mlc.item():.4f}  reg={loss_reg.item():.4f}  "
                f"acc={acc:.4f}  mse={mse:.4f}  ({elapsed:.1f}s)"
            )

    avg_mlc = total_loss_mlc / n_batches
    avg_reg = total_loss_reg / n_batches
    avg_tot = total_loss     / n_batches

    all_logits = np.vstack(all_logits); all_labels = np.vstack(all_labels)
    all_reg = np.vstack(all_reg); all_comps = np.vstack(all_comps)
    per_class_cls = compute_per_class_metrics(all_labels, all_logits)
    per_class_mae = compute_per_class_mae(all_comps, all_reg)

    # ── POST-HOC logit adjustment sweep (inference-only, no retraining) ──
    la_best_tau, la_best_acc, la_table = None, None, {}
    _prior = getattr(_LA, "prior", None)
    if _prior is not None:
        la_table, la_best_tau = logit_adjust_sweep(
            all_logits, all_labels, _prior, _LA.taus, is_logits=True)
        la_best_acc = la_table[la_best_tau][0]

    wandb_logs = {
        "train/loss_mlc":   avg_mlc,
        "train/loss_reg":   avg_reg,
        "train/loss_total": avg_tot,
        "train/lr":         get_current_lr(optimizer),
    }
    for k, v in {**per_class_cls, **per_class_mae}.items():
        wandb_logs[f"train/per_class/{k}"] = v
    if phys_log_n:
        for k, v in phys_log_acc.items():
            wandb_logs[f"train/{k}"] = v / phys_log_n
    wandb.log(wandb_logs, step=epoch)

    return avg_tot


def validate(model, val_loader, epoch, tune_thresholds=False, threshold_seed=0,
             diagnostics=False, reg_loss="l1", reg_smooth_beta=1.0):
    model.eval()

    total_acc     = 0.0
    total_hamming = 0.0
    total_mse     = 0.0
    total_mlc_loss = 0.0
    total_reg_loss = 0.0
    count         = 0
    all_reg, all_comps, all_logits, all_labels = [], [], [], []

    all_resid, all_vac = [], []
    with torch.no_grad():
        for x, y, comp in val_loader:
            x, y, comp = x.cuda(), y.cuda(), comp.cuda()
            mlc_out, reg_out, aux = fwd(model, x)
            if "residual" in aux:
                all_resid.append(aux["residual"].cpu().numpy())
            if "vacuity" in aux:
                all_vac.append(aux["vacuity"].cpu().numpy())

            loss_mlc = F.multilabel_soft_margin_loss(mlc_out, y.float())
            loss_reg = reg_loss_fn(reg_out, comp, reg_loss, reg_smooth_beta)
            total_mlc_loss += loss_mlc.item()
            total_reg_loss += loss_reg.item()

            preds     = threshold_predictions(mlc_out).cpu().numpy()
            reg_preds = reg_out.cpu().numpy()
            y_np      = y.cpu().numpy()
            comp_np   = comp.cpu().numpy()

            n = x.size(0)
            total_acc     += metrics.accuracy_score(y_np, preds) * n
            total_mse     += metrics.mean_squared_error(comp_np, reg_preds) * n
            total_hamming += metrics.hamming_loss(y_np, preds) * n
            count         += n

            all_reg.append(reg_preds)
            all_comps.append(comp_np)
            all_logits.append(mlc_out.cpu().numpy())
            all_labels.append(y_np)

    n_batches  = len(val_loader)
    all_reg    = np.vstack(all_reg)
    all_comps  = np.vstack(all_comps)
    all_logits = np.vstack(all_logits)
    all_labels = np.vstack(all_labels)

    acc     = total_acc     / count
    mse     = total_mse     / count
    hamming = total_hamming / count
    val_loss = (total_mlc_loss + total_reg_loss) / n_batches

    thresh_accs = {
        f"val/acc_within_{t}": cal_error_acc(all_reg, all_comps, t)
        for t in [0, 1, 2, 3, 5, 10]
    }

    per_class_cls = compute_per_class_metrics(all_labels, all_logits)
    per_class_mae = compute_per_class_mae(all_comps, all_reg)

    # ── POST-HOC logit adjustment sweep (inference-only, no retraining) ──
    la_best_tau, la_best_acc, la_table = None, None, {}
    _prior = getattr(_LA, "prior", None)
    if _prior is not None:
        la_table, la_best_tau = logit_adjust_sweep(
            all_logits, all_labels, _prior, _LA.taus, is_logits=True)
        la_best_acc = la_table[la_best_tau][0]

    # ── Physics-specific diagnostics ────────────────────────────────────
    # leakage = mass predicted on truly-absent fibres. This is the number the
    # unmixing decoder is supposed to drive down, and it explains most of the
    # gap between raw MSE and the oracle-gated MSE.
    phys_metrics = {"val/leakage": leakage_metric(all_reg, all_comps)}
    for t, (a_, f_) in la_table.items():
        phys_metrics[f"val/acc_la{t:g}"] = a_
        phys_metrics[f"val/macroF1_la{t:g}"] = f_
    if la_best_tau is not None:
        phys_metrics["val/la_best_tau"] = la_best_tau
        phys_metrics["val/acc_la_best"] = la_best_acc
    if all_resid:
        r = np.concatenate(all_resid)
        phys_metrics["val/recon_residual"] = float(r.mean())
        phys_metrics["val/recon_residual_p95"] = float(np.percentile(r, 95))
    if all_vac:
        phys_metrics["val/vacuity"] = float(np.concatenate(all_vac).mean())

    wandb_logs = {
        "val/acc":        acc,
        "val/mse":        mse,
        "val/hamming":    hamming,
        **phys_metrics,
        "val/loss_mlc":   total_mlc_loss / n_batches,
        "val/loss_reg":   total_reg_loss / n_batches,
        "val/loss_total": val_loss,
        **thresh_accs,
    }
    for k, v in {**per_class_cls, **per_class_mae}.items():
        wandb_logs[f"val/per_class/{k}"] = v

    # ── Tuned + oracle metrics (acc / mse / hamming) ─────────────────────
    tuned, thr_honest, thr_full_val, thr_ham_full_val = {}, None, None, None
    if tune_thresholds:
        probs = 1.0 / (1.0 + np.exp(-all_logits))
        tuned, thr_honest, thr_full_val, thr_ham_full_val = tuned_metrics(
            probs, all_labels, all_reg, all_comps, seed=threshold_seed)
        wandb_logs.update(tuned)

    wandb.log(wandb_logs, step=epoch)

    if diagnostics:
        run_diagnostics(all_logits, all_reg, all_labels, all_comps,
                        epoch=epoch, step=epoch, log_wandb=True)

    results = {
        "acc": acc, "mse": mse, "hamming": hamming, "val_loss": val_loss,
        "thresh_accs": thresh_accs,
        "la_best_tau": la_best_tau, "la_best_acc": la_best_acc, "la_table": la_table,
        **{k.replace("val/", ""): v for k, v in phys_metrics.items()},
        "acc_tuned": tuned.get("val/acc_tuned", None),
        "mse_tuned": tuned.get("val/mse_tuned", None),
        "acc_tuned_oracle": tuned.get("val/acc_tuned_oracle", None),
        "mse_tuned_oracle": tuned.get("val/mse_tuned_oracle", None),
        "thresholds": thr_honest,          # fit on half of val -> honest, use to trust the acc_tuned number
        "thresholds_full_val": thr_full_val,  # fit on all of val -> more stable, use this one to deploy
        "thresholds_hamming_full_val": thr_ham_full_val,  # hamming-optimal, fit on all of val
    }
    return results


# ==================== DATASET ====================

class NIRDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, comp_dir, channels=2, channel_idx=0):
        data_x     = np.load(data_dir)
        data_labels = np.load(label_dir)
        data_comps  = np.load(comp_dir)

        data_x = data_x.reshape((data_labels.shape[0], 2, 200))

        if channels == 1:
            data_x = data_x[:, channel_idx:channel_idx+1, :]
        elif channels != 2:
            raise ValueError("channels must be 1 or 2")

        self.X     = data_x
        self.Y     = data_labels
        self.comps = data_comps

    def __getitem__(self, item):
        return (
            torch.from_numpy(self.X[item]).float(),
            torch.tensor(self.Y[item], dtype=torch.int64),
            torch.from_numpy(self.comps[item]).float(),
        )

    def __len__(self):
        return self.X.shape[0]


# ==================== TRAIN/VAL SPLIT (leakage-free) ==========================
# Ported from clean_convnextv2_trainsplitv2.py. The dataset carries no fabric id,
# so the ~20 scans of one physical fabric must not be split across train/val or
# val leaks. These helpers group by fabric / composition / fiber-set and hold out
# whole groups.

def _combination_keys(dataset, key="composition", round_decimals=4):
    """Per-sample group id so samples sharing a fabric 'combination' share an id.
      key='composition' : exact fiber+percentage vector (all replicate scans of a
                          fabric share it) -> leak-proof, conservative val.
      key='label'       : binary fiber-presence set only (paper's combination
                          key) -> hardest, unseen-fiber-set val."""
    if key == "label":
        mat = np.asarray(dataset.Y).astype(int)
    elif key == "composition":
        mat = np.round(np.asarray(dataset.comps).astype(float), round_decimals)
    else:
        raise ValueError(f"Unknown combination_key '{key}' (use 'composition' or 'label').")
    _, ids = np.unique(mat, axis=0, return_inverse=True)
    return ids.astype(np.int64)


def _fiberset_ids(dataset):
    """Per-sample id of the binary fiber-presence set (paper's 296-class key).
    Used to stratify holdouts so val class-mix matches train."""
    mat = np.asarray(dataset.Y).astype(int)
    _, ids = np.unique(mat, axis=0, return_inverse=True)
    return ids.astype(np.int64)


def fabric_group_ids(dataset, cap=25, merge_threshold=40, round_decimals=4):
    """Recover a per-sample FABRIC group id by run-length over the (order-
    preserved) composition stream. Runs <= merge_threshold are one fabric each;
    over-long pure-fiber mega-runs are chunked into <=cap pieces. Verified on
    FabricSpectra: runs cluster at 20-25, no length-1 shuffling. Returns int
    array length N."""
    C = np.round(np.asarray(dataset.comps).astype(float), round_decimals)
    n = len(C)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    changed = np.any(C[1:] != C[:-1], axis=1)
    bounds = np.concatenate([[0], np.where(changed)[0] + 1, [n]])
    group_ids = np.empty(n, dtype=np.int64)
    gid = 0
    for s, e in zip(bounds[:-1], bounds[1:]):
        run_len = e - s
        if run_len <= merge_threshold:
            group_ids[s:e] = gid; gid += 1
        else:
            for cs in range(s, e, cap):
                ce = min(cs + cap, e); group_ids[cs:ce] = gid; gid += 1
    return group_ids


def _stratified_group_holdout(group_ids, strat_ids, val_frac, seed):
    """Hold out `val_frac` of GROUPS, stratified by `strat_ids` (whole group ->
    one side). Singleton strata stay in train. Returns (train_idx, val_idx)."""
    rng = np.random.default_rng(seed)
    first_pos = {}
    for i, g in enumerate(group_ids):
        if g not in first_pos:
            first_pos[g] = i
    groups = np.array(list(first_pos.keys()))
    g_strat = np.array([strat_ids[first_pos[g]] for g in groups])
    val_groups = set()
    for s in np.unique(g_strat):
        gs = groups[g_strat == s]; rng.shuffle(gs)
        if len(gs) < 2:
            continue
        k = max(1, int(round(val_frac * len(gs))))
        k = min(k, len(gs) - 1)
        val_groups.update(gs[:k].tolist())
    is_val = np.array([g in val_groups for g in group_ids])
    return np.sort(np.where(~is_val)[0]), np.sort(np.where(is_val)[0])


def make_train_val_split(dataset, strategy="fabric", val_frac=0.1, seed=0,
                         combination_key="composition", fabric_cap=25,
                         fabric_merge_threshold=40):
    """(train_idx, val_idx) as numpy int arrays.
      'random'      : per-sample (LEAKS replicate scans; sanity baseline only).
      'combination' : whole combinations out (see combination_key).
      'fabric'      : RECOMMENDED. Whole recovered fabrics out, stratified by
                      fiber-set (novel fabrics, seen combinations)."""
    n = len(dataset)
    rng = np.random.default_rng(seed)
    if strategy == "random":
        idx = np.arange(n); rng.shuffle(idx); n_val = int(round(val_frac * n))
        return np.sort(idx[n_val:]), np.sort(idx[:n_val])
    if strategy == "combination":
        groups = _combination_keys(dataset, key=combination_key)
        uniq = np.unique(groups); rng.shuffle(uniq)
        n_val_groups = max(1, int(round(val_frac * len(uniq))))
        val_groups = set(uniq[:n_val_groups].tolist())
        is_val = np.array([g in val_groups for g in groups])
        return np.sort(np.where(~is_val)[0]), np.sort(np.where(is_val)[0])
    if strategy == "fabric":
        groups = fabric_group_ids(dataset, cap=fabric_cap, merge_threshold=fabric_merge_threshold)
        strat = _fiberset_ids(dataset)
        return _stratified_group_holdout(groups, strat, val_frac, seed)
    raise ValueError(f"Unknown split strategy '{strategy}'.")


def split_fingerprint(train_idx, val_idx):
    """Short, stable hash of a split. Identical on every machine iff the split
    is identical — run the same seed on two nodes and compare these strings."""
    import hashlib
    h = hashlib.sha256()
    h.update(np.asarray(train_idx, dtype=np.int64).tobytes())
    h.update(b"|")
    h.update(np.asarray(val_idx, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def get_or_make_split(dataset, cache_dir=None, **kw):
    """Load a cached split if present, else compute and save it.

    Guarantees every run that shares a seed uses byte-identical indices, which
    paired comparisons across configurations rely on.
    """
    strategy = kw.get("strategy", "fabric"); seed = kw.get("seed", 0)
    val_frac = kw.get("val_frac", 0.1); ckey = kw.get("combination_key", "composition")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        tag = f"{strategy}" + (f"-{ckey}" if strategy == "combination" else "")
        path = os.path.join(cache_dir, f"split_{tag}_vf{int(round(val_frac*100))}_seed{seed}.npz")
        if os.path.exists(path):
            z = np.load(path)
            tr, va = z["train_idx"], z["val_idx"]
            if len(tr) + len(va) != len(dataset):
                raise ValueError(f"cached split {path} covers {len(tr)+len(va)} samples "
                                 f"but dataset has {len(dataset)} — stale cache?")
            print(f"  [split] LOADED cached split <- {path}")
            print(f"  [split] fingerprint = {split_fingerprint(tr, va)}")
            return tr, va
    tr, va = make_train_val_split(dataset, **kw)
    print(f"  [split] fingerprint = {split_fingerprint(tr, va)}  "
          f"(identical across machines iff the split is identical)")
    if cache_dir:
        np.savez_compressed(path, train_idx=tr, val_idx=va)
        print(f"  [split] SAVED -> {path}")
    return tr, va


def assert_split_clean(dataset, train_idx, val_idx, strategy,
                       fabric_cap=25, fabric_merge_threshold=40):
    """Verify + print the split at startup. For fabric/combination it HARD-ASSERTS
    that no recovered fabric group lands on both sides (=> no replicate-scan
    leakage). For 'random' it only warns (random is expected to leak)."""
    tr, va = set(train_idx.tolist()), set(val_idx.tolist())
    assert not (tr & va), "index overlap between train and val!"
    groups = fabric_group_ids(dataset, cap=fabric_cap, merge_threshold=fabric_merge_threshold)
    strat = _fiberset_ids(dataset)
    g_tr, g_va = set(groups[train_idx].tolist()), set(groups[val_idx].tolist())
    shared = g_tr & g_va
    C = np.asarray(dataset.comps).astype(float)
    starved = int(np.sum((C[train_idx] > 0).sum(0) == 0))
    fs_tr, fs_va = set(strat[train_idx].tolist()), set(strat[val_idx].tolist())
    n = len(dataset)
    print(f"  [split-check] strategy={strategy}  train={len(train_idx):,} "
          f"({100*len(train_idx)/n:.1f}%)  val={len(val_idx):,} ({100*len(val_idx)/n:.1f}%)")
    print(f"  [split-check] fabrics train/val={len(g_tr):,}/{len(g_va):,}  "
          f"fiber-sets train/val={len(fs_tr)}/{len(fs_va)}  "
          f"val⊆train={'YES' if fs_va <= fs_tr else 'NO'}  starved_fibers={starved}")
    if strategy == "random":
        if shared:
            print(f"  [split-check] ⚠️  RANDOM split leaks {len(shared):,} fabric groups "
                  f"across train/val (expected for 'random'; use 'fabric' for honest val).")
    else:
        assert not shared, (f"FABRIC LEAKAGE: {len(shared)} groups on both sides "
                            f"under strategy={strategy} — val would be optimistic.")
        print(f"  [split-check] fabric-group leakage: NONE ✅")
    if starved:
        print(f"  [split-check] ⚠️  {starved} fiber(s) absent from TRAIN — training may break.")


def split_tag(args):
    """Compact, filename-safe tag describing the data split, for run_name."""
    if getattr(args, "val_mode", "none") == "none":
        return "valNone"
    vf = int(round(getattr(args, "val_frac", 0.1) * 100))
    strat = getattr(args, "split_strategy", "fabric")
    if strat == "combination":
        key = getattr(args, "combination_key", "composition").capitalize()
        tag = f"combo{key}VF{vf}"
    else:
        tag = f"{strat}VF{vf}"  # e.g. fabricVF10 / randomVF10
    # Only appended when the split seed is DECOUPLED from the training seed, so
    # ordinary coupled-seed runs keep their existing names.
    ss = getattr(args, "split_seed", None)
    if ss is not None and ss != getattr(args, "seed", None):
        tag += f"ss{ss}"
    return tag


# ==================== MAIN ====================

class Args:
    def __init__(self):
        # ── Model / Student architecture ──────────────────────────────────
        self.model = "atto"        # ConvNeXtV2-1D size: atto | femto | pico | nano | tiny | base
        self.student_arch = "conv"     # short label used in run_name / wandb

        # ── Data ──────────────────────────────────────────────────────────
        self.channels = 2          # choices: 1, 2
        self.channel_idx = 1       # choices: 0, 1 (0=absorbance, 1=reflectance)
        self.data_root = "data/FabricSpectra/"   # folder with the released .npy files
        
        # ── Training ──────────────────────────────────────────────────────
        self.epochs = 160
        self.batch_size = 512
        self.lr = 1e-3
        self.weight_decay = 1e-3
        
        # ── Model Params ──────────────────────────────────────────────────
        self.drop_path = 0.2
        
        # ── Scheduler ─────────────────────────────────────────────────────
        self.lr_patience = 7
        self.lr_factor = 0.3
        self.lr_min = 1e-6
        
        # ── Misc ──────────────────────────────────────────────────────────
        self.seed = 0
        
        # ── New Options ───────────────────────────────────────────────────
        self.multiscale = False   # aggregate pooled features from all 4 stages (off in the paper)

        self.use_mixup = True     # Mixup augmentation on spectra + soft targets
        self.mixup_alpha = 0.35
        self.mixup_prob = 0.5

        self.sum_constraint = True  # Add sum-to-total constraint losses
        self.sum_constraint_weight = 0.1

        self.diagnostics = False   # Run per-epoch error/confusion diagnostics on val set
        self.tune_thresholds = True  # Log tuned (honest) + oracle val acc/mse/hamming
        self.save_best_acc = True # Save checkpoint on best TUNED val acc
        self.save_best_mse = False # Save checkpoint on best val MSE

        self.reg_loss = "l1"              # choices: "l1", "smooth_l1", "l2" (paper: l1)
        self.reg_smooth_beta = 1.0 # beta for smooth_l1 (transition point MAE<->MSE)
        
        self.save_dir = "chkpts/"        # Directory for saved checkpoints

        # ── Train/val split (leakage-free holdout carved out of the TRAIN set) ─
        # val_mode="none"    -> original protocol: train on ALL of train, the
        #                       "val" metrics ARE the official test set.
        # val_mode="holdout" -> carve a val split out of train; select/early-stop
        #                       on it, then score the official test set once at the
        #                       end on the best (val-selected) model.
        # ── Scheduler ─────────────────────────────────────────────────────
        # "plateau"  : ReduceLROnPlateau on val loss. Needs a REAL val set —
        #              with val_mode="none" the "val" loss IS the test loss, so
        #              the LR schedule would be driven by test performance.
        # "multistep": fixed milestones, data-independent. The benchmark
        #              authors' setting, and the only safe choice when
        #              val_mode="none".
        self.scheduler = "multistep"          # "plateau" | "multistep"
        self.lr_milestones = [80,110,140,150]   # MultiStepLR milestones
        self.lr_gamma = 0.3                 # MultiStepLR decay factor
        # Save the LAST-epoch weights (needed when there is no val set to select
        # on: full-train protocol => report last epoch, never a test-selected best)
        self.save_last = False
        # Which val metric selects the best checkpoint (holdout mode):
        #   "acc"       -> raw subset accuracy at threshold 0.5  (matches the
        #                  benchmark's evaluation and the main table -> use this
        #                  when both tables must be consistent)
        #   "acc_tuned" -> honest tuned accuracy (thresholds fit on half of val)
        self.select_metric = "acc"

        self.val_mode = "holdout"          # "none" | "holdout"
        self.split_strategy = "fabric"     # "random" | "combination" | "fabric"
        self.combination_key = "composition"  # "composition" | "label" (combination only)
        self.val_frac = 0.10               # fraction held out (of fabrics for 'fabric')
        self.fabric_cap = 25               # max chunk when subdividing a mega-run
        self.fabric_merge_threshold = 40   # runs longer than this are chunked
        # Seed controlling ONLY the train/val partition. None -> use self.seed
        # (coupled: one seed drives init + data order + split; the default, and
        # what you want for headline mean±std). Set it explicitly to DECOUPLE:
        #   fixed split, varying init  -> --split_seed 0 --seed {0..4}
        #   fixed init, varying split  -> --seed 0 --split_seed {0..4}
        self.split_seed = None
        # Directory for cached split indices. Set this on a SHARED filesystem so
        # every node/architecture reuses byte-identical splits per seed.
        self.split_cache_dir = "splits"   # None = no caching (recompute every run)

        # ── Test-set monitoring ───────────────────────────────────────────
        # Evaluate the official test set every epoch (holdout mode only) and log
        # under test/*. ANALYSIS ONLY — never early-stop or pick a checkpoint on
        # these curves; the headline number stays final/test_* at the best VAL
        # epoch. Set freq >1 (or False) if the extra forward pass costs too much.
        self.eval_test_every_epoch = True
        self.test_eval_freq = 1            # evaluate test every N epochs

        # ══════════════════════════════════════════════════════════════════
        #  PHYSICS-INFORMED OPTIONS  (all OFF by default -> baseline is exact)
        #  Test one at a time; each is an independent ablation row.
        # ══════════════════════════════════════════════════════════════════

        # ── (1) Compositional output head ─────────────────────────────────
        # "linear"    : original unconstrained head (baseline)
        # "gated"     : sigmoid(mlc)*softplus(reg), renormalised to 100
        #               -> non-negative, sums to 100, heads cannot disagree
        # "softmax"   : 100*softmax  (simplex, but never exactly zero)
        # "sparsemax" : simplex WITH exact zeros -> matches sparse compositions
        # "simplex"   : differentiable Euclidean projection onto the simplex
        # "dirichlet" : evidential head; gives per-sample uncertainty (vacuity)
        self.comp_head = "linear"

        # ── (2) Beer-Lambert unmixing decoder (analysis-by-synthesis) ─────
        self.physics_decoder = False     # master switch for the decoder
        self.physics_weight = 0.1        # weight on the reconstruction loss
        self.physics_mixing = "linear"   # "linear" | "bilinear" | "ppnm" | "mlp"
                                         # "mlp" = unconstrained decoder CONTROL:
                                         # same recon loss, no physics structure,
                                         # no endmembers -> nothing interpretable.
        self.physics_recon_loss = "mse"  # "mse" | "sad" | "both"
        self.physics_km_target = False   # reconstruct Kubelka-Munk instead of -log10(R)
        # How the decoder gets its abundance vector. "normalize" projects onto
        # the simplex INSIDE the decoder only, so the physics constrains the
        # relative proportions while comp_head can stay "linear" (i.e. you keep
        # whatever accuracy your current head gives you).
        self.physics_abundance = "normalize"   # "normalize" | "softmax" | "as_is"
        self.physics_abundance_softplus = False  # softplus instead of clamp (softer grads)
        self.endmember_init = "data"     # "data" (pure-fibre means) | "random"
        self.endmember_purity = 99.0     # a sample counts as pure at >= this %
        self.endmember_smooth_weight = 0.0    # 2nd-difference prior (try 1e-3)
        self.endmember_volume_weight = 0.0    # minimum-volume prior (try 1e-4)
        self.endmember_anchor_weight = 0.0    # keep endmembers near measured pures

        # ── (3) Evidential / Dirichlet composition likelihood ─────────────
        self.evidential_weight = 0.0     # only used when comp_head="dirichlet"

        # ── (4) Head-consistency (presence <-> mass), label-free ──────────
        self.consistency_weight = 0.0    # try 0.05 - 0.5
        self.consistency_tau = 0.01      # mass fraction treated as "present"

        # ── (5) Physics preprocessing front-end ───────────────────────────
        self.scatter_correct = "none"    # "none" | "snv" | "msc" | "emsc"
        self.input_mode = "raw"          # "raw" | "snv" | "raw+snv"  (backbone input only)
        self.emsc_degree = 2             # polynomial degree for learnable EMSC
        self.km_transform = False        # Kubelka-Munk on reflectance channels

        # ── (6) Beer-Lambert synthetic blend augmentation ─────────────────
        self.synth_blend = False         # manufacture new high-order blends
        self.synth_blend_prob = 0.3
        self.synth_blend_max_fibers = 4  # mix up to this many source spectra
        self.synth_blend_alpha = 1.0     # Dirichlet concentration
        self.synth_blend_hard_labels = True
        self.synth_blend_min_frac = 0.0   # drop trace components below this % and
                                          # renormalise (try 2.0-5.0). Prevents
                                          # labelling 1% components as present.

        # ══════════════════════════════════════════════════════════════════
        #  CLASS-IMBALANCE CORRECTIONS  (Stage 1)
        # ══════════════════════════════════════════════════════════════════
        # Post-hoc logit adjustment is ALWAYS evaluated (it is free and needs
        # no retraining): every epoch the sweep below is scored on val, and the
        # best tau is carried to the test set. tau=0 is the raw model, so the
        # baseline row is always present in the same table.
        self.la_taus = "0,0.25,0.5,0.75,1.0,1.5"   # post-hoc sweep grid

        # ── Held-out-class (open-set) protocol ────────────────────────────
        # e.g. "SP" (isolated -> predict DETECTED) or "L" (degenerate with
        # Cotton -> predict MISSED). Comma-separated names or indices.
        self.holdout_classes = None

        # ── Drop PURE samples from training (endmember-confound ablation) ──
        # Removes any training sample whose dominant fibre is >= this percentage.
        # With pure samples present, a 100%-cotton spectrum forces
        # x_hat = 1.0 * E_cotton, so the reconstruction loss pins that endmember
        # DIRECTLY -- recovery is then partly supervised by the data rather than
        # inferred from blends. Removing them makes every fibre as unconfounded
        # as spandex already is, and turns the pure spectra into a clean
        # held-out reference set. 0 = keep everything (default).
        self.drop_pure = 0.0        # try 99.0

        # TRAIN-time logit adjustment (Menon et al., ICLR 2021): adds
        # tau*log(pi/(1-pi)) to the logits INSIDE the loss, so rare classes are
        # forced to a larger margin rather than just re-thresholded afterwards.
        self.logit_adjust_train = 0.0        # 0 = off; try 0.5 / 1.0

        # Multi-label long-tail loss (Wu et al., ECCV 2020)
        self.cls_loss = "soft_margin"        # "soft_margin" | "db"
        self.db_neg_scale = 5.0              # negative-tolerant scale
        self.db_init_bias = 0.05             # class-specific bias strength
        self.db_focal = True                 # focal term inside DB loss

        # ── (7) Task balance ──────────────────────────────────────────────
        # Regression loss lives on a 0-100 scale and dominates the shared trunk;
        # exact-match accuracy is the headline metric. Try 2-10.
        self.mlc_weight = 1.0


# Usage in notebook:


def _apply_cli_overrides(args):
    """OPTIONAL command-line overrides for the most common knobs, so multi-seed /
    multi-split sweeps can be launched from a shell loop WITHOUT editing Args.
    Only flags actually passed override the Args() default; unknown flags are
    ignored. This is applied ONLY when main() builds Args() itself (i.e. when run
    as a script) — programmatic callers that pass their own Args (e.g. the
    fine-tune sweep) are left untouched.

    Example:
        for s in 0 1 2; do python train.py --seed $s --physics_decoder ...; done
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--data_root", default=None)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--channel_idx", type=int, default=None)
    p.add_argument("--select_metric", choices=["acc", "acc_tuned"], default=None,
                   help="val metric used to pick the best checkpoint (holdout mode)")
    p.add_argument("--scheduler", choices=["plateau", "multistep"], default=None)
    p.add_argument("--lr_gamma", type=float, default=None)
    p.add_argument("--lr_milestones", type=str, default=None,
                   help="comma-separated, e.g. '40,70,90'")
    p.set_defaults(save_last=None)
    p.add_argument("--save_last", dest="save_last", action="store_true",
                   help="save last-epoch weights (use with --val_mode none)")
    p.add_argument("--reg_loss", choices=["l1", "smooth_l1", "l2"], default=None)
    p.add_argument("--reg_smooth_beta", type=float, default=None)
    # split controls
    p.add_argument("--val_mode", choices=["none", "holdout"], default=None)
    p.add_argument("--split_strategy", choices=["random", "combination", "fabric"], default=None)
    p.add_argument("--combination_key", choices=["composition", "label"], default=None)
    p.add_argument("--val_frac", type=float, default=None)
    p.add_argument("--fabric_cap", type=int, default=None)
    p.add_argument("--fabric_merge_threshold", type=int, default=None)
    p.add_argument("--split_cache_dir", type=str, default=None,
                   help="save/reuse split indices here (shared FS) for exact reproducibility")
    p.add_argument("--split_seed", type=int, default=None,
                   help="seed for the train/val partition ONLY; omit to couple it to --seed")
    # per-epoch test monitoring
    p.add_argument("--test_eval_freq", type=int, default=None)
    p.set_defaults(eval_test_every_epoch=None)
    p.add_argument("--eval_test_every_epoch", dest="eval_test_every_epoch",
                   action="store_true", help="log test/* metrics every epoch (analysis only)")
    p.add_argument("--no_eval_test_every_epoch", dest="eval_test_every_epoch",
                   action="store_false", help="skip per-epoch test evaluation")
    p.set_defaults(multiscale=None)
    p.add_argument("--multiscale", dest="multiscale", action="store_true",
                   help="aggregate pooled features from all four stages")
    p.add_argument("--no_multiscale", dest="multiscale", action="store_false")
    # ── physics-informed ablation switches ───────────────────────────────
    p.add_argument("--comp_head", default=None,
                   choices=["linear", "gated", "softmax", "sparsemax",
                            "simplex", "dirichlet"],
                   help="compositional output head (simplex/evidential)")
    p.set_defaults(physics_decoder=None)
    p.add_argument("--physics_decoder", dest="physics_decoder", action="store_true",
                   help="Beer-Lambert unmixing decoder (analysis-by-synthesis)")
    p.add_argument("--no_physics_decoder", dest="physics_decoder", action="store_false")
    p.add_argument("--physics_weight", type=float, default=None)
    p.add_argument("--physics_mixing", default=None,
                   choices=["linear", "bilinear", "ppnm", "mlp"],
                   help="'mlp' = ABLATION CONTROL: free decoder, no physics, "
                        "no endmembers (isolates recon-loss from mixing-structure)")
    p.add_argument("--physics_recon_loss", default=None, choices=["mse", "sad", "both"])
    p.add_argument("--physics_abundance", default=None,
                   choices=["normalize", "softmax", "as_is"],
                   help="how the decoder derives abundance; 'normalize' lets the "
                        "physics bite while comp_head stays linear")
    p.set_defaults(physics_abundance_softplus=None)
    p.add_argument("--physics_abundance_softplus", dest="physics_abundance_softplus",
                   action="store_true")
    p.set_defaults(physics_km_target=None)
    p.add_argument("--physics_km_target", dest="physics_km_target", action="store_true")
    p.add_argument("--endmember_init", default=None, choices=["data", "random"])
    p.add_argument("--endmember_purity", type=float, default=None)
    p.add_argument("--endmember_smooth_weight", type=float, default=None)
    p.add_argument("--endmember_volume_weight", type=float, default=None)
    p.add_argument("--endmember_anchor_weight", type=float, default=None)
    p.add_argument("--evidential_weight", type=float, default=None)
    p.add_argument("--consistency_weight", type=float, default=None)
    p.add_argument("--consistency_tau", type=float, default=None)
    p.add_argument("--scatter_correct", default=None,
                   choices=["none", "snv", "msc", "emsc"])
    p.add_argument("--emsc_degree", type=int, default=None)
    p.add_argument("--input_mode", default=None, choices=list(INPUT_MODES),
                   help="what the BACKBONE sees: raw (default), snv (SNV of each "
                        "channel, same shape), raw+snv (raw and SNV concatenated, "
                        "4 x 200). The decoder always reconstructs raw absorbance, "
                        "so endmembers stay 12 x 2 x 200 in every mode.")
    p.set_defaults(km_transform=None)
    p.add_argument("--km_transform", dest="km_transform", action="store_true",
                   help="Kubelka-Munk remission transform on reflectance channels")
    p.set_defaults(synth_blend=None)
    p.add_argument("--synth_blend", dest="synth_blend", action="store_true",
                   help="Beer-Lambert synthetic high-order blend augmentation")
    p.add_argument("--no_synth_blend", dest="synth_blend", action="store_false")
    p.add_argument("--synth_blend_prob", type=float, default=None)
    p.add_argument("--synth_blend_max_fibers", type=int, default=None)
    p.add_argument("--synth_blend_alpha", type=float, default=None)
    p.add_argument("--synth_blend_min_frac", type=float, default=None,
                   help="drop synthetic components below this %% and renormalise")
    p.set_defaults(synth_blend_hard_labels=None)
    p.add_argument("--synth_blend_soft_labels", dest="synth_blend_hard_labels",
                   action="store_false")
    p.add_argument("--mlc_weight", type=float, default=None)
    # ── class-imbalance corrections (Stage 1) ────────────────────────────
    p.add_argument("--drop_pure", type=float, default=None,
                   help="remove training samples whose dominant fibre is >= this "
                        "%% (e.g. 99). Makes endmember recovery unconfounded for "
                        "ALL fibres, not just spandex.")
    p.add_argument("--holdout_classes", default=None,
                   help="fibres to REMOVE from training and treat as unknown at "
                        "test time, e.g. 'SP' or 'L' or 'CA,M'")
    p.add_argument("--la_taus", default=None,
                   help="comma-separated post-hoc logit-adjust sweep, e.g. '0,0.5,1.0'")
    p.add_argument("--logit_adjust_train", type=float, default=None,
                   help="TRAIN-time logit adjustment tau (0=off, try 0.5/1.0)")
    p.add_argument("--cls_loss", default=None, choices=["soft_margin", "db"],
                   help="'db' = Distribution-Balanced Loss (multi-label long-tail)")
    p.add_argument("--db_neg_scale", type=float, default=None)
    p.add_argument("--db_init_bias", type=float, default=None)
    p.set_defaults(db_focal=None)
    p.add_argument("--no_db_focal", dest="db_focal", action="store_false")
    known, _ = p.parse_known_args()
    if getattr(known, "lr_milestones", None) is not None:
        known.lr_milestones = [int(x) for x in str(known.lr_milestones).split(",") if x.strip()]
    changed = {k: v for k, v in vars(known).items() if v is not None}
    for k, v in changed.items():
        setattr(args, k, v)
    if changed:
        print(f"[cli-overrides] {changed}")
    return args


def build_student(args):
    """Build the ConvNeXtV2-1D backbone. Returns (model, MODEL_REGISTRY); the
    registry is used only for logging depths/dims."""
    bm = build_model
    registry = MODEL_REGISTRY
    n_cls = int(getattr(args, "_num_classes", 12))
    # args.channels is the DATA channel count (what the loader returns and the
    # decoder reconstructs); the stem may be wider under --input_mode raw+snv.
    in_ch = backbone_in_chans(args)
    if in_ch != args.channels:
        print(f"  [input_mode] {input_mode_of(args)}: backbone stem takes "
              f"{in_ch} channels (data has {args.channels})")
    model = bm(args.model, in_chans=in_ch, num_classes=n_cls,
               drop_path_rate=args.drop_path, use_multiscale=args.multiscale)
    return model, registry


def physics_enabled(args):
    """True if ANY physics option is active (so the wrapper is needed)."""
    return (getattr(args, "comp_head", "linear") != "linear"
            or getattr(args, "physics_decoder", False)
            or getattr(args, "scatter_correct", "none") not in (None, "none")
            or input_mode_of(args) != "raw"
            or getattr(args, "km_transform", False)
            or getattr(args, "consistency_weight", 0.0) > 0)


def maybe_wrap_physics(model, args, train_set=None, train_idx=None,
                       length=200, num_classes=12):
    """Attach the physics wrapper to any backbone if requested.
    Endmembers are optionally initialised from observed near-pure fibre spectra,
    which both speeds convergence and makes them interpretable from epoch 1."""
    if not physics_enabled(args):
        return model
    init_E = None
    if getattr(args, "physics_decoder", False) and \
       getattr(args, "physics_mixing", "linear") != "mlp" and \
       getattr(args, "endmember_init", "data") == "data" and train_set is not None:
        refl = reflectance_channel_indices(args.channels, args.channel_idx)
        idx = train_idx if train_idx is not None else range(len(train_set))
        init_E, cnt = estimate_endmembers_from_data(
            train_set, idx, num_classes, args.channels, refl,
            purity=float(getattr(args, "endmember_purity", 99.0)),
            km=bool(getattr(args, "physics_km_target", False)))
        named = {FIBER_NAMES[k]: int(cnt[k]) for k in range(num_classes)}
        print(f"  [physics] endmembers initialised from pure samples: {named}")
        if getattr(args, "endmember_anchor_weight", 0.0) > 0:
            args._endmember_anchor = torch.from_numpy(init_E).float()
    wrapped = PhysicsWrapper(model, args, n_channels=args.channels,
                             length=length, num_classes=num_classes,
                             init_endmembers=init_E)
    args._decoder_ref = wrapped.decoder
    bits = []
    if args.comp_head != "linear":            bits.append(f"comp_head={args.comp_head}")
    if getattr(args, "physics_decoder", False):
        bits.append(f"unmix={getattr(args,'physics_mixing','linear')}"
                    f"(w={getattr(args,'physics_weight',0.0)})")
    if getattr(args, "scatter_correct", "none") not in (None, "none"):
        bits.append(f"scatter={args.scatter_correct}")
    if input_mode_of(args) != "raw":
        bits.append(f"input={input_mode_of(args)} "
                    f"(stem {backbone_in_chans(args)} ch, decoder target raw "
                    f"absorbance {args.channels} ch)")
    if getattr(args, "km_transform", False):  bits.append("kubelka_munk")
    if getattr(args, "consistency_weight", 0.0) > 0:
        bits.append(f"consistency={args.consistency_weight}")
    print(f"  [physics] enabled: {', '.join(bits)}")
    return wrapped


def main(args=None):
    # args = parse_args()
    if args is None:
        args = Args()
        _apply_cli_overrides(args)   # optional shell overrides (script runs only)
    print(f"Model: {args.model}, Reg loss: {args.reg_loss}")
    print(f"Multiscale: {args.multiscale}, Mixup: {args.use_mixup}")
    print(f"Sum constraint: {args.sum_constraint}, Diagnostics: {args.diagnostics}")
    setup_seed(args.seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    root = args.data_root
    full_train_set = NIRDataset(
        os.path.join(root, "train_data.npy"),
        os.path.join(root, "train_multi_label_classification_label.npy"),
        os.path.join(root, "train_composition_label.npy"),
        channels=args.channels, channel_idx=args.channel_idx,
    )
    test_set = NIRDataset(
        os.path.join(root, "test_data.npy"),
        os.path.join(root, "test_multi_label_classification_label.npy"),
        os.path.join(root, "test_composition_label.npy"),
        channels=args.channels, channel_idx=args.channel_idx,
    )
    train_set = full_train_set

    # ── Held-out-class (open-set) protocol ────────────────────────────────
    _holdout = parse_holdout(getattr(args, "holdout_classes", None))
    args._num_classes = 12
    args._holdout, args._keep = _holdout, list(range(12))
    _test_unknown = None
    if _holdout:
        keep = [k for k in range(12) if k not in _holdout]
        args._keep, args._num_classes = keep, len(keep)
        hn = ", ".join(FIBER_NAMES[k] for k in _holdout)
        # IMPORTANT: keep every ROW of the training set so that cached
        # fabric-level split indices stay valid; only the label COLUMNS are
        # remapped here. The held-out samples are removed from the split
        # indices further down, which preserves the exact train/val partition
        # instead of silently building a different one.
        _train_contains = apply_holdout(full_train_set, _holdout, keep,
                                        drop_samples=False)
        args._holdout_train_mask = _train_contains
        # test set keeps ALL rows: the held-out ones are the OOD positives
        _test_unknown = apply_holdout(test_set, _holdout, keep, drop_samples=False)
        print(f"\n  [open-set] holding out: {hn}")
        print(f"    train rows kept for indexing: {len(full_train_set.X):,} "
              f"({int(_train_contains.sum()):,} contain a held-out fibre and will "
              f"be removed from the split)")
        print(f"    classes 12 -> {len(keep)}  ({', '.join(FIBER_NAMES[k] for k in keep)})")
        print(f"    test: {int(_test_unknown.sum()):,} UNKNOWN / "
              f"{int((~_test_unknown).sum()):,} known")
        # Remap the module-level taxonomy to the KEPT subset. Everything
        # downstream (per-class metrics, confusion analysis, family grouping,
        # diagnostics) indexes FIBER_NAMES / _PARENT_GROUPS at call time, so
        # rewriting them here makes the whole pipeline class-count agnostic
        # instead of needing a dozen separate guards.
        globals()["FIBER_NAMES_FULL"] = list(FIBER_NAMES)
        globals()["_PARENT_GROUPS_FULL"] = [list(g) for g in _PARENT_GROUPS]
        _old2new = {old: new for new, old in enumerate(keep)}
        globals()["FIBER_NAMES"] = [FIBER_NAMES[k] for k in keep]
        _pg = []
        for g in _PARENT_GROUPS_FULL:
            g2 = [_old2new[m] for m in g if m in _old2new]
            if g2:
                _pg.append(g2)
        globals()["_PARENT_GROUPS"] = _pg
        print(f"    taxonomy remapped: {len(_pg)} parent families remain")

    # ── Class prior + imbalance corrections (Stage 1) ──────────────────────
    # Prior is computed from TRAIN labels only. It drives both the post-hoc
    # logit adjustment (inference) and the optional train-time variants.
    _prior_rows = (np.asarray(full_train_set.Y)[~args._holdout_train_mask]
                   if getattr(args, "_holdout_train_mask", None) is not None
                   else full_train_set.Y)
    _prior = compute_class_prior(_prior_rows)
    _LA.prior = _prior
    if getattr(args, "la_taus", None):
        _LA.taus = tuple(float(t) for t in str(args.la_taus).split(","))
    args._prior_t = torch.from_numpy(logit_offset(_prior, 1.0)).float()
    _cls_counts = np.asarray(_prior_rows).sum(axis=0)
    print("\n  class prior on TRAIN (P(present)):")
    print("    " + "  ".join(f"{FIBER_NAMES[k]}:{_prior[k]*100:5.2f}%"
                             for k in range(len(_prior))))
    print(f"    imbalance ratio (max/min) = {_prior.max()/_prior.min():.1f}x")
    if getattr(args, "cls_loss", "soft_margin") == "db":
        args._db_loss = DistributionBalancedLoss(
            _cls_counts, len(_prior_rows),
            neg_scale=float(getattr(args, "db_neg_scale", 5.0)),
            init_bias=float(getattr(args, "db_init_bias", 0.05)),
            use_focal=bool(getattr(args, "db_focal", True))).cuda()
        print(f"  [imbalance] Distribution-Balanced Loss ON "
              f"(neg_scale={getattr(args,'db_neg_scale',5.0)}, "
              f"focal={getattr(args,'db_focal',True)})")
    else:
        args._db_loss = None
    if float(getattr(args, "logit_adjust_train", 0.0)) > 0:
        print(f"  [imbalance] TRAIN-time logit adjustment tau="
              f"{args.logit_adjust_train}")

    # The official test set is always available as its own loader.
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Dispatch: original protocol (val==test) OR leakage-free holdout out of train.
    val_mode = getattr(args, "val_mode", "none")
    # Split seed: defaults to args.seed (coupled). Set args.split_seed to decouple
    # the data partition from initialization/data-order randomness.
    split_seed = getattr(args, "split_seed", None)
    if split_seed is None:
        split_seed = args.seed
    if val_mode == "holdout":
        if split_seed != args.seed:
            print(f"  [seeds] DECOUPLED: init/data-order seed={args.seed}  "
                  f"split seed={split_seed}")
        else:
            print(f"  [seeds] coupled: seed={args.seed} drives init, data order, and split")
        train_idx, val_idx = get_or_make_split(
            full_train_set, cache_dir=getattr(args, "split_cache_dir", None),
            strategy=args.split_strategy, val_frac=args.val_frac,
            seed=split_seed, combination_key=args.combination_key,
            fabric_cap=args.fabric_cap, fabric_merge_threshold=args.fabric_merge_threshold)
        assert_split_clean(full_train_set, train_idx, val_idx, args.split_strategy,
                           fabric_cap=args.fabric_cap,
                           fabric_merge_threshold=args.fabric_merge_threshold)
        _hm = getattr(args, "_holdout_train_mask", None)
        if _hm is not None:
            n_tr0, n_va0 = len(train_idx), len(val_idx)
            train_idx = train_idx[~_hm[train_idx]]
            val_idx = val_idx[~_hm[val_idx]]
            print(f"  [open-set] split preserved, held-out samples removed from it:"
                  f"  train {n_tr0:,}->{len(train_idx):,}   val {n_va0:,}->{len(val_idx):,}")
        _dp = float(getattr(args, "drop_pure", 0.0) or 0.0)
        if _dp > 0:
            _pure = (np.asarray(full_train_set.comps).max(axis=1) >= _dp)
            n_tr0 = len(train_idx)
            train_idx = train_idx[~_pure[train_idx]]
            print(f"  [drop_pure] removed TRAIN samples with a fibre >= {_dp:g}%:"
                  f"  {n_tr0:,} -> {len(train_idx):,} "
                  f"({100*(n_tr0-len(train_idx))/max(n_tr0,1):.1f}% dropped)")
            print(f"  [drop_pure] endmember recovery is now unconfounded for EVERY "
                  f"fibre; the pure spectra become a held-out reference set.")
        train_source = torch.utils.data.Subset(full_train_set, train_idx)
        val_loader = DataLoader(torch.utils.data.Subset(full_train_set, val_idx),
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)
        run_test_at_end = True
        print(f"[val_mode=holdout | split={split_tag(args)}]  "
              f"Train {len(train_idx):,} | Val {len(val_idx):,} | Test {len(test_set):,}")
    else:
        train_idx = np.arange(len(full_train_set))
        _hm = getattr(args, "_holdout_train_mask", None)
        if _hm is not None:
            train_idx = train_idx[~_hm]
            print(f"  [open-set] train restricted to {len(train_idx):,} samples "
                  f"without a held-out fibre")
        train_source = (torch.utils.data.Subset(full_train_set, train_idx)
                        if _hm is not None else full_train_set)
        val_loader = test_loader                     # original: 'val' metrics == test
        run_test_at_end = False
        print(f"[val_mode=none]  Train {len(full_train_set):,} | Val(=test) {len(test_set):,}")

    train_loader = DataLoader(train_source, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    print(f"Input shape: {full_train_set[0][0].shape}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model, student_registry = build_student(args)
    student_arch = getattr(args, "student_arch", None) or "conv"

    # ── Physics-informed wrapper (no-op unless a physics flag is set) ─────
    model = maybe_wrap_physics(model, args, train_set=full_train_set,
                               train_idx=(train_idx if val_mode != "none" else None),
                               length=full_train_set.X.shape[-1],
                               num_classes=args._num_classes)
    model = model.cuda()

    n_params = count_parameters(model)
    print(f"\nModel: {student_arch}-{args.model}  |  Params: {n_params:,}"
          f"{'  | multiscale' if args.multiscale else ''}")

    # Threshold tuning is only REQUIRED when the selection metric is acc_tuned.
    # With select_metric="acc" we select on raw acc@0.5, so tuning is optional
    # (still runs if args.tune_thresholds is set, to report tuned numbers).
    select_metric = getattr(args, "select_metric", "acc_tuned")
    tune_thresholds = args.tune_thresholds or (
        select_metric == "acc_tuned" and (args.save_best_acc or args.save_best_mse))
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    try:
        import torch_optimizer as topt
        optimizer = topt.Lamb(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optim_name = "LAMB"
    except ImportError:
        print("torch_optimizer not found, falling back to AdamW.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optim_name = "AdamW"

    sched_kind = getattr(args, "scheduler", "plateau")
    # SAFETY: with val_mode="none" the "val" loader IS the official test set, so
    # ReduceLROnPlateau(val_loss) would make the LR schedule a function of TEST
    # performance, and save_best_* would select a checkpoint on TEST. The
    # full-train protocol must therefore use a fixed schedule + last-epoch
    # weights. We refuse to run the leaky combination silently.
    if getattr(args, "val_mode", "none") == "none":
        if sched_kind == "plateau":
            raise ValueError(
                "val_mode='none' with scheduler='plateau' would step the LR on TEST loss.\n"
                "  Use --scheduler multistep (fixed milestones, data-independent) for the\n"
                "  full-train protocol, or --val_mode holdout to get a real val set.")
        if args.save_best_acc or args.save_best_mse:
            raise ValueError(
                "val_mode='none' with save_best_acc/save_best_mse would select the\n"
                "  checkpoint on TEST. Use --no_save_best_acc --no_save_best_mse --save_last\n"
                "  (report last-epoch weights), or --val_mode holdout.")
        if not getattr(args, "save_last", False):
            print("  [warn] val_mode='none' and save_last=False — no checkpoint will be "
                  "written. Add --save_last to keep the final weights.")

    if sched_kind == "multistep":
        scheduler = MultiStepLR(optimizer, milestones=list(args.lr_milestones),
                                gamma=args.lr_gamma)
        print(f"  Scheduler: MultiStepLR(milestones={list(args.lr_milestones)}, "
              f"gamma={args.lr_gamma})  [fixed, data-independent]")
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",           # monitor val_loss (lower = better)
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.lr_min,
            #verbose=True,
        )
        print(f"  Scheduler: ReduceLROnPlateau(factor={args.lr_factor}, "
              f"patience={args.lr_patience})  [driven by val loss]")

    # ── WandB ─────────────────────────────────────────────────────────────────
    # Name encodes arch+size, channels, seed, data split and every ablation
    # flag, so runs never overwrite each other's .pt files. The fixed tokens
    # "single" and "ablfull" are kept so names match the checkpoints in the
    # paper, e.g. cleanv3conv-atto_ch2_seed0_fabricVF10_single_ablfull_...
    mode_tag = "single"
    parts = [f"cleanv3{student_arch}-{args.model}", f"ch{args.channels}",
             f"seed{args.seed}", split_tag(args), mode_tag, "ablfull"]
    # Regression-loss ablation tag. Only added when it differs from the default
    # ("smooth_l1") so existing run/checkpoint names are unchanged — without this
    # an l1-vs-smooth_l1 ablation at the same seed would OVERWRITE its own .pt.
    if getattr(args, "scheduler", "plateau") != "plateau":
        parts.append(args.scheduler)          # e.g. ..._multistep_...
    if getattr(args, "reg_loss", "smooth_l1") != "smooth_l1":
        parts.append(f"reg{args.reg_loss}")
    # ── physics ablation tags (keeps each variant's checkpoint distinct) ──
    if getattr(args, "comp_head", "linear") != "linear":
        parts.append(f"head-{args.comp_head}")
    if getattr(args, "physics_decoder", False):
        parts.append(f"unmix{args.physics_mixing}{args.physics_weight:g}")
        if getattr(args, "physics_recon_loss", "mse") != "mse":
            parts.append(args.physics_recon_loss)
    if getattr(args, "scatter_correct", "none") not in (None, "none"):
        parts.append(args.scatter_correct)
    if input_mode_tag(input_mode_of(args)):
        parts.append(input_mode_tag(input_mode_of(args)))
    if getattr(args, "km_transform", False):        parts.append("km")
    if getattr(args, "consistency_weight", 0.0) > 0: parts.append(f"cons{args.consistency_weight:g}")
    if getattr(args, "evidential_weight", 0.0) > 0:  parts.append(f"ev{args.evidential_weight:g}")
    if getattr(args, "synth_blend", False):          parts.append(f"synth{args.synth_blend_prob:g}")
    if float(getattr(args, "mlc_weight", 1.0)) != 1.0: parts.append(f"mlcw{args.mlc_weight:g}")
    if float(getattr(args, "logit_adjust_train", 0.0)) > 0:
        parts.append(f"LAtr{args.logit_adjust_train:g}")
    if getattr(args, "cls_loss", "soft_margin") == "db": parts.append("dbloss")
    if getattr(args, "holdout_classes", None):
        parts.append("hold-" + str(args.holdout_classes).replace(",", "-"))
    if float(getattr(args, "drop_pure", 0.0) or 0.0) > 0:
        parts.append(f"nopure{args.drop_pure:g}")
    run_name = "_".join(parts)
    # depths+dims are meaningful only for a matching MODEL_REGISTRY; guard it.
    _reg_extra = {}
    if (student_registry is not None and args.model in student_registry
            and isinstance(student_registry[args.model], dict)):
        _reg_extra = student_registry[args.model]
    wandb.init(
        project="fabric-endmember-diagnostics",
        mode=os.environ.get("WANDB_MODE", "disabled"),
        name=run_name,
        config={
            "model":          f"{student_arch}-{args.model}",
            "student_arch":   student_arch,
            "n_params":       n_params,
            # physics-informed ablation config
            "comp_head":            getattr(args, "comp_head", "linear"),
            "physics_decoder":      getattr(args, "physics_decoder", False),
            "physics_weight":       getattr(args, "physics_weight", 0.0),
            "physics_mixing":       getattr(args, "physics_mixing", "linear"),
            "physics_recon_loss":   getattr(args, "physics_recon_loss", "mse"),
            "endmember_init":       getattr(args, "endmember_init", "data"),
            "endmember_smooth_weight": getattr(args, "endmember_smooth_weight", 0.0),
            "endmember_anchor_weight": getattr(args, "endmember_anchor_weight", 0.0),
            "evidential_weight":    getattr(args, "evidential_weight", 0.0),
            "consistency_weight":   getattr(args, "consistency_weight", 0.0),
            "scatter_correct":      getattr(args, "scatter_correct", "none"),
            "input_mode":           input_mode_of(args),
            "backbone_in_chans":    backbone_in_chans(args),
            "km_transform":         getattr(args, "km_transform", False),
            "synth_blend":          getattr(args, "synth_blend", False),
            "synth_blend_prob":     getattr(args, "synth_blend_prob", 0.0),
            "mlc_weight":           getattr(args, "mlc_weight", 1.0),
            "optimizer":      optim_name,
            "lr":             args.lr,
            "weight_decay":   args.weight_decay,
            "drop_path_rate": args.drop_path,
            "batch_size":     args.batch_size,
            "epochs":         args.epochs,
            "channels":       args.channels,
            "channel_idx":    args.channel_idx,
            "lr_patience":    args.lr_patience,
            "lr_factor":      args.lr_factor,
            "lr_min":         args.lr_min,
            "seed":           args.seed,
            "split_seed":     split_seed,
            "seeds_coupled":  (split_seed == args.seed),
            "val_mode":       getattr(args, "val_mode", "none"),
            "split_strategy": getattr(args, "split_strategy", None),
            "combination_key": getattr(args, "combination_key", None),
            "val_frac":       getattr(args, "val_frac", None),
            "multiscale":     args.multiscale,
            "use_mixup":      args.use_mixup,
            "sum_constraint": args.sum_constraint,
            "tune_thresholds": tune_thresholds,
            "reg_loss":       args.reg_loss,
            "ms_init":        "trunc_normal",  # CHANGED
            **_reg_extra,  # depths + dims (local ConvNeXt student only)
        },
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best = {
        "acc": 0.0, "mse": float("inf"), "hamming": float("inf"),
        "acc_tuned": 0.0, "acc_tuned_oracle": 0.0,
        "mse_tuned": float("inf"), "mse_tuned_oracle": float("inf"),
        **{f"acc_within_{t}": 0.0 for t in [0, 1, 2, 3, 5, 10]},
    }

    # Keep the best model in memory (by tuned acc if tuning, else raw acc) so the
    # end-of-run report describes the BEST model, not the last epoch.
    sel_metric_name = "acc_tuned" if tune_thresholds else "acc"
    best_sel = -1.0
    best_state = None
    best_thr_full_val = None   # val-fit thresholds that pair with the best weights
    best_la_tau = None         # post-hoc logit-adjust tau chosen on VAL at that epoch
    best_thr_ham_full_val = None

    print(f"\n{'='*70}")
    print(f"  Starting training: {run_name}")
    print(f"  Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"  Multi-scale init: trunc_normal_ (all stages active from epoch 1)")
    print(f"{'='*70}\n")

    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"\n{'─'*70}")
        print(f"  Epoch {epoch+1}/{args.epochs}  |  LR: {get_current_lr(optimizer):.2e}")
        print(f"{'─'*70}")

        train_loss = train(model, optimizer, train_loader, epoch,
                           use_mixup=args.use_mixup, mixup_alpha=args.mixup_alpha,
                           mixup_prob=args.mixup_prob,
                           use_sum_constraint=args.sum_constraint,
                           sum_constraint_weight=args.sum_constraint_weight,
                           reg_loss=args.reg_loss, reg_smooth_beta=args.reg_smooth_beta,
                           args=args)

        # Validate
        res = validate(model, val_loader, epoch, tune_thresholds=tune_thresholds,
                       threshold_seed=args.seed, diagnostics=args.diagnostics,
                       reg_loss=args.reg_loss, reg_smooth_beta=args.reg_smooth_beta)
        acc, mse, hamming = res["acc"], res["mse"], res["hamming"]
        val_loss, thresh_accs = res["val_loss"], res["thresh_accs"]

        # ── Per-epoch TEST evaluation (analysis curves only) ───────────────
        # Uses THIS epoch's val-tuned thresholds, so test/acc_tuned_by_val shows
        # how well thresholds fitted on val transfer to test, epoch by epoch.
        # Nothing here feeds the scheduler, early stopping, or checkpointing.
        test_res = None
        if (run_test_at_end and test_loader is not None
                and getattr(args, "eval_test_every_epoch", False)
                and ((epoch + 1) % max(1, getattr(args, "test_eval_freq", 1)) == 0
                     or epoch == args.epochs - 1)):
            test_res = evaluate_on_test(
                model, test_loader,
                thr_acc=res.get("thresholds_full_val"),
                thr_ham=res.get("thresholds_hamming_full_val"),
                step=epoch, prefix="test",
                reg_loss=args.reg_loss, reg_smooth_beta=args.reg_smooth_beta,
                verbose=False, la_tau=res.get("la_best_tau"))

        # Step scheduler. MultiStepLR is metric-free (fixed milestones);
        # ReduceLROnPlateau consumes the val loss.
        if sched_kind == "multistep":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        # Track bests
        new_best_acc     = acc     > best["acc"]
        new_best_mse     = mse     < best["mse"]
        new_best_hamming = hamming < best["hamming"]
        if new_best_acc:     best["acc"]     = acc
        if new_best_mse:     best["mse"]     = mse
        if new_best_hamming: best["hamming"] = hamming
        for k, v in thresh_accs.items():
            key = k.replace("val/", "")
            if v > best.get(key, 0.0):
                best[key] = v
        # Tuned/oracle bests (acc higher-is-better, mse lower-is-better)
        if res["acc_tuned"] is not None and res["acc_tuned"] > best["acc_tuned"]:
            best["acc_tuned"] = res["acc_tuned"]
        if res["acc_tuned_oracle"] is not None and res["acc_tuned_oracle"] > best["acc_tuned_oracle"]:
            best["acc_tuned_oracle"] = res["acc_tuned_oracle"]
        if res["mse_tuned"] is not None and res["mse_tuned"] < best["mse_tuned"]:
            best["mse_tuned"] = res["mse_tuned"]
        if res["mse_tuned_oracle"] is not None and res["mse_tuned_oracle"] < best["mse_tuned_oracle"]:
            best["mse_tuned_oracle"] = res["mse_tuned_oracle"]

        # ── Checkpointing (save the honest tuned thresholds, not the oracle) ──
        _sel_val = (res["acc_tuned"] if (select_metric == "acc_tuned"
                                         and res["acc_tuned"] is not None) else acc)
        if args.save_best_acc and _sel_val is not None:
            if _sel_val >= best.get("_ckpt_acc_tuned", -1.0):
                best["_ckpt_acc_tuned"] = _sel_val
                path = os.path.join(args.save_dir, f"{run_name}_best_acc.pt")
                torch.save({"model": model.state_dict(), **_ckpt_meta(args),
                            "thresholds": res["thresholds"],
                            "thresholds_full_val": res["thresholds_full_val"],
                            "thresholds_hamming_full_val": res.get("thresholds_hamming_full_val"),
                            "epoch": epoch, "acc_tuned": res["acc_tuned"],
                            "select_metric": select_metric, "selected_value": _sel_val,
                            "mse": mse}, path)
                print(f"    ✔ saved best-acc checkpoint "
                      f"({select_metric}={_sel_val:.4f}) -> {path}")
        if args.save_best_mse and new_best_mse:
            path = os.path.join(args.save_dir, f"{run_name}_best_mse.pt")
            torch.save({"model": model.state_dict(), **_ckpt_meta(args),
                        "thresholds": res["thresholds"],
                        "thresholds_full_val": res["thresholds_full_val"],
                        "thresholds_hamming_full_val": res.get("thresholds_hamming_full_val"),
                        "epoch": epoch, "mse": mse}, path)
            print(f"    ✔ saved best-mse checkpoint (mse={mse:.4f}) -> {path}")

        # In-memory snapshot of the best model for the final report.
        # Selection metric (drives best_state -> the final TEST evaluation).
        if select_metric == "acc_tuned" and tune_thresholds and res["acc_tuned"] is not None:
            sel = res["acc_tuned"]
        else:
            sel = acc                      # raw subset accuracy @ 0.5
        if sel is not None and sel > best_sel:
            best_sel = sel
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_thr_full_val = res.get("thresholds_full_val")  # pairs with these weights
            best_la_tau = res.get("la_best_tau")                # pairs with these weights
            best_thr_ham_full_val = res.get("thresholds_hamming_full_val")

        epoch_time = time.time() - epoch_start

        # ── Per-epoch summary ──────────────────────────────────────────────
        print(f"\n  ┌─ Epoch {epoch+1:03d} Summary {'─'*46}┐")
        print(f"  │  Time:        {epoch_time:6.1f}s")
        print(f"  │  Train loss:  {train_loss:.4f}   Val loss: {val_loss:.4f}")
        print(f"  ├─ Classification {'─'*51}")
        print(f"  │  Acc:         {acc:.4f}  {'★ NEW BEST' if new_best_acc else '':12s}  Best: {best['acc']:.4f}")
        print(f"  │  Hamming:     {hamming:.4f}  {'★ NEW BEST' if new_best_hamming else '':12s}  Best: {best['hamming']:.4f}")
        if res.get("la_best_tau") is not None:
            _lt = res["la_table"]
            print(f"  │  LogitAdj:    "
                  + "  ".join(f"t={t:g}:{_lt[t][0]:.4f}" for t in sorted(_lt))
                  + f"   best tau={res['la_best_tau']:g}")
        if res["acc_tuned"] is not None:
            print(f"  │  Acc tuned:   {res['acc_tuned']:.4f}   (oracle: {res['acc_tuned_oracle']:.4f})  Best tuned: {best['acc_tuned']:.4f}")
        print(f"  ├─ Regression {'─'*54}")
        print(f"  │  MSE:         {mse:.4f}  {'★ NEW BEST' if new_best_mse else '':12s}  Best: {best['mse']:.4f}")
        if res["mse_tuned"] is not None:
            print(f"  │  MSE tuned:   {res['mse_tuned']:.4f}   (oracle: {res['mse_tuned_oracle']:.4f})")
        ta = thresh_accs
        print(f"  │  Within ±0:   {ta['val/acc_within_0']:.4f}   ±1: {ta['val/acc_within_1']:.4f}   ±2: {ta['val/acc_within_2']:.4f}")
        print(f"  │  Within ±3:   {ta['val/acc_within_3']:.4f}   ±5: {ta['val/acc_within_5']:.4f}   ±10:{ta['val/acc_within_10']:.4f}")
        if test_res is not None:
            _at = (f"  tuned_by_val: {test_res['acc_tuned_by_val']:.4f}"
                   if test_res["acc_tuned_by_val"] is not None else "")
            _ht = (f"  tuned_by_val: {test_res['hamming_tuned_by_val']:.4f}"
                   if test_res["hamming_tuned_by_val"] is not None else "")
            print(f"  ├─ TEST (monitoring only — never select on this) {'─'*20}")
            print(f"  │  Test acc:    {test_res['acc']:.4f}{_at}")
            print(f"  │  Test hamming:{test_res['hamming']:.4f}{_ht}")
            print(f"  │  Test MSE:    {test_res['mse']:.4f}")
        print(f"  │  LR:          {get_current_lr(optimizer):.2e}")
        print(f"  └{'─'*67}┘")

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Training complete: {run_name}")
    print(f"  Best Accuracy:       {best['acc']:.4f}")
    print(f"  Best Acc (tuned):    {best['acc_tuned']:.4f}   (oracle: {best['acc_tuned_oracle']:.4f})")
    print(f"  Best MSE:            {best['mse']:.4f}")
    print(f"  Best MSE (tuned):    {best['mse_tuned']:.4f}   (oracle: {best['mse_tuned_oracle']:.4f})")
    print(f"  Best Hamming:        {best['hamming']:.4f}")
    print(f"{'='*70}")
    print("  NOTE: the 'Best X' values above are per-metric maxima over epochs and")
    print("  may come from DIFFERENT epochs — they describe no single model.")
    print("  Report the final/reported block printed at the end instead.")

    # ⚠️ PER-METRIC MAXIMA, tracked independently across epochs. best_acc and
    # best_mse may come from different epochs, so together they describe no
    # single model. Kept for monitoring / comparison with the benchmark's
    # reporting style only. NEVER put these in a results table.
    wandb.log({
        "final/best_acc":              best["acc"],
        "final/best_acc_tuned":        best["acc_tuned"],
        "final/best_acc_tuned_oracle": best["acc_tuned_oracle"],
        "final/best_mse":              best["mse"],
        "final/best_mse_tuned":        best["mse_tuned"],
        "final/best_mse_tuned_oracle": best["mse_tuned_oracle"],
        "final/best_hamming":          best["hamming"],
        **{f"final/best_{k}": v for k, v in best.items() if k.startswith("acc_within")},
    })

    # ── Full diagnostics on the BEST model ──────────────────────────────────
    # ── Last-epoch checkpoint (full-train protocol) ─────────────────────────
    # With no val set there is nothing legitimate to select on, so the reported
    # model is simply the final one. Saved BEFORE best_state is restored below.
    if getattr(args, "save_last", False):
        last_path = os.path.join(args.save_dir, f"{run_name}_last.pt")
        torch.save({"model": model.state_dict(), **_ckpt_meta(args),
                    "thresholds": res.get("thresholds"),
                    "thresholds_full_val": res.get("thresholds_full_val"),
                    "thresholds_hamming_full_val": res.get("thresholds_hamming_full_val"),
                    "epoch": args.epochs - 1, "selection": "last_epoch"}, last_path)
        print(f"\n  ✔ saved LAST-epoch checkpoint -> {last_path}")

    # ── THE NUMBERS TO REPORT (full-train protocol) ─────────────────────────
    # val_mode="none": no val set exists, so the reported model is the LAST
    # epoch, evaluated once on the official test set. Model still holds
    # last-epoch weights here (best_state is restored below).
    if val_mode == "none":
        print(f"\n{'#'*70}")
        print(f"  REPORTED RESULT — weights: LAST EPOCH ({args.epochs})")
        print(f"  protocol: full train set, no val split, fixed schedule")
        print(f"{'#'*70}")
        evaluate_on_test(model, test_loader, thr_acc=None, thr_ham=None,
                         step=None, prefix="final/reported",
                         reg_loss=args.reg_loss, reg_smooth_beta=args.reg_smooth_beta,
                         verbose=True)
        try:
            wandb.run.summary["reported_selection"] = "last_epoch"
            wandb.run.summary["reported_epoch"] = args.epochs - 1
            wandb.run.summary["reported_prefix"] = "final/reported"
        except Exception:
            pass
        print(f"\n  ^ REPORT THESE (final/reported/*) — NOT final/best_*")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n{'#'*70}")
    print(f"  FINAL DIAGNOSTICS ON BEST MODEL  ({sel_metric_name}={best_sel:.4f})")
    print(f"{'#'*70}")
    final_report(model, val_loader, step=args.epochs, log_wandb=True)

    # ── Honest TEST-set evaluation of the val-selected best model ────────────
    # Only in holdout mode; in 'none' mode the val metrics ARE the test set.
    # NOTE: model already holds best_state (loaded above), so this is the BEST
    # epoch by val, NOT the last epoch. This is the number to report.
    if run_test_at_end and (test_loader is not None):
        print(f"\n{'#'*70}")
        print(f"  REPORTED RESULT — weights: BEST VAL EPOCH "
              f"({sel_metric_name}={best_sel:.4f})")
        print(f"  protocol: fabric-holdout val split, selection on val, test scored once")
        print(f"{'#'*70}")
        final_test = evaluate_on_test(
            model, test_loader,
            thr_acc=best_thr_full_val, thr_ham=best_thr_ham_full_val,
            step=None, prefix="final/test",
            reg_loss=args.reg_loss, reg_smooth_beta=args.reg_smooth_beta,
            verbose=True, la_tau=best_la_tau)
        try:
            wandb.run.summary["reported_selection"] = f"best_val_epoch({sel_metric_name})"
            wandb.run.summary["reported_prefix"] = "final/test"
        except Exception:
            pass
        print(f"\n  ^ REPORT THESE (final/test/*) — NOT final/best_*  "
              f"(mean ± std across seeds).")

    if _test_unknown is not None:
        evaluate_open_set(model, test_set, _test_unknown)

    wandb.finish()


if __name__ == "__main__":
    main()
