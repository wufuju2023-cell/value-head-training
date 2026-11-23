#!/usr/bin/env python3
"""Value head + losses + metrics for the value-head-training plugin.

Target: **remaining proof steps** for a mission (tactic steps, `log1p`-encoded).
The state is mission-level (one record per mission state / agent response); the
*target* is steps.  `--unit blocks` is a deprecated shim that still targets
`ceil(steps / block_steps)`, kept only so the old checkpoints can be compared.

    INVARIANT: d_hat is in proof steps; N is in proof steps; rounds = ceil(d_hat/N).

The head has two outputs over one trunk (the "dual head" of DESIGN.md §4): a
64-bin log-spaced bucket distribution plus a bounded residual that refines inside
the bucket; the policy head predicts the mission-level action `block` vs
`decompose`.

Pure-python part (no torch import at module load):

  bucket_edges / bucket_centers / two_hot / decode_expected / quantiles
  metrics()            MAE in log space, bucket accuracy, calibration, tail recall
  gate()               re-exported from gate.py (single source of truth)

Torch part, built on first use by `ensure()`:

  Model                 hashed backbone + policy head + value head
  4 value encodings     "uniform" | "log" | "regress" | "dual"
  loss                  value_loss * 1e-3 + policy CE   (VALUE_COEF below)

Why the four encodings are all present at once: DESIGN.md §4 recommends "dual"
(coarse log bucket + residual), but the comparison must be runnable, so every
encoding is one flag away and shares the same metrics.

Torch is imported lazily.  If it is missing, importing this module still works
for the pure functions; `ensure()` raises `TorchUnavailable` and the CLI
scripts turn that into a one-line message plus exit code 2.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import zlib
from typing import Any

import dataset as _dataset  # same directory; provides stable_hash

VALUE_COEF = 1e-3          # AlphaProof convention (docs: value_coef=1e-3)
DEFAULT_BUCKETS = 64
DEFAULT_MAX_STEPS = 32768  # log grid top; >10k tail is inside the grid
ENCODINGS = ("uniform", "log", "regress", "dual")
TAIL_STEPS = 10000

# d_hat is in proof steps; N is in proof steps; rounds = ceil(d_hat/N).
# One unit only.  `blocks` survives as a deprecated shim for old checkpoints; do
# not add a third unit, and never mix them inside one comparison.
DEFAULT_UNIT = "steps"                      # remaining tactic steps
UNITS = ("steps", "blocks", "tactic")       # "blocks" is DEPRECATED
DEPRECATED_UNITS = ("blocks",)
DEFAULT_BLOCK_STEPS = 3                     # only used by the deprecated shim


def unit_of(records: "list[dict]") -> str:
    """Infer the target unit of a record set (default: steps)."""
    for rec in records:
        unit = rec.get("unit")
        if unit in UNITS:
            return str(unit)
        if rec.get("remaining_blocks") is not None:
            return "blocks"
        if rec.get("remaining_steps") is not None:
            return "steps"
    return DEFAULT_UNIT


def target_steps(record: dict, unit: str = DEFAULT_UNIT) -> float:
    """The number the value head regresses on.

    steps:   remaining tactic steps (default; the gate's unit)
    tactic:  same as steps -- legacy alias for step-level records
    blocks:  DEPRECATED `ceil(steps / block_steps)`; only for old checkpoints
    """
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
    if unit == "blocks":
        for key in ("remaining_blocks", "d_hat_target"):
            if record.get(key) is not None:
                return float(record[key])
        if record.get("remaining_steps") is not None:
            return float(math.ceil(max(0.0, float(record["remaining_steps"]))
                                   / DEFAULT_BLOCK_STEPS))
        raise KeyError("blocks record needs remaining_blocks / d_hat_target")
    if record.get("remaining_steps") is None:
        raise KeyError("steps record needs remaining_steps")
    return float(record["remaining_steps"])


# ---------------------------------------------------------------------------
# Torch availability
# ---------------------------------------------------------------------------

class TorchUnavailable(RuntimeError):
    pass


_TORCH = None
_TORCH_ERROR: str | None = None
try:  # noqa: SIM105 - keep the error text for the CLIs
    import torch as _TORCH  # type: ignore
except Exception as exc:  # pragma: no cover - depends on the machine
    _TORCH = None
    _TORCH_ERROR = f"{type(exc).__name__}: {exc}"

_CLASSES: dict[str, Any] = {}
_NS: Any = None


class _Namespace:
    """Attribute + item access, so call sites can use ns["Model"] or ns.Model."""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def torch_available() -> bool:
    return _TORCH is not None


def torch_error() -> str:
    return _TORCH_ERROR or ""


def ensure() -> Any:
    """Build (once) and return a namespace with the torch classes/helpers."""
    global _NS
    if _TORCH is None:
        raise TorchUnavailable(
            "PyTorch is not importable on this Python "
            f"({_TORCH_ERROR}). Install torch in the training venv, e.g. "
            "`python3 -m pip install torch` (CPU wheel is enough for --tiny). "
            "Nothing else in this plugin needs it.")
    if _NS is None:
        if not _CLASSES:
            _CLASSES.update(_define_torch(_TORCH))
        ns = _Namespace()
        for key, value in _CLASSES.items():
            setattr(ns, key, value)
        _NS = ns
    return _NS


# ---------------------------------------------------------------------------
# Bucket geometry (pure python)
# ---------------------------------------------------------------------------

def bucket_edges(encoding: str, buckets: int = DEFAULT_BUCKETS,
                 max_steps: int = DEFAULT_MAX_STEPS) -> list[float]:
    """Bin edges.  Bin 0 is ALWAYS the degenerate terminal bin `[0, 1)`.

    The target is an *integer* remaining step count, so `[0, 1)` contains exactly
    one representable value: 0.  Reserving bin 0 for it is what makes
    `encode(0) == 0` and `decode(0) == 0.0` hold for **every** encoding, and it
    stops the grid from spending a whole bin boundary on the 0-to-1 gap (a log
    grid over `[1, 32768]` would otherwise give 0 and 1 the same resolution and
    put a 4.5x-wide first bin in the middle of the geometry).

    Bins `1 .. buckets-1` then cover `[1, max_steps]` in the encoding's own
    space, so `uniform` vs `log` differs only in *spacing*, never in whether
    terminal is exact.  That is what makes the ablation a fair comparison.
    """
    if buckets < 2:
        raise ValueError("buckets must be >= 2")
    if max_steps <= 1:
        raise ValueError("max_steps must be > 1")
    if encoding == "uniform":
        width = (float(max_steps) - 1.0) / (buckets - 1)
        return [0.0] + [1.0 + i * width for i in range(buckets)]
    lo_l, hi_l = math.log1p(1.0), math.log1p(float(max_steps))
    edges = [0.0] + [math.expm1(lo_l + (hi_l - lo_l) * i / (buckets - 1))
                     for i in range(buckets)]
    edges[-1] = float(max_steps)   # expm1 rounding: pin the top edge exactly
    return edges


def bucket_centers(encoding: str, edges: list[float]) -> list[float]:
    """Centres in step space.  Bin 0 is exactly 0 so terminal is exact.

    Log grids use the log1p midpoint (the space two-hot interpolates in), not
    the arithmetic midpoint: that keeps decoding and the training target in the
    same geometry, which is what makes the tail calibration hold.  Bin 0 is the
    only bin whose centre is not an interior point: it is `[0, 1)` and its only
    integer member is 0.
    """
    centers = [0.0]
    for i in range(1, len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if encoding == "uniform":
            centers.append((lo + hi) / 2)
        else:
            centers.append(math.expm1((math.log1p(lo) + math.log1p(hi)) / 2))
    return centers


def _grid(encoding: str, buckets: int, max_steps: int) -> tuple[list[float], list[float], bool]:
    """(edges, centres, log_scale) for an encoding.  `regress` borrows the log grid."""
    grid_enc = "log" if encoding == "regress" else encoding
    edges = bucket_edges(grid_enc, buckets, max_steps)
    return edges, bucket_centers(grid_enc, edges), grid_enc != "uniform"


def encode(value: float, encoding: str = "log", buckets: int = DEFAULT_BUCKETS,
           max_steps: int = DEFAULT_MAX_STEPS) -> int:
    """Bucket index containing `value`.  `encode(0) == 0` for every encoding.

    This is the canonical encoder: it is monotone non-decreasing in `value` and
    it is the *only* place the bin boundaries are interpreted, so the test suite
    can pin `encode(0) -> 0` without going through torch.
    """
    import bisect
    edges, _, log_scale = _grid(encoding, buckets, max_steps)
    xs = [math.log1p(e) for e in edges] if log_scale else list(edges)
    x = math.log1p(max(0.0, float(value))) if log_scale else max(0.0, float(value))
    if x <= xs[0]:
        return 0
    if x >= xs[-1]:
        return buckets - 1
    return min(buckets - 1, bisect.bisect_right(xs, x) - 1)


def decode(bucket: int, encoding: str = "log", buckets: int = DEFAULT_BUCKETS,
           max_steps: int = DEFAULT_MAX_STEPS) -> float:
    """Representative step value of a bucket: its centre.  `decode(0) == 0.0`."""
    _, centers, _ = _grid(encoding, buckets, max_steps)
    if not 0 <= bucket < len(centers):
        raise IndexError(f"bucket {bucket} out of range 0..{len(centers) - 1}")
    return float(centers[bucket])


def decode_mode(probs: list[float], centers: list[float], log_scale: bool,
                window: int = 3) -> tuple[float, int]:
    """Binned mode decode: local mean of the `window` bins around the argmax.

    Returns `(value_in_steps, mode_bucket)`.  A global expectation over a
    log-spaced distribution is dragged toward the geometric middle, so an
    uncertain head *underestimates* the tail — the exact failure the 500-step
    gate cannot tolerate.  The mode cannot do that: it reports where the mass
    actually is, and the local mean only smooths the quantisation.

    Bucket 0 short-circuits to exactly 0.0: it is `[0, 1)` and its only integer
    member is 0, so "the mode is the terminal bin" must decode as terminal.
    """
    if not probs:
        return 0.0, 0
    nb = len(probs)
    b = max(range(nb), key=lambda i: probs[i])
    if b == 0:
        return 0.0, 0
    # Bin 0 is degenerate (`[0,1)`), so it never enters a local mean: blending a
    # bin whose whole span is "0" into a neighbourhood would bias every small
    # estimate downward.
    lo, hi = max(1, b - window // 2), min(nb - 1, b + window // 2)
    idx = list(range(lo, hi + 1))
    w = [probs[i] for i in idx]
    tot = sum(w) or 1e-9
    if log_scale:
        centre = sum(p * math.log1p(centers[i]) for p, i in zip(w, idx)) / tot
        return math.expm1(max(0.0, centre)), b
    return max(0.0, sum(p * centers[i] for p, i in zip(w, idx)) / tot), b



def two_hot(value: float, edges: list[float], log_scale: bool) -> tuple[int, int, float]:
    """Adjacent-bin soft target. Returns (i, i+1, weight_on_i)."""
    xs = [math.log1p(e) for e in edges] if log_scale else list(edges)
    x = math.log1p(max(0.0, value)) if log_scale else max(0.0, value)
    if x <= xs[0]:
        return 0, 0, 1.0
    if x >= xs[-1]:
        return len(xs) - 2, len(xs) - 2, 1.0
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    span = xs[hi] - xs[lo]
    frac = 0.0 if span <= 0 else (x - xs[lo]) / span
    i = min(lo, len(xs) - 2)
    return i, i + 1, 1.0 - frac


def decode_expected(probs: list[float], centers: list[float], log_scale: bool) -> float:
    if log_scale:
        acc = sum(p * math.log1p(c) for p, c in zip(probs, centers))
        return math.expm1(acc)
    return sum(p * c for p, c in zip(probs, centers))


def quantiles(probs: list[float], centers: list[float], log_scale: bool,
              qs: tuple[float, ...] = (0.1, 0.9)) -> list[float]:
    cum, out, k = 0.0, [], 0
    targets = list(qs)
    for i, p in enumerate(probs):
        cum += p
        while k < len(targets) and cum >= targets[k]:
            c = centers[i]
            out.append(math.expm1(math.log1p(c)) if log_scale else c)
            k += 1
    while len(out) < len(targets):
        c = centers[-1]
        out.append(math.expm1(math.log1p(c)) if log_scale else c)
    return out


# ---------------------------------------------------------------------------
# Metrics (pure python)
# ---------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def bucket_index(value: float, centers: list[float]) -> int:
    best, bd = 0, float("inf")
    for i, c in enumerate(centers):
        d = abs(math.log1p(c) - math.log1p(max(0.0, value))) if c > 0 or value > 0 else abs(c - value)
        if d < bd:
            best, bd = i, d
    return best


def calibration(preds: list[float], trues: list[float], bins: int = 5) -> dict:
    """Monotone calibration in log space over predicted-quantile bins."""
    if len(preds) < 2:
        return {"bins": 0, "violations": 0, "monotone": True, "gap_log": 0.0}
    order = sorted(range(len(preds)), key=lambda i: preds[i])
    bins = max(2, min(bins, max(2, len(preds) // 3)))  # tiny n -> fewer bins
    size = max(1, len(order) // bins)
    rows = []
    for b in range(bins):
        idx = order[b * size: (b + 1) * size] if b < bins - 1 else order[b * size:]
        if not idx:
            continue
        mp = _mean([math.log1p(preds[i]) for i in idx])
        mt = _mean([math.log1p(trues[i]) for i in idx])
        rows.append({"n": len(idx), "pred_log": round(mp, 4), "true_log": round(mt, 4)})
    violations = sum(1 for a, b in zip(rows, rows[1:]) if b["true_log"] < a["true_log"] - 1e-9)
    gap = _mean([abs(r["pred_log"] - r["true_log"]) for r in rows])
    return {"bins": len(rows), "violations": violations,
            "monotone": violations == 0, "gap_log": round(gap, 4), "rows": rows}


def metrics(preds: list[float], trues: list[float], *,
            at: float = TAIL_STEPS, centers: list[float] | None = None) -> dict:
    """The four numbers DESIGN.md §4 asks for.

    mae_log        mean |log1p(pred) - log1p(true)|   (primary, scale free)
    mae_steps      mean |pred - true|                 (interpretable)
    bucket_acc     fraction landing in the true bucket (needs `centers`)
    calibration    monotone check + gap in log space
    tail_recall    P(pred >= at | true >= at), plus tail mae_log
    """
    n = len(preds)
    if n == 0:
        return {"n": 0}
    errs_log = [abs(math.log1p(max(0.0, p)) - math.log1p(max(0.0, t)))
                for p, t in zip(preds, trues)]
    errs_steps = [abs(p - t) for p, t in zip(preds, trues)]
    med = _median([math.log1p(t) for t in trues])
    baseline = [math.expm1(med)] * n
    base_err = [abs(math.log1p(p) - math.log1p(t)) for p, t in zip(baseline, trues)]
    tail_idx = [i for i, t in enumerate(trues) if t >= at]
    out: dict[str, Any] = {
        "n": n,
        "mae_log": round(_mean(errs_log), 4),
        "mae_steps": round(_mean(errs_steps), 1),
        "median_ae_steps": round(_median(errs_steps), 1),
        "baseline_mae_log": round(_mean(base_err), 4),
        "skill_vs_median": round(1.0 - _mean(errs_log) / max(1e-9, _mean(base_err)), 4),
        "calibration": calibration(preds, trues),
    }
    if centers:
        out["bucket_acc"] = round(
            sum(1 for p, t in zip(preds, trues)
                if bucket_index(p, centers) == bucket_index(t, centers)) / n, 4)
    if tail_idx:
        hit = sum(1 for i in tail_idx if preds[i] >= at)
        out["tail"] = {
            "at": at, "n": len(tail_idx),
            "recall": round(hit / len(tail_idx), 4),
            "mae_log": round(_mean([errs_log[i] for i in tail_idx]), 4),
            "median_ratio": round(_median([preds[i] / max(1.0, trues[i]) for i in tail_idx]), 3),
        }
    else:
        out["tail"] = {"at": at, "n": 0, "recall": None}
    zero = [i for i, t in enumerate(trues) if t == 0]
    out["zero"] = {
        "n": len(zero),
        # "exact" = the head says literally 0.0, not 0.4.  A bucket encoding
        # gets this for free (bin 0 decodes to 0.0); a regression head does not.
        "exact": (round(sum(1 for i in zero if abs(preds[i]) < 1e-9) / len(zero), 4)
                  if zero else None),
        "mae_steps": round(_mean([errs_steps[i] for i in zero]), 3) if zero else None,
    }
    return out


# ---------------------------------------------------------------------------
# Baseline (non-learning) predictors, useful as a sanity floor
# ---------------------------------------------------------------------------

def constant_predictor(train_trues: list[float], test_trues: list[float]) -> list[float]:
    med = _median([math.log1p(t) for t in train_trues])
    return [math.expm1(med)] * len(test_trues)


# ---------------------------------------------------------------------------
# Synthetic corpus: the ablation fixture (shared by ablate_head.py and --probe)
# ---------------------------------------------------------------------------

def synthetic_corpus(n: int = 400, seed: int = 7, zero_frac: float = 0.15,
                     tail_frac: float = 0.20, tail_at: float = TAIL_STEPS,
                     max_steps: int = DEFAULT_MAX_STEPS,
                     noise_tokens: float = 1.2) -> list[dict]:
    """Heavy-tailed (text, steps) rows with exact 0s and a real >10k tail.

    Why synthetic and not the bank: the bank has no tail (p50 = 1 remaining
    step, max 15), so it cannot say anything about the encoding's tail
    behaviour.  This fixture is *controlled* instead: the step count is a
    function of a latent difficulty only, and the text exposes `log1p(d)` through
    a token count plus bounded noise.  So a correct head can reach the noise
    floor and no better, which makes differences between encodings encoding
    error rather than data luck.

    Terminal states are identifiable from the text on purpose (`goals: none`,
    zero hypothesis tokens), because in the real pipeline they are: a terminal
    record has no open goals left.  Without that, "zero-step exactness" would
    measure the label noise, not the decoder.
    """
    rng = random.Random(seed)
    rows: list[dict] = []
    for i in range(n):
        u = rng.random()
        if u < zero_frac:
            d = 0.0
        elif u < zero_frac + tail_frac:
            d = math.expm1(rng.uniform(math.log1p(tail_at), math.log1p(max_steps)))
        else:
            d = math.expm1(rng.uniform(0.0, math.log1p(tail_at)))
        d = float(round(max(0.0, min(float(max_steps), d))))
        k = 0 if d == 0 else max(1, int(round(8.0 * math.log1p(d)
                                              + rng.gauss(0.0, noise_tokens))))
        goal = "theorem syn_%04d %s : True" % (i, " ".join("h%d" % j for j in range(k)))
        ctx = "import Mathlib\ngoals: %s\n%s" % (
            "none" if d == 0 else "1",
            "\n".join("-- prior step %d" % j for j in range(min(5, k // 4))))
        rows.append({"id": "syn-%04d" % i, "unit": "steps", "source": "synthetic",
                     "goal": goal, "context": ctx, "text": goal + "\n" + ctx,
                     "remaining_steps": d, "terminal": d == 0,
                     "next_action": "block" if d < 500 else "decompose"})
    return rows


# ---------------------------------------------------------------------------
# Torch classes
# ---------------------------------------------------------------------------

_TOKEN_RE = None


def _tokens(text: str) -> list[str]:
    global _TOKEN_RE
    if _TOKEN_RE is None:
        import re
        _TOKEN_RE = re.compile(r"[A-Za-z0-9_'\u03b1-\u03c9]+|[^\sA-Za-z0-9_'\u03b1-\u03c9]")
    return _TOKEN_RE.findall(text.lower())


_HASH_CACHE: dict[tuple[str, int], tuple[float, ...]] = {}


def hash_vector(text: str, dim: int) -> tuple[float, ...]:
    """Deterministic hashed bag of words + bigrams, L2 normalised."""
    key = (text, dim)
    hit = _HASH_CACHE.get(key)
    if hit is not None:
        return hit
    toks = _tokens(text)
    counts: dict[str, float] = {}
    for t in toks:
        counts[t] = counts.get(t, 0.0) + 1.0
    for a, b in zip(toks, toks[1:]):
        g = "\x00" + a + " " + b
        counts[g] = counts.get(g, 0.0) + 1.0
    vec = [0.0] * dim
    for tok, c in counts.items():
        h = _dataset.stable_hash(tok)
        idx = h % dim
        sign = 1.0 if (h >> 31) & 1 else -1.0
        vec[idx] += sign * (1.0 + math.log(c))
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    out = tuple(v / norm for v in vec)
    if len(_HASH_CACHE) > 200_000:
        _HASH_CACHE.clear()
    _HASH_CACHE[key] = out
    return out


def _define_torch(torch: Any) -> dict[str, Any]:
    import torch.nn as nn
    import torch.nn.functional as F

    def hash_batch(texts: list[str], dim: int):
        return torch.tensor([hash_vector(t, dim) for t in texts], dtype=torch.float32)

    class HashedBackbone(nn.Module):
        """Stand-in for a HF base model: text -> hidden vector.

        Same interface as the real `Backbone` below (`forward(texts)` and
        `.hidden_size`), so train/predict code never branches on tiny mode.
        The hashed bag carries content; three explicit length features carry
        the "how big is this obligation" signal that a real hidden state would
        also encode (goal size, context size, how many steps already ran).
        """

        def __init__(self, hidden: int = 64):
            super().__init__()
            self.hidden_size = hidden
            self.proj = nn.Linear(hidden, hidden)
            self.len_proj = nn.Linear(3, hidden)
            self.norm = nn.LayerNorm(hidden)

        @staticmethod
        def length_features(texts: list[str]):
            rows = []
            for t in texts:
                goal, _, ctx = t.partition("\n")
                rows.append([
                    math.log1p(len(_tokens(goal))) / 4.0,
                    math.log1p(len(_tokens(ctx))) / 4.0,
                    math.log1p(ctx.count("\n")) / 2.0,
                ])
            return torch.tensor(rows, dtype=torch.float32)

        def forward(self, texts=None, goals=None, contexts=None):
            if texts is None:
                if goals is None:
                    raise ValueError("HashedBackbone needs texts or (goals, contexts)")
                texts = [f"{g}\n{c}" for g, c in zip(goals, contexts or [""] * len(goals))]
            x = hash_batch(texts, self.hidden_size)
            h = self.proj(x) + self.len_proj(self.length_features(texts))
            return torch.tanh(self.norm(h))

    class LoRALinear(nn.Module):
        """B@A low-rank delta on a frozen Linear (the delegation pattern)."""

        def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad = False
            self.rank = rank
            self.scale = alpha / max(1, rank)
            self.A = nn.Parameter(torch.zeros(rank, base.in_features))
            self.B = nn.Parameter(torch.zeros(base.out_features, rank))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        def forward(self, x):
            return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)

    class Model(nn.Module):
        def __init__(self, cfg: dict):
            super().__init__()
            self.cfg = dict(cfg)
            self.encoding = cfg.get("encoding", "dual")
            if self.encoding not in ENCODINGS:
                raise ValueError(f"encoding must be one of {ENCODINGS}")
            self.buckets = int(cfg.get("buckets", DEFAULT_BUCKETS))
            self.max_steps = int(cfg.get("max_steps", DEFAULT_MAX_STEPS))
            self.decode = str(cfg.get("decode", "mode"))
            hidden = int(cfg.get("hidden", 64))
            vhidden = int(cfg.get("value_hidden", 128))
            vocab = int(cfg.get("vocab_size", 2))
            # dual = log bucket + a residual that refines inside the bucket;
            # keeping `buckets` (not buckets/2) means dual is "log + resolution",
            # never "log with fewer bins".
            self.coarse = self.buckets
            enc_for_grid = "log" if self.encoding in ("log", "dual") else (
                "uniform" if self.encoding == "uniform" else "regress")
            # The uniform baseline is bounded on purpose: 64 equal-width bins
            # over [0, uniform_span] is what the shipped head can represent, and
            # nothing above uniform_span exists for it.
            self.grid_max = int(cfg.get("uniform_span", 2048)) if self.encoding == "uniform" else self.max_steps
            edges = bucket_edges(enc_for_grid if enc_for_grid != "regress" else "log",
                                 self.coarse, self.grid_max)
            centers = bucket_centers(enc_for_grid if enc_for_grid != "regress" else "log", edges)
            self.register_buffer("edges", torch.tensor(edges, dtype=torch.float32))
            self.register_buffer("log_edges", torch.log1p(self.edges))
            self.register_buffer("centers", torch.tensor(centers, dtype=torch.float32))
            self.register_buffer("log_centers", torch.log1p(self.centers.clamp(min=0)))
            self.register_buffer("width_log", (self.log_edges[1:] - self.log_edges[:-1]))
            self.log_scale = self.encoding in ("log", "dual")
            self.register_buffer("resid_sigma", torch.tensor(
                float(cfg.get("resid_sigma", 0.35)), dtype=torch.float32))

            self.backbone = HashedBackbone(hidden)
            self.value_trunk = nn.Sequential(
                nn.Linear(hidden, vhidden), nn.GELU(),
                nn.Linear(vhidden, vhidden), nn.GELU(),
            )
            n_out = self.coarse if self.encoding != "regress" else 1
            self.value_out = nn.Linear(vhidden, n_out)
            self.value_res = nn.Linear(vhidden, 1) if self.encoding == "dual" else None
            self.policy = nn.Linear(hidden, vocab)
            self.drop = nn.Dropout(float(cfg.get("dropout", 0.0)))

        # -- heads ---------------------------------------------------------
        def forward(self, texts=None, goals=None, contexts=None):
            h = self.drop(self.backbone(texts, goals=goals, contexts=contexts))
            vt = self.value_trunk(h)
            return {
                "h": h,
                "value_logits": self.value_out(vt) if self.encoding != "regress" else None,
                "value_scalar": self.value_out(vt).squeeze(-1) if self.encoding == "regress" else None,
                "value_res": self.value_res(vt).squeeze(-1) if self.value_res is not None else None,
                "policy_logits": self.policy(h),
            }

        # -- distributions -------------------------------------------------
        def value_probs(self, out):
            if self.encoding == "regress":
                return None
            return torch.softmax(out["value_logits"], dim=-1)

        def value_point(self, out):
            """(estimate_steps, estimate_log).

            decode="mode" (default) takes a 3-bucket local mean around the
            argmax bucket.  That matters: a plain expectation over a log-spaced
            distribution is pulled toward the middle, so an uncertain head
            systematically *underestimates* the tail — which is precisely the
            failure the >500 gate cannot tolerate.  decode="expected" keeps the
            shipped d_hat behaviour for A/B.
            """
            if self.encoding == "regress":
                pred_log = out["value_scalar"]
                return torch.expm1(pred_log).clamp(min=0.0), pred_log
            probs = self.value_probs(out)
            base = self.log_centers if self.log_scale else self.centers
            decode = getattr(self, "decode", "mode")
            if decode == "expected":
                if self.log_scale:
                    est_log = (probs * base).sum(-1)
                    est_log = est_log + (torch.tanh(out["value_res"]) * 0.5
                                         * (probs * self.width_log).sum(-1) if self.encoding == "dual" else 0.0)
                    return torch.expm1(est_log).clamp(min=0.0), est_log
                est = (probs * base).sum(-1)
                return est.clamp(min=0.0), torch.log1p(est.clamp(min=0.0))
            b = probs.argmax(-1)
            nb = probs.shape[-1]
            offsets = torch.arange(-1, 2, device=probs.device)
            idx = (b.unsqueeze(-1) + offsets).clamp(0, nb - 1)
            # mirror decode_mode: bin 0 is degenerate, never blended into a
            # neighbourhood for a non-terminal mode.
            idx = torch.where((b > 0).unsqueeze(-1), idx.clamp(min=1), idx)
            w = torch.gather(probs, 1, idx)
            w = w / w.sum(-1, keepdim=True).clamp(min=1e-9)
            local = torch.gather(base.expand(probs.shape[0], -1), 1, idx)
            center = (w * local).sum(-1)
            if self.encoding == "dual":
                width = torch.gather(self.width_log.expand(probs.shape[0], -1), 1, idx)
                wmean = (w * width).sum(-1)
                est_log = center + torch.tanh(out["value_res"]) * 0.5 * wmean
            elif self.log_scale:
                est_log = center
            else:
                est = center.clamp(min=0.0)
                est = torch.where(b == 0, torch.zeros_like(est), est)
                return est, torch.log1p(est.clamp(min=0.0))
            # mode in bin 0 == terminal: decode to exactly 0.0, not to "the mean
            # of a neighbourhood that happens to contain 0".
            est_log = torch.where(b == 0, torch.zeros_like(est_log), est_log)
            return torch.expm1(est_log).clamp(min=0.0), est_log

        def interval(self, out, q=(0.1, 0.9)):
            """80% interval as (lo, hi) step tensors."""
            if self.encoding == "regress":
                pred_log = out["value_scalar"]
                z = 1.2816  # 80% two-sided normal
                return (torch.expm1(pred_log - z * self.resid_sigma).clamp(min=0.0),
                        torch.expm1(pred_log + z * self.resid_sigma).clamp(min=0.0))
            probs = self.value_probs(out)
            cdf = torch.cumsum(probs, dim=-1)
            qs = torch.tensor(q, dtype=probs.dtype, device=probs.device)
            idx = torch.searchsorted(cdf.contiguous(), qs.expand(probs.shape[0], 2).contiguous())
            idx = idx.clamp(max=probs.shape[-1] - 1)
            base = self.log_centers if self.log_scale else self.centers
            picked = torch.gather(base.expand(probs.shape[0], -1), 1, idx)
            if self.encoding == "dual":
                w = (probs * self.width_log).sum(-1, keepdim=True) * 0.5
                lo = torch.expm1(picked[:, 0] - w.squeeze(-1)).clamp(min=0.0)
                hi = torch.expm1(picked[:, 1] + w.squeeze(-1)).clamp(min=0.0)
            elif self.log_scale:
                lo, hi = torch.expm1(picked[:, 0]).clamp(min=0.0), torch.expm1(picked[:, 1]).clamp(min=0.0)
            else:
                lo, hi = picked[:, 0], picked[:, 1]
            return torch.maximum(lo, torch.zeros_like(lo)), torch.maximum(hi, lo)

        # -- losses --------------------------------------------------------
        def two_hot_batch(self, steps, edges, log_scale: bool):
            """Torch twin of `two_hot`: returns (bin_i, weight_on_bin_i).

            THE ZERO-STEP BUG lived here.  The shipped version clamped the input
            to `xs[0] + 1e-6` and then let `searchsorted` pick bin 1 with
            `weight_on_i = 0`, so a terminal record (d = 0) trained the head to
            put all its mass on **bin 1**, one bin above the truth, and every
            later decode inherited a +1-step offset.  Exact 0 (and exact grid
            top) now short-circuit to a single-bin target, which is what the
            pure `two_hot` always did.
            """
            xs = torch.log1p(edges) if log_scale else edges
            x = torch.log1p(steps.clamp(min=0.0)) if log_scale else steps.clamp(min=0.0)
            bins = xs.numel() - 1
            idx = (torch.searchsorted(xs.contiguous(), x.contiguous(), right=True) - 1)
            idx = idx.clamp(0, bins - 1)
            nxt = (idx + 1).clamp(max=bins - 1)
            lo, hi = xs[idx], xs[nxt]
            t = ((x - lo) / (hi - lo).clamp(min=1e-9)).clamp(0.0, 1.0)
            # weight on bin `idx`, same convention as the pure `two_hot`.  The
            # shipped version used `t` (the distance from `lo`) as the weight on
            # `idx`, i.e. it was inverted: a value just above a boundary put all
            # its mass on the *upper* bin.  Combined with the 1e-6 clamp that is
            # why d = 0 trained bin 1.
            frac = 1.0 - t
            # exact 0 -> bin 0 alone (terminal is a representable point, not a
            # two-hot between "nowhere" and "1 step"); grid top -> last bin.
            at_zero = x <= xs[0]
            at_top = x >= xs[-1]
            frac = torch.where(at_zero | at_top, torch.ones_like(frac), frac)
            return idx, frac

        def value_loss(self, out, steps, terminal=None, weights=None):
            if self.encoding == "regress":
                per = F.smooth_l1_loss(out["value_scalar"].flatten(),
                                       torch.log1p(steps.clamp(min=0.0)), beta=0.1, reduction="none")
            else:
                logits = out["value_logits"]
                logp = torch.log_softmax(logits.float(), dim=-1)
                i, frac = self.two_hot_batch(steps, self.edges, self.log_scale)
                j = (i + 1).clamp(max=logits.shape[-1] - 1)
                per = -(frac * logp.gather(1, i.unsqueeze(1)).squeeze(1)
                        + (1 - frac) * logp.gather(1, j.unsqueeze(1)).squeeze(1))
                if self.encoding == "dual":
                    target_log = torch.log1p(steps.clamp(min=0.0))
                    center_log = self.log_centers[i]
                    w = self.width_log[i]
                    pred_log = center_log + torch.tanh(out["value_res"]) * 0.5 * w
                    per = per + F.smooth_l1_loss(pred_log, target_log, beta=0.1, reduction="none")
            if weights is None:
                return per.mean()
            w = weights.to(per.dtype)
            return (per * w).sum() / w.sum().clamp(min=1e-9)

    def value_loss_batch(model, out, steps, terminal=None):
        return model.value_loss(out, steps, terminal)

    def policy_ce(out, targets):
        return F.cross_entropy(out["policy_logits"].float(), targets, ignore_index=-100)

    def combined_loss(model, out, batch, value_coef: float = VALUE_COEF,
                      stop_grad_value: bool = False, policy_weight: float = 1.0):
        steps = batch["steps"]
        v_out = out
        if stop_grad_value:
            # second pass with the value head seeing a detached hidden state:
            # the value gradient cannot reshape the policy features.
            h = out["h"].detach()
            vt = model.value_trunk(h)
            v_out = dict(out)
            if model.encoding == "regress":
                v_out["value_scalar"] = model.value_out(vt).squeeze(-1)
            else:
                v_out["value_logits"] = model.value_out(vt)
                if model.value_res is not None:
                    v_out["value_res"] = model.value_res(vt).squeeze(-1)
        vloss = model.value_loss(v_out, steps, weights=batch.get("weights"))
        if "policy_targets" in batch and bool((batch["policy_targets"] >= 0).any()):
            ploss = policy_ce(out, batch["policy_targets"])
        else:
            ploss = torch.zeros((), dtype=vloss.dtype, device=vloss.device)
        total = value_coef * vloss + policy_weight * ploss
        return total, {"value_loss": float(vloss.detach()), "policy_loss": float(ploss.detach()),
                       "total": float(total.detach())}

    def grad_share(model, out, batch, value_coef: float = VALUE_COEF,
                   policy_weight: float = 1.0) -> dict:
        """Per-tensor value-vs-policy gradient share on the SHARED parameters.

        This is the measurement DESIGN §9.1 asks for: it recomputes both loss
        terms from one forward pass and backprops each *alone*, so the share is
        the gradient budget the value objective would take on the trunk if the
        policy objective were absent.  `share = |g_v|^2 / (|g_v|^2 + |g_p|^2)`
        per tensor, plus the aggregate that stage 1 logs.

        The head's own parameters (`value_*`) are excluded: they are not shared,
        so their gradient is not interference by definition.
        """
        shared = [(n, p) for n, p in model.named_parameters()
                  if not n.startswith("value_")]
        model.zero_grad(set_to_none=True)
        vloss = model.value_loss(out, batch["steps"], weights=batch.get("weights"))
        (value_coef * vloss).backward(retain_graph=True)
        gv = {n: (float(p.grad.detach().norm()) if p.grad is not None else 0.0)
              for n, p in shared}
        model.zero_grad(set_to_none=True)
        if "policy_targets" in batch and bool((batch["policy_targets"] >= 0).any()):
            ploss = policy_ce(out, batch["policy_targets"])
            (policy_weight * ploss).backward()
        gp = {n: (float(p.grad.detach().norm()) if p.grad is not None else 0.0)
              for n, p in shared}
        model.zero_grad(set_to_none=True)
        rows, sv, sp = [], 0.0, 0.0
        for n, _ in shared:
            a, b = gv[n] ** 2, gp[n] ** 2
            sv, sp = sv + a, sp + b
            rows.append({"param": n, "g_value": round(gv[n], 6),
                         "g_policy": round(gp[n], 6),
                         "share": round(a / (a + b), 6) if (a + b) > 0 else 0.0})
        total = sv / (sv + sp) if (sv + sp) > 0 else 0.0
        return {"share_total": round(total, 8), "g_value": round(sv ** 0.5, 6),
                "g_policy": round(sp ** 0.5, 6), "per_param": rows,
                "lambda_eff": round(value_coef, 8)}

    def attach_lora(model, rank: int = 8):
        """Scheme A: freeze the base projection, adapt with B@A.

        For the hashed stand-in the "base" is `backbone.proj`; LayerNorm and the
        length-feature projection stay trainable because they are head-side
        features, not base weights (a real model gets that capacity from the
        adapter's own bias/norm modules).
        """
        bb = model.backbone
        if isinstance(bb.proj, LoRALinear):
            return [p for p in bb.parameters() if p.requires_grad]
        bb.proj = LoRALinear(bb.proj, rank=rank)
        return [p for p in bb.parameters() if p.requires_grad]

    def attach_lora_hf(model, rank: int = 8, targets: list[str] | None = None):
        """Scheme A against a real HF model (skeleton, needs peft)."""
        try:
            from peft import LoraConfig, get_peft_model  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise TorchUnavailable(
                "scheme=lora against a HF base needs `peft` "
                f"({type(exc).__name__}: {exc}). `--scheme full` needs neither peft nor bitsandbytes."
            )
        targets = targets or [
            # every projection the shipped recipe needs, incl. Qwen3.5 GDN
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
        ]
        cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                         target_modules=targets, bias="none", task_type="CAUSAL_LM")
        return get_peft_model(model, cfg)

    def save_checkpoint(model, run_dir: str, name: str = "value-head.pt"):
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, name)
        torch.save({"cfg": model.cfg, "state": model.state_dict()}, path)
        return path

    def load_checkpoint(run_dir: str, name: str = "value-head.pt", backbone=None):
        path = os.path.join(run_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"checkpoint not found: {path}")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        bar = blob["cfg"]
        model = Model(bar)
        if backbone is not None:
            model.backbone = backbone
        if bar.get("lora_rank") and isinstance(model.backbone, HashedBackbone):
            attach_lora(model, rank=int(bar["lora_rank"]))
        model.load_state_dict(blob["state"])
        model.eval()
        return model

    def predict_texts(model, texts=None, batch_size: int = 256,
                      goals: list[str] | None = None,
                      contexts: list[str] | None = None) -> list[dict]:
        model.eval()
        n = len(texts) if texts is not None else len(goals or [])
        out_rows: list[dict] = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                chunk = texts[start:start + batch_size] if texts is not None else None
                g = goals[start:start + batch_size] if goals is not None else None
                c = contexts[start:start + batch_size] if contexts is not None else None
                out = model(chunk, goals=g, contexts=c)
                rows_n = len(chunk) if chunk is not None else len(g)
                est, est_log = model.value_point(out)
                lo, hi = model.interval(out)
                if model.encoding == "regress":
                    modes = [-1] * rows_n
                    # no distribution: fall back to the value test
                    tails = (est >= 0.95 * model.grid_max).tolist()
                else:
                    modes = model.value_probs(out).argmax(-1).tolist()
                    # The tail flag is a *bucket* statement, not a value
                    # threshold: "the mode is in the last bucket" means the head
                    # put its mass at or above the last edge, so the point
                    # estimate is a lower bound.  See DESIGN §11a.
                    tails = [tail_flag(int(m), model.coarse) for m in modes]
                for k in range(rows_n):
                    out_rows.append({
                        "estimate": float(est[k]),
                        "estimate_log": float(est_log[k]),
                        "ci_low": float(lo[k]),
                        "ci_high": float(hi[k]),
                        "mode_bucket": int(modes[k]),
                        "tail_saturated": bool(tails[k]),
                    })
        return out_rows

    def count_params(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"total": int(total), "trainable": int(trainable)}

    def set_trainable(model, *, backbone: bool, heads: bool = True):
        for p in model.backbone.parameters():
            p.requires_grad = backbone
        for name in ("value_trunk", "value_out", "value_res", "policy"):
            mod = getattr(model, name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = heads

    return {
        "Model": Model, "HashedBackbone": HashedBackbone, "LoRALinear": LoRALinear,
        "nn": nn, "F": F, "torch": torch,
        "combined_loss": combined_loss, "policy_ce": policy_ce,
        "attach_lora": attach_lora, "attach_lora_hf": attach_lora_hf,
        "save_checkpoint": save_checkpoint, "load_checkpoint": load_checkpoint,
        "predict_texts": predict_texts, "count_params": count_params,
        "set_trainable": set_trainable, "hash_batch": hash_batch,
        "grad_share": grad_share,
        "value_loss_batch": value_loss_batch,
    }


# ---------------------------------------------------------------------------
# Gate re-export (kept in one place so JS and python cannot drift)
# ---------------------------------------------------------------------------

from gate import DEFAULT_THRESHOLD, HYSTERESIS, decide_gate, tail_flag  # noqa: E402


def gate(estimate: float, threshold: float = DEFAULT_THRESHOLD, prev: str = "prove",
         ci_high: float | None = None, tail_saturated: bool = False) -> dict:
    return decide_gate(estimate, threshold=threshold, prev=prev,
                       ci_high=ci_high, tail_saturated=tail_saturated)


def _probe(seed: int = 7, n: int = 240, warm: int = 40, burst: int = 20) -> dict:
    """Measure the six interference modes DESIGN §9 quantifies (CPU, ~5 s).

    Everything here is a *measurement*, not a model:

    M1 grad share       per-tensor |g_v|^2/(|g_v|^2+|g_p|^2) on shared params
    M2 h-drift          policy-only steps, then ||h-h0||/||h0|| and the induced
                        shift in log1p d_hat (the "head went stale" number)
    M3 adam coupling    cosine between the value-step and policy-step update
                        directions on the shared tensors
    M4 lambda_eff       |g_value| at N_v = 16 vs N_v = 4 in one mixed batch
    M5 terminal bucket  decode(bin 0) and encode(0) before/after the fix
    M6 ema              value loss of raw vs EMA weights after the burst
    """
    torch = _TORCH
    ns = ensure()
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rows = synthetic_corpus(n=n, seed=seed)
    cut = int(n * 0.7)
    train, test = rows[:cut], rows[cut:]
    cfg = {"encoding": "dual", "buckets": DEFAULT_BUCKETS, "max_steps": DEFAULT_MAX_STEPS,
           "hidden": 48, "value_hidden": 96, "vocab_size": 2, "decode": "mode"}
    model = ns["Model"](cfg)
    ns["set_trainable"](model, backbone=True, heads=True)

    def batch(rs, policy: bool = True):
        return {"texts": [r["text"] for r in rs],
                "steps": torch.tensor([r["remaining_steps"] for r in rs], dtype=torch.float32),
                "policy_targets": torch.tensor(
                    [1 if r["next_action"] == "decompose" else 0 for r in rs]
                    if policy else [-100] * len(rs), dtype=torch.long)}

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    m1 = None
    for step in range(warm):
        rs = train[(step * 16) % (len(train) - 16):][:16]
        out = model([r["text"] for r in rs])
        if step == warm - 1:
            # separate forward: grad_share backprops twice and therefore owns
            # (and frees) its own graph.
            out_g = model([r["text"] for r in rs])
            m1 = ns["grad_share"](model, out_g, batch(rs), value_coef=VALUE_COEF)
        total, _ = ns["combined_loss"](model, out, batch(rs), value_coef=VALUE_COEF)
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()

    # M4: the same value loss over 16 vs 4 value rows.
    rs16 = train[:16]
    b16 = batch(rs16)
    out16 = model([r["text"] for r in rs16])
    g16 = ns["grad_share"](model, out16, b16, value_coef=VALUE_COEF)

    def value_grad_norm(b):
        model.zero_grad(set_to_none=True)
        out = model(b["texts"])
        model.value_loss(out, b["steps"]).backward()
        g = sum(float(p.grad.detach().norm()) ** 2 for p in model.parameters()
                if p.grad is not None) ** 0.5
        model.zero_grad(set_to_none=True)
        return g

    g_16 = value_grad_norm(b16)
    g_4 = value_grad_norm(batch(train[:4]))

    # snapshot before the policy burst
    probe_texts = [r["text"] for r in test]
    trues = [r["remaining_steps"] for r in test]
    before = ns["predict_texts"](model, probe_texts)
    h0 = model.backbone(probe_texts).detach().clone()
    vloss_before = value_grad_norm_quiet(ns, model, probe_texts, trues)

    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    beta = 0.99
    for step in range(burst):
        rs = train[(step * 11) % (len(train) - 16):][:16]
        out = model([r["text"] for r in rs])
        # policy-only: no value term at all, i.e. the worst case for the head
        _, parts = ns["combined_loss"](model, out, batch(rs, policy=True), value_coef=0.0)
        opt.zero_grad(set_to_none=True)
        parts_total = ns["policy_ce"](out, batch(rs)["policy_targets"])
        parts_total.backward()
        opt.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(beta).add_(v.detach(), alpha=1 - beta)
                else:
                    ema[k] = v.detach().clone()

    h1 = model.backbone(probe_texts).detach()
    after = ns["predict_texts"](model, probe_texts)
    vloss_after = value_grad_norm_quiet(ns, model, probe_texts, trues)
    raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema)
    vloss_ema = value_grad_norm_quiet(ns, model, probe_texts, trues)
    after_ema = ns["predict_texts"](model, probe_texts)
    model.load_state_dict(raw)

    def drift(rows_a, rows_b):
        d = [abs(math.log1p(a["estimate"]) - math.log1p(b["estimate"]))
             for a, b in zip(rows_a, rows_b)]
        flips = sum(1 for a, b in zip(rows_a, rows_b)
                    if (a["estimate"] >= DEFAULT_THRESHOLD) != (b["estimate"] >= DEFAULT_THRESHOLD))
        return {"median_dlog": round(_median(d), 4), "max_dlog": round(max(d), 4) if d else 0.0,
                "gate_flips": flips, "n": len(d)}

    # M3: cosine between the two update directions on shared tensors.
    shared = [(k, p) for k, p in model.named_parameters() if not k.startswith("value_")]
    opt.zero_grad(set_to_none=True)
    out = model(b16["texts"])
    model.value_loss(out, b16["steps"]).backward(retain_graph=True)
    gv = [p.grad.detach().flatten().clone() if p.grad is not None else torch.zeros_like(p).flatten()
          for _, p in shared]
    opt.zero_grad(set_to_none=True)
    ns["policy_ce"](out, b16["policy_targets"]).backward()
    gp = [p.grad.detach().flatten().clone() if p.grad is not None else torch.zeros_like(p).flatten()
          for _, p in shared]
    opt.zero_grad(set_to_none=True)
    a = torch.cat(gv)
    b = torch.cat(gp)
    cos = float((a @ b) / (a.norm() * b.norm() + 1e-12))

    # M3b: the real coupling — two AdamW steps taken from the SAME optimizer
    # state.  Adam's moments are per-tensor, so a value step after a policy step
    # moves the shared tensors in a direction biased by the policy's second
    # moment.  cos ~ 1 means "they fight/agree in the same subspace"; cos ~ 0
    # means the two objectives are locally orthogonal on the trunk.
    def snap():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    def flat_delta(after, before):
        return torch.cat([(after[k] - before[k]).flatten() for k, p in shared])

    s0 = snap()
    opt.zero_grad(set_to_none=True)
    o = model(b16["texts"])
    ns["policy_ce"](o, b16["policy_targets"]).backward()
    opt.step()
    s1 = snap()
    opt.zero_grad(set_to_none=True)
    o = model(b16["texts"])
    model.value_loss(o, b16["steps"]).backward()
    opt.step()
    s2 = snap()
    dp, dv = flat_delta(s1, s0), flat_delta(s2, s1)
    cos_adam = float((dp @ dv) / (dp.norm() * dv.norm() + 1e-12))

    return {
        "seed": seed, "warm_steps": warm, "burst_steps": burst, "n": n,
        "M1_grad_share": {
            "total": m1["share_total"], "g_value": m1["g_value"], "g_policy": m1["g_policy"],
            "per_param": [r for r in m1["per_param"] if r["g_value"] or r["g_policy"]],
        },
        "M2_h_drift": {
            "rel_h_norm": round(float((h1 - h0).norm() / (h0.norm() + 1e-12)), 4),
            "value_loss_before": round(vloss_before, 4),
            "value_loss_after": round(vloss_after, 4),
            "dhat": drift(after, before),
            "gate_flip_rate": round(drift(after, before)["gate_flips"] / max(1, len(test)), 4),
        },
        "M3_grad_cos": round(cos, 4),
        "M3b_adam_step_cos": round(cos_adam, 4),
        "M3b_adam_step_norms": {"policy_step": round(float(dp.norm()), 6),
                                "value_step": round(float(dv.norm()), 6),
                                "ratio": round(float(dv.norm() / (dp.norm() + 1e-12)), 4)},
        "M4_lambda_eff": {
            "g_value_n16": round(g_16, 4), "g_value_n4": round(g_4, 4),
            "ratio": round(g_4 / max(1e-9, g_16), 3),
            "lambda_eff_at_nv4_of_n16": round(VALUE_COEF * 16 / 4, 6),
        },
        "M5_terminal": {"encode_0": encode(0), "decode_0": decode(0),
                        "encode_0_uniform": encode(0, "uniform"), "decode_0_uniform": decode(0, "uniform")},
        "M6_ema": {
            "value_loss_raw": round(vloss_after, 4),
            "value_loss_ema": round(vloss_ema, 4),
            "dhat_raw": drift(after, before),
            "dhat_ema": drift(after_ema, before),
        },
    }


def _probe_recipe(seed: int = 7, n: int = 240, steps: int = 90, burst: int = 10) -> dict:
    """Measure the two recipe knobs full FT actually has: capacity and LR ratio.

    Mode 6 (capacity placement) and mode 7 (LR ratio) from DESIGN §9.  Every
    variant sees the same data, the same seed, the same number of optimizer
    steps; only the knob moves.  `drift` is the median |Δ log1p d̂| after a
    10-step policy-only burst — the mode-2 damage, i.e. how much of the head's
    accuracy survives a policy update.
    """
    torch = _TORCH
    ns = ensure()
    rows = synthetic_corpus(n=n, seed=seed)
    cut = int(n * 0.7)
    train, test = rows[:cut], rows[cut:]
    probe_texts = [r["text"] for r in test]
    trues = [r["remaining_steps"] for r in test]
    from gate import DEFAULT_THRESHOLD as THR

    def run(value_hidden: int, ratio: float, policy_lr: float = 3e-3) -> dict:
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        cfg = {"encoding": "dual", "buckets": DEFAULT_BUCKETS, "max_steps": DEFAULT_MAX_STEPS,
               "hidden": 48, "value_hidden": value_hidden, "vocab_size": 2, "decode": "mode"}
        model = ns["Model"](cfg)
        ns["set_trainable"](model, backbone=True, heads=True)
        vparams = [p for k, p in model.named_parameters() if k.startswith("value_")]
        oparams = [p for k, p in model.named_parameters() if not k.startswith("value_")]
        opt = torch.optim.AdamW([{"params": vparams, "lr": policy_lr * ratio},
                                 {"params": oparams, "lr": policy_lr}], weight_decay=0.0)
        for step in range(steps):
            rs = train[(step * 13) % (len(train) - 16):][:16]
            b = {"texts": [r["text"] for r in rs],
                 "steps": torch.tensor([r["remaining_steps"] for r in rs], dtype=torch.float32),
                 "policy_targets": torch.tensor(
                     [1 if r["next_action"] == "decompose" else 0 for r in rs], dtype=torch.long)}
            out = model(b["texts"])
            total, _ = ns["combined_loss"](model, out, b, value_coef=VALUE_COEF)
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
        preds = ns["predict_texts"](model, probe_texts)
        m = metrics([p["estimate"] for p in preds], trues,
                    centers=[float(c) for c in model.centers.tolist()])
        model.eval()
        with torch.no_grad():
            logits = model(probe_texts)["policy_logits"]
        acc = float((logits.argmax(-1) == torch.tensor(
            [1 if r["next_action"] == "decompose" else 0 for r in test])).float().mean())
        before = preds
        for step in range(burst):
            rs = train[(step * 17) % (len(train) - 16):][:16]
            b = {"texts": [r["text"] for r in rs],
                 "policy_targets": torch.tensor(
                     [1 if r["next_action"] == "decompose" else 0 for r in rs], dtype=torch.long)}
            out = model(b["texts"])
            loss = ns["policy_ce"](out, b["policy_targets"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        after = ns["predict_texts"](model, probe_texts)
        d = sorted(abs(math.log1p(a["estimate"]) - math.log1p(b0["estimate"]))
                   for a, b0 in zip(after, before))
        flips = sum(1 for a, b0 in zip(after, before)
                    if (a["estimate"] >= THR) != (b0["estimate"] >= THR))
        return {"value_hidden": value_hidden, "lr_ratio": ratio,
                "mae_log": m["mae_log"], "mae_steps": m["mae_steps"],
                "zero_exact": m["zero"]["exact"], "bucket_acc": m.get("bucket_acc"),
                "policy_acc": round(acc, 4),
                "drift_median_dlog": round(_median(d), 4),
                "gate_flips": flips, "params": int(sum(p.numel() for p in model.parameters()))}

    out = []
    for vh in (32, 96, 256):
        out.append(run(vh, 1.0))
    for ratio in (0.1, 3.0):
        out.append(run(96, ratio))
    return {"seed": seed, "steps": steps, "burst": burst, "grid": out}


def value_grad_norm_quiet(ns, model, texts, trues) -> float:
    """Value loss (not gradient) on a fixed set, used only for before/after."""
    torch = _TORCH
    model.eval()
    with torch.no_grad():
        out = model(texts)
        loss = model.value_loss(out, torch.tensor(trues, dtype=torch.float32))
    model.train()
    return float(loss.detach())


if __name__ == "__main__":  # tiny self-check, no torch required
    if "--probe" in sys.argv:
        print(json.dumps(_probe(), indent=1))
        raise SystemExit(0)
    if "--probe-recipe" in sys.argv:
        print(json.dumps(_probe_recipe(), indent=1))
        raise SystemExit(0)
    edges = bucket_edges("log")
    print(json.dumps({
        "torch": torch_available(),
        "torch_error": torch_error(),
        "buckets": len(edges) - 1,
        "edges_head": [round(e, 2) for e in edges[:4]],
        "encode_0": encode(0), "decode_0": decode(0),
        "encode_5": encode(5), "decode_last": round(decode(63), 1),
        "two_hot_1234": two_hot(1234, edges, True),
        "gate_900": gate(900),
    }))
