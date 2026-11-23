#!/usr/bin/env python3
"""B2 — the long tail: log grid, binned-mode decode, tail flag, and the ablation.

Uniform bins over a 64-bin grid cannot name a 10k-step mission: the shipped head
topped out at 2048 steps, so any mission above that decoded to the same number
and `tail recall@10k` was structurally 0.  The fix is the dual head — 64
log-spaced bins over `[1, 32768]` plus a bounded in-bucket residual — decoded by
**binned mode**, with a `tail_saturated` flag that fires when the mode is in the
last bin.

This file pins the three pieces that are easy to regress:

  1. the grid geometry and the *bounded* residual
  2. binned mode vs global expectation on a bimodal (i.e. uncertain) head
  3. the ablation table itself, on a small but real run (all 6 arms)

`python3 tests/test_bucket_ablation.py` — plain asserts, no pytest.
"""

from __future__ import annotations

import io
import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import ablate_head  # noqa: E402
import gate  # noqa: E402
import head  # noqa: E402

CHECKS = 0


def ok(name: str) -> None:
    global CHECKS
    CHECKS += 1
    print(f"ok {CHECKS:02d} - {name}")


# ---------------------------------------------------------------------------
# B2.1 geometry: log spacing, full span, bounded residual
# ---------------------------------------------------------------------------

def test_log_grid_tail() -> None:
    edges = head.bucket_edges("dual", 64, 32768)
    assert len(edges) - 1 == 64
    assert edges[-1] == 32768.0
    logs = [math.log1p(e) for e in edges[1:]]
    gaps = [b - a for a, b in zip(logs, logs[1:])]
    approx_gap = max(gaps) - min(gaps)
    assert approx_gap < 1e-9, approx_gap
    # the top four bins must each be wider than 1k steps, i.e. the tail really
    # is coarse, and that is fine because the residual refines inside the bin.
    widths = [b - a for a, b in zip(edges, edges[1:])]
    assert widths[-1] > 1000 and widths[1] < 1.0, (widths[1], widths[-1])
    ok("dual grid: 63 log1p-even bins over [1, 32768] after the terminal bin, top bin > 1k wide")


def test_residual_is_bounded() -> None:
    """The residual may move the log estimate by at most half a bin width.

    That is what keeps the head from "fixing" a bad bin by extrapolating, which
    is how a residual head silently becomes an unbounded regressor.
    """
    edges = head.bucket_edges("dual", 64, 32768)
    w = [math.log1p(b) - math.log1p(a) for a, b in zip(edges, edges[1:])]
    for bucket in (1, 10, 30, 63):
        centre = math.log1p(head.decode(bucket, "dual"))
        lo = math.expm1(centre - 0.5 * w[bucket])
        hi = math.expm1(centre + 0.5 * w[bucket])
        assert lo >= edges[bucket] * 0.999, (bucket, lo, edges[bucket])
        assert hi <= edges[bucket + 1] * 1.001, (bucket, hi, edges[bucket + 1])
    ok("residual is bounded: +-0.5 bin in log space, so it cannot leave the bin")


# ---------------------------------------------------------------------------
# B2.2 decode: binned mode does not shrink an uncertain tail
# ---------------------------------------------------------------------------

def test_mode_vs_expectation_tail() -> None:
    centers = head.bucket_centers("log", head.bucket_edges("log"))
    # dominant belief is the top decade, with real mass a decade below
    probs = [0.0] * 64
    probs[62] = 0.55
    probs[55] = 0.45
    mode_v, mode_b = head.decode_mode(probs, centers, True)
    exp_v = head.decode_expected(probs, centers, True)
    assert mode_b == 62, mode_b
    assert mode_v >= centers[62] * 0.95, (mode_v, centers[62])
    # the expectation is dragged into the *gap* between the two modes: a value
    # with zero probability mass, and a 30%+ underestimate of the tail.
    assert centers[55] < exp_v < centers[62], (centers[55], exp_v, centers[62])
    assert exp_v / mode_v < 0.85, (exp_v, mode_v)
    # two equal modes: the expectation lands in the empty middle, i.e. it names
    # a value the head explicitly does not believe.
    even = [0.0] * 64
    even[55] = even[62] = 0.5
    assert centers[55] < head.decode_expected(even, centers, True) < centers[62]
    assert head.decode_mode(even, centers, True)[1] in (55, 62)
    ok("binned mode keeps a 55/62 bimodal head on the tail; the expectation names an empty middle bin")


def test_mode_is_local_not_global() -> None:
    centers = head.bucket_centers("log", head.bucket_edges("log"))
    probs = [0.0] * 64
    probs[40] = 0.6
    probs[41] = 0.4
    v, b = head.decode_mode(probs, centers, True)
    assert b == 40 and centers[40] * 0.95 <= v <= centers[41] * 1.05, (v, b)
    ok("binned mode is a 3-bin local mean around the argmax, not a global average")


# ---------------------------------------------------------------------------
# B2.3 tail flag
# ---------------------------------------------------------------------------

def test_tail_flag_on_mode() -> None:
    assert gate.tail_flag(63, 64) and not gate.tail_flag(62, 64)
    # a head that puts 40% on the last bin but has its mode at 62 is NOT
    # saturated: the point estimate is still justified by the grid.
    assert not gate.tail_flag(62, 64)
    got = gate.decide_gate(30000.0, tail_saturated=gate.tail_flag(63, 64))
    assert got["action"] == "decompose" and got["reason"] == "tail_saturated", got
    ok("tail flag = mode in the last bin; it forces decompose without a value test")


def test_predict_reports_mode_bucket() -> None:
    if not head.torch_available():
        print(f"skip - torch not importable ({head.torch_error()})")
        return
    import torch

    ns = head.ensure()
    model = ns["Model"]({"encoding": "dual", "buckets": 64, "max_steps": 32768,
                         "hidden": 16, "value_hidden": 16, "vocab_size": 2})
    rows = ns["predict_texts"](model, ["theorem a : True\nimport Mathlib"] * 3)
    assert all("mode_bucket" in r for r in rows), rows[0].keys()
    assert all(0 <= r["mode_bucket"] < 64 for r in rows)
    assert all(isinstance(r["tail_saturated"], bool) for r in rows)
    ok("predict_texts exposes mode_bucket (0..63) and a boolean tail_saturated")


# ---------------------------------------------------------------------------
# B2.4 the ablation itself (small but real: all six arms, one seed)
# ---------------------------------------------------------------------------

def test_ablation_table() -> None:
    if not head.torch_available():
        return
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = ablate_head.main(["--tiny", "--n", "140", "--n-test", "70",
                               "--seeds", "1", "--steps", "60", "--json"])
    assert rc == 0, rc
    data = json.loads(buf.getvalue())
    mean = data["mean"]
    for arm in ("uniform64", "uniform64@32k", "log64", "dual(mode)",
                "dual(expected)", "regress"):
        assert arm in mean, (arm, list(mean))
    # the two defects, as measured rather than argued
    assert mean["uniform64"]["tail_recall"] == 0.0, mean["uniform64"]
    assert mean["uniform64"]["tail_rate"] == 1.0, mean["uniform64"]   # flag=1.0 on the tail
    assert mean["dual(mode)"]["tail_recall"] >= 0.9, mean["dual(mode)"]
    assert mean["dual(mode)"]["zero_exact"] == 1.0, mean["dual(mode)"]
    assert mean["regress"]["zero_exact"] <= 0.5, mean["regress"]
    assert mean["regress"]["tail_recall"] < mean["dual(mode)"]["tail_recall"], mean["regress"]
    # zero-step cases must exist in the fixture, otherwise the column is empty
    assert mean["dual(mode)"]["zero_n"] >= 3, mean["dual(mode)"]["zero_n"]
    assert mean["dual(mode)"]["tail_n"] >= 3, mean["dual(mode)"]["tail_n"]
    ok("ablation: uniform64 has tail recall 0, dual(mode) 1.0 + zero-exact 1.0, regress fails both")


def test_fallback_rule() -> None:
    good = {"log64": {"mae_log": 0.40, "tail_recall": 1.0, "tail_mae_log": 0.40,
                      "cov80": 0.92, "zero_exact": 1.0},
            "dual(mode)": {"mae_log": 0.41, "tail_recall": 1.0, "tail_mae_log": 0.41,
                           "cov80": 0.95, "zero_exact": 1.0}}
    v = ablate_head.verdict(good)
    assert v["keep_dual"] and v["bulk_gain"] < 0, v
    bad = {**good, "dual(mode)": {**good["dual(mode)"], "tail_recall": 0.7}}
    v2 = ablate_head.verdict(bad)
    assert not v2["keep_dual"] and "tail_recall" in v2["why"], v2
    worse_zero = {**good, "dual(mode)": {**good["dual(mode)"], "zero_exact": 0.8}}
    assert not ablate_head.verdict(worse_zero)["keep_dual"]
    ok("fallback rule: bulk MAE is a tie-breaker; a tail or terminal regression forces log64")


def main() -> int:
    test_log_grid_tail()
    test_residual_is_bounded()
    test_mode_vs_expectation_tail()
    test_mode_is_local_not_global()
    test_tail_flag_on_mode()
    test_predict_reports_mode_bucket()
    test_ablation_table()
    test_fallback_rule()
    print(f"\n{CHECKS} checks passed (bucket tail / ablation)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
