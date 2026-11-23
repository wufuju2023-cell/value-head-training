#!/usr/bin/env python3
"""The >500-step decomposition gate (single source of truth, mirrored in JS).

A value estimate above the threshold means "do not attempt the full proof;
decompose into missions".  Two things make a naive `estimate > 500` unusable:

* hysteresis: an estimate that hovers around 500 would flip the agent between
  "prove" and "decompose" every turn, and each flip restarts planning.  We
  enter decompose at `threshold` and only leave at `HYSTERESIS * threshold`.
* tail: with a 32k log grid the head can saturate.  A saturated estimate is
  not evidence that the problem is *only* 500 steps; it must decompose.

The JS twin is `lib/gate.mjs`; `tests/gate_cases.json` holds the shared table
that both test suites assert against, so the two cannot drift silently.
"""

from __future__ import annotations

import math

DEFAULT_THRESHOLD = 500.0
HYSTERESIS = 0.8          # leave decompose below 400 when threshold=500
FAR_TAIL = 4.0            # ci_high >= 4x threshold => uncertain, decompose
MAX_CHILDREN = 8
MIN_CHILDREN = 2
TAIL_MODE_BUCKETS = 1     # the tail flag fires when the mode is in the last N bins


def tail_flag(mode_bucket: int, buckets: int, window: int = TAIL_MODE_BUCKETS) -> bool:
    """`tail_saturated` for a bucket head: the mode sits in the last `window` bins.

    This replaces the old `estimate >= 0.95 * grid_max` test.  That test never
    fired for a *correctly* saturated head, because the last bin's centre is
    ~30339 on the shipped grid, i.e. below `0.95 * 32768 = 31130`: a head that
    had run out of grid reported `tail_saturated: false` and the gate happily
    blocked on a point estimate the grid could not justify.  The mode bucket is
    the honest statement: "the mass is at or above the last edge, so the point
    estimate is a lower bound".

    A bucket head therefore saturates in the safe direction (`decompose`), and
    only when it really is out of resolution.
    """
    if buckets <= 0 or mode_bucket < 0:
        return False

    return mode_bucket >= max(0, buckets - max(1, window))


def suggested_children(estimate: float, threshold: float) -> int:
    if threshold <= 0:
        return MIN_CHILDREN
    raw = int(math.ceil(estimate / threshold))
    return max(MIN_CHILDREN, min(MAX_CHILDREN, raw))


def decide_gate(estimate: float, *, threshold: float = DEFAULT_THRESHOLD,
                prev: str = "prove", ci_high: float | None = None,
                tail_saturated: bool = False) -> dict:
    threshold = float(threshold) if threshold and threshold > 0 else DEFAULT_THRESHOLD
    enter = threshold
    exit_ = HYSTERESIS * threshold
    prev = prev if prev in ("prove", "decompose") else "prove"
    est = float(estimate)

    def out(action: str, reason: str) -> dict:
        return {
            "action": action,
            "reason": reason,
            "estimate": round(est, 3),
            "threshold": threshold,
            "enter": enter,
            "exit": round(exit_, 3),
            "prev": prev,
            "tail_saturated": bool(tail_saturated),
            "suggested_children": suggested_children(est, threshold) if action == "decompose" else 0,
        }

    if not math.isfinite(est) or est < 0:
        return out("prove", "no_estimate")
    if tail_saturated:
        return out("decompose", "tail_saturated")
    if prev == "decompose":
        if est <= exit_:
            return out("prove", "hysteresis_exit")
        return out("decompose", "hysteresis_hold")
    if est >= enter:
        return out("decompose", "above_threshold")
    if ci_high is not None and ci_high >= FAR_TAIL * enter:
        return out("decompose", "ci_upper_far_above")
    return out("prove", "below_threshold")


if __name__ == "__main__":
    import json
    import sys
    est = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    print(json.dumps(decide_gate(est)))
