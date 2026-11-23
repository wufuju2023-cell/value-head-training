#!/usr/bin/env python3
"""B1 — the zero-step (terminal) bucket contract.

The head's target is an integer remaining step count, so the state `d = 0` is a
first-class value, not a rounding artifact.  Three things must hold and are
pinned here:

  1. `encode(0) == 0`                     the terminal state lands in bin 0
  2. `decode(0) == 0.0`                    bin 0's representative value is 0
  3. the torch training target for d = 0   is bin 0 with weight 1 (not bin 1)

(3) is the actual bug that shipped: `two_hot_batch` clamped the input to
`xs[0] + 1e-6`, `searchsorted` then returned bin 1 with interpolation weight 0,
and the head learned to put all its terminal mass one bin above the truth — so
every later `d_hat` carried a +1-step floor and a "solved" mission looked like
it had one step of work left.

`python3 tests/test_head_geometry.py` — plain asserts, no pytest.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import gate  # noqa: E402
import head  # noqa: E402

CHECKS = 0
ENCODINGS = ("uniform", "log", "dual")


def ok(name: str) -> None:
    global CHECKS
    CHECKS += 1
    print(f"ok {CHECKS:02d} - {name}")


def approx(a: float, b: float, tol: float = 1e-9) -> None:
    assert abs(a - b) <= tol, f"{a} != {b} (tol {tol})"


# ---------------------------------------------------------------------------
# B1.1 bin 0 is the degenerate terminal bin, in every encoding
# ---------------------------------------------------------------------------

def test_bin0_is_terminal() -> None:
    for enc in ENCODINGS:
        edges = head.bucket_edges(enc, 64, 32768)
        assert len(edges) == 65, (enc, len(edges))
        approx(edges[0], 0.0)
        approx(edges[1], 1.0)          # [0, 1): the only integer inside is 0
        assert all(b > a for a, b in zip(edges, edges[1:])), enc
        assert head.encode(0, enc) == 0, enc
        approx(head.decode(0, enc), 0.0)
    ok("bin 0 is [0,1) with centre exactly 0 in uniform64 / log64 / dual")


def test_encode_zero_and_monotone() -> None:
    for enc in ENCODINGS:
        assert head.encode(0, enc) == 0, enc
        approx(head.decode(head.encode(0, enc), enc), 0.0)
        # monotone across the small-value range where the bug was visible
        buckets = [head.encode(d, enc) for d in range(0, 6)]
        assert buckets[0] == 0, (enc, buckets)
        assert all(b >= a for a, b in zip(buckets, buckets[1:])), (enc, buckets)
        assert buckets[-1] >= buckets[0], (enc, buckets)
        # and monotone over four orders of magnitude
        wide = [head.encode(v, enc) for v in (0, 1, 2, 5, 17, 99, 501, 5000, 20000, 32768)]
        assert all(b >= a for a, b in zip(wide, wide[1:])), (enc, wide)
        assert wide[-1] == 63, (enc, wide)
    ok("encode(0)==0, decode(0)==0.0 and encode is monotone over 0..5 and 0..32768")


def test_log_grid_shape() -> None:
    edges = head.bucket_edges("log", 64, 32768)
    logs = [math.log1p(e) for e in edges[1:]]
    gaps = [b - a for a, b in zip(logs, logs[1:])]
    approx(max(gaps), min(gaps), 1e-9)          # uniform in log1p space
    approx(edges[-1], 32768.0)
    assert edges[1] == 1.0
    ok("log64 covers [1, 32768] in 63 even log1p steps, with [0,1) reserved")


def test_decode_mode_terminal() -> None:
    centers = head.bucket_centers("log", head.bucket_edges("log"))
    one_hot = [0.0] * 64
    one_hot[0] = 1.0
    value, b = head.decode_mode(one_hot, centers, True)
    approx(value, 0.0)
    assert b == 0
    # a neighbouring mass must not drag a terminal mode off zero
    blur = [0.7, 0.3] + [0.0] * 62
    value2, b2 = head.decode_mode(blur, centers, True)
    assert b2 == 0, b2
    approx(value2, 0.0)
    # and bin 0 never leaks into a non-terminal local mean
    near = [0.4] + [0.6] + [0.0] * 62
    value3, b3 = head.decode_mode(near, centers, True)
    assert b3 == 1 and value3 >= centers[1] * 0.9, (value3, centers[1])
    ok("decode_mode: a terminal mode decodes to exactly 0 and bin 0 never biases 1-step modes")


def test_expectation_vs_mode_on_terminal() -> None:
    centers = head.bucket_centers("log", head.bucket_edges("log"))
    probs = [0.55, 0.45] + [0.0] * 62
    mode_value, _ = head.decode_mode(probs, centers, True)
    exp_value = head.decode_expected(probs, centers, True)
    approx(mode_value, 0.0)
    assert exp_value > 0.2, exp_value     # expectation cannot say "terminal"
    ok("mode decodes terminal exactly where the expectation cannot (this is why B1 needs mode)")


# ---------------------------------------------------------------------------
# B1.2 the torch target: d = 0 -> bin 0 with weight 1
# ---------------------------------------------------------------------------

def test_two_hot_batch_zero() -> None:
    if not head.torch_available():
        print(f"skip - torch not importable ({head.torch_error()})")
        return
    import torch

    ns = head.ensure()
    model = ns["Model"]({"encoding": "dual", "buckets": 64, "max_steps": 32768,
                         "hidden": 16, "value_hidden": 16, "vocab_size": 2})
    steps = torch.tensor([0.0, 1.0, 5.0, 1234.0, 32767.0, 32768.0, 1e9])
    idx, w = model.two_hot_batch(steps, model.edges, True)
    assert idx[0].item() == 0, idx.tolist()
    assert abs(w[0].item() - 1.0) < 1e-6, w.tolist()      # ALL mass on bin 0
    assert idx[1].item() == 1, idx.tolist()               # 1 step -> bin 1
    assert idx[3].item() > 0
    assert idx[-1].item() == 63 and abs(w[-1].item() - 1.0) < 1e-6
    # monotone in d
    wide = torch.tensor([float(i) for i in range(0, 200)]
                        + [300.0, 1000.0, 5000.0, 20000.0])
    idxs, _ = model.two_hot_batch(wide, model.edges, True)
    got = idxs.tolist()
    assert all(b >= a for a, b in zip(got, got[1:])), got
    ok("two_hot_batch: d=0 -> bin 0 with weight 1.0 (was bin 1), grid top -> bin 63")


def test_two_hot_batch_matches_pure() -> None:
    if not head.torch_available():
        return
    import torch

    ns = head.ensure()
    model = ns["Model"]({"encoding": "dual", "buckets": 64, "max_steps": 32768,
                         "hidden": 16, "value_hidden": 16, "vocab_size": 2})
    vals = [0.0, 0.5, 1.0, 3.0, 42.0, 999.0, 12345.0, 32768.0]
    idx, w = model.two_hot_batch(torch.tensor(vals), model.edges, True)
    for k, v in enumerate(vals):
        i, j, pure_w = head.two_hot(v, [float(e) for e in model.edges.tolist()], True)
        assert idx[k].item() == i, (v, idx[k].item(), i)
        assert abs(w[k].item() - pure_w) < 1e-5, (v, w[k].item(), pure_w)
    ok("two_hot_batch agrees with pure two_hot on 0 / 0.5 / grid top (no drift between twins)")


def test_model_decodes_terminal_exactly() -> None:
    if not head.torch_available():
        return
    import torch

    ns = head.ensure()
    for enc in ("uniform", "log", "dual"):
        model = ns["Model"]({"encoding": enc, "buckets": 64, "max_steps": 32768,
                             "uniform_span": 2048, "hidden": 16, "value_hidden": 16,
                             "vocab_size": 2, "decode": "mode"})
        out = model(["theorem a : True\nimport Mathlib"])
        with torch.no_grad():
            logits = out["value_logits"].clone()
            # force all mass onto bin 0, with a nonzero residual
            logits[:] = -1e4
            logits[:, 0] = 1e4
            out["value_logits"] = logits
            if out["value_res"] is not None:
                out["value_res"] = torch.full_like(out["value_res"], 0.9)
        est, est_log = model.value_point(out)
        assert float(est[0]) == 0.0, (enc, float(est[0]))
        assert float(est_log[0]) == 0.0, (enc, float(est_log[0]))
    ok("Model.value_point: bin-0 mode with max residual still decodes to exactly 0.0")


# ---------------------------------------------------------------------------
# B1.3 the flag and the gate see the same terminal story
# ---------------------------------------------------------------------------

def test_tail_flag_semantics() -> None:
    assert gate.tail_flag(63, 64) is True
    assert gate.tail_flag(62, 64) is False
    assert gate.tail_flag(0, 64) is False
    assert gate.tail_flag(-1, 64) is False          # regress has no mode
    # the old value test missed a saturated head: last centre 30339 < 0.95*32768
    assert head.decode(63, "log") < 0.95 * 32768
    got = gate.decide_gate(head.decode(63, "log"), tail_saturated=gate.tail_flag(63, 64))
    assert got["action"] == "compose" if False else got["action"] == "decompose", got
    assert got["reason"] == "tail_saturated", got
    ok("tail_flag fires on the last bin only; the saturated head now decomposes")


def main() -> int:
    test_bin0_is_terminal()
    test_encode_zero_and_monotone()
    test_log_grid_shape()
    test_decode_mode_terminal()
    test_expectation_vs_mode_on_terminal()
    test_two_hot_batch_zero()
    test_two_hot_batch_matches_pure()
    test_model_decodes_terminal_exactly()
    test_tail_flag_semantics()
    print(f"\n{CHECKS} checks passed (head geometry / zero-step)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
