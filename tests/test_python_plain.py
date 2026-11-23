#!/usr/bin/env python3
"""Plain-assert tests (no pytest): `python3 tests/test_python_plain.py`.

Covers dataset.py (proof -> prefix records), head.py (bucket geometry, two-hot,
metrics, gate parity, and — if torch is importable — a forward/backward pass)
and the shared gate table in tests/gate_cases.json.

Every check prints `ok <name>`; the last line is a summary.  Exit code is 1 on
the first failure, so it drops straight into CI or a shell chain.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import dataset  # noqa: E402
import gate  # noqa: E402
import head  # noqa: E402

CHECKS = 0


def ok(name: str) -> None:
    global CHECKS
    CHECKS += 1
    print(f"ok {CHECKS:02d} - {name}")


def approx(a: float, b: float, tol: float = 1e-6) -> None:
    assert abs(a - b) <= tol, f"{a} != {b} (tol {tol})"


# ---------------------------------------------------------------------------
# dataset.py
# ---------------------------------------------------------------------------

def test_split_steps() -> None:
    proof = "by\n  intro h\n  have h2 : x = y := by\n    rfl\n  simp [h, h2]"
    steps = dataset.split_tactic_steps(proof)
    assert steps == ["intro h", "have h2 : x = y := by\n    rfl", "simp [h, h2]"], steps
    ok("split_tactic_steps keeps nested blocks attached to their step")
    assert dataset.split_tactic_steps("") == []
    assert dataset.split_tactic_steps("by") == []
    ok("split_tactic_steps handles empty/`by`-only proofs")


def test_records_from_proof() -> None:
    steps = ["intro h", "simp [h]", "rfl"]
    rows = dataset.records_from_proof(context="import Mathlib", goal="theorem t : True",
                                      steps=steps, source="unit", problem_id="p")
    assert len(rows) == 4, len(rows)
    assert [r["remaining_steps"] for r in rows] == [3, 2, 1, 0]
    assert rows[0]["next_tactic"] == "intro h"
    assert rows[2]["context"].endswith("intro h\nsimp [h]")
    assert rows[3]["terminal"] is True and rows[3]["next_tactic"] is None
    ok("records_from_proof: remaining_steps = n - i and a terminal 0 record exists")


def test_tiny_dataset() -> None:
    a = dataset.synthesize_tiny(50, 7)
    b = dataset.synthesize_tiny(50, 7)
    assert len(a) == 50 and a == b
    steps = [r["remaining_steps"] for r in a]
    assert max(steps) > 10000, max(steps)
    assert min(steps) >= 1
    assert all(r["next_tactic"].startswith("tac_") for r in a)
    ok(f"tiny dataset: 50 deterministic records, max remaining={max(steps)}")


def test_dataset_cli_tiny(tmp: Path) -> None:
    out = tmp / "tiny.jsonl"
    proc = subprocess.run([sys.executable, str(ROOT / "train" / "dataset.py"),
                           "--tiny", "--out", str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary["records"] == 50 and summary["unit"] == "steps", summary
    assert summary["target"] == "remaining_steps", summary
    assert summary["decompose_actions"] > 0 and summary["remaining_steps_max"] > 500, summary
    rows = dataset.read_jsonl(str(out))
    assert len(rows) == 50 and rows[0]["unit"] == "steps"
    ok("dataset.py --tiny CLI writes 50 mission records targeting proof steps")


# ---------------------------------------------------------------------------
# head.py: geometry + metrics (torch-free)
# ---------------------------------------------------------------------------

def test_bucket_geometry() -> None:
    for enc in ("uniform", "log"):
        edges = head.bucket_edges(enc, 64, 32768)
        assert len(edges) == 65
        assert edges[0] == 0.0
        assert all(b > a for a, b in zip(edges, edges[1:])), enc
    default = head.bucket_edges("log")
    log_edges = [math.log1p(e) for e in default]
    gaps = [b - a for a, b in zip(log_edges[1:], log_edges[2:])]
    approx(max(gaps), min(gaps), 1e-6)  # log grid is uniform in log1p space
    ok("bucket_edges: 64 monotone bins, bin 0 starts at 0, log grid even in log space")


def test_two_hot() -> None:
    edges = head.bucket_edges("log", 64, 32768)
    i, j, w = head.two_hot(1234, edges, True)
    assert j == i + 1
    approx(w + (1 - w), 1.0)
    assert 0 <= i < 64
    i2, j2, w2 = head.two_hot(10 ** 9, edges, True)
    assert (i2, j2, w2) == (63, 63, 1.0)
    i0, _, w0 = head.two_hot(0, edges, True)
    assert i0 == 0 and w0 == 1.0
    ok("two_hot: adjacent bins for mid values, exact top/bottom clamp at 0 and 10^9")


def test_decode_and_quantiles() -> None:
    centers = head.bucket_centers("log", head.bucket_edges("log"))
    one_hot = [0.0] * 64
    one_hot[10] = 1.0
    approx(head.decode_expected(one_hot, centers, True), centers[10], 1e-6)
    lo, hi = head.quantiles(one_hot, centers, True)
    approx(lo, centers[10], 1e-6)
    approx(hi, centers[10], 1e-6)
    ok("decode_expected / quantiles: one-hot recovers the bucket centre")


def test_metrics() -> None:
    centers = head.bucket_centers("log", head.bucket_edges("log"))
    trues = [1, 5, 20, 300, 1200, 12000, 15000]
    perfect = head.metrics(trues, trues, centers=centers)
    assert perfect["mae_log"] == 0.0 and perfect["mae_steps"] == 0.0
    assert perfect["bucket_acc"] == 1.0
    assert perfect["tail"]["recall"] == 1.0 and perfect["tail"]["n"] == 2
    assert perfect["calibration"]["monotone"] is True
    bad = head.metrics([3] * len(trues), trues, centers=centers)
    assert bad["tail"]["recall"] == 0.0
    assert bad["mae_log"] > perfect["mae_log"]
    assert bad["skill_vs_median"] <= 0.0
    ok("metrics: perfect scores 0 MAE / 1.0 tail recall; a constant 3 misses the tail")


def test_metrics_no_tail() -> None:
    m = head.metrics([2, 4], [2, 4])
    assert m["tail"] == {"at": head.TAIL_STEPS, "n": 0, "recall": None}
    ok("metrics: no >10k samples reports tail n=0 instead of inventing a recall")


# ---------------------------------------------------------------------------
# gate parity with the JS suite
# ---------------------------------------------------------------------------

def test_gate_cases() -> None:
    table = json.loads((ROOT / "tests" / "gate_cases.json").read_text())
    for case in table["cases"]:
        est = float(case["estimate"])
        got = gate.decide_gate(
            est,
            threshold=case.get("threshold", gate.DEFAULT_THRESHOLD),
            prev=case.get("prev", "prove"),
            ci_high=case.get("ci_high"),
            tail_saturated=case.get("tail_saturated", False),
        )
        exp = case["expect"]
        assert got["action"] == exp["action"], (case["name"], got, exp)
        assert got["reason"] == exp["reason"], (case["name"], got, exp)
        if "suggested_children" in exp:
            assert got["suggested_children"] == exp["suggested_children"], (case["name"], got)
        if "threshold" in exp:
            assert got["threshold"] == exp["threshold"], (case["name"], got)
    ok(f"gate: {len(table['cases'])} shared cases match lib/gate.mjs expectations")
    assert gate.decide_gate(500)["action"] == "decompose"
    assert gate.decide_gate(400, prev="decompose")["action"] == "prove"
    ok("gate: 500 enters decompose, 400 releases the hysteresis band")


# ---------------------------------------------------------------------------
# head.py: torch (skipped cleanly when torch is absent)
# ---------------------------------------------------------------------------

def test_torch_model() -> None:
    if not head.torch_available():
        print(f"skip - torch not importable ({head.torch_error()}); torch checks skipped")
        return
    ns = head.ensure()
    import torch

    cfg = {"encoding": "dual", "buckets": 64, "max_steps": 32768, "hidden": 32,
           "value_hidden": 64, "vocab_size": 4, "decode": "mode"}
    model = ns["Model"](cfg)
    texts = ["theorem a : True\nintro h", "theorem b : 1 = 1\nsimp"]
    out = model(texts)
    assert out["value_logits"].shape == (2, 64), out["value_logits"].shape
    assert out["policy_logits"].shape == (2, 4)

    steps = torch.tensor([0.0, 15000.0])
    batch = {"texts": texts, "steps": steps,
             "policy_targets": torch.tensor([1, 2])}
    total, parts = ns["combined_loss"](model, out, batch, value_coef=head.VALUE_COEF)
    assert math.isfinite(parts["value_loss"]) and math.isfinite(parts["policy_loss"])
    total.backward()
    assert any(p.grad is not None for p in model.parameters())
    assert head.VALUE_COEF == 1e-3
    ok("torch Model: dual head shapes, finite loss, gradient flows")

    i, frac = model.two_hot_batch(torch.tensor([0.0, 20000.0, 32768.0]), model.edges, True)
    assert i.max().item() <= 63 and frac.min().item() >= 0.0
    assert i[2].item() == 63, i.tolist()
    ok("two_hot_batch: 0 / 20000 / grid-top all land inside 64 valid bins")

    preds = ns["predict_texts"](model, texts)
    assert len(preds) == 2 and preds[0]["ci_low"] <= preds[0]["ci_high"]
    ok("predict_texts: estimate, 80% interval, saturation flag for both texts")

    for enc in head.ENCODINGS:
        m = ns["Model"]({**cfg, "encoding": enc})
        o = m(texts)
        pts, _ = m.value_point(o)
        assert pts.shape == (2,) and bool((pts >= 0).all()), enc
        lo, hi = m.interval(o)
        assert bool((hi >= lo).all()), enc
        loss = m.value_loss(o, steps)
        assert math.isfinite(float(loss.detach())), enc
    ok("all 4 encodings: non-negative point estimate, ordered interval, finite loss")


# ---------------------------------------------------------------------------
# Mission-level unit (default) — dataset + head + stage-1 run
# ---------------------------------------------------------------------------

def test_mission_unit_default() -> None:
    rows = dataset.build(bank_dir=None, mathlib=None, tiny=True, seed=7)
    assert dataset.DEFAULT_UNIT == "steps"
    assert rows[0]["unit"] == "steps"
    assert {"mission_id", "env", "context", "goal", "action_type",
            "n_child_missions", "remaining_steps"} <= set(rows[0])
    assert "d_hat_target" not in rows[0] and "remaining_blocks" not in rows[0]
    summary = dataset.summarize(rows)
    assert summary["unit"] == "steps" and summary["missions"] == 50
    assert summary["target"] == "remaining_steps"
    assert summary["decompose_actions"] + summary["block_actions"] == 50
    assert summary["children_total"] > 0
    tactic = dataset.build(bank_dir=None, mathlib=None, tiny=True, seed=7, unit="tactic")
    assert dataset.summarize(tactic)["unit"] == "tactic"
    blocks = dataset.build(bank_dir=None, mathlib=None, tiny=True, seed=7, unit="blocks")
    assert dataset.summarize(blocks)["deprecated"] is True
    ok("dataset: default unit is steps, summarize reports the step target + action split")


def test_mission_target() -> None:
    assert head.DEFAULT_UNIT == "steps"
    rec = {"remaining_steps": 36, "next_action": "block"}
    assert head.target_steps(rec) == 36.0
    assert head.target_steps({"remaining_steps": 36, "next_tactic": "simp"}, "tactic") == 36.0
    # the deprecated shim is the only place a ceil() survives
    assert head.target_steps({"remaining_blocks": 12, "d_hat_target": 12}, "blocks") == 12.0
    assert head.unit_of([rec]) == "steps"
    assert head.unit_of([{"remaining_blocks": 3}]) == "blocks"
    ok("head: target_steps defaults to remaining proof steps; blocks is the shim")


def test_mission_stage1(tmp: Path) -> None:
    run = tmp / "mission-run"
    proc = subprocess.run([sys.executable, str(ROOT / "train" / "train_stage1.py"),
                           "--tiny", "--out", str(run)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["unit"] == "steps"
    assert metrics["data"]["unit"] == "steps"
    assert metrics["data"]["target"] == "remaining_steps"
    assert metrics["test"]["unit"] == "steps"
    assert metrics["test"]["tail"]["at"] == 500.0        # the gate is the tail event
    assert metrics["test"]["policy_pass_at_1"] is not None
    ok("train_stage1 --tiny: trains on remaining proof steps, reports unit in metrics")


def test_mission_batch() -> None:
    if not head.torch_available():
        print(f"skip - torch not importable ({head.torch_error()}); mission batch skipped")
        return
    sys.path.insert(0, str(ROOT / "train"))
    import train_stage1  # noqa: E402  (needs head.ensure() at call time only)
    rows = dataset.build(bank_dir=None, mathlib=None, tiny=True, seed=7)
    vocab = {t: i for i, t in enumerate(train_stage1.build_vocab(rows, "steps"))}
    assert sorted(vocab) == ["<pad>", "block", "decompose"]
    rows = rows[:3]
    ns = head.ensure()
    batch = train_stage1.make_batch(rows, vocab, ns, "steps")
    got = [float(x) for x in batch["steps"]]
    want = [float(r["remaining_steps"]) for r in rows]
    assert got == want, (got, want)
    targets = batch["policy_targets"].tolist()
    assert targets == [vocab[r["next_action"]] for r in rows]
    ok("train_stage1.make_batch: mission steps = remaining_steps, vocab = block|decompose")


def main() -> int:
    tmp = Path("/tmp/vht-tests")
    tmp.mkdir(parents=True, exist_ok=True)
    test_split_steps()
    test_records_from_proof()
    test_tiny_dataset()
    test_dataset_cli_tiny(tmp)
    test_bucket_geometry()
    test_two_hot()
    test_decode_and_quantiles()
    test_metrics()
    test_metrics_no_tail()
    test_gate_cases()
    test_torch_model()
    test_mission_unit_default()
    test_mission_target()
    test_mission_stage1(tmp)
    test_mission_batch()
    print(f"\n{CHECKS} checks passed (python)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
