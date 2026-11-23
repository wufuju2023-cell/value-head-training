#!/usr/bin/env python3
"""Plain-assert tests for the mission-level design (no pytest).

    python3 tests/test_mission_plain.py

Covers:
  traj.py      explicit mask spans, {input_ids, loss_mask}, main/sub pools
  cache.py     mission registry (reuse), sibling dedupe, compile cache (lock)
  mcts/        sampling trigger (child-return, never a timer), gate, AND record
  dataset.py   mission-level records (default unit) + --unit tactic legacy
  head.py      target_steps in both units
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "train"), str(ROOT / "mcts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cache          # noqa: E402
import dataset        # noqa: E402
import head           # noqa: E402
import schedule       # noqa: E402
import traj           # noqa: E402

CHECKS = 0


def ok(name: str) -> None:
    global CHECKS
    CHECKS += 1
    print(f"ok {CHECKS:02d} - {name}")


def raises(fn, exc=Exception) -> str:
    try:
        fn()
    except exc as err:  # noqa: BLE001
        return str(err)
    raise AssertionError(f"expected {exc.__name__}")


TOK = traj.HashTokenizer(vocab_size=512)


def decompose_traj() -> traj.Trajectory:
    return traj.Trajectory(
        mission_id="m-root", env="tiny",
        goal="theorem root : f a + f b = 0", lean_context="import Mathlib",
        action_type="decompose", action_text="split: prove f a = 0 and f b = 0",
        child_missions=["m-a", "m-b"],
        child_returns=[
            {"mission_id": "m-a", "summary": "closed by simp", "status": "accepted",
             "solved": True},
            {"mission_id": "m-b", "summary": "needs induction", "status": "failed",
             "solved": False, "remaining_blocks": 40},
        ],
        messages=[
            {"role": "system", "text": "you are a Lean prover; obey the workspace"},
            {"role": "user", "text": "theorem root : f a + f b = 0"},
            {"role": "assistant", "text": "split: prove f a = 0 and f b = 0"},
            {"role": "assistant", "blocks": [
                {"kind": "text", "text": "checking the split is well founded"},
                {"kind": "tool_result", "text": "goals: 2 (f a = 0, f b = 0)"}]},
            {"role": "subagent", "text": "m-a returned: closed by simp"},
        ],
        outcome={"solved": False, "remaining_blocks": 40}, depth=0,
    )


def block_traj() -> traj.Trajectory:
    return traj.Trajectory(
        mission_id="m-leaf", env="tiny", goal="theorem leaf : 1 = 1",
        action_type="block",
        action_text="by\n  norm_num",
        messages=[
            {"role": "system", "text": "you are a Lean prover"},
            {"role": "user", "text": "theorem leaf : 1 = 1"},
            {"role": "assistant", "text": "by\n  norm_num"},
            {"role": "tool", "text": "no goals"},
        ],
        outcome={"solved": True},
    )


# ---------------------------------------------------------------------------
# traj.py
# ---------------------------------------------------------------------------

def test_tokenizer() -> None:
    a = TOK.encode("theorem t : 1 = 1")
    b = traj.HashTokenizer(vocab_size=512).encode("theorem t : 1 = 1")
    assert a == b and len(a) >= 4
    assert all(0 <= t < 512 for t in a)
    ok("HashTokenizer: deterministic, in-vocabulary token ids")


def test_mask_spans() -> None:
    spans = traj.mask_spans(decompose_traj(), TOK)
    kinds_masked = {s["kind"] for s in spans["segments"]
                    if any(s["start"] >= a and s["end"] <= b for a, b in spans["masked"])}
    assert kinds_masked == {"system", "tool_result", "subagent_return"}, kinds_masked
    kept = {s["kind"] for s in spans["segments"] if s["kind"] not in traj.DEFAULT_MASK_POLICY["mask_kinds"]}
    assert kept == {"user", "assistant"}, kept
    assert spans["masked"] and spans["kept"]
    # merged, disjoint, in order
    flat = sorted(spans["masked"] + spans["kept"])
    assert flat[0][0] == 0 and flat[-1][1] == spans["n_tokens"]
    for (s1, e1), (s2, e2) in zip(flat, flat[1:]):
        assert e1 <= s2
    ok("traj.mask_spans: system + tool_result + subagent_return masked, user/assistant kept")


def test_input_ids_loss_mask() -> None:
    out = traj.to_input_ids_loss_mask(decompose_traj(), tokenizer=TOK)
    assert len(out["input_ids"]) == len(out["loss_mask"]) == out["n_tokens"]
    assert set(out["loss_mask"]) == {0, 1}
    assert out["n_masked"] > 0 and out["n_kept"] > 0
    # every masked span is exactly the complement of the kept spans
    masked = sum(e - s for s, e in out["spans"])
    assert masked == out["n_masked"], (masked, out["n_masked"])
    padded = traj.to_input_ids_loss_mask(decompose_traj(), tokenizer=TOK, pad_to=256)
    assert len(padded["input_ids"]) == 256
    assert padded["loss_mask"][out["n_tokens"]:] == [0] * (256 - out["n_tokens"])
    assert padded["n_kept"] == out["n_kept"]
    ok("traj.to_input_ids_loss_mask: aligned ids/mask, spans sum to n_masked, pad masked")


def test_mask_policy_override() -> None:
    pol = {"mask_kinds": ["system", "tool_result", "subagent_return", "user"],
           "keep_kinds": ["assistant"]}
    out = traj.to_input_ids_loss_mask(decompose_traj(), tokenizer=TOK, policy=pol)
    base = traj.to_input_ids_loss_mask(decompose_traj(), tokenizer=TOK)
    assert out["n_kept"] < base["n_kept"]           # the user turn is gone too
    # keep_kinds always wins: asking to mask `assistant` cannot remove the decision
    pol2 = {"mask_kinds": ["assistant", "system", "tool_result", "subagent_return"]}
    same = traj.to_input_ids_loss_mask(decompose_traj(), tokenizer=TOK, policy=pol2)
    assistant_spans = [(s["start"], s["end"]) for s in same["segments"]
                       if s["kind"] == "assistant"]
    assert assistant_spans
    for start, end in assistant_spans:
        assert all(same["loss_mask"][i] == 1 for i in range(start, end))
    assert same["n_kept"] == base["n_kept"]
    ok("traj mask_policy: configurable spans, keep_kinds always wins over mask_kinds")


def test_pools() -> None:
    pools = traj.split_pools([decompose_traj(), block_traj()])
    assert set(pools) == {"main_pool", "sub_pool"}
    assert len(pools["main_pool"]) == 2
    main = {r["mission_id"]: r for r in pools["main_pool"]}
    assert set(main["m-root"]) & traj.CHILD_INTERNAL_KEYS == set()
    assert main["m-root"]["n_child_missions"] == 2
    assert [s["mission_id"] for s in main["m-root"]["child_summaries"]] == ["m-a", "m-b"]
    assert "closed by simp" in main["m-root"]["child_summaries"][0]["summary"]
    assert set(pools["sub_pool"]) == {"m-root", "m-leaf"}
    assert len(pools["sub_pool"]["m-root"][0]["messages"]) == 5   # full internals live here
    for rec in pools["main_pool"]:
        traj.assert_main_record_clean(rec)
    leaked = {**main["m-root"], "messages": []}
    msg = raises(lambda: traj.assert_main_record_clean(leaked), ValueError)
    assert "messages" in msg
    ok("traj.split_pools: main_pool keeps summaries only, sub_pool keeps full trajectories")


def test_children_pending() -> None:
    t = decompose_traj()
    assert t.children_pending() is False       # both children returned
    t.child_returns = t.child_returns[:1]
    assert t.children_pending() is True        # m-b still out
    assert block_traj().children_pending() is False
    ok("traj.children_pending: a dispatched child that has not returned keeps the parent waiting")


# ---------------------------------------------------------------------------
# cache.py — mission registry, dedupe, compile cache
# ---------------------------------------------------------------------------

def test_mission_identity() -> None:
    key = cache.statement_hash("bank", "theorem  t :   True")
    assert key == cache.statement_hash("bank", "theorem t : True")   # whitespace normalised
    assert key != cache.statement_hash("mathlib", "theorem t : True")  # env is part of the key
    mid = cache.mission_id_for("bank", "theorem t : True")
    assert mid == "m-" + key[:12] and len(mid) == 14
    ok("cache: sha256(env, statement_formal) key, whitespace-stable, env-sensitive")


def test_registry_reuse(tmp: Path) -> None:
    reg = cache.MissionRegistry(tmp / "reg")
    a = reg.get_or_create("tiny", "theorem t : True")
    b = reg.get_or_create("tiny", "theorem t : True")
    c = reg.get_or_create("tiny", "theorem u : 1 = 1")
    assert a["reused"] is False and b["reused"] is True and c["reused"] is False
    assert a["mission_id"] == b["mission_id"] and a["workspace"] == b["workspace"]
    assert a["mission_id"] != c["mission_id"]
    assert Path(b["workspace"]).is_dir()
    assert reg.stats()["reused"] == 1 and len(reg) == 2
    reloaded = cache.MissionRegistry(tmp / "reg")
    again = reloaded.get_or_create("tiny", "theorem t : True")
    assert again["reused"] is True and again["workspace"] == a["workspace"]
    ok("cache.MissionRegistry: 2 identical + 1 different -> 2 missions, reuse survives reload")


def test_registry_children(tmp: Path) -> None:
    reg = cache.MissionRegistry(tmp / "kids")
    parent = reg.get_or_create("tiny", "theorem root : True")
    out = cache.register_children(
        reg, "tiny",
        [{"goal": "theorem a : True"}, {"goal": "theorem a : True"},
         {"goal": "theorem b : True"}],
        parent_mission_id=parent["mission_id"], depth=1)
    assert len(out["missions"]) == 2, [m["statement_formal"] for m in out["missions"]]
    assert len(out["dropped"]) == 1 and out["dropped"][0]["reused"] is True
    assert out["log"][1]["reused"] is True
    assert all(m["reused"] is False for m in out["missions"])
    # the same two children appearing in a later decomposition are reused
    again = cache.register_children(reg, "tiny",
                                   [{"goal": "theorem a : True"}, {"goal": "theorem b : True"}],
                                   parent_mission_id=parent["mission_id"], depth=1)
    assert [m["reused"] for m in again["missions"]] == [True, True]
    assert {m["mission_id"] for m in again["missions"]} == {m["mission_id"] for m in out["missions"]}
    ok("cache.register_children: identical siblings collapse, second decomposition reuses ids")


def test_registry_status(tmp: Path) -> None:
    reg = cache.MissionRegistry(tmp / "status")
    rec = reg.get_or_create("tiny", "theorem s : True")
    assert reg.set_status(rec["mission_id"], "accepted")["status"] == "accepted"
    assert reg.stats()["by_status"] == {"accepted": 1}
    raises(lambda: reg.set_status(rec["mission_id"], "solved"), ValueError)
    raises(lambda: reg.set_status("m-nope", "accepted"), KeyError)
    ok("cache.MissionRegistry.set_status: validated statuses, unknown mission rejected")


def test_compile_cache(tmp: Path) -> None:
    calls: list[str] = []

    def runner(task_dir, file_name, olean):
        calls.append(file_name)
        olean.parent.mkdir(parents=True, exist_ok=True)
        olean.write_text("ok", encoding="utf-8")
        return {"returncode": 0, "stderr": ""}

    cc = cache.CompileCache(tmp, runner=runner)
    proof = "theorem t : True := by trivial"
    e1 = cc.compile("tiny", "Problem.lean", proof)
    e2 = cc.compile("tiny", "Problem.lean", proof)
    assert e1["cache_hit"] is False and e2["cache_hit"] is True
    assert len(calls) == 1
    assert e1["task_dir"] == e2["task_dir"]
    assert Path(e2["olean"]).is_file()
    other = cc.compile("tiny", "Problem.lean", proof + "\n-- v2")
    assert other["cache_hit"] is False and other["task_dir"] != e1["task_dir"]
    stats = cc.stats()
    assert stats == {"compiles": 3, "cache_hits": 1, "misses": 2,
                     "hit_rate": 0.3333, "envs": ["tiny"]}
    assert cc.project_dir("tiny") == tmp / "olean" / "tiny" / "project"
    jvm = cc.compile("mathlib", "Problem.lean", proof)
    assert jvm["task_dir"] != e1["task_dir"]     # one shared project per env
    ok("cache.CompileCache: identical content hits once, different content/env -> new dir")


def test_compile_lock(tmp: Path) -> None:
    calls = []
    barrier = threading.Barrier(4)

    def slow_runner(task_dir, file_name, olean):
        calls.append(threading.current_thread().name)
        olean.parent.mkdir(parents=True, exist_ok=True)
        import time
        time.sleep(0.05)                     # widen the window a lock must close
        olean.write_text("ok", encoding="utf-8")
        return {"returncode": 0, "stderr": ""}

    cc = cache.CompileCache(tmp, runner=slow_runner)
    proof = "theorem same : True := by trivial"
    results: list[dict] = []
    threads = []
    for _ in range(4):
        def work():
            barrier.wait()
            results.append(cc.compile("tiny", "Problem.lean", proof))
        th = threading.Thread(target=work)
        threads.append(th)
        th.start()
    for th in threads:
        th.join()
    assert len(calls) == 1, calls
    assert sum(1 for r in results if r["cache_hit"]) == 3
    assert len({r["task_dir"] for r in results}) == 1
    ok("cache.CompileCache: 4 concurrent compiles of one file -> 1 runner call, 3 hits")


# ---------------------------------------------------------------------------
# mcts/schedule.py — trigger, gate, AND bookkeeping
# ---------------------------------------------------------------------------

def test_trigger() -> None:
    tick = schedule.should_sample({"pending_children": [], "in_flight": 0}, "tick")
    assert tick["sample"] is False and tick["reason"] == "timer_is_not_a_trigger"
    start = schedule.should_sample({}, "start", k=6)
    assert start["sample"] is True and start["k"] == 6
    ret = schedule.should_sample({"sampled_this_state": 1}, "child_return", k=4)
    assert ret["sample"] is True and ret["reason"] == "child_return"
    blk = schedule.should_sample({"sampled_this_state": 1}, "block_finished")
    assert blk["sample"] is True
    ok("schedule.should_sample: child_return / block_finished trigger, tick never does")


def test_trigger_blocked() -> None:
    pending = schedule.should_sample({"pending_children": ["m-a"], "sampled_this_state": 1},
                                     "child_return")
    assert pending["sample"] is False and pending["reason"] == "children_pending"
    assert pending["wait_for"] == "child_return"
    flying = schedule.should_sample({"in_flight": 2}, "block_finished")
    assert flying["sample"] is False and flying["reason"] == "samples_in_flight"
    twice = schedule.should_sample({"sampled_this_state": 1}, "start")
    assert twice["sample"] is False and twice["reason"] == "already_sampled"
    deep = schedule.should_sample({"depth": 9, "max_depth": 3}, "child_return")
    assert deep["sample"] is False and deep["reason"] == "depth_exceeded"
    ok("schedule.should_sample: no expansion while children pending / samples in flight")


def test_gate_action() -> None:
    assert schedule.gate_action(900)["action"] == "decompose"
    assert schedule.gate_action(900)["mandatory"] is True
    assert schedule.gate_action(500)["action"] == "decompose"      # enter at the threshold
    assert schedule.gate_action(120)["action"] == "block"
    capped = schedule.gate_action(900, depth=3, max_depth=3)
    assert capped["action"] == "block" and capped["reason"] == "depth_capped"
    assert capped["mandatory"] is False and capped["suggested_children"] == 0
    none = schedule.gate_action(None)
    assert none["action"] == "block" and none["reason"] == "no_estimate"
    sat = schedule.gate_action(12, tail_saturated=True)
    assert sat["action"] == "decompose" and sat["mandatory"] is True
    # a child above the threshold is allowed: it decomposes further (depth 1 < 3)
    child = schedule.gate_action(4000, depth=1, max_depth=3)
    assert child["action"] == "decompose"
    ok("schedule.gate_action: d>500 must decompose, depth cap forces a block")


def test_gate_steps_units() -> None:
    """The five cases that define the unit contract (d_hat and N in steps)."""
    n = 100
    # 1. 900 steps at N=100 -> 9 rounds -> decompose
    g1 = schedule.gate_action(900, action_block_steps=n)
    assert (g1["action"], g1["rounds_left"], g1["single_shot"]) == ("decompose", 9, False)
    # 2. 400 steps -> 4 rounds -> prove, and 4 > 1 so it is NOT single-shot
    g2 = schedule.gate_action(400, action_block_steps=n)
    assert (g2["action"], g2["rounds_left"], g2["single_shot"]) == ("block", 4, False)
    assert g2["learning_signal"] is True
    # 3. 80 steps <= N=100 -> single shot, no learning signal
    g3 = schedule.gate_action(80, action_block_steps=n)
    assert (g3["action"], g3["rounds_left"], g3["single_shot"]) == ("block", 1, True)
    assert g3["learning_signal"] is False and g3["no_signal_band"] is True
    # 4. 480 with prev=decompose stays decomposed (hysteresis exits at 0.8*500=400)
    g4 = schedule.gate_action(480, prev="decompose", action_block_steps=n)
    assert g4["action"] == "decompose" and g4["reason"] == "hysteresis_hold"
    assert schedule.gate_action(400, prev="decompose")["reason"] == "hysteresis_exit"
    # 5. rounds is ceil, not round/floor
    assert [schedule.rounds_left(d, n) for d in (99, 100, 101, 150, 1000)] == [1, 1, 2, 2, 10]
    ok("gate: 900->decompose(9), 400->prove(4), 80->single_shot, 480 stays decomposed")


def test_unit_regression() -> None:
    """Regression for the blocks/steps bug this revision fixes.

    A 900-step mission used to be re-read as ceil(900/3) = 300 "blocks", which is
    below the 500 gate, so it passed as `prove` and the agent attempted a 900-step
    proof inline.  Under one unit (steps) it must decompose, and the old
    conversion must be visibly the wrong answer.
    """
    rec = dataset.upgrade_tactic_record(
        {"id": "big#0", "source": "unit", "context": "", "goal": "theorem big : True",
         "remaining_steps": 900})
    target = head.target_steps(rec)                       # 900 steps, not 300
    assert target == 900.0
    assert schedule.gate_action(target)["action"] == "decompose"
    assert schedule.gate_action(target)["mandatory"] is True
    # the old reading: 900 steps -> 300 blocks -> under the gate -> "prove"
    legacy_blocks = dataset.remaining_blocks(900, 3)
    assert legacy_blocks == 300
    assert schedule.gate_action(float(legacy_blocks))["action"] == "block"
    # and the old band formula (d_hat * block_steps <= N) was wrong in the other
    # direction: 4 blocks * 3 = 12 steps <= N=10 would have *not* been a band,
    # while 12 steps <= 10 correctly is not; 3 blocks * 3 = 9 <= 10 is a band.
    assert schedule.in_no_signal_band(9, 10) is True
    assert schedule.in_no_signal_band(12, 10) is False
    assert schedule.in_no_signal_band(30, 10) is False    # 30 steps is 3 rounds, not 1
    ok("regression: 900 steps decomposes; reading it as 300 blocks was the bug")


def test_and_bookkeeping() -> None:
    children = [{"mission_id": "m-a", "solved": True, "remaining_blocks": 0},
                {"mission_id": "m-b", "solved": False, "remaining_blocks": 40},
                {"mission_id": "m-c", "solved": False, "remaining_blocks": 7}]
    rec = schedule.and_state("m-root", children)
    assert rec["kind"] == "AND" and rec["n_children"] == 3
    assert rec["discharged"] is False and rec["pending"] == ["m-b", "m-c"]
    assert schedule.aggregate_children_value(children) == 40.0          # max, not sum
    assert schedule.aggregate_children_value(children, mode="min") == 7.0
    done = [{"mission_id": "m-a", "solved": True}]
    assert schedule.and_state("m-root", done)["discharged"] is True
    ok("schedule.and_state: AND recorded, parent value aggregates by max over pending")


# ---------------------------------------------------------------------------
# dataset.py mission unit + head.target_steps
# ---------------------------------------------------------------------------

REQUIRED = ["mission_id", "env", "context", "goal", "action_type",
            "n_child_missions", "remaining_steps"]


def test_mission_dataset() -> None:
    rows = dataset.build(bank_dir=None, mathlib=None, tiny=True, seed=7)
    assert len(rows) == 50 and rows[0]["unit"] == "steps"
    for rec in rows:
        for key in REQUIRED:
            assert key in rec, (key, rec)
        # one unit only: no blocks field on the default path
        assert "d_hat_target" not in rec and "remaining_blocks" not in rec
        assert rec["remaining_steps"] >= 1
    assert rows == dataset.build(bank_dir=None, mathlib=None, tiny=True, seed=7)  # deterministic
    ok("dataset: --unit steps (default) emits 7 mission fields, target is steps")


def test_mission_steps_target() -> None:
    rec = dataset.upgrade_tactic_record(
        {"id": "p#0", "source": "unit", "context": "c", "goal": "theorem t : True",
         "remaining_steps": 7})
    # no ceil: the target is literally the remaining tactic steps
    assert rec["remaining_steps"] == 7 and rec["unit"] == "steps"
    assert "remaining_blocks" not in rec
    assert rec["mission_id"] == cache.mission_id_for("unit", "theorem t : True")
    # the deprecated shim is the only place the ceil conversion survives
    legacy = dataset.upgrade_tactic_record(
        {"id": "p#0", "source": "unit", "context": "c", "goal": "theorem t : True",
         "remaining_steps": 7}, unit="blocks", block_steps=3)
    assert legacy["remaining_blocks"] == 3 and legacy["d_hat_target"] == 3
    assert head.target_steps(legacy, "blocks") == 3.0
    ok("dataset: steps target is exact (no ceil); blocks survives only as the shim")


def test_mission_actions() -> None:
    rows = dataset.build(bank_dir=None, mathlib=None, tiny=True, seed=7, threshold=500)
    dec = [r for r in rows if r["action_type"] == "decompose"]
    blk = [r for r in rows if r["action_type"] == "block"]
    assert dec and blk
    assert all(r["remaining_steps"] >= 500 for r in dec)
    assert all(2 <= r["n_child_missions"] <= 8 for r in dec)
    assert all(r["n_child_missions"] == 0 for r in blk)
    assert all(r["remaining_steps"] < 500 for r in blk)
    assert all(r["next_action"] == r["action_type"] for r in rows)
    ok("dataset: decompose iff remaining steps >= 500, 2..8 children, block has 0")


def test_mission_from_tactic() -> None:
    steps = ["intro h", "simp [h]", "rw [h]", "rfl", "ring", "norm_num", "omega"]
    tactic_rows = dataset.records_from_proof(context="import Mathlib",
                                             goal="theorem t : True", steps=steps,
                                             source="unit", problem_id="p")
    rows = dataset.mission_from_tactic(tactic_rows)
    assert [r["id"] for r in rows] == [f"p#{i}" for i in range(8)], [r["id"] for r in rows]
    assert [r["remaining_steps"] for r in rows] == [7, 6, 5, 4, 3, 2, 1, 0]
    assert all(r["mission_id"] == rows[0]["mission_id"] for r in rows)   # one mission
    ok("dataset.mission_from_tactic: every prefix, remaining_steps = total - i")


def test_dataset_cli_units(tmp: Path) -> None:
    steps_out = tmp / "steps.jsonl"
    tactic_out = tmp / "tactic.jsonl"
    blocks_out = tmp / "blocks.jsonl"
    p1 = subprocess.run([sys.executable, str(ROOT / "train" / "dataset.py"),
                         "--tiny", "--out", str(steps_out)], capture_output=True, text=True)
    assert p1.returncode == 0, p1.stderr
    s1 = json.loads(p1.stdout.strip().splitlines()[-1])
    assert s1["unit"] == "steps" and s1["decompose_actions"] > 0, s1
    assert s1["target"] == "remaining_steps" and s1["deprecated"] is False, s1
    p2 = subprocess.run([sys.executable, str(ROOT / "train" / "dataset.py"),
                         "--tiny", "--unit", "tactic", "--out", str(tactic_out)],
                        capture_output=True, text=True)
    assert p2.returncode == 0, p2.stderr
    s2 = json.loads(p2.stdout.strip().splitlines()[-1])
    assert s2["unit"] == "tactic" and s2["over_10k"] > 0, s2
    assert "remaining_blocks" not in dataset.read_jsonl(str(tactic_out))[0]
    p3 = subprocess.run([sys.executable, str(ROOT / "train" / "dataset.py"),
                         "--tiny", "--unit", "blocks", "--out", str(blocks_out)],
                        capture_output=True, text=True)
    assert p3.returncode == 0, p3.stderr
    s3 = json.loads(p3.stdout.strip().splitlines()[-1])
    assert s3["unit"] == "blocks" and s3["deprecated"] is True, s3
    assert "DEPRECATED" in p3.stderr, p3.stderr
    ok("dataset CLI: steps default; tactic legacy; blocks deprecated with a warning")


def test_target_steps() -> None:
    rec = {"remaining_steps": 27, "next_action": "block"}
    tactic = {"remaining_steps": 27, "next_tactic": "simp"}
    assert head.target_steps(rec) == 27.0
    assert head.target_steps(rec, "tactic") == 27.0
    assert head.target_steps({"remaining_blocks": 9, "d_hat_target": 9}, "blocks") == 9.0
    assert head.target_steps({"remaining_steps": 8, "remaining_blocks": 3}, "blocks") == 3.0
    assert head.unit_of([rec]) == "steps" and head.unit_of([tactic]) == "steps"
    assert head.unit_of([{"remaining_blocks": 3}]) == "blocks"
    assert head.DEFAULT_UNIT == "steps"
    raises(lambda: head.target_steps({"goal": "x"}, "steps"), KeyError)
    raises(lambda: head.target_steps(rec, "mission"), ValueError)
    ok("head.target_steps: steps is the default, blocks is explicit and deprecated")


# ---------------------------------------------------------------------------
# Terminal-only learning signal + N-step block
# ---------------------------------------------------------------------------

def solved_traj(mission_id: str = "m-ok", **outcome) -> traj.Trajectory:
    base = {"solved": True, "status": "accepted"}
    base.update(outcome)
    return traj.Trajectory(
        mission_id=mission_id, env="tiny", goal="theorem ok : True",
        action_type="block", action_text="by\n  trivial",
        messages=[{"role": "user", "text": "theorem ok : True"},
                  {"role": "assistant", "text": "by\n  trivial"}],
        outcome=base)


def failed_traj(mission_id: str = "m-bad", **outcome) -> traj.Trajectory:
    base = {"solved": False, "status": "failed"}
    base.update(outcome)
    return traj.Trajectory(
        mission_id=mission_id, env="tiny", goal="theorem bad : False",
        action_type="decompose", action_text="split it",
        child_missions=["m-c"], child_returns=[
            {"mission_id": "m-c", "summary": "stuck", "status": "failed", "solved": False}],
        messages=[{"role": "user", "text": "theorem bad : False"},
                  {"role": "assistant", "text": "split it"}],
        outcome=base)


def test_terminal_signal() -> None:
    ok_t = solved_traj()
    bad = failed_traj()
    exhausted = failed_traj("m-out", solved=True, budget_exhausted=True,
                            status="budget_exhausted")
    unfinished = traj.Trajectory(
        mission_id="m-and", env="tiny", goal="theorem and : True",
        action_type="decompose", child_missions=["m-a", "m-b"],
        child_returns=[{"mission_id": "m-a", "solved": True, "summary": "done"},
                       {"mission_id": "m-b", "solved": False, "summary": "open"}],
        messages=[{"role": "assistant", "text": "split"}],
        outcome={"solved": True})
    assert traj.is_terminal_success(ok_t) is True
    assert traj.is_terminal_success(bad) is False
    assert traj.is_terminal_success(exhausted) is False
    assert traj.is_terminal_success(unfinished) is False        # AND(children) not discharged
    pool = traj.filter_ce_pool([ok_t, bad, exhausted, unfinished])
    assert [t.mission_id for t in pool["kept"]] == ["m-ok"]
    assert pool["reason_counts"] == {"not_success": 2, "children_unproved": 1}, pool
    assert pool["signal"] == "terminal_success_only"
    assert {d["drop_reason"] for d in pool["dropped"]} == {"not_success", "children_unproved"}
    pools = traj.ce_pools([ok_t, bad])
    assert [r["mission_id"] for r in pools["main_pool"]] == ["m-ok"]
    assert set(pools["sub_pool"]) == {"m-ok"}
    # the searched-node shape (stage 2 jsonl) is understood too
    assert traj.is_terminal_success({"solved": True}) is True
    assert traj.is_terminal_success({"solved": False}) is False
    assert traj.is_terminal_success({"solved": True, "budget_exhausted": True}) is False
    ok("traj: CE pool keeps terminal success only; failures stay as diagnostics")


def test_no_auxiliary_reward() -> None:
    bad = failed_traj()
    bad.outcome["decomposition_bonus"] = 0.3
    msg = raises(lambda: traj.is_terminal_success(bad), ValueError)
    assert "decomposition_bonus" in msg and traj.CE_SIGNAL in msg
    rec = {"mission_id": "m-x", "outcome": {"subagent_reward": 1.0, "solved": True}}
    raises(lambda: traj.filter_ce_pool([rec]), ValueError)
    raises(lambda: traj.assert_no_auxiliary_reward({"credit_assignment": 2}), ValueError)
    traj.assert_no_auxiliary_reward({"mission_id": "m-y", "action_type": "block"})
    ok("traj: decomposition/subagent/credit reward keys are rejected, not ignored")


def test_block_progress() -> None:
    old = {"goal": "theorem t : f a = 0", "lean_context": "import Mathlib"}
    new_goal = {"goal": "theorem t : f a = 0 ∧ g a = 0", "lean_context": "import Mathlib"}
    same = {"goal": "theorem t : f a = 0", "lean_context": "import Mathlib"}
    grew = {"goal": "theorem t : f a = 0", "lean_context": "import Mathlib\nhave h : f a = 0"}
    done = {"goal": "theorem t : f a = 0", "lean_context": "import Mathlib", "solved": True}
    assert traj.block_progress(old, new_goal)["progress"] is True
    assert traj.block_progress(old, new_goal)["goal_changed"] is True
    assert traj.block_progress(old, same)["no_op"] is True
    assert traj.block_progress(old, same)["progress"] is False
    assert traj.block_progress(old, grew)["progress"] is True
    assert traj.block_progress(old, grew)["context_delta"] > 0
    assert traj.block_progress(old, done)["solved"] is True
    assert schedule.block_progress is traj.block_progress     # one definition
    ok("traj.block_progress: a block that moves the goal/context is progress, identical is a no-op")


def test_no_signal_band() -> None:
    assert schedule.DEFAULT_ACTION_BLOCK_STEPS == 10
    assert schedule.ACTION_BLOCK_STEPS_CANDIDATES == (10, 20, 50, 100)
    # plain steps on both sides: d_hat <= N, no multiplication anywhere
    assert schedule.in_no_signal_band(9, 10) is True
    assert schedule.in_no_signal_band(10, 10) is True
    assert schedule.in_no_signal_band(11, 10) is False
    assert schedule.in_no_signal_band(80, 100) is True
    assert schedule.in_no_signal_band(101, 100) is False
    assert schedule.in_no_signal_band(None) is False
    first = schedule.should_sample({"d_hat_steps": 80, "action_block_steps": 100,
                                    "sampled_this_state": 0}, "start")
    assert first["sample"] is True and first["mode"] == "single_shot"
    assert first["learning_signal"] is False and first["no_signal_band"] is True
    skipped = schedule.should_sample({"d_hat_steps": 80, "action_block_steps": 100,
                                      "sampled_this_state": 1}, "block_finished")
    assert skipped["sample"] is False and skipped["reason"] == "no_signal_band"
    big = schedule.should_sample({"d_hat_steps": 4000, "action_block_steps": 100,
                                  "sampled_this_state": 1}, "block_finished")
    assert big["sample"] is True and big["learning_signal"] is True
    assert big["mode"] == "multi_round" and big["rounds_left"] == 40
    assert schedule.gate_action(80, action_block_steps=100)["no_signal_band"] is True
    assert schedule.gate_action(900, action_block_steps=100)["no_signal_band"] is False
    steps = [row["action_block_steps"] for row in schedule.CURRICULUM]
    assert steps == sorted(steps) and steps[0] == 10
    ok("schedule: d̂ <= N in steps is a one-shot, skipped by the round scheduler")


def test_dataset_trajectories(tmp: Path) -> None:
    trajs = [solved_traj(), failed_traj(), failed_traj("m-out", solved=True,
                                                       budget_exhausted=True)]
    kept = dataset.records_from_trajectories(trajs)
    assert len(kept) == 1 and kept[0]["success"] is True and kept[0]["status"] == "accepted"
    both = dataset.records_from_trajectories(trajs, include_failed=True)
    assert len(both) == 3
    assert sum(1 for r in both if r["success"] is False) == 2
    assert all(r["unit"] == "steps" for r in both)
    path = tmp / "trajs.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for t in trajs:
            fh.write(json.dumps(t.to_json()) + "\n")
    default_out = tmp / "from-trajs.jsonl"
    p1 = subprocess.run([sys.executable, str(ROOT / "train" / "dataset.py"),
                         "--trajectories", str(path), "--out", str(default_out)],
                        capture_output=True, text=True)
    assert p1.returncode == 0, p1.stderr
    s1 = json.loads(p1.stdout.strip().splitlines()[-1])
    assert s1["records"] == 1 and s1["failed"] == 0, s1
    all_out = tmp / "from-trajs-all.jsonl"
    p2 = subprocess.run([sys.executable, str(ROOT / "train" / "dataset.py"),
                         "--trajectories", str(path), "--include-failed",
                         "--out", str(all_out)], capture_output=True, text=True)
    assert p2.returncode == 0, p2.stderr
    s2 = json.loads(p2.stdout.strip().splitlines()[-1])
    assert s2["records"] == 3 and s2["failed"] == 2, s2
    ok("dataset --trajectories: successes only by default, --include-failed keeps diagnostics")


def main() -> int:
    tmp_root = Path(tempfile.mkdtemp(prefix="vht-mission-"))
    test_tokenizer()
    test_mask_spans()
    test_input_ids_loss_mask()
    test_mask_policy_override()
    test_pools()
    test_children_pending()
    test_mission_identity()
    test_registry_reuse(tmp_root / "reuse")
    test_registry_children(tmp_root / "children")
    test_registry_status(tmp_root / "status")
    test_compile_cache(tmp_root / "compile")
    test_compile_lock(tmp_root / "lock")
    test_trigger()
    test_trigger_blocked()
    test_gate_action()
    test_gate_steps_units()
    test_unit_regression()
    test_and_bookkeeping()
    test_mission_dataset()
    test_mission_steps_target()
    test_mission_actions()
    test_mission_from_tactic()
    test_dataset_cli_units(tmp_root)
    test_target_steps()
    test_terminal_signal()
    test_no_auxiliary_reward()
    test_block_progress()
    test_no_signal_band()
    test_dataset_trajectories(tmp_root)
    print(f"\n{CHECKS} checks passed (mission-level)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
