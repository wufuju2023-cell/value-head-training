# SELFTEST — pasted evidence

**What changed and why (this revision).** Three things. (1) DESIGN §9 was a prose
argument about λ and Adam; it is now a measured treatment of **full-parameter
fine-tuning only** (LoRA dropped from scope), with eight interference modes, one
exact log quantity per mode, and a single numbered default recipe
(`python3 train/head.py --probe`, `--probe-recipe`). The headline measurement is
that λ is *not* the interference knob — AdamW normalises it away, a value-only
batch moves the shared trunk 1.76× further than a policy batch — and that the
dominant mode is policy→value drift: 20 policy-only steps move the trunk by 0.90
of its own norm and flip 25% of probe states across the 500-step gate while the
head is untouched. (2) Both 64-bucket defects are fixed, not argued: the terminal
state trained **bin 1** because `two_hot_batch` clamped to `1e-6` *and* used an
inverted interpolation weight, and 64 uniform bins could not name a 10k mission at
all; bin 0 is now `[0,1)` with centre exactly 0 (`encode(0)==0`, `decode(0)==0.0`)
and the dual head adds a bounded residual, binned-mode decoding and a tail flag on
the mode. (3) A new `train/ablate_head.py --tiny` measures six encodings on a
heavy-tailed fixture (20% >10k, 15% exactly 0, 5 seeds); dual(mode) wins on the two
columns the gate reads (zero-exact 1.000, tail recall 1.000, cov80 0.949) while
plain regression wins the bulk MAE and loses the tail (recall 0.731). New tests:
9 + 8 checks in `tests/test_head_geometry.py` and `tests/test_bucket_ablation.py`,
for **87 checks green** (21 node + 20 + 29 + 9 + 8). One transient cross-agent
breakage in `tests/test_python_plain.py` (another agent's concurrent
`train_stage1.py` / `dataset.py` rewrite) was reported to its owner and is
resolved — §23 keeps the record.

Machine: this Linux box, Python 3.14.4, torch 2.14.0+cu130 (CPU used), Node
v22.22.1. All commands run from
`./plugin/value-head-training`. No network, nothing installed,
`opencode.json` untouched.

**Unit contract:** `d_hat` is in proof steps; `N` (`action_block_steps`) is in
proof steps; `rounds = ceil(d_hat / N)`. One unit everywhere — head target, gate,
noise band, dataset, trajectories. `blocks` survives only as a deprecated shim.

Determinism: stage 1 seeds torch *before* building the model and pins one CPU
thread for `--tiny`, so the same seed gives bit-identical MAE (verified: three
runs at `--epochs 150` all reported 0.6217).

| # | command | result |
| --- | --- | --- |
| 1 | `python3 train/dataset.py --tiny --out …` | 50 records, unit=steps, target=remaining_steps |
| 2 | `python3 train/dataset.py --tiny --unit blocks …` | deprecated shim, warning on stderr |
| 3 | `python3 train/dataset.py --tiny --unit tactic …` | 50 tactic records, legacy shape |
| 4 | `python3 train/dataset.py --bank-dir … --tier T1 --limit 300` | 300 records / 95 missions, target steps |
| 5 | `python3 train/dataset.py --mathlib … --max-files 40` | 400 records / 85 missions, target steps |
| 6 | `python3 train/dataset.py --trajectories …` | 3 trajectories → 1 record (2 failures dropped) |
| 7 | `python3 train/train_stage1.py --tiny` | unit=steps, MAE(log) 0.6217, tail@500 1.0 (n=7) |
| 8 | `python3 train/train_stage1.py --tiny --unit blocks` | deprecated: MAE(log) 0.5269 (not comparable) |
| 9 | `python3 train/train_stage1.py --tiny --unit tactic` | legacy shape: MAE(log) 0.6217 (same target) |
| 10 | `python3 train/train_stage2.py --run … --tiny` | signal=terminal_success_only, MAE 0.606 → 0.232 |
| 11 | `python3 train/predict.py …` | 7 steps → prove; 15734 steps → decompose (8 children) |
| 12 | `node --test tests/gate.test.mjs` | 21/21 pass |
| 13 | `python3 tests/test_python_plain.py` | 20/20 checks pass |
| 14 | `python3 tests/test_mission_plain.py` | 29/29 checks pass |
| 15 | `tsc -p tsconfig.json --noEmit` | exit 0 |
| 16 | `PYTHONPATH=<torch-stub> python3 train/*.py` | stage1/2/predict exit 2; dataset still works |

---

## 1. dataset.py --tiny (default unit = steps)

```text
$ python3 train/dataset.py --tiny --out /tmp/u2/steps.jsonl
{"ok": true, "out": "/tmp/u2/steps.jsonl", "records": 50, "unit": "steps",
 "deprecated": false, "target": "remaining_steps", "sources": ["tiny"], "envs": ["tiny"],
 "missions": 50, "remaining_steps_p50": 2090, "remaining_steps_p90": 19997,
 "remaining_steps_max": 20000, "over_500": 32, "decompose_actions": 32,
 "block_actions": 18, "children_total": 225, "success": 50, "failed": 0, "terminal": 0}
```

One record (first line, wrapped):

```text
{"id": "tiny-000", "unit": "steps", "source": "tiny", "env": "tiny",
 "mission_id": "m-8d5da414b093", "context": "import Mathlib\npositivity",
 "goal": "theorem tiny_00 (a b c : ℕ) : f0(a, b, c) + … = g0(a, b, c) + …",
 "action_type": "decompose", "next_action": "decompose", "n_child_missions": 4,
 "remaining_steps": 1615, "depth": 0, "parent_mission_id": null,
 "status": "accepted", "success": true, "terminal": false}
```

No `remaining_blocks`, no `d_hat_target`, no ceil anywhere on the default path.

## 2. dataset.py --unit blocks (deprecated)

```text
$ python3 train/dataset.py --tiny --unit blocks --out /tmp/u2/blocks.jsonl
dataset: --unit blocks is DEPRECATED (the target is proof steps; a blocks target is
only for comparing old checkpoints)
{"ok": true, "out": "/tmp/u2/blocks.jsonl", "records": 50, "unit": "blocks",
 "deprecated": true, "target": "remaining_blocks", "missions": 50,
 "remaining_blocks_p50": 697, "remaining_blocks_p90": 6666,
 "remaining_blocks_max": 6667, "over_500": 31, "decompose_actions": 32, …}
```

The shim exists so old checkpoints can be compared; it prints its deprecation on
stderr and `summarize` marks it `"deprecated": true`.

## 3. dataset.py --unit tactic (legacy record shape, same target)

```text
$ python3 train/dataset.py --tiny --unit tactic --out /tmp/u2/tactic.jsonl
{"ok": true, "out": "/tmp/u2/tactic.jsonl", "records": 50, "unit": "tactic",
 "sources": ["tiny"], "remaining_p50": 2090, "remaining_p90": 19997,
 "remaining_max": 20000, "over_500": 32, "over_10k": 19, "terminal": 0}
```

## 4. Real prove2me bank, steps unit

```text
$ python3 train/dataset.py --bank-dir ./data/bank_local \
    --tier T1 --limit 300 --out /tmp/u2/bank.jsonl
{"ok": true, "out": "/tmp/u2/bank.jsonl", "records": 300, "unit": "steps",
 "deprecated": false, "target": "remaining_steps", "sources": ["bank"], "envs": ["bank"],
 "missions": 95, "remaining_steps_p50": 1, "remaining_steps_p90": 6,
 "remaining_steps_max": 11, "over_500": 0, "decompose_actions": 0, "block_actions": 300,
 "children_total": 0, "success": 300, "failed": 0, "terminal": 95}
```

300 tactic prefixes → 300 mission states over 95 missions: every prefix is a
state, and the target is the remaining step count verbatim. Reference proofs are
short (p50 = 1 step); the >500-step tail still has to come from MCTS
trajectories, not from the bank.

## 5. Mathlib corpus, steps unit

```text
$ python3 train/dataset.py --mathlib /data/projects/lean-corpus/mathlib4/Mathlib \
    --max-files 40 --mathlib-limit 400 --out /tmp/u2/mathlib.jsonl
{"ok": true, "out": "/tmp/u2/mathlib.jsonl", "records": 400, "unit": "steps",
 "deprecated": false, "target": "remaining_steps", "sources": ["mathlib"],
 "envs": ["mathlib"], "missions": 85, "remaining_steps_p50": 2,
 "remaining_steps_p90": 6, "remaining_steps_max": 15, "over_500": 0,
 "decompose_actions": 0, "block_actions": 400, "children_total": 0,
 "success": 400, "failed": 0, "terminal": 85}
```

## 6. Terminal-only signal: failed trajectories are dropped

Fixture = 1 success (`block`, solved), 1 failure (`solved: false`), 1 budget
exhaustion (`solved: true, budget_exhausted: true`):

```text
$ python3 train/dataset.py --trajectories /tmp/u2/trajs.jsonl --out …
{"ok": true, "records": 1, "unit": "steps", "missions": 1, "success": 1, "failed": 0,
 "terminal": 1, "decompose_actions": 0, "block_actions": 1, "children_total": 0, …}

$ python3 train/dataset.py --trajectories /tmp/u2/trajs.jsonl --include-failed --out …
{"ok": true, "records": 3, "unit": "steps", "missions": 2, "success": 1, "failed": 2,
 "terminal": 3, "decompose_actions": 2, "block_actions": 1, "children_total": 2, …}
```

`missions: 2` for 3 records: the two failures share a goal, so they resolve to the
same content-addressed `mission_id`. Failures stay visible as diagnostics
(`success: false`, `status: failed`) and never become loss.

## 7. Stage 1, dual encoding, LoRA, steps unit (CPU, 1.0 s)

```text
$ python3 train/train_stage1.py --tiny --out /tmp/u2/run
[stage1] scheme=lora encoding=dual unit=steps records=50 steps=450
[stage1] value MAE(log) = 0.6217  MAE(steps) = 2154.4  baseline = 1.8730  bucket_acc = 0.4  pass@1 = 1.0
[stage1] calibration monotone=True violations=0 gap=0.4629  tail>500 recall=1.0 n=7
[stage1] run dir: /tmp/u2/run  (1.01s)
```

`MAE(steps)` is in proof steps and `tail>500` is the gate read as a tail event.
`pass@1 = 1.0` is the mission-level policy (`block` vs `decompose`), learnable
from size alone in the tiny generator — plumbing, not evidence (§12 of DESIGN).

## 8. Stage 1, deprecated blocks target (for comparison only)

```text
$ python3 train/train_stage1.py --tiny --unit blocks --out /tmp/u2/run-blocks
[stage1] scheme=lora encoding=dual unit=blocks records=50 steps=450
[stage1] value MAE(log) = 0.5269  MAE(steps) = 412.9  baseline = 1.8401  bucket_acc = 0.3  pass@1 = 1.0
[stage1] calibration monotone=True violations=0 gap=0.3797  tail>500 recall=0.8571 n=7
```

These numbers are **not** comparable to §7: the target is `ceil(steps/3)`, so both
the scale and the tail threshold move. This is exactly the confusion that produced
the unit bug; the shim prints a deprecation warning and `metrics.unit` records it.

## 9. Stage 1, legacy tactic record shape (same target as §7)

```text
$ python3 train/train_stage1.py --tiny --unit tactic --out /tmp/u2/run-tactic
[stage1] value MAE(log) = 0.6217  MAE(steps) = 2154.4  baseline = 1.8730  bucket_acc = 0.4  pass@1 = 0.8
[stage1] calibration monotone=True violations=0 gap=0.4629  tail>500 recall=1.0 n=7
```

Same MAE as §7 by construction: the target unit did not change, only the record
shape (and therefore the policy vocabulary: `tac_*` instead of `block|decompose`).

## 10. Stage 2, terminal-only CE (CPU, 0.6 s)

```text
$ python3 train/train_stage2.py --run /tmp/u2/run --tiny
[stage2] signal=terminal_success_only terminal-success trajectories=40 dropped=0 {}
[stage2] trajectories=40 value MAE(log) 0.6056 -> 0.232
[stage2] policy agree(search posterior) 0.975 -> 1.0  gate decompose_rate 0.65 (true 0.65)
[stage2] run dir: /tmp/u2/run (0.6s)
```

The `signal=` line is printed on every run: if the input contains no
terminal-success trajectory, stage 2 exits 2 with `nothing to distil` instead of
training on failures.

## 11. predict.py — steps unit, both sides of the gate

```text
$ python3 train/predict.py --run /tmp/u2/run \
    --goal "theorem demo (a b c : N) : f0(a,b,c) + f1(a,b,c) = 0" --context "import Mathlib"
{"ok": true, "estimate": 7.07, "ci_low": 7.0, "ci_high": 8.33, "tail_saturated": false,
 "gate": {"action": "prove", "reason": "below_threshold", "estimate": 7.074,
          "threshold": 500.0, "enter": 500.0, "exit": 400.0, "prev": "prove",
          "tail_saturated": false, "suggested_children": 0},
 "model": {"run": "/tmp/u2/run", "checkpoint": "value-head-stage2.pt",
           "encoding": "dual", "unit": "steps", "block_steps": 3,
           "scheme": "lora", "base": "hashed"}}
```

A 20-term statement, same checkpoint:

```text
{"estimate": 15734.62, "ci_low": 3250.07, "ci_high": 17695.25, "tail_saturated": false,
 "gate": {"action": "decompose", "reason": "above_threshold", "estimate": 15734.616,
          "threshold": 500.0, "enter": 500.0, "exit": 400.0, "prev": "prove",
          "tail_saturated": false, "suggested_children": 8}}
```

7 steps → prove; 15734 steps → decompose into 8 children. Both numbers are in the
same unit as the threshold, so `d̂ ≥ 500` is a statement about work.

## 12. Module self-checks (no torch)

```text
$ python3 traj.py
{"tokens": 25, "kept": 9, "masked": 16, "spans": [[0, 4], [13, 25]]}

$ python3 cache.py
{"same_id": true, "reused": [false, true, false],
 "missions": {"missions": 2, "created": 2, "reused": 1, "by_status": {"pending": 2}},
 "cache_hits": [false, true],
 "compile_stats": {"compiles": 2, "cache_hits": 1, "misses": 1, "hit_rate": 0.5, "envs": ["tiny"]}}

$ python3 mcts/schedule.py
{'tick':         {'sample': False, 'reason': 'timer_is_not_a_trigger', ...},
 'child_return': {'sample': True,  'reason': 'child_return', 'k': 4, 'mode': 'multi_round'},
 'pending':      {'sample': False, 'reason': 'children_pending', 'wait_for': 'child_return'},
 'gate_900':     {'action': 'decompose', 'reason': 'above_threshold', 'mandatory': True,
                  'action_block_steps': 10, 'rounds_left': 90, 'single_shot': False,
                  'suggested_children': 2},
 'gate_120_capped': {'action': 'block', 'reason': 'depth_capped', 'mandatory': False}}
```

## 13. The five gate cases (pasted from the suite)

```text
$ python3 tests/test_mission_plain.py
ok 16 - gate: 900->decompose(9), 400->prove(4), 80->single_shot, 480 stays decomposed
ok 17 - regression: 900 steps decomposes; reading it as 300 blocks was the bug
```

| case | input | expected | actual |
| --- | --- | --- | --- |
| 1 | `d̂=900, N=100` | decompose, rounds 9 | `decompose`, `rounds_left=9`, `single_shot=False` |
| 2 | `d̂=400, N=100` | prove, 4 rounds, not single-shot | `block`, `rounds_left=4`, `single_shot=False`, `learning_signal=True` |
| 3 | `d̂=80, N=100` | single-shot, no signal | `block`, `rounds_left=1`, `single_shot=True`, `learning_signal=False` |
| 4 | `d̂=480, N=100, prev=decompose` | stays decompose (hysteresis 400) | `decompose`, `reason=hysteresis_hold` (and 400 → `hysteresis_exit`) |
| 5 | `rounds_left` for 99/100/101/150/1000 at N=100 | 1/1/2/2/10 | `[1, 1, 2, 2, 10]` |

Regression (`test_unit_regression`): a 900-step mission now targets `900`, so the
gate decomposes (mandatory). The old reading `ceil(900/3) = 300` is asserted to
produce `block`, which is the bug — 900 steps would have been attempted inline.

## 14. Encoding comparison, 3 seeds, steps unit (tail @500)

*Historical: measured before the bucket fixes and at a 500-step tail, where
every arm scores recall 1.0 — the column cannot discriminate. Superseded by
§21, which measures at the 10k tail, adds the zero-step column and the 32k-span
uniform control, and runs 5 seeds on a heavy-tailed fixture.*

```text
$ for enc in dual log uniform regress; do for seed in 7 11 23; do
    python3 train/train_stage1.py --tiny --seed $seed --encoding $enc --out /tmp/u-enc/$enc-$seed
  done; done

dual     s7   0.6217  mae_steps=2154.4  base=1.8730  acc=0.4  tail=1.0 n=7  cov80=0.7 sat=0
dual     s11  1.2167  mae_steps=6508.0  base=1.7287  acc=0.0  tail=1.0 n=7  cov80=0.8 sat=0
dual     s23  0.8734  mae_steps=4370.8  base=2.2117  acc=0.3  tail=1.0 n=6  cov80=0.7 sat=0
log      s7   0.4736  mae_steps=2332.5  base=1.8730  acc=0.5  tail=1.0 n=7  cov80=0.2 sat=0
log      s11  0.3510  mae_steps=2075.0  base=1.7287  acc=0.2  tail=1.0 n=7  cov80=0.5 sat=0
log      s23  0.8853  mae_steps=3066.6  base=2.2117  acc=0.3  tail=1.0 n=6  cov80=0.5 sat=0
uniform  s7   1.7385  mae_steps=10845.8 base=1.8730  acc=0.7  tail=1.0 n=7  cov80=0.1 sat=6
uniform  s11  1.3581  mae_steps=6316.3  base=1.7287  acc=0.7  tail=1.0 n=7  cov80=0.3 sat=7
uniform  s23  1.3299  mae_steps=5988.4  base=2.2117  acc=0.6  tail=1.0 n=6  cov80=0.2 sat=6
regress  s7   0.8077  mae_steps=6100.0  base=1.8730  acc=0.0  tail=1.0 n=7  cov80=0.3 sat=0
regress  s11  0.4848  mae_steps=3355.2  base=1.7287  acc=0.0  tail=1.0 n=7  cov80=0.4 sat=0
regress  s23  0.9202  mae_steps=8365.8  base=2.2117  acc=0.0  tail=1.0 n=6  cov80=0.3 sat=3
```

At the 500-step tail every encoding hits recall 1.0 on this data, so the tail
column no longer discriminates; coverage does. Uniform saturates on 6–7 of 10
held-out states and covers 0.1–0.3 of the truths — disqualified on the same
grounds as before. n = 10, so nothing is decided here; the acceptance run is ≥5k
held-out states.

## 15. Gate tests — `node --test` (unchanged shared table)

```text
$ node --test tests/gate.test.mjs
1..21
# tests 21
# pass 21
# fail 0
# duration_ms 141.677782
```

## 16. Python tests

```text
$ python3 tests/test_python_plain.py
ok 01 - split_tactic_steps keeps nested blocks attached to their step
ok 02 - split_tactic_steps handles empty/`by`-only proofs
ok 03 - records_from_proof: remaining_steps = n - i and a terminal 0 record exists
ok 04 - tiny dataset: 50 deterministic records, max remaining=20000
ok 05 - dataset.py --tiny CLI writes 50 mission records targeting proof steps
ok 06 - bucket_edges: 64 monotone bins, bin 0 starts at 0, log grid even in log space
ok 07 - two_hot: adjacent bins for mid values, exact top/bottom clamp at 0 and 10^9
ok 08 - decode_expected / quantiles: one-hot recovers the bucket centre
ok 09 - metrics: perfect scores 0 MAE / 1.0 tail recall; a constant 3 misses the tail
ok 10 - metrics: no >10k samples reports tail n=0 instead of inventing a recall
ok 11 - gate: 17 shared cases match lib/gate.mjs expectations
ok 12 - gate: 500 enters decompose, 400 releases the hysteresis band
ok 13 - torch Model: dual head shapes, finite loss, gradient flows
ok 14 - two_hot_batch: 0 / 20000 / grid-top all land inside 64 valid bins
ok 15 - predict_texts: estimate, 80% interval, saturation flag for both texts
ok 16 - all 4 encodings: non-negative point estimate, ordered interval, finite loss
ok 17 - dataset: default unit is steps, summarize reports the step target + action split
ok 18 - head: target_steps defaults to remaining proof steps; blocks is the shim
ok 19 - train_stage1 --tiny: trains on remaining proof steps, reports unit in metrics
ok 20 - train_stage1.make_batch: mission steps = remaining_steps, vocab = block|decompose

20 checks passed (python)
```

```text
$ python3 tests/test_mission_plain.py
ok 01 - HashTokenizer: deterministic, in-vocabulary token ids
ok 02 - traj.mask_spans: system + tool_result + subagent_return masked, user/assistant kept
ok 03 - traj.to_input_ids_loss_mask: aligned ids/mask, spans sum to n_masked, pad masked
ok 04 - traj mask_policy: configurable spans, keep_kinds always wins over mask_kinds
ok 05 - traj.split_pools: main_pool keeps summaries only, sub_pool keeps full trajectories
ok 06 - traj.children_pending: a dispatched child that has not returned keeps the parent waiting
ok 07 - cache: sha256(env, statement_formal) key, whitespace-stable, env-sensitive
ok 08 - cache.MissionRegistry: 2 identical + 1 different -> 2 missions, reuse survives reload
ok 09 - cache.register_children: identical siblings collapse, second decomposition reuses ids
ok 10 - cache.MissionRegistry.set_status: validated statuses, unknown mission rejected
ok 11 - cache.CompileCache: identical content hits once, different content/env -> new dir
ok 12 - cache.CompileCache: 4 concurrent compiles of one file -> 1 runner call, 3 hits
ok 13 - schedule.should_sample: child_return / block_finished trigger, tick never does
ok 14 - schedule.should_sample: no expansion while children pending / samples in flight
ok 15 - schedule.gate_action: d>500 must decompose, depth cap forces a block
ok 16 - gate: 900->decompose(9), 400->prove(4), 80->single_shot, 480 stays decomposed
ok 17 - regression: 900 steps decomposes; reading it as 300 blocks was the bug
ok 18 - schedule.and_state: AND recorded, parent value aggregates by max over pending
ok 19 - dataset: --unit steps (default) emits 7 mission fields, target is steps
ok 20 - dataset: steps target is exact (no ceil); blocks survives only as the shim
ok 21 - dataset: decompose iff remaining steps >= 500, 2..8 children, block has 0
ok 22 - dataset.mission_from_tactic: every prefix, remaining_steps = total - i
ok 23 - dataset CLI: steps default; tactic legacy; blocks deprecated with a warning
ok 24 - head.target_steps: steps is the default, blocks is explicit and deprecated
ok 25 - traj: CE pool keeps terminal success only; failures stay as diagnostics
ok 26 - traj: decomposition/subagent/credit reward keys are rejected, not ignored
ok 27 - traj.block_progress: a block that moves the goal/context is progress, identical is a no-op
ok 28 - schedule: d̂ <= N in steps is a one-shot, skipped by the round scheduler
ok 29 - dataset --trajectories: successes only by default, --include-failed keeps diagnostics

29 checks passed (mission-level)
```

Totals: **21 node + 20 + 29 + 9 + 8 python = 87 checks**, all passing (70 from
the previous revision, plus 9 head-geometry and 8 bucket-ablation; the two
`test_python_plain.py` checks that another agent's concurrent `train_stage1.py` /
`dataset.py` rewrite had broken are green again after they restored the keys — see
§23).

## 17. Torch missing → clean exit, no traceback

```text
$ PYTHONPATH=/tmp/vht-notorch2 python3 train/train_stage1.py --tiny --out /tmp/x
value-head-training/stage1: PyTorch is not importable (ImportError: simulated: torch not installed).
Install torch in the training venv; --tiny runs on the CPU wheel. Aborting without training.
stage1 exit=2

$ PYTHONPATH=/tmp/vht-notorch2 python3 train/train_stage2.py --run … --tiny   -> exit 2
$ PYTHONPATH=/tmp/vht-notorch2 python3 train/predict.py --run … --goal x    -> exit 2
$ PYTHONPATH=/tmp/vht-notorch2 python3 train/dataset.py --tiny --out /tmp/x.jsonl
{"ok": true, ..., "records": 50, ...}        # dataset, traj, cache, schedule keep working
```

`python3 tests/test_mission_plain.py` also passes under the stub (29/29), because
nothing in the mission-level core imports torch.

## 18. Type check

```text
$ tsc -p tsconfig.json --noEmit
tsc exit=0
```

## 19. Interference probe, full FT (`train/head.py --probe`)

Every number DESIGN §9 quotes. 240 synthetic mission rows (72 held out), full FT,
AdamW `lr 3e-3`, 40 warm-up + 20 **policy-only** burst steps, seed 7:

```text
$ python3 train/head.py --probe        # 2.4 s, CPU, 1 thread
  M1 grad share (shared params only)
     g_value 4.8e-04   g_policy 4.08e-01   value_share 1.37e-06
     per tensor: proj.weight 3e-06, proj.bias 2e-06, len_proj.weight 1e-06,
                 len_proj.bias 2e-06, norm.weight 3e-06, norm.bias 5e-06
  M2 policy->value drift (20 steps, lambda = 0 => head provably not trained)
     |dh|/|h|              0.9024
     value_loss frozen     3.0114 -> 3.0675   (+1.9%)
     median |dlog1p d_hat| 0.0014
     max    |dlog1p d_hat| 9.4099
     gate flips @500       18 / 72  (0.25)
  M3 grad cosine (terms)              0.2173
  M3b AdamW step cosine (same state)  0.3477
      |dvalue step| / |dpolicy step|  1.7583   (0.0762 / 0.0433)
  M4 lambda_eff  |g_value| n16 1.5756 vs n4 2.3589  ratio 1.497
      lambda_eff at n_v=4, N=16 = 4e-03 (4x nominal)
  M5 terminal    encode(0)=0 decode(0)=0.0  (uniform: 0 / 0.0)
  M6 EMA (beta 0.99, all float params)
                 raw        EMA
     value_loss  3.0675     2.9995
     median dlog 0.0014     0.0006
     max dlog    9.4099     8.6400
     gate flips  18/72      4/72
```

Read the flips, not the median: the median says the head barely moved, while a
quarter of the decisions changed side of the gate. The 1.37e-06 share is why
`--stop-grad-value` is an *alarm response*, not a default — and why the share must
be computed on shared parameters only (over all parameters it is ≈1.0, because the
value loss is the only thing touching `value_*`).

## 20. Capacity and LR-ratio grid (`train/head.py --probe-recipe`)

```text
$ python3 train/head.py --probe-recipe    # 3 s
 value_hidden  ratio  mae_log  mae_steps  policy_acc  dhat_drift  flips  params
 32            1.0    1.2188   2403       0.9722      0.0040       0       7 507
 96            1.0    0.4769   1652       0.9722      0.0010       3      23 059
 256           1.0    0.3702   1519       0.9722      0.0171       1      97 779
 96            0.1    2.0792   2534       0.9722      0.0000       3      23 059
 96            3.0    0.3969   1664       0.9722      0.0023       3      23 059
```

Capacity in the **value tower** is nearly free (−70% MAE from width 32 → 256, policy
accuracy bit-identical at all three). Capacity in the **trunk** is not, which is why
the recipe widens the tower and never the trunk. On LR: 0.1× is not "stable", it is
undertrained (4.4× the MAE of 1×); 3× is 17% better with no measured stability cost
over this horizon, and is the documented first escalation rather than the default.

## 21. 64-bucket ablation (`train/ablate_head.py --tiny`)

Fixture: 300 train / 150 held-out synthetic missions, 20% at 10k..32768, 15%
exactly 0, text→steps with bounded noise; 5 seeds; 120 optimizer steps; one trainer
for all arms.

```text
$ python3 train/ablate_head.py --tiny --seeds 5
n_train=300 n_test=150 seeds=[7, 11, 23, 37, 41] steps=120 tail_at=10000 buckets=64 uniform_span=2048

| arm | MAE(log) | MAE(steps) | zero-exact | tail recall@10k | tail MAE(log) | flag@tail | cov80 | mono |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uniform64 | 1.0063 | 4046.8 | 1.000 | 0.000 | 2.2303 | 1.000 | 0.413 | 5/5 |
| uniform64@32k | 1.2281 | 2018.2 | 1.000 | 0.979 | 0.3724 | 0.000 | 0.584 | 5/5 |
| log64 | 0.4282 | 2542.5 | 1.000 | 1.000 | 0.4079 | 0.000 | 0.913 | 4/5 |
| dual(mode) | 0.4321 | 2511.0 | 1.000 | 1.000 | 0.4194 | 0.000 | 0.949 | 4/5 |
| dual(expected) | 0.3218 | 2021.9 | 0.786 | 1.000 | 0.4170 | 0.000 | 0.949 | 5/5 |
| regress | 0.2784 | 1842.4 | 0.234 | 0.731 | 0.5114 | 0.000 | 0.760 | 5/5 |

[ablate] dual(mode) vs log64: ok (bulk MAE(log) gain -0.9%)
[ablate] fallback rule -> KEEP dual(mode)
```

`flag@tail` = share of the >10k rows whose mode is the last bin. `uniform64` flags
all of them, which is the only reason it is safe: it cannot name a 10k mission, but
it knows it cannot, and the gate decomposes.

Winner: **dual(mode)**. It ties log64 on bulk MAE (within 1%), wins 80% coverage
(0.913 → 0.949), keeps terminal exactness at 1.000 (the expectation decoder drops
to 0.786), and the bounded residual is what gives the top three bins any resolution
at all. `regress` is the honest counter-example: best bulk MAE (0.2784) and
disqualified on exactly the two columns the gate reads (tail recall 0.731,
zero-exact 0.234).

**Fallback rule** (implemented in `ablate_head.verdict`): ship plain log64 instead
of dual only if, on ≥5k held-out records, dual is worse on any of `tail_recall@10k`
by >0.02, `tail MAE(log)` by >0.02, `cov80` by >0.03, or `zero_exact` at all. Bulk
MAE is a tie-breaker, never the decision, because the gate is a threshold on a tail
quantity with asymmetric cost.

## 22. New test suites (B1 and B2)

```text
$ python3 tests/test_head_geometry.py
ok 01 - bin 0 is [0,1) with centre exactly 0 in uniform64 / log64 / dual
ok 02 - encode(0)==0, decode(0)==0.0 and encode is monotone over 0..5 and 0..32768
ok 03 - log64 covers [1, 32768] in 63 even log1p steps, with [0,1) reserved
ok 04 - decode_mode: a terminal mode decodes to exactly 0 and bin 0 never biases 1-step modes
ok 05 - mode decodes terminal exactly where the expectation cannot (this is why B1 needs mode)
ok 06 - two_hot_batch: d=0 -> bin 0 with weight 1.0 (was bin 1), grid top -> bin 63
ok 07 - two_hot_batch agrees with pure two_hot on 0 / 0.5 / grid top (no drift between twins)
ok 08 - Model.value_point: bin-0 mode with max residual still decodes to exactly 0.0
ok 09 - tail_flag fires on the last bin only; the saturated head now decomposes

9 checks passed (head geometry / zero-step)
```

```text
$ python3 tests/test_bucket_ablation.py
ok 01 - dual grid: 63 log1p-even bins over [1, 32768] after the terminal bin, top bin > 1k wide
ok 02 - residual is bounded: +-0.5 bin in log space, so it cannot leave the bin
ok 03 - binned mode keeps a 55/62 bimodal head on the tail; the expectation names an empty middle bin
ok 04 - binned mode is a 3-bin local mean around the argmax, not a global average
ok 05 - tail flag = mode in the last bin; it forces decompose without a value test
ok 06 - predict_texts exposes mode_bucket (0..63) and a boolean tail_saturated
ok 07 - ablation: uniform64 has tail recall 0, dual(mode) 1.0 + zero-exact 1.0, regress fails both
ok 08 - fallback rule: bulk MAE is a tie-breaker; a tail or terminal regression forces log64

8 checks passed (bucket tail / ablation)
```

Check 06 is the regression pin for B1: `two_hot_batch(d=0)` used to return
`(bin 0, weight 0)` — i.e. all mass on bin 1 — because the input was clamped to
`xs[0]+1e-6` and the interpolation weight was inverted. The test asserts bin 0 with
weight 1.0 and that the torch twin and the pure `two_hot` agree on `0`, `0.5` and
the grid top.

## 23. The one transient cross-agent breakage (resolved)

`train/train_stage1.py` and `train/dataset.py` were rewritten by another agent
(they own both) while this revision was in flight. For ~30 minutes two checks in
`tests/test_python_plain.py` failed for reasons unrelated to the head:

```text
AssertionError: metrics["data"]["unit"] -> KeyError: 'unit'             (test 19)
TypeError: build_vocab() takes 1 positional argument but 2 were given   (test 20)
```

Cause: stage 1 had switched to `dataset.summarize_sft(records)` (which dropped the
`unit`/`target` keys) and `build_vocab` lost its second positional argument. Both
are outside this file set (`DESIGN.md`, `train/head.py`, `train/gate.py`,
`tests/*head*`, `tests/*bucket*`), so they were reported to the owning agent rather
than patched here. They restored both; the current run is 20/20:

```text
$ python3 tests/test_python_plain.py
ok 18 - head: target_steps defaults to remaining proof steps; blocks is the shim
ok 19 - train_stage1 --tiny: trains on remaining proof steps, reports unit in metrics
ok 20 - train_stage1.make_batch: mission steps = remaining_steps, vocab = block|decompose

20 checks passed (python)
```

Lesson kept rather than hidden: the two pinned contracts (`metrics["data"].unit`,
`build_vocab(records, unit)`) are what caught a silent change of the target's
provenance, which is exactly the class of bug §1 of DESIGN exists to prevent.

## Known gaps (deliberate)

- The plugin is staged: `index.ts` is type-checked and its gate core is unit
  tested, but the five tools were not exercised inside a running OpenCode host
  (that would mean installing it, out of scope). `index.ts` is unchanged by the
  mission-level revision.
- Tiny numbers are a harness signal (DESIGN.md §12), not value accuracy.
- The ablation fixture is **synthetic on purpose**: the real bank has no tail
  (p50 = 1 remaining step, max 15), so it cannot rank encodings for the tail at
  all. What the fixture cannot tell you is the *shape* of the real tail; the ≥5k
  held-out acceptance run (§21 rule, DESIGN §11) is still the decision.
- The interference numbers are harness scale: a 48-wide hashed trunk, 20-step
  bursts. The ranking (§9) is what transfers; the magnitudes are not predictions
  for a 9B model.
- `--probe` and `--probe-recipe` own their own forward passes and step the
  optimizer, so they measure a model, not a training run; they are diagnostics,
  not an acceptance test.
- The HF `--scheme lora|full` path is a skeleton: it needs `transformers` /
  `peft` and real weights, and was not run here (no network, no GPU).
- The MCTS loop itself is specified (`mcts/SPEC.md`) and its three decisions are
  unit-tested (`mcts/schedule.py`), but no live agent driver was run: there is no
  OpenCode host in this staged tree, and the compile cache only calls its
  injectable runner (the default runner writes the `.olean` marker).
