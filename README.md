# value-head-training

Trains a **value head** that predicts how many proof steps are left for a
**mission** — one `(env, goal, lean_context)` state, one agent response — for any
base model W, then uses mission-level MCTS + CE to internalise search. The estimate is used in three places: it initialises `Q` for unvisited
children and is backed up in PUCT; it is the CE value target; and it powers the
deployment gate — at `d̂ ≥ 500` proof steps the agent *must* decompose into child
missions instead of attempting the full proof.

One unit only: `d̂` is in proof steps, `N` (`action_block_steps`) is in proof
steps, and `rounds = ceil(d̂ / N)`.

Search unit is the mission, not the tactic step. One action is one agent
response: an **N-step proof block** (`action_block_steps`, default 10) or a
decomposition. A block that leaves `(context, goal)` unchanged is a no-op; one
that moves it is a state transition and triggers the next K-sample round.
Missions with `d̂ ≤ N` are single-shot and produce no learning signal.

The learning signal is **terminal-only**: no reward for decomposing, none for
calling a subagent. Only trajectories that discharge the original goal enter the
CE pool (failures stay as diagnostics). Trajectories are split into `main_pool`
(decisions + child summaries) and `sub_pool` (child transcripts), and CE on
`main_pool` is masked over system prompts, tool results and injected subagent
returns.

Staged: nothing is installed, `opencode.json` is untouched, no network is used.

```text
value-head-training/
  DESIGN.md              mission-level state/action/sampling/gate, two pools,
                         masking, reuse/caching, what changed vs step level
  README.md              this file
  SELFTEST.md            pasted test + tiny-run evidence
  index.ts               OpenCode V2 plugin: vht_plan / predict / gate / stage2 / status
  lib/gate.mjs|.d.mts    pure gate logic (shared with train/gate.py)
  traj.py                Trajectory, mask_spans, to_input_ids_loss_mask, split_pools,
                         is_terminal_success, filter_ce_pool, block_progress
  cache.py               mission registry + sibling dedupe + compile lock/cache
  mcts/
    SPEC.md              loop pseudocode + diagram (state -> K-sample -> wait),
                         N-step block, state-change test, no-signal band
    schedule.py          gate_action / should_sample / and_state / CURRICULUM
  train/
    dataset.py           bank/Mathlib/tiny -> mission records, steps target (default)
    head.py              backbone, 4 value encodings, losses, metrics, target unit
    train_stage1.py      SFT: value init + policy (lora | full), unit-aware
    train_stage2.py      MCTS + CE over searched trajectories (unit-aware)
    predict.py           score one mission state -> estimate + gate
    gate.py              gate rules (python twin of lib/gate.mjs)
  tests/
    gate.test.mjs        node --test (21 tests)
    test_python_plain.py plain asserts, no pytest (20 checks)
    test_mission_plain.py mission-level: traj/cache/schedule/dataset (22 checks)
    gate_cases.json      shared table, asserted by both gate suites
```

## Quickstart — CPU, no model, no corpus (2 s)

```bash
cd ./plugin/value-head-training

python3 train/dataset.py --tiny --out /tmp/vht/data.jsonl        # steps target
python3 train/train_stage1.py --tiny --out /tmp/vht/run          # dual head, full FT
python3 train/train_stage2.py --run /tmp/vht/run --tiny          # 40 trajectories
python3 train/predict.py --run /tmp/vht/run \
    --goal "theorem d : f0(a,b,c) + f1(a,b,c) = 0" --context "import Mathlib"
```

Stage 1 prints the numbers to watch:

```text
[stage1] scheme=lora encoding=dual unit=steps records=50 steps=450
[stage1] value MAE(log) = 0.6217  MAE(steps) = 2154.4  baseline = 1.8730  bucket_acc = 0.4  pass@1 = 1.0
[stage1] calibration monotone=True violations=0 gap=0.4629  tail>500 recall=1.0 n=7
```

`MAE(steps)` is in proof steps, and `tail>500` is the gate read as a tail event.

Two compatibility modes, both explicitly labelled:

```bash
python3 train/dataset.py --tiny --unit tactic --out /tmp/vht/tactic.jsonl   # legacy shape
python3 train/dataset.py --tiny --unit blocks --out /tmp/vht/blocks.jsonl  # DEPRECATED target
```

## Reuse: mission registry, dedupe, compile cache

```python
import cache
reg = cache.MissionRegistry("state/cache")
a = reg.get_or_create("bank:T1", "theorem foo : 1 + 1 = 2")   # reused: False
b = reg.get_or_create("bank:T1", "theorem foo : 1 + 1 = 2")   # reused: True, same workspace

out = cache.register_children(reg, "bank:T1", [{"goal": "...A"}, {"goal": "...A"}])
# len(out["missions"]) == 1, out["dropped"][0]["reused"] is True

cc = cache.CompileCache("state/cache")
cc.compile("bank:T1", "Problem.lean", proof)   # cache_hit: False
cc.compile("bank:T1", "Problem.lean", proof)   # cache_hit: True (flock + content address)
```

## Pools and masking

```python
import traj
pools = traj.split_pools(trajectories)          # main_pool / sub_pool
rec = pools["main_pool"][0]                     # no child internals, summaries only
ids = traj.to_input_ids_loss_mask(trajectories[0])
# loss_mask == 0 on system prompts, tool results, injected subagent returns, pad

# terminal-only: failures never reach CE
pool = traj.filter_ce_pool(trajectories)        # {"kept", "dropped", "reason_counts"}
assert pool["signal"] == "terminal_success_only"
```

## Stage 1 data — SFT from the shape-classified lemma bank

`train/build_sft.py` streams `bank_v2/lemma/T{1..5}.jsonl` (read-only) and writes
the stage-1 SFT set: one record per mission state, `mission_start` plus 3 prefix
variants per reference proof.

```bash
python3 train/build_sft.py --out ./data/sft --prefixes 3
# -> sft/stage1/T{1..5}-000.jsonl, sft/stats.json, sft/README.md

python3 train/train_stage1.py --tiny \
    --data ./data/sft/stage1 --stratify 300 \
    --max-steps 2000 --hidden 128 --value-hidden 256 --grid-max 1024 \
    --out state/runs/tiny-sft
```

```json
{"mission_id": "m-…", "env": "0df444a",
 "messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", "content": "by …"}],
 "value_target": 9, "value_target_kind": "mission_start",
 "tier": "T1", "problem_type": "lemma", "theorem_id": "…",
 "prefix_len": 0, "total_steps": 9}
```

`value_target` is remaining tactic steps (`total_steps - prefix_len`) — the same
unit as the gate. `--stratify N` is what makes a bounded CPU run informative: the
shards are tier-ordered, so the first N lines are dominated by one-step proofs.

## Scheme A — LoRA (secondary path)

Kept for the compare-against-full-FT case and for a frozen base; **full FT is the
default scheme** (`--scheme full`).

```bash
python3 train/dataset.py --bank-dir ./data/bank_local \
    --tier T1 --tier T2 --limit 60000 --out data/bank.jsonl

python3 train/train_stage1.py \
    --scheme lora --base /data/models/Qwen3.5-0.8B \
    --data data/bank.jsonl --encoding dual --value-hidden 256 \
    --epochs 2 --value-lr 3e-4 --policy-lr 1e-4 \
    --out runs/qwen08-lora
```

LoRA target modules must include the GDN projections (`in_proj_qkv`,
`in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`) or 3/4 of a Qwen3.5 model's
attention stays frozen; `head.attach_lora_hf` ships that list. If `peft` or
`transformers` is missing the script exits 2 with a one-line message.

## Scheme B — full-parameter

```bash
python3 train/train_stage1.py \
    --scheme full --base /data/models/Qwen3.5-4B \
    --data data/bank.jsonl --encoding dual --epochs 1 \
    --value-lr 3e-4 --policy-lr 1e-4 --stop-grad-value \
    --out runs/qwen4b-full
```

### Long contexts: goal-full + context pooling

The state is tokenised as `goal` and `context` **separately**; each is mean-pooled
with its own attention mask and the two pools are averaged. The default output is
still `hidden_size`, so existing checkpoints load unchanged.

| flag | default | meaning |
| --- | --- | --- |
| `--context-pool` | `tail` | `tail` = last `--context-window` tokens; `segments` = last `segments*window` tokens in window-sized chunks, mean of chunk pools; `mean` = whole context up to `--max-length`; `full` = legacy joined 2048-token truncation (A/B only) |
| `--goal-max-tokens` | 1024 | goal tokens kept from the front |
| `--context-window` | 3072 | tokens per context chunk |
| `--segments` | 4 | ceiling on context chunks (`segments` mode) |
| `--max-length` | 8192 | hard token budget for one state, goal + context |
| `--pool-combine` | `mean` | `mean` = elementwise mean (parameter-free); `projection` = learned `Linear(2H,H)`, which **adds `pool_proj.*`** to the checkpoint |

```bash
python3 tests/tests_context_pool_plain.py    # 16 checks, fake tokenizer/backbone
```

The test pins the bug directly: a signal at token 5000 changes the pooled vector
under `tail`/`segments`/`mean` and does not under `full` (2048 cap), and padding
never moves it.


Full FT moves the hidden state much faster than LoRA: keep `--stop-grad-value`
on and plan the re-alignment protocol from DESIGN.md §9 (affine recalibration +
re-freeze + gate safety receipt) on every checkpoint promotion.

## Stage 2 — MCTS + CE

Input is one searched node per line, the shape the MCTS-p2m runs write:

```json
{"mission_id": "m-…", "goal": "...", "context": "...", "action_type": "block",
 "block": "by\n  simp", "outcome": "success", "steps_remaining": 37,
 "visits": [{"tactic": "simp", "visits": 12}]}
```

`outcome` may be a word (`success` / `sorry`), a status dict
(`{"solved": true, "status": "accepted"}`) or absent; `visits` is optional (the
search posterior). Each record is encoded through `traj.to_input_ids_loss_mask`
(the masking rule: system / tool_result / subagent_return = 0, user + assistant
kept), and only terminal-success records enter the CE pool.

```bash
python3 train/train_stage2.py --run runs/qwen08-lora --tiny --dry-run      # shapes + masks, no torch
python3 train/train_stage2.py --run runs/qwen08-lora \
    --trajectories trajectories/wave_001.jsonl --steps 200 --resume
```

Stage 2 loads the stage-1 checkpoint (re-attaching the LoRA adapter), reads the
run's `unit` from `config.json`, does visit-weighted value CE + soft policy CE,
and writes `metrics_stage2.json`, `value-head-stage2.pt` and `stage2_state.json`.
`--resume` continues from that state file; `--dry-run` prints batch shapes, mask
spans and labels without importing torch.
Unknown actions are dropped and the visit distribution is renormalised — never
invent a policy class at stage 2.

`python3 train/predict.py` automatically prefers `value-head-stage2.pt`.

## Numbers to watch

| number | where | target | why |
| --- | --- | --- | --- |
| `mae_log` | `metrics.json → test` | ≤ 0.35 | primary value metric (scale free, steps) |
| `baseline_mae_log` | same | must be beaten | sanity floor (median predictor) |
| `tail.recall` | same | ≥ 0.8, n ≥ 200 | the gate's expensive failure |
| `coverage_80` | `test.interval` | 0.75–0.85 | interval honesty |
| `calibration.monotone` | `test.calibration` | true, 0 violations | d̂ must order missions |
| `bucket_acc` | `test` | ≥ 0.6 | resolution vs quantization |
| `policy_pass_at_1` | `test` | > always-block base rate | the action head must learn the gate |
| `gate.decompose_rate` | `metrics_stage2.json → after.gate` | ≈ `true_decompose_rate` | stage 2 must not move the gate |
| `reused` share | registry `stats()` | rising over a wave | duplication was real if this stays low |

Tiny mode's absolute numbers only prove the harness; see DESIGN.md §12.

## Q4 vs Q8 policy drift (is the gate safe on a quantised policy?)

If the policy runs at `Q4_K` while the value head was validated against `Q8_0`,
does the policy drift enough to invalidate the `d̂ ≥ 500` gate? `train/q4_q8_drift.py`
measures exactly that on real bank statements: same prompt, `temperature=0`,
`max_tokens=256`, one `lake env lean` per mission on the model's own proof.

```bash
# 1. real missions, reference proof verified standalone (easy + moderate strata)
python3 train/q4_q8_drift.py select --per-stratum 6 --tier T1 \
    --project /data/projects/lean-4.28-reap/env-0df444a \
    --out state/experiments/missions.jsonl

# 2. one endpoint at a time, under the GPU lease
flock -x -w 3600 /data/gpu.lock bash /tmp/vht-exp/one_endpoint.sh q4   # :8097
flock -x -w 3600 /data/gpu.lock bash /tmp/vht-exp/one_endpoint.sh q8   # :8098

# 3. judge + d_hat (CPU), then the Markdown table
python3 train/q4_q8_drift.py score --responses state/experiments/q4.jsonl \
    --head-run state/runs/tiny-sft --out state/experiments/q4_scored.jsonl
python3 train/q4_q8_drift.py report --q4 state/experiments/q4_scored.jsonl \
    --q8 state/experiments/q8_scored.jsonl --out state/experiments/q4_vs_q8_drift.md
```

`solved` is the Lean exit code (plus a `sorry` rejection), not a text match.
`d_hat` is **SYNTHETIC** unless the checkpoint was trained on the endpoint being
measured; only the Q4-minus-Q8 difference is interpreted. The run is recorded in
`state/experiments/q4_vs_q8_drift.md`.


## Plugin tools (installed)

`index.ts` exposes `vht_plan`, `vht_predict`, `vht_gate`, `vht_stage2`,
`vht_status` in the `vht` namespace. Each returns a compact JSON string.
Installed at `~/.opencode2/plugins/value-head-training` and registered in
`~/.opencode2/config/opencode/opencode.json`; `DEFAULT_ROOT` resolves from the
plugin file itself, so the copy trains against its own `train/` and `state/runs`
(override with `options.root` or `VHT_ROOT`). The plugin itself never edits
`opencode.json` and never calls the network.

```text
vht_plan({base, scheme, data?, out?, tiny?, prepare?})   -> run dir + command
vht_predict({goal, context, model, threshold?})          -> estimate + CI + gate
vht_gate({estimate, key?, threshold?, prev?, ci_high?})  -> action + reason (hysteresis persisted)
vht_stage2({run, trajectories?, tiny?, execute?, steps?})-> command or metrics
vht_status({run})                                        -> stage-1/2 metrics
```

## Tests

```bash
node --test tests/gate.test.mjs        # 21 tests
python3 tests/test_python_plain.py     # 20 checks (torch checks skipped if absent)
python3 tests/tests_context_pool_plain.py  # 16 checks (goal-full + context pooling)
python3 tests/test_mission_plain.py    # 29 checks (mission unit, pools, cache, trigger,
                                       #            terminal-only signal, N-step block, band,
                                       #            five gate unit cases, unit regression)
~/repos/opencode-llm-trace/node_modules/.bin/tsc -p tsconfig.json --noEmit
```

Without torch, `train_stage1.py`, `train_stage2.py` and `predict.py` exit 2 with
`PyTorch is not importable (...)`; `dataset.py`, `traj.py`, `cache.py`,
`mcts/schedule.py` and the gate keep working.
