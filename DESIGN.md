# DESIGN — value head for "how many steps are left?"

Status: staged, not installed. Everything here is runnable from the plugin
directory on CPU (`--tiny`), and every claim that can be measured locally is
marked as measured, with the number and the command that produced it.

Scope: given a base model W (any causal LM), learn

    d̂(mission state) ≈ number of remaining proof steps for that mission

where a **mission** is one (env, goal) pair with a Lean context, and the search
action is an **N-step block**: one agent response that contributes at most
`action_block_steps` proof steps. The head's only deployment job is the **gate**:
at `d̂ > 500` steps the agent *must* decompose into child missions instead of
attempting the full proof (§5).

    INVARIANT: d_hat is in proof steps; N is in proof steps; rounds = ceil(d_hat/N).

There is exactly one unit in this plugin — proof steps — for the head target, the
gate, the noise band, the dataset and the trajectories. An earlier revision mixed
"blocks" into the target and steps into the gate; that made a 900-step mission
look like 300 blocks and pass the 500 gate without decomposing. §10 records the
fix, and `tests/test_mission_plain.py::test_unit_regression` pins it.

| stage | what moves | objective | output |
| --- | --- | --- | --- |
| 1 | value head + policy (LoRA or full FT) | `1e-3 * value_loss + policy_CE` on prove2me + Mathlib proofs | calibrated head at W0 |
| 2 | policy + value | MCTS posterior: visit-weighted value CE + soft policy CE, masked | head aligned with the *searched* policy |

The head never certifies a proof, and its estimate is never a reward.

---

## 1. Two different questions: state granularity vs target unit

They were conflated once, and that is the bug this revision fixes.

**State granularity = mission.** A human writes proof blocks (`have …`,
`constructor` plus two `·` branches, a five-line `calc`), and so does the agent:
one assistant response is a whole attempt. Between responses it re-reads the
goal, so the response — not the single tactic — is the decision unit. Hence one
record per mission state, one action per agent response.

**Target unit = proof steps.** The quantity being estimated is the remaining
amount of proof *work*, and the only unit that is stable across block sizes,
curricula and agent verbosity is the tactic step:

    d̂ = remaining_steps          # total_tactic_steps - i, log1p-encoded

An N-step block is the *action*, not the measurement. It changes how much work
one response may do (`action_block_steps = 10`, rising with the curriculum), and
therefore how many rounds are left:

    rounds_left = ceil(d̂ / N)

Keeping the target in steps means the gate does not move when N moves: raising N
from 10 to 100 must not silently redefine "500". This is exactly what went wrong
when the target was `ceil(steps / 3)`: 900 steps became "300", which is under the
500 gate, so the agent attempted a 900-step proof inline.

`--unit blocks` still exists as a **deprecated** shim (`ceil(steps/block_steps)`,
one warning on stderr) only so old checkpoints can be compared. `--unit tactic`
keeps the legacy record shape with the same step target. Everything defaults to
`--unit steps`.

What this changes is listed in §10.

---

## 2. Mission-level state model

```text
State := {
  env            : str          # lean project / task env ("bank:T1", "mathlib", ...)
  mission_id     : str          # sha256(env, statement_formal) -> "m-" + 12 hex
  lean_context   : str          # imports + preamble + what is already proved
  goal           : str          # statement_formal of THIS mission
  depth          : int          # 0 at the root
  parent         : str | None   # parent mission id
}
```

Initial state = the root mission's statement, empty proof context. `d̂` is a
property of the state — the predicted number of **remaining proof steps** —
predicted by the head and never stored as truth during search. Dataset records
carry the same quantity as `remaining_steps = total_tactic_steps - i`, with no
conversion anywhere on the path.

`mission_id` is content-addressed: `sha256(env, statement_formal)` truncated to
12 hex characters. That single choice gives reuse for free (§7): the same lemma
reached from two different decompositions is the same mission, with the same
workspace and the same compiled `.olean`. The registry, not the tree, is the
identity authority.

## 3. Action set

```text
Action :=
  | Block(proof_text)                  # at most `action_block_steps` proof steps
  | Decompose([child_statement ...])   # one agent response splitting the mission
```

Both are one agent message. Single-tactic actions are not modelled.

`action_block_steps` (config, default **10**; candidates 10/20/50/100) is the
maximum number of proof steps one sampled action may contribute. Inside one
action the agent is free: plan, sketch, compile, read feedback, search the web,
call tools. Only the emitted proof block is capped, which is what makes the
action atomic for MCTS without pretending a human writes one `simp` at a time.
It rises with the model through `schedule.CURRICULUM` (10 → 20 → 50 → 100).

Both `d̂` and `N` are in proof steps, so the arithmetic is direct:

    rounds_left = ceil(d̂ / N)
    single_shot = d̂ <= N        # one action finishes it: no search, no CE

Transitions, with the state-change test:

```text
old := (lean_context, goal)
execute Block (<= action_block_steps emitted steps) -> new := (lean_context', goal')

block_progress(old, new).progress  iff  goal changed | lean_context changed | solved
    progress  -> state transition, next K-sample round legal
    identical -> no-op / failed action for this node: no transition, no CE
```

`traj.block_progress` is the single definition (re-exported by
`mcts/schedule.py`). A block that rewrites the goal is progress; one that only
grows the context is progress; one that changes nothing is a failed action, kept
for diagnostics and excluded from CE.

```text
apply(State, Block)
    -> State' with a longer lean_context   # goal unchanged, nothing proved yet

apply(State, Decompose(children))
    -> State' = AND(children)
    # "the original goal is discharged iff all child missions are discharged"
    # the state changes even though nothing has been proved yet
```

`AND(children)` is what makes decomposition a real move rather than a
side-channel: after a decomposition the search is looking at a *different
obligation set*, and the parent cannot be sampled again until those children
return (§4).

There is no third action. Retry is another K-sample at the same state; giving up
is a `Block` whose outcome is `failed`.

### No-signal band

A mission whose estimate fits in one action never enters multi-round MCTS:

    single_shot  iff  d̂ ≤ action_block_steps          # same unit, no scaling
    learning_signal = not single_shot

It is sampled once (the mission still has to be solved), never expanded, and it
writes no CE record. With N = 100, everything at or below 100 steps is silent;
with N = 10 only ≤10-step missions are. That is why the band is a *curriculum*
quantity and not a constant. That is intended: the base model is assumed to already
handle missions this small in the free-agent setting, so there is nothing to
learn from them and search budget is better spent above the band. If the model
solves everything at N = 100/200/300/400, stop generating signal for missions
≤ 1000 and raise N instead — the band moves with N.

## 4. Sampling schedule: the trigger is a child returning

At a state, sample **K responses** from the main agent at temperature > 0,
execute each in OpenCode, and keep them as sibling actions:

```text
sample(S):
    for i in 1..K:
        r := agent.respond(S)              # temperature sampling, one response
        if r is Block:  execute_lean(r.proof_text) -> outcome; push(block_finished)
        if r is Decompose:
            kids := dedupe_siblings(r.children)      # §7: same statement = one mission
            register(kids); dispatch(kids); S.pending += ids(kids)
    S.sampled += 1
```

The trigger for the next K-sample is **an event**, never a clock:

| event | samples? | why |
| --- | --- | --- |
| `start` | yes, once | nothing sampled at this state yet |
| `child_return` | yes | the state moved: a child's result is now in the pool |
| `block_finished` | yes | a sibling response finished executing; the state is evaluable |
| `tick` | **never** | a timer expands states that have already moved |

and never while children are pending:

```text
sample(S, trigger) is allowed iff
    trigger in {start, child_return, block_finished}
and S.pending == []          # a decomposition is outstanding: wait, do not expand
and S.in_flight == 0         # the current K responses are still executing
and not (trigger == start and S.sampled > 0)
```

This is `mcts/schedule.py::should_sample`, unit-tested (tick never samples;
`children_pending` returns `wait_for: "child_return"`; `in_flight` blocks).
Pseudocode of the whole loop, with a diagram, is `mcts/SPEC.md`.

Two sampler-side consequences of §4a and §3:

- a `Block` that leaves `(lean_context, goal)` byte-identical is a **no-op**: no
  transition, no next round. Only a real state change is progress.
- a mission with `d̂ ≤ action_block_steps` (both in proof steps) is **single-shot**:
  one sample, no expansion, no CE record (the no-signal band, §3).

Why event-driven matters for the value head: siblings sampled against the same
state are the only honest comparison set. A timer creates siblings whose state
has silently changed, and any preference learned between them is noise. The pool
therefore writes one record per (state, action), not per wall-clock slot.

## 4a. 为什么没有分解奖励（terminal-only signal）

There is no reward for decomposing and no reward for calling a subagent. The only
learning signal is expert iteration + teacher forcing on trajectories that reach
**terminal success**:

```text
keep   outcome.solved ∧ status ∉ FAILED_STATUSES ∧ every child solved
drop   not_solved | not_success (failed / budget_exhausted / stuck / sorry) | children_unproved
```

- A trajectory enters the CE posterior pool because the original
  `(context, goal)` was discharged, never because it was shaped well.
- A trajectory that decomposes into 60 missions and later succeeds is rewarded
  *because it succeeded*. A one-block success at the same state gets the same
  weight. Planning quality is learned implicitly: the plans that reach the
  terminal state are the ones whose tokens get reinforced.
- Failures stay in the pool file as diagnostics (`drop_reason`, `status`), and are
  excluded from CE — `dataset.py` drops them by default and only keeps them behind
  `--include-failed`; `train_stage2.py` filters before it builds a batch.
- There is no `decomposition_bonus`, no `subagent_reward`, and no
  credit-assignment-to-action. `traj.assert_no_auxiliary_reward` raises if any of
  `decomposition_bonus / subagent_reward / credit / advantage / shaping / bonus`
  appears in a pool record or an outcome, so the rule is enforced by code.
- Value targets remain **remaining-work estimates**, never rewards: `d̂` predicts
  how much work is left, it is not a return.

Two consequences worth stating: (1) a mission inside the no-signal band
(`d̂ ≤ N`, both in proof steps) contributes nothing, by design — the base model
already solves it; (2) the curriculum (raising N) is what keeps signal flowing as
the model gets better, because the band moves with N.

## 5. The gate: d̂ > 500 steps ⇒ decompose, and depth is the only cap

```text
input:  d̂ (remaining proof steps), ci_high, tail_saturated, prev_action,
        depth, max_depth, threshold = 500, action_block_steps = N
output: action ∈ {block, decompose}, reason, mandatory, rounds_left,
        single_shot, suggested_children
```

`rounds_left = ceil(d̂ / N)` is reported, not used: the gate is a budget boundary
on *work*, not on the number of responses. The threshold comparison is `>=`,
identical to the shared table (`train/gate.py`, `lib/gate.mjs`), which enters
decompose at exactly 500.

Rules, in order (`train/gate.py`, `lib/gate.mjs`, `mcts/schedule.py`, one shared
test table):

1. `d̂ < 0` or NaN → **block** (`no_estimate`: never block on a missing value).
2. `tail_saturated` → **decompose** (`tail_saturated`, mandatory).
3. `depth >= max_depth` → **block** (`depth_capped`, not mandatory). The gate
   wanted children but the tree is deep enough; escalate instead.
4. `prev = decompose` and `d̂ ≤ 0.8 × threshold` → **block** (`hysteresis_exit`).
5. `prev = decompose` otherwise → **decompose** (`hysteresis_hold`, mandatory).
6. `d̂ ≥ threshold` → **decompose** (`above_threshold`, mandatory).
7. `ci_high ≥ 4 × threshold` → **decompose** (`ci_upper_far_above`).
8. otherwise → **block** (`below_threshold`).

Decomposition is a first-class action, not a fallback of last resort:

- Above the threshold the agent **must** decompose. It is not "try the big proof
  and split if it fails"; the estimate is the plan.
- A child is **not** required to satisfy `d̂ ≤ 500`. A child may itself be huge
  and decompose again. Requiring small children would turn the gate into a
  one-level filter and would forbid exactly the shape that hard problems have: a
  60-mission opening fan-out (the Anthropic Fable-5.2 Fermat run started with 60
  missions). The bound on depth is `max_depth` (config, default 3), not the gate.
- `suggested_children = clamp(2, 8, ceil(d̂ / threshold))` is a planning hint.
  A 20k-block estimate asks for 40 children, the cap says 8, so the agent
  decomposes recursively. The gate answers "decompose?", not "into how many?".
- Hysteresis at 0.8: an estimate hovering at 495–505 would flip the plan every
  event, and each flip disposes of an outstanding mission set. Enter at 500,
  leave at 400. Both numbers are config and must be recalibrated together.
- The gate threshold is a *budget boundary* ("a mission bigger than 500 steps is
  not worth attempting inline"), not a quantile of the head's output. Keep it
  fixed across base versions and recalibrate the head instead.
- N does not move the gate. `rounds_left = ceil(d̂/N)` changes with the
  curriculum; the 500-step boundary does not. That separation is the reason the
  target is steps and not "blocks".

Asymmetry is deliberate. The costly error is a false `block` on a tail mission
(the agent burns its whole budget on a 10k-step proof); the cheap error is a
false `decompose` on a 300-step mission (one extra mission level). Rule 7 is
the guard for the first, and rule 1 refuses to block on a missing estimate.

## 6. Two trajectory pools, and the CE mask

```text
                  sampled action (one agent response)
                              |
              +---------------+----------------+
              |                                |
        action = Block                  action = Decompose
              |                                |
    lean executes the block            children dispatched (mission DAG)
    outcome: solved / failed           each child returns a summary
              |                                |
              +---------------+----------------+
                              |
                      main_pool write
        {state, action_text, action_type, child ids, child SUMMARIES, outcome}
                              |
                    (child sessions, separately)
                              |
                         sub_pool[mission_id]
              the child's full transcript, for that child's own CE
```

| pool | key | contains | never contains |
| --- | --- | --- | --- |
| `main_pool` | (state, action) | agent text, action type, child mission ids, child **summaries**, outcome | child messages, child tool output, child token streams |
| `sub_pool` | `mission_id` | that child session's full trajectory | the parent's decisions |

`traj.split_pools` builds both, and `traj.assert_main_record_clean` raises if a
main record carries a key from `CHILD_INTERNAL_KEYS` (`messages`, `transcript`,
`tool_outputs`, `tokens`, …). Separation is enforced, not documented.

Admission to CE is **terminal success only** (§4a): `traj.filter_ce_pool` keeps a
trajectory iff `outcome.solved`, the status is not a failure status, and every
child mission it created is solved. Both pools are still *written* for failures;
they just never become loss.

### The exact masking rule

CE on `main_pool` is computed only on the agent's own tokens:

| kind | examples | `loss_mask` |
| --- | --- | --- |
| `system` | harness prompt, tool docs, workspace rules | **0** |
| `tool_result` | anything Lean printed, `lake` output, file reads | **0** |
| `subagent_return` | the injected message a child sent back | **0** |
| `assistant` | the decision, the mission contract, the proof block | 1 |
| `user` | the task statement | 1 |
| pad | padding to batch length | **0** |

Spans are **explicit** `[start, end)` token ranges, recorded per message and per
block inside a message, then merged; nothing is inferred from text patterns.
`mask_policy` in config fixes the kinds (`DEFAULT_MASK_POLICY` in `traj.py`);
`keep_kinds` always wins over `mask_kinds`, so a config typo cannot delete the
decision it is supposed to train.

`traj.to_input_ids_loss_mask(traj)` returns `{input_ids, loss_mask}` plus
diagnostics (`n_kept`, `n_masked`, `spans`, `segments`).

Why mask at all: with unmasked CE the cheapest way to lower the loss on a
decomposition record is to reproduce the child's returned proof. The model then
learns "a subagent returns a proof" as the continuation, i.e. it learns *to call
a subagent* from the content of returns instead of from the quality of the
decision to decompose. That is a shortcut that looks like learning on the loss
curve and destroys the policy in deployment. Masking leaves the decision and the
contract — the part whose probability we actually want to raise.

## 7. Reuse and caching

Three kinds of duplicated work, three mechanisms (`cache.py`):

**Mission registry.** `key = sha256(env, statement_formal)`, value = mission id,
workspace path, lean project, created_at, status, depth, parent. Registering the
same statement in the same env returns the same id and workspace with
`reused: true`; `register_children` dedupes first, then registers. Status is one
of `pending | running | accepted | failed | split | blocked`, so a resumed run
picks up the mission table rather than re-deriving it. The table is written
atomically (`os.replace`), so a killed run never leaves half a registry.

**Sibling dedupe.** Identical missions created in the *same* decomposition (same
env, same normalised statement) are dropped before dispatch: proving the same
lemma twice in parallel spends budget and teaches the value head nothing. The
log records `{reused: true, statement_hash, kept_index, dropped_index}`.

**Compile cache.** One shared Lean project per env, and content-addressed task
dirs:

```text
<root>/olean/<env>/project/                 # one project per env
<root>/olean/<env>/tasks/<sha256(content)[:16]>/
    <Name>.lean   <Name>.olean   .lock
```

`CompileCache.compile(env, file, content)` takes an exclusive `flock` on the task
dir's `.lock` (plus a per-key thread lock), re-checks the `.olean` inside the
lock, and reports `cache_hit` for every call. Two children can therefore never
compile the same file at once, and the second one does not compile it at all.
Measured: 4 concurrent compiles of one file → 1 runner call, 3 hits.

## 8. The value head itself

**Attachment point:** the last hidden state of the final decoder layer, after the
final RMSNorm, read at the **last non-pad token** of the prompt.

```text
prompt = [system][user: mission state = goal + lean_context]
h_t    = hidden_states[-1][t]              # [B, T, H]
h      = h_{last non-pad}                  # [B, H]   <-- readout
value  = head(h)  =  Linear(H→V) → GELU → Linear(V→V) → GELU → out
```

Why that position and not mean pooling:

- the model is causal, so only the last prompt token has attended to the whole
  mission state; earlier positions see a prefix and would systematically
  underestimate (they have not seen the goal's hardest hypotheses yet);
- mean pooling mixes the system prompt and the statement into the readout; its
  scale then depends on prompt length, which is exactly the variable we are
  trying to predict. That is a leak in the wrong direction: a longer prompt
  would look like "more blocks left" for a purely lexical reason.

`H` is whatever the base reports (Qwen3.5 0.8B = 1024, 4B = 2560, 9B = 4096); the
tower is `H → 256 → 64` for the bucket encodings, matching `ValueHead64`. Nothing
in the head assumes H.

**Long contexts: goal-full + pooled context (the 2048 cap is gone).** The first
implementation joined `f"{goal}\n{context}"` and tokenised with
`max_length=2048`, so any Lean context longer than ~2K tokens was cut from the
right — i.e. exactly the recent proof history that carries `remaining_steps`.
The state is now tokenised in two pieces:

```text
goal_ids    = first --goal-max-tokens of goal
context     = --context-pool over context_ids:
  tail        last --context-window tokens                 (default)
  segments    last segments*context_window tokens, chunked (mean of chunk pools)
  mean        whole context up to --max-length
  full        legacy joined truncation, kept for A/B only
h = tanh( 0.5 * (pool(goal_ids) + mean_k pool(chunk_k)) )   # dim H, no new params
```

Each piece is mean-pooled with its own attention mask, so padding never moves the
vector. The default combine is the elementwise mean of the goal pool and the
context pool: the output is still `hidden_size`, so **every existing checkpoint
loads unchanged**. `--pool-combine projection` replaces the mean with a learned
`Linear(2H, H)`; that adds `pool_proj.*` to the checkpoint, so a run trained with
it must be loaded with the same flag, and it is off by default.

Known deviation from the "read the last token" argument above: the shipped
backbone mean-pools (it always did). Reading `h_{last non-pad}` is a one-line
change but changes the checkpoint's input distribution, so it is a separate
promotion with the §9 re-alignment protocol, not a silent edit.

**Readout content:** `h` alone is version-fragile (§9). Recommended input is
`[h ; φ]` with a handful of *version-invariant* scalars:

    φ = (log1p(goal tokens), log1p(context tokens), open-goal count,
         mission depth, tier id, Mathlib graph degree bucket)

`φ` costs ~16 parameters and gives the head a prior that survives a base swap.
This is the cheap answer to "the base moved and the head drifted", not depth.

**Dual head (kept, and now the default).** Two outputs over one trunk: a 64-bin
log-spaced bucket distribution plus a bounded residual that refines inside the
bucket, decoded with a **binned mode** (3-bin local mean around the argmax) and a
CI from the bucket CDF. The residual is what buys resolution; the log grid is what
makes the tail representable at all; the mode is what stops an uncertain head from
reporting the geometric middle of its own uncertainty. The binned mode is not a
detail: with `decode="expected"` the same head loses terminal exactness (0.79 vs
1.00 measured, §11a) and names bins the head assigns no mass to.

`tail_saturated` is now a property of the *mode*, not of the decoded value: it
fires when the argmax is the last bin (§11a). The old test
(`estimate >= 0.95 * grid_max`) could never fire on a correctly saturated head,
because the last bin's centre is 30339 steps, below `0.95 * 32768 = 31130`.


## 9. Interference under full-parameter fine-tuning

*(§2 of the review; this replaces the old "λ/Adam" section, which argued instead
of measuring.)*

**Scope: W is fully trainable.** No LoRA, no frozen base, no adapters. One AdamW
over three parameter classes:

| class | what it is | trained by |
| --- | --- | --- |
| `backbone` | W: decoder blocks + final norm | **value and policy** |
| `value_*` | value tower, bucket logits, residual | value only |
| `policy` | `block` vs `decompose` head | policy only |

```text
L = CE_policy + λ · L_value        λ = 1e-3        L_value = mean over value rows
```

Every number below is **measured** on the tiny harness, seed 7, 240 synthetic
mission rows (72 held out), full FT, AdamW, 40 warm-up + 20 policy-burst steps:

```text
python3 train/head.py --probe          # modes 1,2,3,4,5,8   (~2.5 s, CPU)
python3 train/head.py --probe-recipe   # modes 6,7           (~3 s, CPU)
```

Harness scale is not model scale. What transfers is the **ranking**, the **sign**,
and the fact that each mode has one exact quantity you can log. Where a number is
analytic rather than measured, the text says so.

Two facts run through everything:

1. **λ is not the interference knob; the learning rate is.** AdamW normalises by
   the second moment, so once `λ|g_value|` clears the ε floor, a value term moves
   a shared tensor by ~`lr` regardless of λ. Measured from the same Adam state:
   one value-only batch moves the shared parameters **1.76×** further than one
   policy batch (Δ norms 0.0762 vs 0.0433) at λ = 1e-3. λ decides *whether* the
   value term joins the direction; `lr` decides how far the step goes.
2. **The dominant mode is policy→value, not value→trunk.** Measured: after 20
   policy-only steps (λ = 0, so the head provably did not train), the trunk moved
   `‖Δh‖/‖h‖ = 0.90`, the frozen head's value loss rose 3.011 → 3.068 (+1.9%),
   the worst estimate moved `9.41` in log1p space, and **18 of 72** probe states
   flipped across the 500-step gate (25%). The head was fitted correctly; the
   representation under it moved.

### The eight modes

| # | mode | mechanism | symptom | detection metric (log exactly this) | ranked mitigation |
| --- | --- | --- | --- | --- | --- |
| 1 | value→trunk leakage | value CE backprops through the value tower into `h`, rotating the features the policy head reads | policy loss plateaus after value-heavy batches; value falls while pass@1 stalls | `value_share_shared = ‖g_v‖²/(‖g_v‖²+‖g_p‖²)` over params **not** named `value_*`, per tensor + total | alarm only (see 9.1) |
| 2 | policy→value shift | the policy update moves `h`; the head is a fixed function of `h` | d̂ drifts on unchanged missions; gate flips | `h_drift = ‖Δh‖/‖h‖`, `dhat_drift = median\|Δlog1p d̂\|`, `gate_flip_rate`, all on a frozen 256-row probe set | 1. joint EMA, 2. frozen-trunk warm-up |
| 3 | optimizer-state coupling | `m, v` are **per tensor**; a value-heavy batch writes them for the policy too | a value-only micro-batch nudges the policy; step size depends on batch order | `adam_cos` = cos(Δv-step, Δp-step) on shared params from the same Adam state; `‖Δv‖/‖Δp‖` | 4. fixed denominator + per-term clip; never `opt.step()` on a value-only micro-batch |
| 4 | normalisation drift | value loss averaged over `N_v` while policy is over `N` ⇒ `λ_eff = λ·N/N_v` | value gradient spikes when few value rows are in the batch | `n_v/N` and `lambda_eff` every step | 5. fixed denominator `N_batch` (9.4) |
| 5 | terminal-0 bias | two-hot target put `d=0` on bin **1** (inverted interpolation weight + 1e-6 clamp) | a solved mission looks ~1.16 steps deep; d̂ floor of +1 | `zero_exact = P(d̂ == 0.0 \| d == 0)`, `median(d̂ \| d=0)` | 3. bucket 0 = `[0,1)`, exact target, mode decode (implemented, §11a) |
| 6 | capacity placement | extra width in the trunk is shared; extra width in the tower is not | widening the trunk raises drift and buys nothing; widening the tower cuts MAE | `mae_log` vs `value_hidden` at equal steps, with `policy_acc` and `dhat_drift` beside it | 6. widen the **tower** only (9.6) |
| 7 | LR ratio / warm-up | one `lr` for the trunk means the value term inherits the policy's step size; the trunk moves most in the first steps | head fits an `h` that no longer exists by step 200 | `mae_log` and `dhat_drift` over a ratio grid; `h_drift` in the first 200 steps | 7. `value_lr = 1×`, `λ = 0` + frozen trunk for warm-up (9.7) |
| 8 | EMA / ckpt averaging | averaging the head alone lags it behind the trunk; averaging **jointly** keeps the served pair `(h, head)` consistent | d̂ jitter, gate flips that no loss curve shows | `ema_gap = median\|log1p d̂_ema − log1p d̂_raw\|`, plus `dhat_drift` and `gate_flip_rate` of the EMA weights | 8. joint EMA, serve EMA (9.8) |

Ranked by measured effect per unit of engineering, the order is
**8 > 3 > 7 > 5 > 4 > 6 > 1 > 2-rest**, and the default recipe below is exactly
that order with the losers demoted to alarms.

### 9.1 value→trunk leakage: real, but 5 orders of magnitude below the policy

Measured at λ = 1e-3 on the shared parameters only:

```text
g_value_shared  = 0.000477      g_policy_shared = 0.408128
value_share     = 1.37e-06      (per tensor: 1e-06 .. 5e-06, max on norm.bias)
```

So the value objective asks for **0.0001%** of the shared gradient budget. That is
not a reason to be complacent, it is a reason to *measure the right denominator*:
the share over **all** parameters is ≈1.0, because the value loss is the only thing
that touches `value_*` and the head is most of the parameter count. A trainer that
prints the all-parameter share reports "value dominates" for a head that is barely
training. (The sibling trainer's `--tiny` probe prints `value_share=0.99960` for
this reason; the same run's *shared*-parameter share is ≈1e-6.)

**Mitigation:** log the shared-parameter share and alarm at `> 0.05` for three
consecutive logs. Do **not** default to `--stop-grad-value`: at 1e-6 it costs the
value head its only route to better features and buys nothing. Stop-grad is a
*response* to the alarm, not a default.

### 9.2 policy→value: the one that actually breaks the gate

The same probe, before and after a 20-step policy-only burst:

```text
‖Δh‖/‖h‖              0.9024        trunk moved ~90% of its own norm
value_loss (frozen)   3.0114 → 3.0675     +1.9% on a head that was not trained
median |Δlog1p d̂|     0.0014        most states barely move …
max    |Δlog1p d̂|     9.4099        … but the worst state moves 9.4 (e^9 ≈ 12000×)
gate flips @500       18 / 72      25% of unchanged missions changed plan
```

Read that as: the *median* is a comfortable lie. The damage is concentrated — a
handful of states get re-classified, and those are the tail states whose plan
matters most. That is why `gate_flip_rate` and `max|Δlog1p d̂|` are in the log set
and `mae_log` alone is not enough.

**Mitigation, ranked:** (a) joint EMA of trunk + head (§9.8, measured 18 → 4
flips); (b) frozen-trunk warm-up so the head is fitted to `h_0` and the trunk
starts moving only once the head is useful (§9.7); (c) periodic refresh: unfreeze
the value term for a short window every k policy updates — only if (a)+(b) leave
`dhat_drift > 0.05`.

### 9.3 optimizer-state coupling: two param groups is not two optimizers

AdamW keeps `m, v` per **tensor**, not per parameter group. `backbone.*` tensors
are touched by both losses, so the value batch writes the moments that the next
policy step reads. Measured from one Adam state:

```text
adam_cos (Δvalue-step, Δpolicy-step)  = 0.348     (gradient-only cosine: 0.217)
‖Δvalue-step‖ / ‖Δpolicy-step‖         = 1.76
```

Not orthogonal, and the value step is *larger*. So batch order matters more than λ:
a value-only micro-batch that calls `opt.step()` is a 1.76× kick in a direction
34% aligned with the policy's.

**Mitigation:** (a) never step on a value-only micro-batch — accumulate gradients
for the micro-batches of one logical batch, then step once; (b) keep `value_*` in
its own optimizer (disjoint tensors, so it genuinely is decoupled) and leave
`backbone.*` in one group; (c) clip each loss term separately before summing
(`clip_grad_norm_` per term, then add), so a spike in one cannot set the clip
scale for both.

### 9.4 normalisation drift: `λ_eff = λ·N/N_v`

Averaging the value loss over its own `N_v` rows makes the per-row weight `λ/N_v`,
so the coefficient the trunk actually sees is `λ·N/N_v`. Measured |g_value| for the
same objective with `N_v = 16` vs `N_v = 4`:

```text
g_value(N_v=16) = 1.576     g_value(N_v=4) = 2.359     ratio 1.50
λ_eff at N_v=4, N=16 = 4e-3   (4× the nominal 1e-3)
```

(The ratio is 1.5 rather than 4 because the two subsets have different per-row
losses; with identical rows it is exactly `N/N_v` by construction.)

**Mitigation:** divide the value *sum* by the **configured batch size**, not by the
number of value rows:

```text
L_value = (Σ_{i∈V} ℓ_i) / N_batch        ⇒  λ_eff = λ · n_v/N  ≤ λ
```

Then a value-heavy batch cannot exceed λ, an empty value batch contributes exactly
0, and `λ_eff` is a *coverage* statement rather than a surprise. Log `n_v/N` and
`λ_eff` every step.

### 9.5 terminal-0 bucket bias: a structural +1 step, fixed

Two defects compounded. `two_hot_batch` (a) clamped the input to `xs[0] + 1e-6` and
(b) used the *distance from the lower edge* as the weight on the lower bin, i.e.
the interpolation was inverted. For `d = 0` that produced `idx = 0, weight_on_0 = 0
⇒ weight 1 on bin 1`, whose centre is **1.16 steps**. The head then learned a
floor of one bin on every terminal record, and the bias is invisible in `mae_log`
(it is one bin out of 63) while it is decisive at the gate: "solved" and "one step
left" are different plans.

Fixed in §11a: bin 0 is `[0,1)`, the target for `d = 0` is bin 0 with weight 1, and
the binned-mode decoder short-circuits bin 0 to exactly `0.0`.

**Mitigation:** structural, no hyperparameter. Detection: `zero_exact` and
`median(d̂ | d=0)` on the probe set; both are in `head.metrics()["zero"]`.

### 9.6 capacity placement: widen the tower, never the trunk

Same data, same seed, same 90 steps, only `value_hidden` moves:

| `value_hidden` | params | MAE(log) | MAE(steps) | policy acc | `dhat_drift` after burst |
| --- | --- | --- | --- | --- | --- |
| 32 | 7.5k | 1.2188 | 2403 | 0.972 | 0.0040 |
| 96 | 23.1k | 0.4769 | 1652 | 0.972 | 0.0010 |
| 256 | 97.8k | 0.3702 | 1519 | 0.972 | 0.0171 |

Capacity in the **tower** is close to free: −70% MAE from 32 → 256 while policy
accuracy is bit-identical at all three widths, because the tower's parameters are
not shared with the policy. Capacity in the **trunk** is not free: it is shared, so
the extra width also gives the value term a bigger surface to rotate (§9.1) and
gives the policy features more parameters to overfit.

**Mitigation / decision rule:** the tower is `H → 2H → 256 → 64` (the shipped
`ValueHead64` shape, with `2H` as the first hidden layer). Widen the *tower* if
`mae_log` improves ≥10% at equal steps. Widen the *trunk* for the value head never;
and if the value head is the bottleneck after the tower is already 2H wide, the
answer is the readout features `φ` (§8), not depth.

### 9.7 LR ratio and warm-up

`value_lr` names a **head-only** group (`value_*`). The trunk has no separate value
LR — it is shared, and that is the point. Ratio grid, same seed, 90 steps:

| `value_lr / policy_lr` | MAE(log) | MAE(steps) | `dhat_drift` | gate flips |
| --- | --- | --- | --- | --- |
| 0.1 | 2.0792 | 2534 | 0.0000 | 3 |
| 1.0 | 0.4769 | 1652 | 0.0010 | 3 |
| 3.0 | 0.3969 | 1664 | 0.0023 | 3 |

0.1× is not "stable", it is **undertrained** (4.4× the MAE of 1×). 3× is 17%
better on MAE with no measured stability cost over this horizon. The default is
**1×** — the measured-safe point — with 3× documented as the first escalation once
the drift log shows headroom (`dhat_drift < 0.01` over a 5k-record run).

Warm-up is a separate knob and it does more than the ratio. In full FT the first
~200 steps move `h` the most (measured 0.90 relative drift over 20 policy steps at
a high LR), so a head trained through them fits a representation that is about to
disappear.

**Mitigation:** warm-up = `λ = 0` **and** the trunk frozen for `max(200, 3%)` steps;
the head fits `h_0` first, and the trunk starts moving only after the head has
something to say. Then `λ` on, cosine decay to `0.1 × lr_base`.

### 9.8 EMA and checkpoint averaging: average jointly, or not at all

After the same 20-step policy burst, with `β = 0.99` over **all** float parameters:

```text
                    raw weights    EMA weights
value_loss          3.0675         2.9995
median |Δlog1p d̂|   0.0014         0.0006
max    |Δlog1p d̂|   9.4099         8.6400
gate flips @500     18 / 72        4 / 72
```

Two things to keep straight:

- **Read the flips, not the median.** The median says both are fine; only the flip
  count shows the 4.5× difference.
- **EMA must cover the trunk and the head together.** An EMA of `value_*` alone
  would freeze the head at an old `h` while the trunk keeps moving — that *is* mode
  2, manufactured on purpose. The invariant is that the served pair `(h, head)` is
  one point in weight space; averaging only half of it breaks the pairing.
- Honest caveat: a 20-step burst with a 100-step EMA window mostly reports the
  pre-burst weights, so `0.0014 → 0.0006` is an **upper bound** on the benefit.
  Over a full run the mechanism is the same but the magnitude shrinks, and EMA
  trades variance for lag. That is why the recipe also logs `ema_gap`: when the gap
  exceeds the drift it is meant to absorb, the EMA is stale and must be re-seeded.

**Checkpoint averaging (SWA)** is the same operation at a coarser grain: average the
last 5 checkpoints spaced 250 steps, **all** parameters, only when `cfg` is
identical (`encoding`, `buckets`, `max_steps`, `unit`, `data_hash`). Never average
across an LR restart or a data-mix change — the average of two models fitted to
different targets is neither. After averaging, re-run the gate receipt (§9.9 step
9); if it fails, serve the EMA instead and do not re-init.

### 9.9 Default recipe (full FT, LoRA explicitly out of scope)

Not a menu: this is the configuration to run, and the escalation list is what to
change **if** the receipts fail.

1. **Optimizer.** AdamW, `β = (0.9, 0.95)`, `weight_decay = 0.0`, grad clip 1.0
   per loss term.
2. **Parameter groups.** `backbone.*` → lr `lr_base`; `value_*` → `1.0 × lr_base`;
   `policy` → `lr_base`. `lr_base = 2e-5` for 0.8–4B, `1e-5` for ≥9B full FT.
   There is **no** separate value LR on the trunk; the trunk is shared by
   construction.
3. **Objective.** `L = CE + λ · (Σ_{i∈V} ℓ_i)/N_batch`, `λ = 1e-3`. Denominator is
   the configured batch size, never `N_v` (§9.4).
4. **Warm-up.** First `max(200, 3% of steps)`: trunk frozen, `λ = 0`, heads at
   `1 × lr_base`. The value head fits `h_0`.
5. **Main phase.** Unfreeze the trunk, `λ = 1e-3`, cosine decay to `0.1 × lr_base`.
   One `opt.step()` per logical batch, never per value-only micro-batch (§9.3).
6. **EMA.** `β = 0.999` over all float parameters, updated after every step; serve
   the EMA weights; keep raw weights for receipts and for the checkpoint.
7. **Averaging.** SWA over the last 5 checkpoints (every 250 steps) when `cfg` is
   identical; re-seed the EMA (`ema := raw`) after any LR restart.
8. **Head capacity.** Tower `H → 2H → 256 → 64`; decode `dual` + **binned mode** +
   tail flag; never `expected` (§11a).
9. **Receipts, every 250 steps, on a frozen 256-row probe set.** `value_share_shared`
   (alarm > 0.05), `h_drift`, `dhat_drift` (alarm > 0.05), `gate_flip_rate`,
   `value_loss_probe`, `zero_exact`, `tail_recall@10k`, `cov80`, `ema_gap`.
   The gate is **blocked** (all decompose, the safe direction) if
   `|Δ median log1p d̂| > 0.25` or tail recall drops > 0.10 — the existing
   post-base-swap rule, now applied continuously instead of only after a swap.
10. **Escalation, in order, only when a receipt fails.** 3× head LR (§9.7) →
    `--stop-grad-value` for 200 steps (§9.1) → value-tower refresh window (§9.2) →
    1-block two-tower split (§9.6). Re-init is never on the list.

### 9.10 What is deliberately *not* in the recipe

- **LoRA / adapters as an interference fix.** Out of scope, and the mechanism does
  not require it: mode 2 is drift of a shared representation, and an adapter makes
  the adapter's `h` drift too. Separate capacity is a *capacity* answer (§9.6).
- **Stop-gradient by default.** At a measured share of 1e-6 it buys nothing and
  costs the head its route to better features.
- **A λ sweep.** λ barely moves the trunk step (§9.1); it is a *ratio of objectives*
  you set once (1e-3, AlphaProof convention) and then leave alone. Tuning λ to fix
  a drift problem is tuning the wrong quantity.
- **Re-initialising the head after a base swap.** Re-align instead: affine
  recalibration of `log1p d̂` on ~2k held-out states, `value_lr × 0.1` for the first
  `N` updates, and the receipt above. Version-tag the checkpoint (`cfg.base`,
  `cfg.base_rev`, `cfg.unit`, `cfg.data_hash`).

## 10. What this changes vs the old step-level design

| | old (step-level) | new (mission-level) |
| --- | --- | --- |
| unit | one tactic | one agent response: proof block or decomposition |
| state | tactic state after a prefix | `{env, mission_id, lean_context, goal}` |
| action | tactic string | `Block` \| `Decompose(children)`; a block emits at most `action_block_steps` (10) proof steps |
| transition | prefix grows | block extends context; decompose replaces the goal with `AND(children)` |
| value target | remaining tactic steps | remaining **proof steps** (unchanged unit; the *state* is mission-level) |
| policy target | next tactic (vocab of tactics) | `block` vs `decompose` |
| gate input | step estimate, 500 steps | step estimate, 500 steps (same unit, N-step actions) |
| sampling | expand per node as budget allows | K responses per state, re-sample on **child return** / block finish |
| decomposition | fallback when the proof stalls | first-class action, *mandatory* above the gate |
| depth bound | implicit (step budget) | explicit `max_depth` (default 3), gate never overrides it |
| trajectories | one pool of searched nodes | `main_pool` (decisions, summaries) + `sub_pool` (child transcripts) |
| CE | full sequence, visit-weighted | masked (system / tool results / subagent returns = 0), and terminal-success trajectories only |
| reward | none stated; decomposition was an escape hatch | none at all: no decomposition bonus, no subagent reward, no credit assignment |
| small missions | always trained on | no-signal band: `d̂ ≤ N` (steps) is single-shot and yields no CE record |
| reuse | none | mission registry + sibling dedupe + compile cache |
| structure | strict AND-OR tree | mission DAG primary, AND recorded only |

Consequences worth stating plainly:

- **There is one unit, and mixing it is a bug, not a scale choice.** An earlier
  revision targeted `ceil(steps / 3)` and kept the gate at 500 steps; 900 steps
  read as 300 "blocks" and passed the gate. `cfg.unit` is part of the checkpoint
  identity, and `tests/test_mission_plain.py::test_unit_regression` fails if the
  conversion comes back on the default path.
- **N is a curriculum knob, the gate is not.** Raising N from 10 to 100 changes
  `rounds_left` (and the no-signal band) but must not change what 500 means. If
  raising N ever changes the decompose rate on a fixed mission, a conversion has
  crept back in.
- **Tiny numbers did not move when the unit was fixed**, because the tiny
  generator's target was always steps: MAE(log) 0.6217 / MAE(steps) 2154 (dual,
  s7, 50 records). The earlier 0.5269 / 413 figures were the blocks target and
  are not comparable; they are kept only in SELFTEST.md as the deprecated shim.
  After the bucket fixes the same command reports **0.4762 / 2405.7** (dual, s7,
  50 records) — the MAE(log) drop is the terminal-bucket and interpolation fix
  (§9.5, §11a), not a data change; `MAE(steps)` stays large because the tiny
  generator's tail is 20k and one bin near 20k is ~500 steps wide.
- **Decomposition can no longer be used as "the thing after failure".** The gate
  runs *before* the attempt, so a mission above 500 never burns a full proof
  attempt first. That is the point.

## 11. Metrics

Used by `head.metrics`, printed by stage 1:

| metric | definition | accept |
| --- | --- | --- |
| `mae_log` | mean `\|log1p d̂ − log1p d\|` | ≤ 0.35 on ≥5k held out |
| `baseline_mae_log` | same with the train median (sanity floor) | must beat it |
| `bucket_acc` | nearest-centre bucket matches truth | ≥ 0.6 |
| `monotone` | over predicted-quantile bins, mean truth never decreases | 0 violations |
| `gap_log` | mean `\|mean pred_log − mean true_log\|` per bin | ≤ 0.3 |
| `tail.recall` | `P(d̂ ≥ at \| d ≥ at)`, with `tail.n` reported | ≥ 0.8 with n ≥ 200 |
| `tail.median_ratio` | median `d̂/d` on the tail | 0.5–2.0 |
| `coverage_80` | fraction of truths inside the 80% interval | 0.75–0.85 |
| `zero.exact` | `P(d̂ == 0.0 \| d == 0)`, with `zero.n` | 1.0 (bucket head) |
| `zero.mae_steps` | mean absolute error on terminal rows | ≤ 1.0 |
| `tail_saturated_rate` | share of rows whose **mode** is the last bin | report only; >0.3 = grid too small |
| `value_share_shared` | `‖g_v‖²/(‖g_v‖²+‖g_p‖²)` on non-`value_*` params (§9.1) | alarm > 0.05 |
| `dhat_drift` | `median \|Δlog1p d̂\|` on the frozen probe set (§9.2) | alarm > 0.05 |
| `gate_flip_rate` | fraction of probe states changing side of 500 | alarm > 0.05 |
| `ema_gap` | `median \|log1p d̂_ema − log1p d̂_raw\|` (§9.8) | must be ≤ `dhat_drift` |
| `policy_pass_at_1` | mission-level action agreement | must beat the "always block" base rate |
| `gate.decompose_rate` | predicted decompose share, stage 2 | ≈ `true_decompose_rate` |

A head that passes MAE but fails tail recall is not shippable: the gate's worst
error is a false "block" on a 20k-step mission.

## 11a. The 64-bucket geometry: two defects, both fixed and measured

*(§4 of the review. This replaces the old "keep dual, fall back to log" paragraph
and the 2024-style encoding table, which measured at a 500-step tail where every
arm already scored 1.0 — a tail column that cannot discriminate.)*

### The two defects

**D1 — the terminal bin was off by one.** `two_hot_batch` clamped its input to
`xs[0] + 1e-6` and then used the distance *from the lower edge* as the weight on
the lower bin. Both halves are wrong, and together they mean `d = 0` trains bin 1
(centre 1.16 steps) with weight 1. Consequences: every terminal record taught a
+1-bin floor, `decode` inherited it, and `d̂ = 0` was unreachable. `mae_log` cannot
see this (one bin in 63); the gate can ("solved" vs "one step left").

**D2 — 64 uniform bins cannot name a 10k-step mission.** The shipped uniform grid
topped out at `uniform_span = 2048`, so every mission at or above 2048 decoded to
the same number and `tail recall@10k` was structurally **0** (measured 0.000 across
5 seeds). Giving the uniform grid the *full* 32k span does not rescue it either:
linear spacing puts 44 of 63 usable bins above 10k and 19 bins below 100 steps, so
small missions are unrepresentable and the calibration breaks (measured MAE(log)
1.23 vs 0.43 for log64, and the monotone check fails on 1 of 5 seeds).

### The fix, as geometry

```text
bin 0            = [0, 1)                       centre exactly 0
bins 1 .. b-1    = [1, 32768]                   b-1 = 63 bins
  log / dual       even in log1p space           width_log = ln(2^15)/63 ≈ 0.154
  uniform          even in step space            width = 520 steps
decode(0)        = 0.0 exactly                  (the only integer in [0,1))
decode(k)        = centre(k)                    log1p midpoint for log/dual
mode decode      = 3-bin local mean at argmax    never a global expectation
residual         = ±0.5 · width_log[bin]         bounded: it cannot leave the bin
tail flag        = argmax == b-1                  a bucket statement, not a value test
```

Three properties that follow, and are pinned by tests
(`tests/test_head_geometry.py`, `tests/test_bucket_ablation.py`):

1. `encode(0) == 0` and `decode(0) == 0.0` for **every** encoding, and `encode` is
   monotone across `0..5` and across `0..32768`. The terminal bin is a real bin, not
   a rounding artifact.
2. The residual is bounded to `±0.5` bin in log space, so a head cannot "fix" a bad
   bin by extrapolating; the bucket decides the decade and the residual decides
   inside it.
3. `tail_saturated` fires when the **mode** is the last bin. The old test
   (`estimate >= 0.95 · grid_max`) was dead code: the last centre is 30339 steps
   and `0.95 · 32768 = 31130`, so a head that had run out of grid reported
   `tail_saturated: false` and the gate blocked on a number the grid could not
   justify. With the mode test, `uniform64` (whose grid tops at 2048) flags **1.00**
   of its >10k rows, so uniform's failure is at least *safe*: it decomposes.

### The decoder: mode, not expectation

Measured on a bimodal head (0.55 on bin 62 ≈ 28k, 0.45 on bin 55 ≈ 12k): the mode
reports bin 62; the expectation reports 15.2k — a value the head assigns **zero**
probability and a 46% underestimate of its dominant belief. With two equal modes
the expectation names the empty middle bin. That is the mechanism behind "the tail
always looks smaller than it is", and it is why `decode="expected"` is not the
default even though it wins on bulk MAE.

### Ablation

`python3 train/ablate_head.py --tiny` — 300 train / 150 held-out synthetic missions
(20% at 10k..32768, 15% exactly 0), text→steps with bounded noise, 5 seeds, 120
steps, identical trainer:

| arm | MAE(log) | MAE(steps) | zero-exact | tail recall@10k | tail MAE(log) | flag@tail | cov80 | mono |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uniform64 | 1.0063 | 4046.8 | 1.000 | 0.000 | 2.2303 | 1.000 | 0.413 | 5/5 |
| uniform64@32k | 1.2281 | 2018.2 | 1.000 | 0.979 | 0.3724 | 0.000 | 0.584 | 5/5 |
| log64 | 0.4282 | 2542.5 | 1.000 | 1.000 | 0.4079 | 0.000 | 0.913 | 4/5 |
| **dual(mode)** | 0.4321 | 2511.0 | **1.000** | **1.000** | **0.4194** | 0.000 | **0.949** | 4/5 |
| dual(expected) | **0.3218** | 2021.9 | 0.786 | 1.000 | 0.4170 | 0.000 | 0.949 | 5/5 |
| regress | **0.2784** | 1842.4 | 0.234 | 0.731 | 0.5114 | 0.000 | 0.760 | 5/5 |

Winner: **dual(mode)**. Reading, column by column:

- **uniform64** is the D2 defect made visible: tail recall 0.000, tail MAE 2.23
  (every >10k mission is reported as ~2048). Its flag@tail is 1.000, so the gate
  still decomposes — the flag is the only reason this arm is not actively
  dangerous. `uniform64@32k` shows the defect is *spacing*, not just span: better
  tail recall, but MAE(log) 3× worse than log64 and 20% 80%-coverage.
- **regress** wins the bulk (0.2784) and loses the two columns that matter:
  tail recall 0.731 and zero-exact 0.234. A regression head cannot say "exactly
  zero" and systematically under-shoots the tail, which is the one error the gate
  cannot afford.
- **dual(expected)** has the best bucket-arm bulk MAE (0.3218) and pays for it with
  zero-exact 0.786: 21% of terminal missions decode to a nonzero number. That is
  the +1-bin defect coming back through the decoder instead of the target.
- **log64** and **dual(mode)** are within 1% of each other on bulk MAE, but dual
  wins the two columns the gate reads: tail MAE 0.4079 → 0.4194 is a wash, cov80
  0.913 → 0.949 (+4 pts), and the residual is what gives the last three bins any
  resolution at all. One seed (of 5) violates the monotone calibration check in
  both bucket arms — a harness-size warning, not a geometry one.

**Fallback rule (in code: `ablate_head.verdict`).** Bulk MAE is a *tie-breaker*,
not the decision, because the gate is a threshold on a tail quantity with
asymmetric cost (a false `block` on a 10k mission burns the whole budget; a false
`decompose` on a 300-step mission costs one level). Ship plain **log64** instead of
dual only if, on ≥5k held-out records, dual is worse on any of: `tail_recall@10k`
by > 0.02, `tail MAE(log)` by > 0.02, `cov80` by > 0.03, or `zero_exact` at all. A
5% bulk gain is **not** required, and a 1% bulk loss is **not** a reason to fall
back. On this fixture the rule says KEEP dual(mode), with a bulk delta of −0.9%
and a coverage delta of +3.6 pts.

## 12. What tiny mode proves, and what it does not

Proves: the mission-level data pipeline (one record per state, `remaining_steps`
target, action labels, child counts), the mission registry (identical statement →
one mission, `reused: true`, survives reload), sibling dedupe, the compile cache
(lock, `cache_hit`, content-addressed dirs, 4 concurrent compiles → 1 runner
call), the two pools (`main_pool` carries summaries only), the mask spans (system
/ tool_result / subagent_return), the sampling trigger (child-return, never a
tick), the gate in steps with the depth cap and the five contract cases
(900→decompose/9, 400→block/4, 80→single-shot, 480+prev→decompose,
ceil rounds), the unit-regression test, the three units (`steps` default,
`tactic` legacy, `blocks` deprecated), the four encodings, the metrics,
checkpoint reload, stage 2 consuming `data.jsonl` in any unit, and the CLIs
exiting cleanly without torch. Also, since the bucket revision: the terminal
contract (`encode(0)==0`, `decode(0)==0.0`, monotone over `0..5`), the torch
two-hot target agreeing with the pure one on `0`, `0.5` and the grid top, the
bounded residual, the mode-vs-expectation gap on a bimodal head, the tail flag
firing on the last bin only, and the 6-arm ablation with its fallback verdict —
`tests/test_head_geometry.py` (9 checks) and `tests/test_bucket_ablation.py`
(8 checks).

Does not prove: any real value accuracy. The tiny generator makes
`log1p(remaining)` an explicit smooth function of token counts, so MAE there is a
plumbing signal. The real acceptance run is ≥5k held-out mission states from
`bank_local` + MCTS trajectories, judged by §11.

## 13. Open risks

1. `d` labels from reference proofs are *short* (bank p50 = 1 remaining tactic
   step). They teach the head "small" and never "10k". Stage 2's MCTS
   trajectories are the only source of tail labels; the first real run must
   oversample hard missions or tail recall stays at chance.
2. `action_block_steps` (N) is a modelling choice. It does not change the target
   (steps) but it does decide the no-signal band and the number of rounds; a
   mission just above N sits right at the boundary, where one action may or may
   not finish it. Watch the band's actual hit rate, not just N.
3. `h` at the last token is sensitive to prompt formatting (chat template,
   `enable_thinking`). The head must be trained and served through the *same*
   template; this is the #1 silent drift source in the shipped stack.
4. Masking is only as good as the span bookkeeping. If a driver injects child
   output as plain `user` text instead of `subagent_return`, the mask silently
   fails; the harness must classify injected messages, and a pool-writer audit
   should assert that a decomposition record has no unmasked token from a child.
5. A K-sample of `K` decomposition responses can multiply children fast
   (K × 8). `max_depth` bounds depth, not fan-out; the mission registry makes the
   duplicates cheap, but the *dispatch* count still needs a budget cap.
6. `action_block_steps` is the curriculum knob and it is not free: raising N
   widens the no-signal band, so the mix of trainable missions changes with N.
   Compare runs at equal *band*, not equal N, or the value MAE will move for
   reasons that have nothing to do with the head.
7. Terminal-only signal means a model that fails everything learns nothing, and a
   model that succeeds at everything inside the band also learns nothing. The
   healthy regime is narrow: the band must sit *below* the current success
   frontier. Watch `dropped / total` in stage 2; if it goes to 1.0, raise N.
8. The AND aggregate (`max` over pending children) is a heuristic. It is recorded,
   not enforced, precisely because a strict AND backup would throw away the
   information that some children are already solved.
