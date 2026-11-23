# SPEC — mission-level MCTS loop

Status: design spec. Pseudocode + the invariants it must hold. The executable part of
this spec is `mcts/schedule.py` (`gate_action`, `should_sample`, `and_state`),
`traj.py` (pools + masking) and `cache.py` (registry, compile lock, dedupe).
Everything below is CPU-only and network-free.

Unit of search = **mission**, not tactic step. One action = one agent response.

    INVARIANT: d_hat is in proof steps; N is in proof steps; rounds = ceil(d_hat/N).

`d_hat`, the gate threshold (500), `action_block_steps` (N), the no-signal band,
the dataset target and the trajectories are all in **proof steps**. There is no
blocks/steps conversion anywhere in the loop.

---

## 1. State

```text
State := {
  env            : str            # lean project / task env id ("bank:T1", "mathlib", ...)
  mission_id     : str            # cache.mission_id_for(env, statement_formal)
  lean_context   : str            # imports + preamble + what is already proved
  goal           : str            # statement_formal of THIS mission
  depth          : int            # 0 at the root mission
  parent         : str | None
  pending        : list[str]      # child missions dispatched, not yet returned
  in_flight      : int            # responses of the current K-sample still executing
  sampled        : int            # how many K-samples already happened here
  and_record     : dict | None    # schedule.and_state(...) bookkeeping, not enforced
}
```

The initial state is the root mission's statement with an empty proof context.
`remaining_steps` (the value target, in proof steps) is a property of the state,
predicted by the head as `d_hat_steps`; it is never stored as truth during search.

## 2. Action: an N-step block (or a decomposition)

```text
Action :=
  | Block(proof_text)                  # at most `action_block_steps` proof steps
  | Decompose([child_statement ...])   # one agent response that splits the mission
```

`action_block_steps` (config, default **10**; candidates 10 / 20 / 50 / 100) is
the maximum number of proof steps a single sampled action may contribute. It is
the same unit as `d_hat_steps`, so `rounds_left = ceil(d_hat_steps / N)` and
`single_shot = d_hat_steps <= N` need no scaling. Inside
one action the agent is completely free: plan, sketch, compile, read feedback,
search the web, call tools. Only the **emitted proof block** is capped at N, so
the action is atomic for MCTS without pretending that a human writes one `simp`.

Both actions are one agent response. Single-tactic actions are not modelled.

### The state-change test (what makes a block an action)

```text
old := (lean_context, goal)
execute Block in the mission workspace, emitting at most N steps
new := (lean_context', goal')          # from the Lean run, not from the agent

if block_progress(old, new).progress:   # goal changed | context grew | solved
      -> state transition: push(block_finished), next K-sample round is legal
else:                                   # identical state
      -> no-op / failed action for this node: no transition, no CE
```

`traj.block_progress` is the definition; `mcts/schedule.py` re-exports it so
there is exactly one comparison in the codebase. A block that rewrites the goal
counts as progress, and so does one that only grows the context — both move the
mission. A byte-identical result is a failed action: recorded for diagnostics,
excluded from CE (terminal-only, §6), and it does not trigger the next round.

Transitions:

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

### The no-signal band (d̂ ≤ N)

```text
if d_hat_steps <= action_block_steps:     # both in proof steps, no conversion
      mode = "single_shot"        # sampled once, never expanded
      learning_signal = False     # no CE record from this mission
else:
      mode = "multi_round"        # K-samples, child returns, pool writes
```

A mission whose estimated remaining work fits in one N-step block never enters
multi-round MCTS: there is nothing to search, and the trajectory teaches nothing
the base model cannot already do. The sampler still *attempts* it (the mission
has to be solved), it just does not iterate or write a CE record. That is
intended: the base model is assumed to handle missions this small in the free
agent setting.

Curriculum: `action_block_steps` rises with the model (10 → 20 → 50 → 100; table
`schedule.CURRICULUM`). If the model solves everything at N = 100/200/300/400,
stop generating learning signal for missions ≤ 1000 and raise N instead — the
band moves with N, so the curriculum is what keeps the signal alive.

## 3. The loop

```text
LOOP(root_mission, K, max_depth, threshold):
  S := initial_state(root_mission)
  frontier := {S}
  events := []                       # child returns / block completions
  sample(S, "start")

  while frontier nonempty or in_flight > 0:
      e := WAIT(events)              # blocks on a message; there is no timer
      if e is None: break
      S := state_of(e)

      # ---- 3a. bookkeeping: child returns update the parent's AND record
      if e.kind == "child_return":
          S.pending.remove(e.mission_id)
          S.and_record := and_state(S.mission_id, children_of(S))
          write_sub_pool(e.mission_id)           # child's own full trajectory
          if S.pending is empty:
              write_main_pool(S)                 # parent action is now evaluable

      # ---- 3b. can we sample here?  (never while children are pending)
      d := should_sample(S, e.kind, k=K)
      if d.sample:
          sample(S, e.kind)
      else:
          events.extend(d.wait_for)              # park until the next real event

  # ---- 3c. one K-sample: K sibling responses at one state
  sample(S, trigger):
      children_before := S.pending
      for i in 1..K:
          r := agent.respond(S)                  # temperature > 0
          if r is Decompose:
              kids := dedupe_siblings(r.children)  # same statement -> one mission
              for c in kids:
                  m := registry.get_or_create(c.env, c.statement_formal)
                  # log reused:true when the mission already existed
                  dispatch(m); S.pending.add(m.mission_id)
              record_action(S, r, child_ids, reused_flags)
          else:                                  # N-step Block
              in_flight += 1
              old := (S.lean_context, S.goal)
              execute_lean(r.proof_text, cwd=m.workspace)  # at most N emitted steps
              new := state_from_lean_run()
              in_flight -= 1
              if block_progress(old, new).progress:
                  record_action(S, r, outcome)
                  push_event(block_finished, S)   # state moved: next round legal
              else:
                  record_action(S, r, outcome, no_op=True)   # no transition
      S.sampled += 1
```

Trigger rule, in one place:

```text
sample(S, trigger) is allowed iff
    trigger in {start, child_return, block_finished}
and S.pending == []
and S.in_flight == 0
and not (trigger == start and S.sampled > 0)
```

A timer (`tick`) never samples: siblings must be sampled against the state they
were actually sampled in, not against a state that has since moved. If children
are pending, the only legal move is to wait — expanding a state with an
outstanding decomposition would create two trees for one obligation.

Depth cap is on decomposition, not on sampling: `depth >= max_depth` forces
`Block` even when the gate wants `Decompose` (`reason = depth_capped`).

## 4. Gate at a state (why decompose)

```text
gate_action(d_hat_steps, action_block_steps=N, depth, max_depth, threshold=500):
    rounds_left := ceil(d_hat_steps / N)
    single_shot := d_hat_steps <= N
    if d_hat_steps is None:                       -> Block   (no_estimate)
    if tail_saturated:                            -> Decompose (mandatory)
    if d_hat_steps >= threshold and depth < max_depth:  -> Decompose (mandatory)
    if d_hat_steps >= threshold and depth >= max_depth: -> Block   (depth_capped)
    else:                                         -> Block   (below_threshold)
```

`d_hat_steps` estimates **remaining proof steps for this mission**. Above the
threshold, decomposition is required, not merely permitted. The threshold is a
budget boundary in work, not in responses: N may rise with the curriculum without
moving what 500 means. A child is *not*
required to be small: it may itself be above 500 and decompose again. That is
how a hard root fans out to dozens of missions on the first move (a 60-mission
opening decomposition is a normal shape, not a pathology). The only bound on
fan-out depth is `max_depth` (config, default 3).

`Block` here means the N-step block of §2: one response, at most
`action_block_steps` emitted proof steps. Gate first, band second — a mission
that passes the gate as `block` and also lands in the no-signal band (§2) is
attempted once and produces no CE record.

Children are cached: `sha256(env, statement_formal)` -> mission id, so the same
lemma reached by two different decompositions is the same mission with the same
workspace and the same compiled `.olean`.

## 5. Pools

```text
                  sampled action (one agent response)
                              |
              +---------------+----------------+
              |                                |
        action = Block                  action = Decompose
              |                                |
    lean executes the block           children dispatched (mission DAG)
    outcome: solved / failed          each child returns a summary
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
| `main_pool` | (state, action) | agent's own text, action type, child mission ids, child **summaries**, outcome | child messages, child tool output, child token streams |
| `sub_pool` | `mission_id` | that child session's full trajectory | the parent's decisions |

`traj.split_pools(trajectories)` builds both and raises if a main record carries
child internals (`CHILD_INTERNAL_KEYS`).

**Admission to CE is terminal success only.** Both pools are *written* for every
action (diagnostics want the failures), but CE reads them through
`traj.filter_ce_pool`, which keeps a trajectory only if the original
`(context, goal)` was discharged and its budget was not exhausted:

```text
keep   outcome.solved and status not in FAILED_STATUSES and every child solved
drop   not_solved | not_success (failed / budget_exhausted / stuck / sorry) | children_unproved
```

There is no reward for decomposing, and none for calling a subagent: a
60-mission plan that eventually succeeds is rewarded *because it succeeded*.
Planning quality is learned implicitly, through which plans reach the terminal
state. Any record carrying `decomposition_bonus`, `subagent_reward`, `credit`,
`advantage` or `shaping` is rejected by `traj.assert_no_auxiliary_reward` — the
signal is `CE_SIGNAL = "terminal_success_only"` and it is enforced, not assumed.
Value targets stay remaining-work estimates; they are never rewards.

## 6. CE masking on main_pool

Loss is computed only on the agent's own tokens:

```text
loss_mask[i] = 1  for  assistant tokens   (decision, mission contract, proof block)
                        user tokens      (the task statement)
loss_mask[i] = 0  for  system messages
                        tool_result blocks (anything Lean printed)
                        subagent_return messages (anything a child returned)
                        pad tokens
```

Spans are explicit `[start, end)` token ranges recorded per message and per
block inside a message, merged, then applied. `traj.to_input_ids_loss_mask`
returns `{input_ids, loss_mask}`.

Why: with unmasked CE the cheapest way to lower the loss on a decomposition
record is to reproduce the child's returned proof, so the model learns "a
subagent returns a proof" as the continuation and treats calling a subagent as a
lexical shortcut. Masking leaves only the decision and the contract, which is the
thing being reinforced.

## 7. AND bookkeeping (secondary)

`and_state(parent, children)` records `{kind: "AND", children, solved, pending,
discharged}` and `aggregate_children_value` takes `max` over unresolved children
(optimistic aggregate, not a strict AND backup). The **mission DAG**, keyed by
`mission_id`, is the primary structure; the AND record exists so a reader can see
the shape and so the pool writer knows when a parent action is evaluable.

## 8. Reuse / caching touchpoints

| point | call | guarantee |
| --- | --- | --- |
| mission identity | `registry.get_or_create(env, statement_formal)` | same statement+env -> same id/workspace, `reused: true` |
| sibling dedupe | `dedupe_siblings(children)` before dispatch | identical siblings dropped once |
| compile | `CompileCache.compile(env, file, content)` | content-addressed task dir, `cache_hit` per call, `flock` so two children never compile the same file at once |

## 9. Invariants (asserted in tests)

1. No sample while `pending` is non-empty.
2. `tick` never samples.
3. Exactly one of `Block` / `Decompose` per action; no single-tactic action, and
   a `Block` emits at most `action_block_steps` proof steps.
4. `Decompose` at `depth >= max_depth` is impossible (`depth_capped`).
5. `main_pool ∩ child internals = ∅`.
6. `loss_mask` is 0 on every system / tool_result / subagent_return token.
7. Same `(env, statement)` -> same `mission_id` -> `reused: true` on the second
   registration.
8. Same file content compiled twice -> second call `cache_hit: true`, runner ran
   once.
9. A trajectory that did not reach terminal success never enters the CE pool.
10. A block that leaves the state identical does not create a transition and does
    not trigger the next K-sample round.
11. A mission with `d̂ ≤ action_block_steps` (both in proof steps) is
    `single_shot`: sampled once, `learning_signal: false`, and skipped by the
    round scheduler afterwards.
12. `rounds_left == ceil(d̂ / N)`; a 900-step mission at N=100 is 9 rounds and
    decomposes, and it is never re-read as 300 "blocks" (the unit bug).
13. No `decomposition_bonus` / `subagent_reward` / `credit` / `advantage` /
    `shaping` key survives into a pool record or an outcome.
