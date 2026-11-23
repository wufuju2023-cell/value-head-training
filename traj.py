#!/usr/bin/env python3
"""Mission-level trajectories: two pools, explicit token spans, CE masking.

The unit of work is a **mission**, not a tactic step.  One trajectory is one
*agent response* at one state:

    state  = {env, mission_id, lean_context, goal}
    action = a proof block attempt          (action_type = "block")
           | a decomposition into children   (action_type = "decompose")

Two pools, strictly separated (DESIGN.md §7, mcts/SPEC.md §5):

    main_pool  one record per (state, action).  Contains the agent's own text,
               the action type, the child mission ids it created, the
               *summaries* of child returns, and the outcome.  It must never
               contain a child's internals -- no child messages, no child
               token stream, no raw tool output.
    sub_pool   mission_id -> that child session's FULL trajectory.  Used only
               for the child's own CE.

Cross-entropy on `main_pool` is masked.  Unmasked CE would teach the model that
"a subagent returned a proof" is itself the continuation, i.e. it would learn to
call a subagent from the *content of returns* instead of from the quality of the
decision to decompose.  So loss is computed only on the agent's own tokens:

    mask (loss_mask = 0)   every `system` message, every `tool_result` block,
                           every injected `subagent_return` message, pad tokens
    keep (loss_mask = 1)   `assistant` tokens (the decision, the mission
                           contract, the proof block) and `user` tokens

The masking is expressed as *explicit token spans*, recorded per message (and
per block inside a message), never inferred from text.  `mask_spans` returns the
spans; `to_input_ids_loss_mask` merges them into `{input_ids, loss_mask}`.

No torch, no network, no tokenizer download: the default tokenizer is a
deterministic hashed tokenizer, and any object with `.encode(text) -> list[int]`
can be injected instead (a HF tokenizer works unchanged).

    python3 -c "import traj; print(traj.DEFAULT_MASK_POLICY)"
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

# Message / block kinds the trajectory format understands.
KINDS = ("system", "user", "assistant", "tool_result",
         "subagent_return", "subagent_prompt", "pad")

ROLE_TO_KIND = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool_result",
    "tool_result": "tool_result",
    "function": "tool_result",
    "subagent": "subagent_return",
    "subagent_return": "subagent_return",
}

#: The exact rule.  Anything not listed in `mask_kinds` is kept.
DEFAULT_MASK_POLICY: dict[str, Any] = {
    "version": 1,
    "mask_kinds": ["system", "tool_result", "subagent_return"],
    "keep_kinds": ["user", "assistant", "subagent_prompt"],
    "mask_pad": True,
    # diagnostics only: if a masked span is 0 tokens long, say so instead of
    # silently dropping it (a formatting bug otherwise becomes a leak).
    "require_nonempty_masks": False,
}

#: Keys that must never appear in a main_pool record.
CHILD_INTERNAL_KEYS = frozenset({
    "messages", "transcript", "sub_transcript", "child_messages",
    "child_trajectory", "child_tokens", "tool_output", "tool_outputs",
    "full_text", "raw", "tokens",
})

POOLS = ("main_pool", "sub_pool")

#: The only learning signal in this design.  Expert iteration + teacher forcing on
#: trajectories that reach terminal success; planning quality is learned
#: *implicitly* (which plans reach the terminal state), never from a shaping term.
#: A 60-mission decomposition that eventually succeeds is rewarded because it
#: succeeded, not because it decomposed.
CE_SIGNAL = "terminal_success_only"

#: Keys that would reintroduce an auxiliary reward.  If any of them shows up in a
#: trajectory or a pool record, something upstream is paying for "decomposing
#: well" or "using a subagent"; that is a bug, not a tuning choice.
AUXILIARY_REWARD_KEYS = frozenset({
    "decomposition_bonus", "decompose_bonus", "subagent_reward",
    "subagent_bonus", "credit", "credit_assignment", "advantage",
    "shaping", "shaping_reward", "reward", "bonus",
})

FAILED_STATUSES = frozenset({
    "failed", "error", "aborted", "budget_exhausted", "stuck", "sorry",
    "timeout", "rejected",
})


# ---------------------------------------------------------------------------
# Tokenizer stand-in
# ---------------------------------------------------------------------------

class HashTokenizer:
    """Deterministic tokenizer: word/punct split -> crc32 -> [0, vocab_size).

    Good enough to give every span real token offsets in tests on a machine with
    no model.  `encode` is the whole interface `traj.py` needs, so a HF
    tokenizer (`tok.encode`) can be passed to every function here instead.
    """

    def __init__(self, vocab_size: int = 4096, salt: str = "vht"):
        self.vocab_size = int(vocab_size)
        self.salt = salt

    def encode(self, text: str) -> list[int]:
        text = "" if text is None else str(text)
        out: list[int] = []
        cur: list[str] = []
        for ch in text:
            if ch.isalnum() or ch in "_'":
                cur.append(ch)
            else:
                if cur:
                    out.append(self._id("".join(cur)))
                    cur = []
                if not ch.isspace():
                    out.append(self._id("p:" + ch))
        if cur:
            out.append(self._id("".join(cur)))
        return out

    def _id(self, token: str) -> int:
        return zlib.crc32((self.salt + "\x00" + token).encode("utf-8")) % self.vocab_size

    def __call__(self, text: str) -> list[int]:  # convenience
        return self.encode(text)


# ---------------------------------------------------------------------------
# Message segments: (kind, start, end) token spans
# ---------------------------------------------------------------------------

def message_kind(msg: dict) -> str:
    kind = msg.get("kind")
    if kind:
        return kind
    return ROLE_TO_KIND.get(str(msg.get("role", "user")), "user")


def message_segments(msg: dict, tokenizer: Any) -> list[dict]:
    """Split one message into token segments, each with its own kind.

    A message may carry `blocks` (a tool result nested inside an assistant
    turn, a text block next to a call).  Segments are computed on the *tokens*,
    so the span is exact even when the text is concatenated.

    Block kinds that describe *formatting* (`text`, `markdown`, `reasoning`)
    inherit the message's kind: an assistant's prose block is assistant text.
    Only `tool_result` / `subagent_return` introduce a new masking class.
    """
    blocks = msg.get("blocks")
    if blocks:
        segs: list[dict] = []
        for blk in blocks:
            kind = blk.get("kind")
            if kind in (None, "", "text", "markdown", "reasoning"):
                kind = message_kind(msg)
            toks = blk.get("tokens")
            if toks is None:
                toks = tokenizer.encode(blk.get("text", ""))
            segs.append({"kind": kind, "tokens": list(toks)})
        return segs
    toks = msg.get("tokens")
    if toks is None:
        toks = tokenizer.encode(msg.get("text", ""))
    return [{"kind": message_kind(msg), "tokens": list(toks)}]


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """One agent response at one mission-level state."""

    mission_id: str
    env: str = "default"
    goal: str = ""
    lean_context: str = ""
    action_type: str = "block"            # "block" | "decompose"
    action_text: str = ""                 # the agent's own response text
    child_missions: list[str] = field(default_factory=list)
    child_returns: list[dict] = field(default_factory=list)  # summaries only
    messages: list[dict] = field(default_factory=list)
    outcome: dict = field(default_factory=dict)
    depth: int = 0
    parent_mission_id: str | None = None
    tokens: list[int] = field(default_factory=list)

    # -- derived -----------------------------------------------------------
    @property
    def state(self) -> dict:
        return {"env": self.env, "mission_id": self.mission_id,
                "lean_context": self.lean_context, "goal": self.goal}

    @property
    def n_child_missions(self) -> int:
        return len(self.child_missions)

    def children_pending(self) -> bool:
        done = {str(r.get("mission_id")) for r in self.child_returns}
        return any(str(c) not in done for c in self.child_missions)

    # -- encoding ----------------------------------------------------------
    def encode(self, tokenizer: Any | None = None) -> list[int]:
        tok = tokenizer or HashTokenizer()
        ids: list[int] = []
        for msg in self.messages:
            for seg in message_segments(msg, tok):
                ids.extend(seg["tokens"])
        self.tokens = ids
        return ids

    def spans(self, tokenizer: Any | None = None,
              policy: dict | None = None) -> dict:
        """Explicit spans: {segments: [...], masked: [...], kept: [...]}."""
        return mask_spans(self, tokenizer=tokenizer, policy=policy)

    def to_json(self) -> dict:
        return asdict(self)


def _as_trajectory(obj: "Trajectory | dict") -> Trajectory:
    if isinstance(obj, Trajectory):
        return obj
    known = {k: obj[k] for k in Trajectory.__dataclass_fields__ if k in obj}
    return Trajectory(**known)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def _normalise_policy(policy: dict | None) -> dict:
    pol = dict(DEFAULT_MASK_POLICY)
    if policy:
        pol.update(policy)
    # keep_kinds always wins over mask_kinds: "keep the agent's own tokens"
    # is the invariant, not a tunable.
    mask = [k for k in pol.get("mask_kinds", []) if k not in set(pol.get("keep_kinds", []))]
    pol["mask_kinds"] = mask
    return pol


def mask_spans(traj: "Trajectory | dict", tokenizer: Any | None = None,
               policy: dict | None = None) -> dict:
    """Token spans to mask, as explicit ranges.

    Returns {"segments": [{kind, start, end}], "masked": [...], "kept": [...],
             "n_tokens": int}.  `masked`/`kept` are merged, disjoint [start, end)
    ranges so they can be applied in one pass.
    """
    t = _as_trajectory(traj)
    tok = tokenizer or HashTokenizer()
    pol = _normalise_policy(policy)
    mask_kinds = set(pol["mask_kinds"])

    segments: list[dict] = []
    offset = 0
    for msg in t.messages:
        for seg in message_segments(msg, tok):
            end = offset + len(seg["tokens"])
            segments.append({"kind": seg["kind"], "start": offset, "end": end})
            offset = end

    masked = _merge([(s["start"], s["end"]) for s in segments if s["kind"] in mask_kinds])
    kept = _merge([(s["start"], s["end"]) for s in segments if s["kind"] not in mask_kinds])
    return {"segments": segments, "masked": masked, "kept": kept, "n_tokens": offset,
            "mask_kinds": sorted(mask_kinds), "policy_version": pol.get("version", 1)}


def _merge(ranges: list[tuple[int, int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for start, end in sorted(r for r in ranges if r[1] > r[0]):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def to_input_ids_loss_mask(traj: "Trajectory | dict", *,
                           tokenizer: Any | None = None,
                           policy: dict | None = None,
                           pad_to: int | None = None,
                           pad_id: int = 0) -> dict:
    """Trajectory JSON -> {input_ids, loss_mask} (+ diagnostics).

    loss_mask[i] == 1 only for the agent's own tokens (decision, mission
    contract, proof block) and the task/user message.  System prompts, tool
    results and injected subagent returns are 0, as are pad tokens.
    """
    t = _as_trajectory(traj)
    tok = tokenizer or HashTokenizer()
    spans = mask_spans(t, tokenizer=tok, policy=policy)
    ids = t.encode(tok) if not t.tokens else list(t.tokens)
    if len(ids) != spans["n_tokens"]:
        # tokens were pre-computed by a different tokenizer: recompute.
        ids = t.encode(tok)
        spans = mask_spans(t, tokenizer=tok, policy=policy)

    loss_mask = [1] * len(ids)
    for start, end in spans["masked"]:
        for i in range(max(0, start), min(end, len(ids))):
            loss_mask[i] = 0

    if pad_to is not None:
        n_pad = max(0, int(pad_to) - len(ids))
        ids = ids + [pad_id] * n_pad
        loss_mask = loss_mask + [0] * n_pad

    return {
        "input_ids": ids,
        "loss_mask": loss_mask,
        "n_tokens": len(ids),
        "n_kept": sum(loss_mask),
        "n_masked": len(loss_mask) - sum(loss_mask),
        "spans": spans["masked"],
        "segments": spans["segments"],
        "mask_kinds": spans["mask_kinds"],
    }


def assert_no_auxiliary_reward(rec: dict, *, where: str = "record") -> None:
    """Fail if an auxiliary reward leaked in (decomposition/subagent/credit)."""
    leaked = sorted(k for k in rec if k in AUXILIARY_REWARD_KEYS)
    if leaked:
        raise ValueError(
            f"{where} carries an auxiliary reward signal {leaked}: this design is "
            f"{CE_SIGNAL}, so planning is scored only by reaching terminal success")


# ---------------------------------------------------------------------------
# Terminal-only signal
# ---------------------------------------------------------------------------

def is_terminal_success(traj: "Trajectory | dict",
                        require_children: bool = True) -> bool:
    """Did this trajectory reach terminal success for its own mission?

    Terminal success = the original (context, goal) is discharged and the budget
    is not exhausted.  For a decomposition that means every child it created is
    itself discharged — the parent is only solved when `AND(children)` is.

    This is the *only* admission test for the CE pool.  Nothing here looks at
    how the success was reached: a 60-mission plan and a 1-block plan are equal.
    """
    if isinstance(traj, Trajectory):
        out = traj.outcome or {}
        children = list(traj.child_missions)
        returns = list(traj.child_returns)
    else:
        out = traj.get("outcome")
        if out is None:
            # a searched-node record: {solved, status?, budget_exhausted?}
            out = {k: traj[k] for k in ("solved", "status", "budget_exhausted")
                   if k in traj}
        children = list(traj.get("child_missions") or [])
        returns = list(traj.get("child_returns") or traj.get("child_summaries") or [])
    assert_no_auxiliary_reward(out, where="outcome")
    status = str(out.get("status", "accepted")).lower()
    if out.get("budget_exhausted") or status in FAILED_STATUSES:
        return False
    if not bool(out.get("solved")):
        return False
    if not require_children or not children:
        return True
    solved = {str(r.get("mission_id")): bool(r.get("solved")) for r in returns}
    for child in children:
        if not solved.get(str(child), False):
            return False       # pending or failed child: the parent is not discharged
    return True


def filter_ce_pool(items: list, *, include_failed: bool = False,
                   diagnostic: bool = True) -> dict:
    """Drop every non-terminal-success trajectory from the CE pool.

    Returns {"kept", "dropped", "reason_counts", "signal"}.  Dropped items keep a
    `drop_reason` so failures stay available as diagnostics — they are excluded
    from CE, not deleted.
    """
    kept: list = []
    dropped: list = []
    reasons: dict[str, int] = {}

    def drop(item, reason: str):
        reasons[reason] = reasons.get(reason, 0) + 1
        if diagnostic:
            rec = dict(item) if isinstance(item, dict) else {"mission_id": item.mission_id}
            assert_no_auxiliary_reward(rec, where="CE pool item")
            rec["drop_reason"] = reason
            dropped.append(rec)

    for item in items:
        if is_terminal_success(item):
            kept.append(item)
            continue
        raw = item if isinstance(item, dict) else asdict(item)
        out = raw.get("outcome") or {}
        status = str(out.get("status", "accepted")).lower()
        if out.get("budget_exhausted") or status in FAILED_STATUSES:
            drop(item, "not_success")
        elif not bool(out.get("solved")):
            drop(item, "not_solved")
        else:
            drop(item, "children_unproved")
    return {"kept": kept, "dropped": dropped, "reason_counts": reasons,
            "signal": CE_SIGNAL, "include_failed": bool(include_failed)}


def block_progress(old_state, new_state) -> dict:
    """Did applying one N-step block move the state?

    `old_state`/`new_state` are mission states (dicts with `goal`, `lean_context`,
    optional `solved`), produced by the agent's response and the Lean run.  A
    block is an atomic action, so "progress" is exactly "the state changed":

        changed goal                     -> progress (the block rewrote the goal)
        longer/different lean_context    -> progress (something was proved)
        solved                           -> progress (mission discharged)
        byte-identical state             -> no-op, a failed action for this node

    A no-op is not an error: it is a failed action, kept for diagnostics and
    excluded from CE by the terminal-only rule.
    """
    def norm(state) -> dict:
        if isinstance(state, dict):
            return {"goal": str(state.get("goal", "")),
                    "lean_context": str(state.get("lean_context", state.get("context", ""))),
                    "solved": bool(state.get("solved", False))}
        return {"goal": str(state), "lean_context": "", "solved": False}

    old, new = norm(old_state), norm(new_state)
    goal_changed = old["goal"].strip() != new["goal"].strip()
    context_changed = old["lean_context"].strip() != new["lean_context"].strip()
    solved = bool(new["solved"]) and not bool(old["solved"])
    changed = [name for name, flag in (("goal", goal_changed),
                                       ("lean_context", context_changed),
                                       ("solved", solved)) if flag]
    return {
        "progress": bool(changed),
        "no_op": not changed,
        "changed": changed,
        "goal_changed": goal_changed,
        "context_delta": len(new["lean_context"]) - len(old["lean_context"]),
        "solved": solved,
    }


# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------

def main_record(traj: "Trajectory | dict") -> dict:
    """Project one trajectory to its main_pool record (no child internals)."""
    t = _as_trajectory(traj)
    summaries = []
    for ret in t.child_returns:
        summaries.append({
            "mission_id": str(ret.get("mission_id")),
            "summary": str(ret.get("summary", "")),
            "status": str(ret.get("status", ret.get("outcome", "unknown"))),
            "solved": bool(ret.get("solved", False)),
            "remaining_blocks": ret.get("remaining_blocks"),
        })
    rec = {
        "mission_id": t.mission_id,
        "env": t.env,
        "state": t.state,
        "action_type": t.action_type,
        "action_text": t.action_text,
        "child_missions": list(t.child_missions),
        "child_summaries": summaries,
        "n_child_missions": t.n_child_missions,
        "outcome": dict(t.outcome),
        "depth": t.depth,
        "parent_mission_id": t.parent_mission_id,
        # terminal-only signal: recorded, but the pool filter is the authority
        "terminal_success": is_terminal_success(t),
        "ce_eligible": is_terminal_success(t),
        "signal": CE_SIGNAL,
    }
    assert_main_record_clean(rec)
    assert_no_auxiliary_reward(rec, where="main_pool record")
    return rec


def assert_main_record_clean(rec: dict) -> None:
    """Fail loudly if a child internal leaked into a main_pool record."""
    leaked = sorted(k for k in rec if k in CHILD_INTERNAL_KEYS)
    if leaked:
        raise ValueError(f"main_pool record carries child internals: {leaked}")
    for summary in rec.get("child_summaries", []):
        bad = sorted(k for k in summary if k in CHILD_INTERNAL_KEYS)
        if bad:
            raise ValueError(f"child summary carries internals: {bad}")


def split_pools(trajectories: list["Trajectory | dict"]) -> dict:
    """Split a mixed trajectory list into {main_pool, sub_pool}.

    main_pool  list of per-(state, action) records, summaries only.
    sub_pool   {mission_id: [full trajectory dict, ...]} for that mission's CE.

    Both pools are returned in full; CE reads them through `filter_ce_pool`, which
    keeps only terminal successes.
    """
    main_pool: list[dict] = []
    sub_pool: dict[str, list[dict]] = {}
    for raw in trajectories:
        t = _as_trajectory(raw)
        main_pool.append(main_record(t))
        sub_pool.setdefault(t.mission_id, []).append(asdict(t))
    return {"main_pool": main_pool, "sub_pool": sub_pool}


def ce_pools(trajectories: list["Trajectory | dict"], *,
             include_failed: bool = False) -> dict:
    """The pools as CE actually consumes them: failures dropped from both."""
    pools = split_pools(trajectories)
    main = filter_ce_pool(pools["main_pool"], include_failed=include_failed)
    flat_subs = [t for group in pools["sub_pool"].values() for t in group]
    subs = filter_ce_pool(flat_subs, include_failed=include_failed)
    return {
        "main_pool": main["kept"],
        "sub_pool": {t["mission_id"]: t for t in subs["kept"]},
        "dropped": main["dropped"] + subs["dropped"],
        "reason_counts": main["reason_counts"],
        "signal": CE_SIGNAL,
    }


def write_pools(path: str, pools: dict) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(pools, fh, ensure_ascii=False, indent=2)
    return path


if __name__ == "__main__":  # tiny self-check, no torch
    demo = Trajectory(
        mission_id="m-demo", env="tiny", goal="theorem t : True",
        lean_context="import Mathlib", action_type="decompose",
        action_text="split into two lemmas",
        child_missions=["m-a", "m-b"],
        child_returns=[{"mission_id": "m-a", "summary": "closed by simp",
                        "status": "accepted", "solved": True}],
        messages=[
            {"role": "system", "text": "you are a prover"},
            {"role": "user", "text": "theorem t : True"},
            {"role": "assistant", "text": "split into two lemmas"},
            {"role": "assistant", "blocks": [
                {"kind": "text", "text": "checking"},
                {"kind": "tool_result", "text": "ok: no goals"}]},
            {"role": "subagent", "text": "m-a returned: closed by simp"},
        ],
        outcome={"solved": False},
    )
    out = to_input_ids_loss_mask(demo)
    print(json.dumps({"tokens": out["n_tokens"], "kept": out["n_kept"],
                      "masked": out["n_masked"], "spans": out["spans"]}))
