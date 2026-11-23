#!/usr/bin/env python3
"""Mission-level MCTS scheduling primitives (pure functions, no torch, no I/O).

These are the three decisions the loop in `mcts/SPEC.md` makes, kept as
functions so they are testable instead of living only in prose:

1. `gate_action`   — block or decompose at this state (d̂ > 500 ⇒ must decompose,
                     depth cap ⇒ cannot decompose further);
2. `should_sample` — is the next K-sample triggered *now*?  The trigger is a
                     child returning or a block attempt finishing.  A timer is
                     explicitly not a trigger, and nothing is expanded while
                     children are pending.
3. `and_state`     — bookkeeping for `AND(children)`.  Recorded, not enforced:
                     the mission DAG is the primary structure and the parent's
                     value aggregates over children.

    python3 -c "import sys; sys.path.insert(0,'mcts'); import schedule; \
print(schedule.should_sample({}, 'tick'))"
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "train"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gate import DEFAULT_THRESHOLD, decide_gate  # noqa: E402
from traj import block_progress  # noqa: E402  (re-exported: one definition of "progress")

# d_hat is in proof steps; N is in proof steps; rounds = ceil(d_hat/N).
# There is exactly one unit in this loop.  Do not reintroduce a blocks/steps
# conversion here: that mismatch made a 900-step mission read as 300 "blocks"
# and pass the 500 gate without decomposing.

#: Why the loop sampled.  `tick` is deliberately absent from the *trigger* set.
EVENTS = ("start", "child_return", "block_finished", "tick")
TRIGGERS = ("start", "child_return", "block_finished")
DEFAULT_MAX_DEPTH = 3
DEFAULT_K = 4

#: One action is an **N-step block**: the maximum number of proof steps a single
#: sampled action may contribute.  Inside one action the agent is free (plan,
#: sketch, compile, read feedback, call tools); only the emitted proof block is
#: capped.  Single-tactic actions are not modelled.
DEFAULT_ACTION_BLOCK_STEPS = 10
ACTION_BLOCK_STEPS_CANDIDATES = (10, 20, 50, 100)

#: Curriculum: raise N as the model improves.  A mission whose estimate is inside
#: the band `d̂ <= N` can be finished by one sample, so it never enters
#: multi-round MCTS and yields no learning signal — intended, because the base
#: model is assumed to handle those in the free-agent setting.  If the model
#: solves everything at N = 100/200/300/400, stop generating signal for
#: missions <= 1000 and raise N instead.
CURRICULUM = (
    {"stage": 1, "action_block_steps": 10, "note": "base model: warm start"},
    {"stage": 2, "action_block_steps": 20, "note": "after the pass rate plateaus"},
    {"stage": 3, "action_block_steps": 50, "note": "long blocks become writable"},
    {"stage": 4, "action_block_steps": 100, "note": "ceiling: stop at missions <= 1000"},
)


def in_no_signal_band(d_hat_steps: float | None,
                      action_block_steps: int = DEFAULT_ACTION_BLOCK_STEPS) -> bool:
    """True when one N-step sample can finish the mission (no CE signal).

    Both arguments are in **proof steps**: `d̂ <= N`.  No conversion, no
    multiplication — `d̂` and `N` are the same unit by construction.
    """
    value = _finite(d_hat_steps)
    if value is None:
        return False
    return value <= max(1, int(action_block_steps))


def _finite(x) -> float | None:
    """Coerce to a finite non-negative float, else None."""
    if x is None:
        return None
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or value < 0:
        return None
    return value


def rounds_left(d_hat_steps: float | None,
                action_block_steps: int = DEFAULT_ACTION_BLOCK_STEPS) -> int | None:
    """How many N-step actions the mission still needs: ceil(d̂ / N)."""
    value = _finite(d_hat_steps)
    if value is None:
        return None
    return max(1, math.ceil(value / max(1, int(action_block_steps))))


def gate_action(d_hat_steps: float | None, *,
                threshold: float = DEFAULT_THRESHOLD,
                prev: str = "prove", ci_high: float | None = None,
                tail_saturated: bool = False,
                depth: int = 0, max_depth: int = DEFAULT_MAX_DEPTH,
                action_block_steps: int = DEFAULT_ACTION_BLOCK_STEPS) -> dict:
    """Mission-level gate on **proof steps**: `block` or `decompose`.

    `d_hat_steps` is the head's estimate of remaining proof steps for this
    mission, and `action_block_steps` is N, the steps one action may emit.  Same
    unit on both sides, so:

        rounds_left = ceil(d_hat_steps / N)
        single_shot = d_hat_steps <= N          # one action finishes it
        decompose   = d_hat_steps >= threshold  # hysteresis: exit at 0.8*threshold

    Above the threshold decomposition is required, not a fallback; a child is not
    required to be small (it may decompose again), so the only bound on depth is
    `max_depth`.  The threshold comparison is `>=` to stay identical to the shared
    gate table (`train/gate.py` + `lib/gate.mjs`), which enters decompose at
    exactly 500.
    """
    depth = int(depth or 0)
    max_depth = int(max_depth)
    n_block = max(1, int(action_block_steps))
    value = _finite(d_hat_steps)
    base = decide_gate(float("nan") if value is None else value,
                       threshold=threshold, prev=prev, ci_high=ci_high,
                       tail_saturated=tail_saturated)
    action = "decompose" if base["action"] == "decompose" else "block"
    reason = base["reason"]
    mandatory = action == "decompose"
    if action == "decompose" and depth >= max_depth:
        # The gate wants children but the tree is deep enough: the mission has
        # to be attempted as an N-step block.  Escalation is the caller's problem.
        action, reason, mandatory = "block", "depth_capped", False
    single_shot = in_no_signal_band(value, n_block)
    return {"action": action, "reason": reason, "mandatory": bool(mandatory),
            "depth": depth, "max_depth": max_depth,
            "threshold": base["threshold"], "estimate": base["estimate"],
            "action_block_steps": n_block,
            "rounds_left": rounds_left(value, n_block),
            "single_shot": bool(single_shot),
            "no_signal_band": bool(single_shot),
            "learning_signal": bool(value is not None and not single_shot),
            "suggested_children": base["suggested_children"] if action == "decompose" else 0}


def should_sample(state: dict, event: str, *, k: int = DEFAULT_K) -> dict:
    """Is the next K-sample triggered?  Returns {sample, reason, k, wait_for}.

    state keys (all optional):
        pending_children     child mission ids not yet returned
        in_flight            responses of the current K-sample still executing
        sampled_this_state   how many K-samples already happened at this state
        depth, max_depth
        d_hat_steps          head estimate for this state, in proof steps
        action_block_steps   N for this curriculum stage, in proof steps
    """
    state = state or {}
    pending = list(state.get("pending_children") or [])
    in_flight = int(state.get("in_flight") or 0)
    sampled = int(state.get("sampled_this_state") or 0)
    depth = int(state.get("depth") or 0)
    max_depth = int(state.get("max_depth", DEFAULT_MAX_DEPTH))
    n_block = int(state.get("action_block_steps", DEFAULT_ACTION_BLOCK_STEPS))
    # `d_hat` is accepted as an alias; both are proof steps.
    d_hat = state.get("d_hat_steps", state.get("d_hat"))
    band = in_no_signal_band(d_hat, n_block)

    def no(reason: str, wait_for: str | None = None) -> dict:
        return {"sample": False, "reason": reason, "k": 0,
                "wait_for": wait_for, "event": event,
                "no_signal_band": band, "learning_signal": False}

    if event == "tick":
        # Owner decision: a timer never triggers a new K-sample.  Expanding on a
        # clock is how a tree ends up with siblings that were sampled against a
        # state that has already moved.
        return no("timer_is_not_a_trigger", _wait_hint(pending, in_flight))
    if event not in TRIGGERS:
        return no("unknown_event", _wait_hint(pending, in_flight))
    if pending:
        # Children exist and have not returned: never expand again here.
        return no("children_pending", "child_return")
    if in_flight > 0:
        return no("samples_in_flight", "block_finished")
    if band and sampled >= 1:
        # d_hat_steps <= N: one sample finishes it.  Never enters multi-round MCTS
        # and therefore produces no learning signal — the base model already
        # solves missions this small in the free-agent setting.
        return no("no_signal_band", None)
    if event == "start" and sampled > 0:
        return no("already_sampled", None)
    if depth > max_depth:
        return no("depth_exceeded", None)
    return {"sample": True, "reason": event, "k": max(1, int(k)),
            "wait_for": None, "event": event,
            "no_signal_band": band,
            "learning_signal": not band,
            "rounds_left": rounds_left(d_hat, n_block),
            "mode": "single_shot" if band else "multi_round"}


def _wait_hint(pending: list[str], in_flight: int) -> str | None:
    if pending:
        return "child_return"
    if in_flight > 0:
        return "block_finished"
    return None


def and_state(parent_mission_id: str, children: list[dict], *,
              kind: str = "AND") -> dict:
    """Bookkeeping record for "the parent is discharged iff all children are".

    Recorded only.  The DAG of missions is the primary structure; this record
    exists so a reader can see the AND shape without the loop enforcing a strict
    AND-OR tree.
    """
    ids = [str(c.get("mission_id")) for c in children]
    solved = {str(c.get("mission_id")): bool(c.get("solved")) for c in children}
    pending = [i for i in ids if not solved.get(i, False)]
    return {
        "kind": kind,
        "parent_mission_id": parent_mission_id,
        "children": ids,
        "solved": solved,
        "pending": pending,
        "discharged": bool(ids) and not pending,
        "n_children": len(ids),
    }


def aggregate_children_value(children: list[dict], *, mode: str = "max",
                             key: str = "remaining_blocks") -> float | None:
    """Parent value from children: `max` over unresolved, not a strict AND backup.

    A decomposition succeeds only when every child does, but the *cost* still
    left is dominated by the hardest outstanding child, so `max` is the useful
    aggregate (and the optimistic `min` is available for comparison).
    """
    vals: list[float] = []
    for child in children:
        if child.get("solved"):
            continue
        v = child.get(key)
        if v is None:
            continue
        vals.append(float(v))
    if not vals:
        return 0.0 if children and all(c.get("solved") for c in children) else None
    return max(vals) if mode == "max" else min(vals)


if __name__ == "__main__":
    print({"tick": should_sample({}, "tick"),
           "child_return": should_sample({}, "child_return"),
           "pending": should_sample({"pending_children": ["m-a"]}, "child_return"),
           "gate_900": gate_action(900),
           "gate_120_capped": gate_action(900, depth=3)})
