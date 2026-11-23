#!/usr/bin/env python3
"""Build (context, goal, target) records for value-head training.

The state unit is a **mission** and the target unit is **proof steps**
(DESIGN.md §1):

    {"mission_id": "m-1a2b3c4d5e6f", "env": "bank",
     "context": "<preamble + everything proved so far>",
     "goal": "<statement_formal>",
     "action_type": "block" | "decompose",
     "n_child_missions": 0 | k,
     "remaining_steps": 7}         # value target: remaining tactic steps

One record is one mission-level state, i.e. one agent response opportunity.
`remaining_steps = total_tactic_steps - i` — no conversion, no ceil: d_hat is in
proof steps and so is the gate. A proof *block* is the action (an agent response
may emit at most `action_block_steps` steps), not the measurement.

`--unit blocks` is a DEPRECATED shim for `ceil(steps / block_steps)`; it exists
only so old checkpoints can be compared, warns on stderr, and is marked
`"deprecated": true` in the summary. `--unit tactic` still emits the old
step-level records (same target, different record shape).

Two sources:

1. prove2me bank shards  (./data/bank_local/T*.jsonl)
   Each record has `statement_formal` (the goal), `context_pre` (namespace +
   docstring) and `best_solution.proof` (a `by ...` tactic block).
2. A Mathlib-shaped proof corpus (a directory of *.lean files).  The parser is
   deliberately shallow: it finds `theorem|lemma ... := by`, takes the indented
   block that follows, and splits it into top-level tactic steps.

A tactic-level record is one *prefix* of one proof:

    {"id": ..., "source": "bank|mathlib|tiny",
     "context": "<preamble + every tactic already applied>",
     "goal": "<the statement>",
     "remaining_steps": 7,
     "history_len": 3, "total_steps": 10,
     "next_tactic": "simp [add_comm]" | null,
     "terminal": false}

`remaining_steps` is the number of tactic steps still to be applied at that
prefix.  The prefix after the last step has `remaining_steps = 0` and
`terminal = true`; that record exists on purpose, because the old 64-bucket
head mapped terminal 0 onto bucket 1 (see DESIGN.md §2).

`context` here is an offline proxy for the real Lean tactic state (goals after
the prefix).  With a live lean-mcts session the caller should replace it with
the pretty-printed goals of the saved state handle; the training code does not
care which one it gets.

CLI
---
    python3 train/dataset.py --tiny --out /tmp/vht/data.jsonl
    python3 train/dataset.py --tiny --unit tactic --out /tmp/vht/tactic.jsonl
    python3 train/dataset.py --bank-dir <dir> --limit 20000 --out data.jsonl
    python3 train/dataset.py --mathlib /path/to/Mathlib --max-files 200 --out data.jsonl
    python3 train/dataset.py --bank-dir <dir> --mathlib <dir> --out data.jsonl

No network, no torch needed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:      # cache.py (mission identity) lives at the plugin root
    sys.path.insert(0, str(ROOT))

import traj as _traj                       # noqa: E402  (terminal-success test)
from cache import mission_id_for, statement_hash  # noqa: E402

BANK_DEFAULT = "./data/bank_local"
MATHLIB_DEFAULT = "/data/projects/lean-corpus/mathlib4/Mathlib"
SFT_DEFAULT = "./data/sft/stage1"   # stage-1 SFT shards (build_sft.py)

# d_hat is in proof steps; N is in proof steps; rounds = ceil(d_hat/N).
# One unit only: mission records target `remaining_steps` (tactic steps).
# "blocks" is a DEPRECATED shim kept so old checkpoints/data can be compared.
UNIT_STEPS = "steps"          # default: mission records, target = remaining steps
UNIT_BLOCKS = "blocks"         # DEPRECATED: target = ceil(steps / block_steps)
UNIT_TACTIC = "tactic"         # legacy record shape: one record per tactic prefix
UNITS = (UNIT_STEPS, UNIT_BLOCKS, UNIT_TACTIC)
DEPRECATED_UNITS = (UNIT_BLOCKS,)
DEFAULT_UNIT = UNIT_STEPS
BLOCK_STEPS_DEFAULT = 3        # only used by the deprecated blocks shim
BLOCK_THRESHOLD_DEFAULT = 500.0  # the gate, in proof steps

TACTIC_KEYWORDS = ("by", "have", "show", "suffices", "obtain", "rcases", "cases",
                   "induction", "refine", "exact", "apply", "intro", "simp",
                   "rw", "ring", "linarith", "nlinarith", "omega", "norm_num",
                   "positivity", "gcongr", "calc", "constructor", "use", "let")


def stable_hash(text: str) -> int:
    """Process-stable hash (Python's hash() is salted per process)."""
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Value target: the one number the head regresses on (shared by stage 1/2)
# ---------------------------------------------------------------------------
#
# Three record families reach the trainer:
#   SFT example   {"messages", "value_target", "value_target_kind"}
#   mission        {"remaining_steps", ...}      (dataset.py's own output)
#   trajectory     {"steps_remaining", ...}      (MCTS-p2m runs)
# They all mean the same thing: remaining *proof steps* for this state.  One
# function so stage 1 (SFT), stage 2 (CE) and the gate cannot drift.

VALUE_TARGET_KEYS = ("value_target", "remaining_steps", "steps_remaining", "d_hat_target")

#: Leading tactic keyword -> policy class.  A stage-1 SFT example's policy target
#: is the class of the *next* tactic step; stage 2 maps its searched tactic
#: strings through the same function, so a stage-1 vocab and a stage-2
#: trajectory meet in one space.
TACTIC_CLASSES = (
    "intro", "have", "show", "suffices", "obtain", "rcases", "cases", "induction",
    "refine", "exact", "apply", "constructor", "use", "let", "calc", "specialize",
    "rw", "erw", "nth_rewrite", "simp", "simp_all", "rwa", "field_simp", "ring",
    "nlinarith", "linarith", "omega", "norm_num", "positivity", "gcongr", "aesop",
    "convert", "subst", "ext", "funext", "push_neg", "by_contra", "interval_cases",
    "decide", "native_decide", "trivial", "assumption", "contradiction",
    "norm_cast", "exact_mod_cast", "by_cases", "split_ifs", "haveI", "set",
)


def value_target(record: dict, unit: str = UNIT_STEPS) -> float:
    """Remaining proof steps for a record of any of the three families."""
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
    if unit == UNIT_BLOCKS:      # DEPRECATED shim, old checkpoints only
        for key in ("remaining_blocks", "d_hat_target"):
            if record.get(key) is not None:
                return float(record[key])
        steps = value_target(record, UNIT_STEPS)
        return float(math.ceil(max(0.0, steps) / max(1, BLOCK_STEPS_DEFAULT)))
    for key in VALUE_TARGET_KEYS:
        if record.get(key) is not None:
            return float(record[key])
    raise KeyError(f"record has no value target ({', '.join(VALUE_TARGET_KEYS)})")


def value_target_kind(record: dict) -> str:
    return str(record.get("value_target_kind") or record.get("kind") or "mission_start")


def tactic_class(tactic: str) -> str:
    """Leading tactic keyword, restricted to the class vocabulary."""
    head = str(tactic or "").strip().split("\n", 1)[0].strip()
    if not head:
        return "other"
    if head.startswith("·") or head.startswith("case "):
        return "case"
    word = head.split(None, 1)[0].strip("()[],·")
    if word in TACTIC_CLASSES:
        return word
    return "other"


def tactic_classes(records: "list[dict]") -> list[str]:
    """Vocabulary for the policy head: the classes actually present."""
    seen = {tactic_class(r.get("next_tactic") or "") for r in records
            if r.get("next_tactic")}
    seen.discard("")
    return ["<pad>"] + sorted(seen)


# ---------------------------------------------------------------------------
# Stage-1 SFT loader
# ---------------------------------------------------------------------------
#
# `build_sft.py` writes stage1/T{1..5}-000.jsonl.  One line is one SFT example:
# system/user/assistant messages + the value target of that state.  The trainer
# wants (goal, context, target, policy class), so this loader is the one place
# that turns a chat example into a training record; stage 2 reads its value
# targets through `value_target` above.

_SFT_FENCE = "```lean\n"


def _sft_parts(messages: "list[dict]") -> dict:
    """user/assistant -> {goal, context, proof, applied, imports}."""
    user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    proof = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
    fences = user.split(_SFT_FENCE)
    statement = fences[1].rsplit("```", 1)[0].strip() if len(fences) > 1 else ""
    applied = ""
    if len(fences) > 2:
        applied = fences[2].rsplit("```", 1)[0].strip()
    imports = ""
    for line in user.splitlines():
        if line.startswith("Imports: "):
            imports = line[len("Imports: "):].strip()
    if imports == "(none)":
        imports = ""
    # The statement is the `theorem|lemma ...` part of the fence; the rest of
    # the fence is the surrounding context (namespace, docstring).
    lines = statement.split("\n")
    cut = next((i for i, l in enumerate(lines)
                if l.lstrip().startswith(("theorem ", "lemma "))), 0)
    goal = "\n".join(lines[cut:]).strip() or statement
    return {"goal": goal, "context": applied, "proof": proof.strip(),
            "imports": imports, "fence": statement}


def sft_to_record(ex: dict) -> dict:
    """One SFT example -> one mission-shaped training record."""
    parts = _sft_parts(ex.get("messages") or [])
    steps = split_tactic_steps(parts["proof"]) if parts["proof"] else []
    ctx = parts["context"]
    ctx = (f"import {parts['imports']}\n{ctx}" if parts["imports"] else ctx).strip()
    return {
        "id": f"{ex.get('mission_id', 'sft')}@{int(ex.get('prefix_len') or 0)}",
        "mission_id": ex.get("mission_id"),
        "env": ex.get("env", "default"),
        "tier": ex.get("tier"),
        "theorem_id": ex.get("theorem_id"),
        "source": "sft",
        "goal": parts["goal"],
        "context": ctx,
        "value_target": value_target(ex),
        "value_target_kind": value_target_kind(ex),
        "remaining_steps": value_target(ex),
        "next_tactic": steps[0] if steps else None,
        "history_len": int(ex.get("prefix_len") or 0),
        "total_steps": int(ex.get("total_steps") or 0) or int(value_target(ex))
                       + int(ex.get("prefix_len") or 0),
        "terminal": False,
    }


def load_sft(path: str = SFT_DEFAULT, limit: int | None = None,
             kinds: "list[str] | None" = None, stratify: int = 0) -> "list[dict]":
    """Read SFT shards (a dir of *.jsonl or one file) as training records.

    `stratify=k > 0` keeps at most k records per log1p(target) bucket instead of
    taking the first `limit` lines.  The shards are ordered by tier, and a tier's
    early records are dominated by one-step proofs, so a naive `limit` gives a
    near-constant target and a useless tiny run.  Stratifying is what makes the
    bounded CPU run's metric mean something.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no SFT data at {path}; run train/build_sft.py")
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"no *.jsonl under {path}")
    want = set(kinds) if kinds else None
    out: "list[dict]" = []
    if stratify <= 0:
        for f in files:
            with f.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    ex = json.loads(line)
                    if want and value_target_kind(ex) not in want:
                        continue
                    out.append(sft_to_record(ex))
                    if limit is not None and len(out) >= limit:
                        return out
        return out

    buckets: dict[int, list[dict]] = {}
    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                if want and value_target_kind(ex) not in want:
                    continue
                b = min(15, int(math.log1p(max(0.0, value_target(ex)))))
                bucket = buckets.setdefault(b, [])
                if len(bucket) < stratify:
                    bucket.append(sft_to_record(ex))
    for b in sorted(buckets):
        out.extend(buckets[b])
    return out[:limit] if limit is not None else out


def iter_sft(path: str = SFT_DEFAULT, kinds: "list[str] | None" = None):
    """Stream SFT examples without holding the shards in memory."""
    p = Path(path)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    want = set(kinds) if kinds else None
    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                if want and value_target_kind(ex) not in want:
                    continue
                yield ex


def summarize_sft(records: "list[dict]") -> dict:
    """Counts the trainer prints: steps per kind, targets, policy classes."""
    if not records:
        return {"records": 0}
    targets = [value_target(r) for r in records]
    ordered = sorted(targets)

    def q(p: float) -> float:
        return ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1)))]

    kinds: dict[str, int] = {}
    for r in records:
        kinds[value_target_kind(r)] = kinds.get(value_target_kind(r), 0) + 1
    return {
        "unit": UNIT_STEPS,
        "target": "value_target",
        "records": len(records),
        "missions": len({r.get("mission_id") for r in records}),
        "kinds": kinds,
        "tiers": sorted({str(r.get("tier")) for r in records if r.get("tier")}),
        "target_p50": q(0.5), "target_p90": q(0.9), "target_max": ordered[-1],
        "over_500": sum(1 for v in targets if v >= 500),
        "policy_classes": len(tactic_classes(records)),
    }


# ---------------------------------------------------------------------------
# Proof -> steps
# ---------------------------------------------------------------------------

def split_tactic_steps(proof_body: str) -> list[str]:
    """Split a Lean proof body into top-level tactic steps.

    Steps are the lines at the smallest indentation; deeper lines are
    continuations of the step above (nested `case`/`·` blocks stay attached).
    A leading bare `by` is dropped.
    """
    if not isinstance(proof_body, str):
        return []
    lines = proof_body.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return []
    stripped_first = lines[0].strip()
    if stripped_first == "by":
        lines = lines[1:]
    elif stripped_first.startswith("by\n"):
        lines = [stripped_first[2:].lstrip()] + lines[1:]
    indents = [len(l) - len(l.lstrip(" ")) for l in lines if l.strip()]
    if not indents:
        return []
    base = min(indents)
    steps: list[str] = []
    cur: list[str] = []
    for raw in lines:
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if indent <= base:
            if cur:
                steps.append("\n".join(cur))
            cur = [raw.strip()]
        else:
            if not cur:
                cur = [raw.strip()]
            else:
                cur.append(raw.rstrip())
    if cur:
        steps.append("\n".join(cur))
    # A step that is only a comment carries no information; keep it anyway so
    # that remaining_steps matches the raw step count of the reference proof.
    return steps


def _context_block(context_pre: str, imports: list[str] | None) -> str:
    head = ""
    if imports:
        head = "\n".join(f"import {m}" for m in imports) + "\n"
    return f"{head}{context_pre or ''}".strip()


def records_from_proof(*, context: str, goal: str, steps: list[str],
                       source: str, problem_id: str,
                       include_terminal: bool = True) -> list[dict]:
    """One record per prefix of the proof (plus the terminal prefix)."""
    total = len(steps)
    out: list[dict] = []
    for i in range(total):
        history = "\n".join(steps[:i])
        out.append({
            "id": f"{problem_id}#{i}",
            "source": source,
            "context": (context + ("\n" + history if history else "")).strip(),
            "goal": goal,
            "remaining_steps": total - i,
            "history_len": i,
            "total_steps": total,
            "next_tactic": steps[i],
            "terminal": False,
        })
    if include_terminal:
        out.append({
            "id": f"{problem_id}#{total}",
            "source": source,
            "context": (context + ("\n" + "\n".join(steps) if steps else "")).strip(),
            "goal": goal,
            "remaining_steps": 0,
            "history_len": total,
            "total_steps": total,
            "next_tactic": None,
            "terminal": True,
        })
    return out


# ---------------------------------------------------------------------------
# Mission-level records (the default unit)
# ---------------------------------------------------------------------------
#
# A mission-level state is {env, mission_id, lean_context, goal}; one record is
# one state with the two numbers the mission-level head needs:
#
#   remaining_steps   how many proof steps are left for this mission (the target)
#   action_type       "block" (attempt an N-step block) | "decompose"
#   n_child_missions  how many children the decomposition asks for (0 for block)
#
# `mission_id` is sha256(env, statement_formal) — the *same* key the mission
# registry uses, so a dataset record and a registered mission are the same
# object (cache.py, DESIGN.md §8).
#
# d_hat is in proof steps; N is in proof steps; rounds = ceil(d_hat/N).
# There is no blocks/steps conversion in this path.  `--unit blocks` keeps the
# old ceil conversion as a DEPRECATED comparison shim only.

def remaining_blocks(remaining_steps: int, block_steps: int = BLOCK_STEPS_DEFAULT) -> int:
    """DEPRECATED: ceil(steps / block_steps). Only for `--unit blocks`."""
    n = max(0, int(remaining_steps))
    if n == 0:
        return 0
    return max(1, math.ceil(n / max(1, int(block_steps))))


def planned_children(remaining_steps: int, threshold: float = BLOCK_THRESHOLD_DEFAULT,
                     min_children: int = 2, max_children: int = 8) -> int:
    """Planning hint only: how many children a decomposition asks for."""
    if threshold <= 0:
        return min_children
    return max(min_children, min(max_children, math.ceil(max(0, remaining_steps) / threshold)))


def mission_record(*, env: str, context: str, goal: str, remaining_steps: int,
                   source: str, record_id: str,
                   threshold: float = BLOCK_THRESHOLD_DEFAULT,
                   action_type: str | None = None,
                   n_child_missions: int | None = None,
                   depth: int = 0, parent_mission_id: str | None = None,
                   status: str = "accepted", success: bool | None = None,
                   unit: str = UNIT_STEPS,
                   block_steps: int = BLOCK_STEPS_DEFAULT) -> dict:
    steps = max(0, int(remaining_steps))
    if success is None:
        success = status == "accepted"
    if action_type is None:
        action_type = "decompose" if steps >= threshold else "block"
    if action_type not in ("block", "decompose"):
        raise ValueError(f"action_type must be block|decompose, got {action_type!r}")
    if n_child_missions is None:
        n_child_missions = planned_children(steps, threshold) if action_type == "decompose" else 0
    rec = {
        "id": record_id,
        "unit": unit,
        "source": source,
        "env": env,
        "mission_id": mission_id_for(env, goal),
        "context": context,
        "goal": goal,
        "action_type": action_type,
        "next_action": action_type,          # what the policy head predicts
        "n_child_missions": int(n_child_missions),
        "remaining_steps": steps,            # value target: proof steps
        "depth": int(depth),
        "parent_mission_id": parent_mission_id,
        "status": status,
        # terminal-only signal: only successful trajectories are CE-eligible
        "success": bool(success),
        "terminal": steps == 0,
    }
    if unit == UNIT_BLOCKS:
        # DEPRECATED shim: the old target, kept only to compare checkpoints.
        rec["block_steps"] = int(block_steps)
        rec["remaining_blocks"] = remaining_blocks(steps, block_steps)
        rec["d_hat_target"] = rec["remaining_blocks"]
    return rec


def upgrade_tactic_record(rec: dict, *, unit: str = UNIT_STEPS,
                          threshold: float = BLOCK_THRESHOLD_DEFAULT,
                          block_steps: int = BLOCK_STEPS_DEFAULT,
                          env: str | None = None) -> dict:
    """One tactic-level record -> one mission-level record (same state, same target)."""
    return mission_record(
        env=env or rec.get("source", "default"),
        context=rec.get("context", ""),
        goal=rec.get("goal", ""),
        remaining_steps=int(rec.get("remaining_steps", 0)),
        source=rec.get("source", "unknown"),
        record_id=str(rec.get("id", "")),
        threshold=threshold, unit=unit, block_steps=block_steps,
    )


def mission_from_tactic(records: "list[dict]", *,
                        unit: str = UNIT_STEPS,
                        threshold: float = BLOCK_THRESHOLD_DEFAULT,
                        block_steps: int = BLOCK_STEPS_DEFAULT,
                        env: str | None = None) -> "list[dict]":
    """One mission record per tactic prefix, target = remaining tactic steps.

    Every prefix is a distinct state (its context differs), so none are dropped:
    `remaining_steps = total_tactic_steps - i` exactly as the source recorded it.
    The deprecated blocks unit keeps the ceil conversion inside `mission_record`.
    """
    out: "list[dict]" = []
    for rec in records:
        out.append(upgrade_tactic_record(rec, unit=unit, threshold=threshold,
                                         block_steps=block_steps, env=env))
    return out


def records_from_trajectories(trajectories: "list[dict]", *,
                              include_failed: bool = False,
                              unit: str = UNIT_STEPS,
                              threshold: float = BLOCK_THRESHOLD_DEFAULT,
                              block_steps: int = BLOCK_STEPS_DEFAULT) -> "list[dict]":
    """Mission records from agent trajectories, failures dropped by default.

    The learning signal is terminal-only: a trajectory enters the CE pool because
    the original (context, goal) was discharged, never because it decomposed or
    called a subagent.  `include_failed=True` keeps the failures as diagnostics
    (they carry `success: false` and `status: failed`, and the trainer still has
    to filter them out before CE).
    """
    out: "list[dict]" = []
    for item in trajectories:
        row = item if isinstance(item, dict) else _traj.asdict(item)
        ok = _traj.is_terminal_success(row)
        if not ok and not include_failed:
            continue
        goal = row.get("goal") or row.get("statement_formal") or ""
        steps = row.get("remaining_steps", row.get("steps_remaining"))
        if steps is None:
            steps = row.get("remaining_blocks", row.get("d_hat_target")) or 0
            if unit != UNIT_BLOCKS:      # a blocks record: convert back to steps
                steps = int(steps) * max(1, int(block_steps))
        out.append(mission_record(
            env=row.get("env", "default"),
            context=row.get("lean_context", row.get("context", "")),
            goal=goal,
            remaining_steps=int(steps or 0),
            source=row.get("source", "trajectory"),
            record_id=str(row.get("mission_id", f"traj-{len(out)}")),
            threshold=threshold, unit=unit, block_steps=block_steps,
            action_type=row.get("action_type"),
            n_child_missions=len(row.get("child_missions") or []),
            depth=int(row.get("depth", 0)),
            parent_mission_id=row.get("parent_mission_id"),
            status="accepted" if ok else "failed",
            success=ok,
        ))
    return out


def synthesize_tiny_missions(n: int = 50, seed: int = 7,
                             unit: str = UNIT_STEPS,
                             threshold: float = BLOCK_THRESHOLD_DEFAULT,
                             block_steps: int = BLOCK_STEPS_DEFAULT) -> "list[dict]":
    """Mission-level tiny set: same 50 states, target = remaining proof steps."""
    return [upgrade_tactic_record(r, unit=unit, threshold=threshold,
                                  block_steps=block_steps, env="tiny")
            for r in synthesize_tiny(n, seed)]


# ---------------------------------------------------------------------------
# Source 1: prove2me bank
# ---------------------------------------------------------------------------

def iter_bank(bank_dir: str, limit: int | None = None,
              tiers: list[str] | None = None) -> "list[dict]":
    root = Path(bank_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"bank dir not found: {bank_dir}")
    shards = sorted(root.glob("T*.jsonl"))
    if not shards:
        shards = sorted(root.glob("*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"no T*.jsonl shards under {bank_dir}")
    records: list[dict] = []
    for shard in shards:
        if tiers and shard.stem not in tiers:
            continue
        with shard.open("r", encoding="utf-8") as fh:
            for line in fh:
                if limit is not None and len(records) >= limit:
                    return records
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sol = row.get("best_solution") or {}
                proof = sol.get("proof")
                goal = row.get("statement_formal") or row.get("statement_nl")
                if not proof or not goal:
                    continue
                steps = split_tactic_steps(proof)
                if not steps:
                    continue
                pid = str(row.get("id") or row.get("theorem_id") or len(records))
                ctx = _context_block(row.get("context_pre", ""), row.get("imports"))
                records.extend(records_from_proof(
                    context=ctx, goal=goal, steps=steps,
                    source="bank", problem_id=pid))
    return records


# ---------------------------------------------------------------------------
# Source 2: Mathlib-shaped .lean corpus
# ---------------------------------------------------------------------------

_DECL_RE = re.compile(r"^(?P<indent>\s*)(?:theorem|lemma)\s+(?P<name>[A-Za-z0-9_'.\u03b1-\u03c9]+)")
_IMPORT_RE = re.compile(r"^import\s+(\S+)")


def _lean_blocks(text: str):
    """Yield (name, signature, body_lines) for `theorem|lemma ... := by`."""
    lines = text.split("\n")
    i = 0
    imports = [m.group(1) for m in (_IMPORT_RE.match(l) for l in lines[:60]) if m]
    while i < len(lines):
        m = _DECL_RE.match(lines[i])
        if not m:
            i += 1
            continue
        base = len(m.group("indent"))
        start = i
        j = i
        sig_lines = []
        body_start = None
        while j < len(lines) and j - start < 80:
            sig_lines.append(lines[j])
            if ":=" in lines[j]:
                tail = lines[j].split(":=", 1)[1].strip()
                if tail == "by" or tail.startswith("by "):
                    body_start = j
                    break
                break  # `:= term` proof, not a tactic block
            j += 1
        if body_start is None:
            i = max(j + 1, i + 1)
            continue
        body: list[str] = []
        k = body_start + 1
        while k < len(lines):
            raw = lines[k]
            if raw.strip():
                ind = len(raw) - len(raw.lstrip(" "))
                if ind <= base:
                    break
            body.append(raw)
            k += 1
        yield m.group("name"), "\n".join(sig_lines).strip(), body, imports
        i = max(k, start + 1)


def iter_mathlib(root: str, max_files: int = 200, max_records: int | None = None,
                 seed: int = 7) -> "list[dict]":
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"corpus dir not found: {root}")
    files = sorted(base.rglob("*.lean"))
    if max_files and len(files) > max_files:
        rng = random.Random(seed)
        files = sorted(rng.sample(files, max_files))
    records: list[dict] = []
    for path in files:
        if max_records is not None and len(records) >= max_records:
            break
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text) > 400_000:  # keep the shallow parser fast
            text = text[:400_000]
        for name, sig, body, imports in _lean_blocks(text):
            steps = split_tactic_steps("by\n" + "\n".join(body))
            if len(steps) < 2:
                continue
            ctx = _context_block(str(path.parent.name), imports)
            records.extend(records_from_proof(
                context=ctx, goal=sig, steps=steps,
                source="mathlib", problem_id=f"{path.stem}.{name}"))
            if max_records is not None and len(records) >= max_records:
                break
    return records


# ---------------------------------------------------------------------------
# Source 3: synthetic tiny set (runs with no corpus at all)
# ---------------------------------------------------------------------------

TINY_TACTICS = [
    "intro h", "simp only [add_comm, mul_comm]", "rw [h]", "exact h",
    "apply le_trans", "omega", "ring", "norm_num", "linarith", "positivity",
    "constructor", "refine ⟨?_, ?_⟩", "cases h with", "induction n with",
    "gcongr", "field_simp", "nlinarith [sq_nonneg x]", "aesop",
]


def synthesize_tiny(n: int = 50, seed: int = 7) -> "list[dict]":
    """50 records with a *deliberate* learnable signal.

    log1p(remaining_steps) ~ 0.55 * (goal token count) + 0.35 * history_len
    + noise, i.e. longer statements / longer prefixes imply longer proofs.
    The point of the tiny set is to exercise the pipeline and the metrics
    (including the >10k tail), not to demonstrate real proof-value learning.
    """
    rng = random.Random(seed)
    records: list[dict] = []
    for idx in range(n):
        # Wide enough that expm1(0.55 * n_words) reaches the >10k tail, so the
        # tail is *predictable from the text* and tail recall is meaningful.
        n_words = rng.randint(3, 20)
        history = rng.randint(0, 6)
        noise = rng.gauss(0.0, 0.45)
        signal = max(0.0, 0.55 * n_words + 0.35 * history + noise)
        total = int(min(20000, max(1, round(math.expm1(signal))))) if signal < 9.9 else 20000
        total = max(total, history + 1)
        remaining = max(1, total - history)
        goal = ("theorem tiny_%02d (a b c : \u2115) : " % idx) + " + ".join(
            f"f{i}(a, b, c)" for i in range(n_words)) + " = " + " + ".join(
            f"g{i}(a, b, c)" for i in range(max(1, n_words - 1)))
        prior = [TINY_TACTICS[(stable_hash(f"{idx}:{k}") % len(TINY_TACTICS))]
                 for k in range(history)]
        # Next tactic is a coarse function of the remaining horizon, so the
        # policy branch is measurable (pass@1 well above 1/8) in tiny mode.
        bucket = min(7, int(math.log2(max(2, remaining))))
        records.append({
            "id": f"tiny-{idx:03d}",
            "source": "tiny",
            "context": ("import Mathlib\n" + "\n".join(prior)).strip(),
            "goal": goal,
            "remaining_steps": remaining,
            "history_len": history,
            "total_steps": total,
            "next_tactic": f"tac_{bucket}",
            "terminal": False,
        })
    return records


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_jsonl(records: "list[dict]", out_path: str) -> str:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return str(path)


def read_jsonl(path: str) -> "list[dict]":
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(records: "list[dict]") -> dict:
    if not records:
        return {"records": 0}
    unit = records[0].get("unit", UNIT_TACTIC)
    if unit in (UNIT_STEPS, UNIT_BLOCKS):
        key = "remaining_blocks" if unit == UNIT_BLOCKS else "remaining_steps"
        values = [int(r[key]) for r in records]
        ordered = sorted(values)

        def qv(p: float) -> int:
            return ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1)))]

        return {
            "records": len(records),
            "unit": unit,
            "deprecated": unit in DEPRECATED_UNITS,
            "target": key,
            "sources": sorted({r["source"] for r in records}),
            "envs": sorted({r["env"] for r in records}),
            "missions": len({r["mission_id"] for r in records}),
            f"{key}_p50": qv(0.5),
            f"{key}_p90": qv(0.9),
            f"{key}_max": ordered[-1],
            "over_500": sum(1 for v in values if v >= 500),
            "decompose_actions": sum(1 for r in records if r["action_type"] == "decompose"),
            "block_actions": sum(1 for r in records if r["action_type"] == "block"),
            "children_total": sum(int(r["n_child_missions"]) for r in records),
            "success": sum(1 for r in records if r.get("success")),
            "failed": sum(1 for r in records if r.get("success") is False),
            "terminal": sum(1 for r in records if r.get("terminal")),
        }
    steps = [int(r["remaining_steps"]) for r in records]
    steps_sorted = sorted(steps)

    def q(p: float) -> int:
        return steps_sorted[min(len(steps_sorted) - 1, int(p * (len(steps_sorted) - 1)))]

    return {
        "records": len(records),
        "unit": UNIT_TACTIC,
        "sources": sorted({r["source"] for r in records}),
        "remaining_p50": q(0.5),
        "remaining_p90": q(0.9),
        "remaining_max": steps_sorted[-1],
        "over_500": sum(1 for s in steps if s > 500),
        "over_10k": sum(1 for s in steps if s > 10000),
        "terminal": sum(1 for r in records if r.get("terminal")),
    }


def build(*, bank_dir: str | None, mathlib: str | None, tiny: bool,
          limit: int | None = None, max_files: int = 200,
          mathlib_limit: int | None = None, tiers: list[str] | None = None,
          seed: int = 7, include_terminal: bool = True,
          unit: str = DEFAULT_UNIT,
          block_steps: int = BLOCK_STEPS_DEFAULT,
          threshold: float = BLOCK_THRESHOLD_DEFAULT,
          trajectories: "list[dict] | None" = None,
          include_failed: bool = False) -> "list[dict]":
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
    records: "list[dict]" = []
    if tiny:
        if unit == UNIT_TACTIC:
            return synthesize_tiny(50, seed)
        return synthesize_tiny_missions(50, seed, unit=unit, threshold=threshold,
                                        block_steps=block_steps)
    if bank_dir:
        rows = iter_bank(bank_dir, limit=limit, tiers=tiers)
        if not include_terminal:
            rows = [r for r in rows if not r["terminal"]]
        records.extend(rows)
    if mathlib:
        rows = iter_mathlib(mathlib, max_files=max_files, max_records=mathlib_limit,
                            seed=seed)
        if not include_terminal:
            rows = [r for r in rows if not r["terminal"]]
        records.extend(rows)
    if trajectories:
        records = records_from_trajectories(
            trajectories, include_failed=include_failed, unit=unit,
            threshold=threshold, block_steps=block_steps)
    if not records:
        raise SystemExit(
            "dataset: no records built. Pass --tiny, --bank-dir <dir>, "
            "--mathlib <dir> or --trajectories <jsonl>.")
    if unit == UNIT_TACTIC or trajectories:
        return records
    return mission_from_tactic(records, unit=unit, threshold=threshold,
                               block_steps=block_steps)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank-dir", default=None, help=f"prove2me shard dir (default {BANK_DEFAULT} when no other source)")
    ap.add_argument("--mathlib", default=None, help="directory of *.lean proofs (Mathlib-shaped)")
    ap.add_argument("--tiny", action="store_true", help="50 synthetic records, no corpus needed")
    ap.add_argument("--out", default="data.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="max records from the bank")
    ap.add_argument("--mathlib-limit", type=int, default=None, help="max records from the corpus")
    ap.add_argument("--max-files", type=int, default=200)
    ap.add_argument("--tier", action="append", default=None, help="bank tier filter, repeatable")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-terminal", action="store_true", help="drop remaining_steps=0 records")
    ap.add_argument("--trajectories", default=None,
                    help="jsonl of agent trajectories; only terminal successes are kept")
    ap.add_argument("--include-failed", action="store_true",
                    help="keep failed trajectories as diagnostics (never CE-eligible)")
    ap.add_argument("--unit", choices=list(UNITS), default=DEFAULT_UNIT,
                    help="steps (default): one record per mission state, target = "
                         "remaining proof steps; tactic: legacy record shape, same "
                         "target; blocks: DEPRECATED ceil(steps/block_steps) target")
    ap.add_argument("--block-steps", type=int, default=BLOCK_STEPS_DEFAULT,
                    help="DEPRECATED, only for --unit blocks")
    ap.add_argument("--block-threshold", type=float, default=BLOCK_THRESHOLD_DEFAULT,
                    help="decompose gate, in proof steps (d_hat > 500 => decompose)")
    args = ap.parse_args(argv)

    bank_dir = args.bank_dir
    mathlib = args.mathlib
    if not args.tiny and not args.trajectories and not bank_dir and not mathlib:
        bank_dir = BANK_DEFAULT if os.path.isdir(BANK_DEFAULT) else None
        if bank_dir is None:
            mathlib = MATHLIB_DEFAULT

    if args.unit in DEPRECATED_UNITS:
        print(f"dataset: --unit {args.unit} is DEPRECATED (the target is proof steps; "
              "a blocks target is only for comparing old checkpoints)", file=sys.stderr)
    records = build(bank_dir=bank_dir, mathlib=mathlib, tiny=args.tiny,
                    limit=args.limit, max_files=args.max_files,
                    mathlib_limit=args.mathlib_limit, tiers=args.tier,
                    seed=args.seed, include_terminal=not args.no_terminal,
                    unit=args.unit, block_steps=args.block_steps,
                    threshold=args.block_threshold,
                    trajectories=(read_jsonl(args.trajectories) if args.trajectories else None),
                    include_failed=args.include_failed)
    path = write_jsonl(records, args.out)
    print(json.dumps({"ok": True, "out": path, **summarize(records)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
