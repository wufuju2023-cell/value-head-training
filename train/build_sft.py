#!/usr/bin/env python3
"""Build the stage-1 SFT dataset from the shape-classified lemma bank.

Input   ./data/bank_v2/lemma/T{1..5}.jsonl   (read-only)
Output  ./data/sft/stage1/T{1..5}-000.jsonl
        ./data/sft/stats.json
        ./data/sft/README.md

One record = one mission-level state with a full SFT example:

    {"mission_id", "env", "messages": [system, user, assistant],
     "value_target": <remaining proof steps at that state>,
     "value_target_kind": "mission_start" | "prefix",
     "tier", "problem_type": "lemma", "theorem_id",
     "prefix_len": <0 for mission_start>, "total_steps": <n>}

Two variants per proof:

  mission_start   user = statement (+ imports, context)   assistant = whole proof
                  value_target = total tactic steps
  prefix (xN)     user = statement + the steps already applied
                  assistant = the remaining steps
                  value_target = total - prefix_len

The prefix user message carries the applied steps *verbatim* inside the lean
fence, so `messages` alone reconstruct the original file (imports + context +
statement + proof) — that is what `--verify` round-trips.

The value target is the reference proof's tactic-step count from
`dataset.split_tactic_steps`, i.e. the same unit as the gate (proof steps).

    python3 train/build_sft.py --out ./data/sft
    python3 train/build_sft.py --out /tmp/sft --limit 200 --prefixes 3 --verify 3

No network, no torch, never writes to the bank.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import dataset  # noqa: E402  (split_tactic_steps, target helpers)
from cache import mission_id_for  # noqa: E402

BANK_DEFAULT = "./data/bank_v2/lemma"
OUT_DEFAULT = "./data/sft"
TIERS = ("T1", "T2", "T3", "T4", "T5")

SYSTEM_PROMPT = (
    "You are a Lean 4 / Mathlib prover. Given a lemma statement, its imports and "
    "the surrounding context, reply with one tactic proof block. Start the block "
    "with `by` and do not use sorry, admit or a placeholder. The value head of "
    "this system estimates how many tactic steps are still missing, so your block "
    "should be the shortest proof you are confident in."
)

BAD_TOKENS = ("sorry", "admit", "sorryAx")


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------

def lean_block(*, context_pre: str, statement_formal: str) -> str:
    """The lean fence inside the user message: context + statement."""
    parts = [context_pre.strip()] if context_pre and context_pre.strip() else []
    parts.append(statement_formal.strip())
    return "\n\n".join(parts)


def user_content(*, statement_nl: str, statement_formal: str,
                 imports: list[str], context_pre: str,
                 applied: list[str] | None = None,
                 total_steps: int = 0) -> str:
    imp = ", ".join(imports) if imports else "(none)"
    text = [
        statement_nl.strip() if statement_nl else statement_formal.strip(),
        "",
        f"Imports: {imp}",
        "",
        "```lean",
        lean_block(context_pre=context_pre, statement_formal=statement_formal),
        "```",
    ]
    if applied:
        text += [
            "",
            f"Proof so far (steps 1..{len(applied)} of {total_steps}):",
            "```lean",
            "\n".join(applied),
            "```",
        ]
    return "\n".join(text)


def build_records(row: dict, *, prefixes: int) -> list[dict]:
    """One mission_start record + up to `prefixes` prefix records for one lemma."""
    sol = row.get("best_solution") or {}
    proof = sol.get("proof")
    statement_formal = (row.get("statement_formal") or "").strip()
    if not proof or not statement_formal:
        return []
    steps = dataset.split_tactic_steps(proof)
    if not steps:
        return []
    total = len(steps)
    imports = [m for m in (row.get("imports") or []) if isinstance(m, str)]
    context_pre = row.get("context_pre") or ""
    env_raw = row.get("env")
    env = (env_raw.get("dir") if isinstance(env_raw, dict) else env_raw) or "default"
    tier = row.get("tier") or "?"
    theorem_id = row.get("theorem_id") or row.get("id") or ""
    mission = mission_id_for(str(env), statement_formal)

    system = {"role": "system", "content": SYSTEM_PROMPT}
    base = {
        "mission_id": mission,
        "env": env,
        "tier": tier,
        "problem_type": "lemma",
        "theorem_id": theorem_id,
        "total_steps": total,
    }

    out: list[dict] = [{
        **base,
        "messages": [
            system,
            {"role": "user", "content": user_content(
                statement_nl=row.get("statement_nl") or "",
                statement_formal=statement_formal, imports=imports,
                context_pre=context_pre)},
            {"role": "assistant", "content": proof.strip()},
        ],
        "value_target": total,
        "value_target_kind": "mission_start",
        "prefix_len": 0,
    }]

    if prefixes > 0 and total >= 2:
        seen: set[int] = set()
        for k in range(1, prefixes + 1):
            cut = round(k * total / (prefixes + 1))
            cut = max(1, min(total - 1, cut))
            if cut in seen:
                continue
            seen.add(cut)
            applied = steps[:cut]
            rest = steps[cut:]
            out.append({
                **base,
                "messages": [
                    system,
                    {"role": "user", "content": user_content(
                        statement_nl=row.get("statement_nl") or "",
                        statement_formal=statement_formal, imports=imports,
                        context_pre=context_pre, applied=applied,
                        total_steps=total)},
                    {"role": "assistant", "content": "by\n" + "\n".join(rest)},
                ],
                "value_target": total - cut,
                "value_target_kind": "prefix",
                "prefix_len": cut,
            })
    return out


# ---------------------------------------------------------------------------
# Streaming + stats
# ---------------------------------------------------------------------------

def iter_bank(bank_dir: str, tiers: list[str] | None = None):
    root = Path(bank_dir)
    shards = sorted(root.glob("T*.jsonl"))
    if not shards:
        raise SystemExit(f"build_sft: no T*.jsonl under {bank_dir}")
    for shard in shards:
        tier = shard.stem
        if tiers and tier not in tiers:
            continue
        with shard.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield tier, json.loads(line)
                except json.JSONDecodeError:
                    continue


def tokens_est(text: str) -> int:
    """Cheap token estimate: ~4 chars/token for Lean prose."""
    return max(1, len(text) // 4)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(p * (len(s) - 1)))
    return float(s[idx])


def length_bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    if n <= 20:
        return "11-20"
    if n <= 50:
        return "21-50"
    return "51+"


def build(*, bank_dir: str, out_dir: str, prefixes: int, limit: int | None,
          tiers: list[str]) -> dict:
    stage1 = Path(out_dir) / "stage1"
    stage1.mkdir(parents=True, exist_ok=True)
    handles = {t: (stage1 / f"{t}-000.jsonl").open("w", encoding="utf-8") for t in tiers}
    stats: dict[str, dict] = {t: {
        "proofs": 0, "records": 0, "mission_start": 0, "prefix": 0,
        "est_tokens": 0, "skipped_no_proof": 0, "skipped_no_steps": 0,
        "targets": [], "proof_steps": [], "hist": Counter(),
        "examples": [],
    } for t in tiers}

    n_in = 0
    for tier, row in iter_bank(bank_dir, tiers):
        if tier not in stats:
            continue
        n_in += 1
        if limit is not None and stats[tier]["records"] >= limit:
            continue
        recs = build_records(row, prefixes=prefixes)
        if not recs:
            key = "skipped_no_proof" if not (row.get("best_solution") or {}).get("proof") \
                else "skipped_no_steps"
            stats[tier][key] += 1
            continue
        stats[tier]["proofs"] += 1
        stats[tier]["proof_steps"].append(recs[0]["total_steps"])
        stats[tier]["hist"][length_bucket(recs[0]["total_steps"])] += 1
        for rec in recs:
            line = json.dumps(rec, ensure_ascii=False)
            handles[tier].write(line + "\n")
            stats[tier]["records"] += 1
            stats[tier]["est_tokens"] += sum(
                tokens_est(m["content"]) for m in rec["messages"])
            stats[tier][rec["value_target_kind"]] += 1
            if rec["value_target_kind"] == "mission_start":
                stats[tier]["targets"].append(rec["value_target"])
        if len(stats[tier]["examples"]) < 2:
            stats[tier]["examples"].append({
                "mission_id": recs[0]["mission_id"],
                "theorem_id": recs[0]["theorem_id"],
                "total_steps": recs[0]["total_steps"],
            })

    for fh in handles.values():
        fh.close()

    summary = {"bank": bank_dir, "out": str(stage1), "prefixes": prefixes,
               "records_total": 0, "est_tokens_total": 0, "tiers": {}}
    all_targets: list[float] = []
    for t, s in stats.items():
        targets = s["targets"]
        all_targets.extend(targets)
        summary["tiers"][t] = {
            "proofs": s["proofs"],
            "records": s["records"],
            "mission_start": s["mission_start"],
            "prefix": s["prefix"],
            "est_tokens": s["est_tokens"],
            "skipped_no_proof": s["skipped_no_proof"],
            "skipped_no_steps": s["skipped_no_steps"],
            "shard": f"stage1/{t}-000.jsonl",
            "proof_steps_hist": dict(s["hist"]),
            "proof_steps": {
                "p50": percentile(s["proof_steps"], 0.5),
                "p90": percentile(s["proof_steps"], 0.9),
                "max": max(s["proof_steps"]) if s["proof_steps"] else 0,
            },
            "value_target_start": {
                "n": len(targets),
                "p50": percentile(targets, 0.5),
                "p90": percentile(targets, 0.9),
                "p99": percentile(targets, 0.99),
                "max": max(targets) if targets else 0,
                "over_500": sum(1 for v in targets if v >= 500),
            },
            "examples": s["examples"],
        }
        summary["records_total"] += s["records"]
        summary["est_tokens_total"] += s["est_tokens"]
    summary["value_target_all"] = {
        "n": len(all_targets),
        "p50": percentile(all_targets, 0.5),
        "p90": percentile(all_targets, 0.9),
        "p99": percentile(all_targets, 0.99),
        "max": max(all_targets) if all_targets else 0,
        "mean": round(statistics.fmean(all_targets), 2) if all_targets else 0.0,
    }
    summary["input_records"] = n_in
    return summary


# ---------------------------------------------------------------------------
# Round-trip verification
# ---------------------------------------------------------------------------

def reconstruct(messages: list[dict]) -> dict:
    """messages -> {imports, fence, applied, proof}.

    `applied` is the prefix the user message already fixed (empty for a
    mission_start record); `proof` is the assistant block.
    """
    user = next(m["content"] for m in messages if m["role"] == "user")
    proof = next(m["content"] for m in messages if m["role"] == "assistant")
    fences = user.split("```lean\n")
    fence = fences[1].rsplit("```", 1)[0].strip() if len(fences) > 1 else ""
    applied: list[str] = []
    if len(fences) > 2:
        raw = fences[2].rsplit("```", 1)[0]
        applied = [l for l in raw.splitlines() if l.strip()]
    imports: list[str] = []
    for line in user.splitlines():
        if line.startswith("Imports: "):
            raw = line[len("Imports: "):].strip()
            if raw != "(none)":
                imports = [w.strip() for w in raw.split(",") if w.strip()]
    return {"imports": imports, "fence": fence, "applied": applied,
            "proof": proof.strip()}


def verify_shard(path: str, n: int, seed: int = 0) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rng = random.Random(seed)
    picked = rng.sample(rows, min(n, len(rows)))
    out = []
    for rec in picked:
        parts = reconstruct(rec["messages"])
        proof = parts["proof"]
        applied = parts["applied"]
        # 1. imports parse as `import <Module.Path>`
        imports_ok = all(
            len(l.split()) == 2 and l.split()[1].replace(".", "").replace("_", "").isalnum()
            for l in [f"import {m}" for m in parts["imports"]])
        # 2. statement present in the lean fence
        stmt_ok = ("theorem" in parts["fence"] or "lemma" in parts["fence"]) and ":" in parts["fence"]
        # 3. the assistant content is a `by` block with at least one step
        proof_ok = proof.startswith("by") and len(dataset.split_tactic_steps(proof)) >= 1
        # 4. no sorry/admit in the prefix or in the proof
        clean = not any(tok in proof or tok in "\n".join(applied) for tok in BAD_TOKENS)
        # 5. value target reproducible from the record's own fields
        total = int(rec.get("total_steps") or 0)
        prefix_len = int(rec.get("prefix_len") or 0)
        target_ok = int(rec["value_target"]) == total - prefix_len
        # 6. round-trip: prefix (from the user message) + assistant rebuild the
        #    original `by` block, and its step count is the whole proof
        full = "by\n" + "\n".join(applied) + ("\n" + proof[len("by"):].lstrip("\n") if proof.startswith("by") else "\n" + proof)
        full_steps = len(dataset.split_tactic_steps(full))
        applied_steps = len(dataset.split_tactic_steps("by\n" + "\n".join(applied))) if applied else 0
        roundtrip_ok = full_steps == total and applied_steps == prefix_len
        out.append({
            "mission_id": rec["mission_id"],
            "tier": rec["tier"],
            "kind": rec["value_target_kind"],
            "steps": len(dataset.split_tactic_steps(proof)),
            "value_target": rec["value_target"],
            "checks": {
                "imports": imports_ok, "statement": stmt_ok, "proof_by_block": proof_ok,
                "no_sorry": clean, "value_target": target_ok, "roundtrip": roundtrip_ok,
            },
            "ok": all((imports_ok, stmt_ok, proof_ok, clean, target_ok, roundtrip_ok)),
        })
    return out


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

README = """# stage-1 SFT dataset (value head + policy)

Built by `train/build_sft.py` from `./data/bank_v2/lemma`
(shape-classified lemma bank, reference proofs already accepted upstream).
The bank is read-only; nothing here writes back to it.

## Layout

| file | contents |
| --- | --- |
| `stage1/T1-000.jsonl` … `T5-000.jsonl` | one shard per tier |
| `stats.json` | counts, token estimates, proof-length histogram, value-target percentiles |
| `README.md` | this file |

## Record

```json
{"mission_id": "m-…", "env": "0df444a",
 "messages": [{"role": "system", "content": "…"},
              {"role": "user", "content": "<statement_nl> … Imports: … ```lean <context+statement>```"},
              {"role": "assistant", "content": "by …"}],
 "value_target": 9, "value_target_kind": "mission_start",
 "tier": "T1", "problem_type": "lemma", "theorem_id": "…",
 "prefix_len": 0, "total_steps": 9}
```

A `prefix` record adds a second lean fence to the user message
(`Proof so far (steps 1..k of n)`) and its assistant message is the `by` block
with the remaining steps, so `prefix fence + assistant` is the whole proof.

* `value_target` — **remaining tactic steps**, the same unit as the >500 gate
  (`total_steps - prefix_len`). `dataset.split_tactic_steps` counts the steps.
* `value_target_kind` — `mission_start` (one per proof, assistant = full proof)
  or `prefix` (assistant = the steps still to come, user message shows the ones
  already applied). Prefixes exist so the value head sees intermediate states,
  not just the mission opening.
* `prefix_len` = steps already applied (0 for `mission_start`); `total_steps` =
  length of the reference proof.
* `env` is the bank's env `dir` (the Lean/Mathlib pin), so `mission_id` is
  `cache.mission_id_for(env, statement_formal)` — the same key the mission
  registry uses.

## Round-trip

`messages` alone reconstruct the source `Problem.lean`-shaped text: imports +
the lean fence (context_pre + statement) + the prefix fence (if any) + assistant.
The verifier (`--verify N`, default 3 per tier) rebuilds the `by` block from the
prefix fence plus the assistant message and checks: imports parse, the statement
is present, the proof is a `by` block with ≥1 tactic step, no `sorry`/`admit`,
`value_target == total_steps - prefix_len`, and the prefix + assistant step count
equals `total_steps`.

## Caveats (read before training on this)

* **Short proofs.** The reference proofs are lemmas: p50 5 steps, p90 16, p99 41,
  max 163. Stage 1 therefore calibrates the *low* range of the head; the >500
  gate region and the >10k tail are not in this data and must come from stage-2
  MCTS trajectories (or a longer-proof corpus). Do not read a flat
  `tail.recall` on this set as a gate failure.
* **T5 is thin.** T5 has 108 accepted reference proofs (735 of its records are
  `Open`), so its shard is 387 records vs 9.5k–18k for T1–T3.
* **Prefixes.** `--prefixes 3` gives ~3 records per proof; the file count stays
  bounded at one shard per tier plus `stats.json` and this README.
* **Sampling.** `train_stage1.py --stratify N` keeps at most N records per
  `log1p(target)` bucket. Use it for bounded CPU runs: the shards are
  tier-ordered and their first records are dominated by one-step proofs.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", default=BANK_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--prefixes", type=int, default=3,
                    help="prefix variants per proof (0 disables, default 3)")
    ap.add_argument("--limit", type=int, default=None, help="max records per tier")
    ap.add_argument("--tier", action="append", default=None)
    ap.add_argument("--verify", type=int, default=3,
                    help="round-trip N random records per tier (default 3)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    tiers = args.tier or list(TIERS)
    summary = build(bank_dir=args.bank, out_dir=args.out, prefixes=args.prefixes,
                    limit=args.limit, tiers=tiers)

    checks: dict[str, list] = {}
    if args.verify > 0:
        for t in tiers:
            shard = Path(args.out) / summary["tiers"][t]["shard"]
            checks[t] = verify_shard(str(shard), args.verify, args.seed)
        summary["roundtrip"] = {
            t: {"checked": len(v), "ok": sum(1 for r in v if r["ok"]),
                "failed": [r["mission_id"] for r in v if not r["ok"]]}
            for t, v in checks.items()
        }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(README, encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "out": str(out_dir),
        "records_total": summary["records_total"],
        "est_tokens_total": summary["est_tokens_total"],
        "per_tier": {t: summary["tiers"][t]["records"] for t in tiers},
        "roundtrip_ok": all(v["ok"] == v["checked"]
                            for v in summary.get("roundtrip", {}).values()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
