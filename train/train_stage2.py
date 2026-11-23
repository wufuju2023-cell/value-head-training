#!/usr/bin/env python3
"""Stage 2: CE expert iteration on searched MCTS-p2m trajectories.

Stage 1 gives `(goal, context) -> remaining_steps` plus a tactic-class policy.
Stage 2 distils *searched* trajectories back into both heads:

    L = VALUE_LOSS_WEIGHT * value_CE(steps_remaining; weight = kept tokens)
        + soft_CE(visits || policy)

Input is one searched node per line, the shape the MCTS-p2m runs write:

    {"mission_id": "m-…", "goal": "<statement>", "context": "<p2m preamble+state>",
     "action_type": "block" | "decompose",
     "block": "<the agent's proof/tactic block>",
     "outcome": "success" | {"solved": true, "status": "accepted"},
     "steps_remaining": 37,
     "visits": [{"tactic": "simp [add_comm]", "visits": 12}]}   # optional

Two things are built per record, and they are the design's, not this script's:

1. `(input_ids, loss_mask)` from `traj.to_input_ids_loss_mask` — the masking
   rule is `traj.DEFAULT_MASK_POLICY`: system / tool_result / subagent_return
   are 0, user + assistant are kept.  The mask is not decoration: a record whose
   assistant span is entirely masked is dropped, and the kept-token count is the
   weight of its value/CE term.
2. The CE pool itself: `traj.filter_ce_pool` keeps only terminal-success
   trajectories (`traj.CE_SIGNAL`), because nothing in this design rewards
   decomposing or calling a subagent — only reaching the terminal state.

`--dry-run` prints batch shapes, mask spans and labels without importing torch.
`--resume` continues a previous stage-2 run from `<run>/stage2_state.json`.

    python3 train/train_stage2.py --run runs/tiny-sft --tiny
    python3 train/train_stage2.py --run runs/tiny-sft --tiny --dry-run
    python3 train/train_stage2.py --run runs/real-full --trajectories traj.jsonl \\
        --steps 200 --resume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))     # traj.py

import dataset  # noqa: E402
import head  # noqa: E402
import traj  # noqa: E402

SYSTEM_PROMPT = (
    "You are a Lean 4 / Mathlib prover inside a mission DAG. Reply with one "
    "tactic block for the current mission, or decompose the mission when the "
    "estimate says more than 500 proof steps remain."
)

SUCCESS_WORDS = {"success", "solved", "ok", "proved", "accepted", "done", "true"}


# ---------------------------------------------------------------------------
# Records: MCTS-p2m trajectory jsonl -> (traj, input_ids, loss_mask, targets)
# ---------------------------------------------------------------------------

def normalise_outcome(row: dict) -> dict:
    out = row.get("outcome")
    if isinstance(out, dict):
        return out
    if isinstance(out, str):
        return {"solved": out.strip().lower() in SUCCESS_WORDS, "status": out}
    if out is None and "solved" in row:
        return {"solved": bool(row["solved"])}
    return {"solved": False, "status": "unknown"}


def as_trajectory(row: dict) -> dict:
    """Trajectory-shaped dict for traj.py (messages carry the mask spans)."""
    goal = row.get("goal") or row.get("statement_formal") or ""
    context = row.get("context") or row.get("lean_context") or ""
    block = row.get("block") or row.get("action_text") or row.get("proof") or ""
    action_type = str(row.get("action_type") or "block")
    messages = [
        {"role": "system", "text": SYSTEM_PROMPT},
        {"role": "user", "text": f"{goal}\n\n{context}".strip()},
        {"role": "assistant", "text": block},
    ]
    return {
        "mission_id": str(row.get("mission_id") or row.get("id") or ""),
        "env": row.get("env", "default"),
        "goal": goal,
        "lean_context": context,
        "action_type": action_type,
        "action_text": block,
        "messages": messages,
        "outcome": normalise_outcome(row),
        "child_missions": list(row.get("child_missions") or []),
        "child_returns": list(row.get("child_returns") or []),
    }


def policy_label(row: dict) -> str:
    """The class the policy head should emit for this node."""
    action = str(row.get("action_type") or "block")
    if action == "decompose":
        return "decompose"
    for key in ("block", "action_text", "proof"):
        text = str(row.get(key) or "").strip()
        if not text:
            continue
        if text in ("block", "decompose"):     # mission action already, not a tactic
            return text
        steps = dataset.split_tactic_steps(text)
        if steps:
            return dataset.tactic_class(steps[0])
    return "other"


def visit_target(row: dict, vocab: dict[str, int]) -> dict[int, float] | None:
    """Search posterior over the policy vocab, or None (single-label CE)."""
    visits = row.get("visits")
    if not visits:
        return None
    acc: dict[int, float] = {}
    for v in visits:
        tactic = v.get("tactic")
        if tactic is None:
            continue
        idx = vocab.get(str(tactic))
        if idx is None:
            idx = vocab.get(dataset.tactic_class(str(tactic)))
        if idx is None:
            continue                       # unknown class: never invent one
        acc[idx] = acc.get(idx, 0.0) + float(v.get("visits", 0) or 0)
    total = sum(acc.values())
    if total <= 0:
        return None
    return {i: w / total for i, w in acc.items()}


def build_records(rows: list[dict], vocab_list: list[str],
                  tokenizer=None) -> list[dict]:
    """Encode every row: input_ids + loss_mask + value/policy targets + weight."""
    tok = tokenizer or traj.HashTokenizer()
    vocab = {t: i for i, t in enumerate(vocab_list)}
    out: list[dict] = []
    for row in rows:
        t = as_trajectory(row)
        enc = traj.to_input_ids_loss_mask(t, tokenizer=tok)
        n_tok = max(1, enc["n_tokens"])
        kept = enc["n_kept"]
        label = policy_label(row)
        posterior = visit_target(row, vocab)
        if posterior is None:
            idx = vocab.get(label)
            posterior = {idx: 1.0} if idx is not None else None
        out.append({
            "mission_id": t["mission_id"],
            "trajectory": t,
            "input_ids": enc["input_ids"],
            "loss_mask": enc["loss_mask"],
            "n_tokens": enc["n_tokens"],
            "n_kept": kept,
            "spans": enc["spans"],
            "value_target": dataset.value_target(row),
            "policy_label": label,
            "posterior": posterior,
            "weight": 1.0 + math.log1p(kept),
            "token_weight": kept / n_tok,
            "text": f"{t['goal']}\n{t['lean_context']}",
            "goal": t["goal"], "context": t["lean_context"],
            "action_type": t["action_type"],
        })
    return out


# ---------------------------------------------------------------------------
# Tiny trajectories (no MCTS, no corpus)
# ---------------------------------------------------------------------------

def synthesize_tiny_trajectories(run: dict, n: int = 40, seed: int = 7) -> list[dict]:
    """40 nodes in the MCTS-p2m shape, from the stage-1 run's own data."""
    rng = random.Random(seed)
    data_path = run["dir"] / "data.jsonl"
    rows: list[dict] = []
    if data_path.is_file():
        rows = [json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines()
                if l.strip()]
    labels = [t for t in run["vocab_list"] if t != "<pad>"] or ["omega", "simp", "ring"]
    out: list[dict] = []
    for i in range(n):
        base = rows[i % len(rows)] if rows else {
            "goal": f"theorem traj_{i} : True", "context": "", "remaining_steps": 20}
        goal = base.get("goal", f"theorem traj_{i} : True")
        context = base.get("context", "")
        total = float(dataset.value_target(base, dataset.UNIT_STEPS)) \
            if any(k in base for k in dataset.VALUE_TARGET_KEYS) else 20.0
        # search finds a shorter proof than SFT did: the posterior value of the
        # node is a little below the SFT target, which is what stage 2 should learn.
        steps = max(1.0, round(total * rng.uniform(0.55, 0.85)))
        # The synthetic posterior must obey the same rule stage 1 learned
        # (>= 500 steps left => decompose), otherwise stage 2 is asked to teach
        # the policy something that contradicts the value target on the very
        # same state and the CE only grows.  On a tactic-class vocab the rule is
        # unrepresentable, so fall back to the record's own next tactic.
        wanted = "decompose" if steps >= float(head.DEFAULT_THRESHOLD) else "block"
        cand = wanted if wanted in labels else None
        if cand is None and base.get("next_tactic"):
            cls = dataset.tactic_class(str(base["next_tactic"]))
            cand = cls if cls in labels else None
        if cand is None and base.get("next_action") in labels:
            cand = base["next_action"]
        best = cand or labels[i % len(labels)]
        visits = [{"tactic": best, "visits": rng.randint(24, 60)}]
        for other in labels[:4]:
            if other != best:
                visits.append({"tactic": other, "visits": rng.randint(1, 8)})
        out.append({
            "mission_id": f"tiny-traj-{i:03d}",
            "goal": goal,
            "context": context,
            "action_type": "block",
            "block": best,
            "outcome": {"solved": True, "status": "accepted"},
            "steps_remaining": steps,
            "visits": visits,
        })
    return out


# ---------------------------------------------------------------------------
# Run dir
# ---------------------------------------------------------------------------

def load_run(run_dir: str) -> dict:
    root = Path(run_dir)
    if not (root / "config.json").is_file():
        raise SystemExit(f"stage2: {run_dir} is not a stage-1 run dir (no config.json)")
    if not (root / "value-head.pt").is_file():
        raise SystemExit(f"stage2: {run_dir} has no value-head.pt; run train_stage1.py first")
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    vocab_path = root / "vocab.json"
    vocab_list = json.loads(vocab_path.read_text(encoding="utf-8")) if vocab_path.is_file() \
        else ["<pad>"]
    metrics = {}
    metrics_path = root / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {"dir": root, "config": cfg, "vocab_list": vocab_list,
            "vocab": {t: i for i, t in enumerate(vocab_list)},
            "unit": dataset.UNIT_STEPS, "metrics": metrics}


def read_trajectories(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Batching (torch only at the last step)
# ---------------------------------------------------------------------------

def target_matrix(records: list[dict], n_classes: int, torch_ns=None):
    """Dense [B, V] posterior; records without a usable label stay all-zero."""
    rows = []
    for rec in records:
        vec = [0.0] * n_classes
        post = rec.get("posterior")
        if post:
            for i, w in post.items():
                if 0 <= i < n_classes:
                    vec[i] = w
        rows.append(vec)
    if torch_ns is None:
        return rows
    import torch
    return torch.tensor(rows, dtype=torch.float32)


def batch_shapes(records: list[dict]) -> dict:
    """The dry-run contract: shapes and masks, no torch."""
    seq = [len(r["input_ids"]) for r in records]
    kept = [r["n_kept"] for r in records]
    masked = [r["n_tokens"] - r["n_kept"] for r in records]
    return {
        "records": len(records),
        "input_ids": [len(records), max(seq) if seq else 0],
        "loss_mask": [len(records), max(seq) if seq else 0],
        "policy_target": [len(records), None],
        "value_target": [len(records)],
        "seq_len_min": min(seq) if seq else 0,
        "seq_len_max": max(seq) if seq else 0,
        "kept_tokens": sum(kept),
        "masked_tokens": sum(masked),
        "kept_ratio": round(sum(kept) / max(1, sum(kept) + sum(masked)), 4),
        "dropped_fully_masked": sum(1 for r in records if r["n_kept"] == 0),
    }


def encode_batch(records: list[dict], vocab_list: list[str], torch_ns):
    import torch
    texts = [r["text"] for r in records]
    goals = [str(r.get("goal", "")) for r in records]
    contexts = [str(r.get("context", "")) for r in records]
    steps = torch.tensor([r["value_target"] for r in records], dtype=torch.float32)
    weights = torch.tensor([r["weight"] * r["token_weight"] for r in records],
                           dtype=torch.float32)
    target = target_matrix(records, len(vocab_list), torch_ns)
    keep = torch.tensor([r["n_kept"] > 0 for r in records], dtype=torch.bool)
    return {"texts": texts, "goals": goals, "contexts": contexts,
            "steps": steps, "weights": weights,
            "policy_target": target, "keep": keep}


def soft_policy_ce(out, target, keep):
    torch = head.ensure()["torch"]
    logq = torch.log_softmax(out["policy_logits"].float(), dim=-1)
    per = -(target * logq).sum(-1)
    if not bool(keep.any()):
        return per.sum() * 0.0
    return per[keep].mean()


def value_loss(model, out, steps, weights, stop_grad: bool = False):
    """Value CE; with stop_grad the head reads detach(h) (no value -> policy leak)."""
    if not stop_grad:
        return model.value_loss(out, steps, weights=weights)
    h = out["h"].detach()
    vt = model.value_trunk(h)
    v_out: dict = {}
    if model.encoding == "regress":
        v_out["value_scalar"] = model.value_out(vt).squeeze(-1)
    else:
        v_out["value_logits"] = model.value_out(vt)
        if model.value_res is not None:
            v_out["value_res"] = model.value_res(vt).squeeze(-1)
    return model.value_loss(v_out, steps, weights=weights)


def evaluate(model, records, vocab_list, torch_ns) -> dict:
    if not records:
        return {"n": 0}
    batch = encode_batch(records, vocab_list, torch_ns)
    preds = torch_ns["predict_texts"](model, batch["texts"],
                                      goals=batch["goals"], contexts=batch["contexts"])
    est = [p["estimate"] for p in preds]
    true = [r["value_target"] for r in records]
    m = head.metrics(est, true, centers=[float(c) for c in model.centers.tolist()])
    out = model(batch["texts"], goals=batch["goals"], contexts=batch["contexts"])
    top1 = out["policy_logits"].argmax(-1).tolist()
    agree = [i for i, r in enumerate(records)
             if r["posterior"] and max(r["posterior"], key=r["posterior"].get) == top1[i]]
    m["policy_agree_with_search"] = round(len(agree) / len(records), 4) if records else None
    m["unit"] = dataset.UNIT_STEPS
    m["gate"] = {
        "decompose_rate": round(sum(1 for e in est
                                    if head.gate(e)["action"] == "decompose") / len(est), 4),
        "true_decompose_rate": round(sum(1 for t in true if t >= 500) / len(true), 4),
    }
    return m


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="stage-1 run dir")
    ap.add_argument("--trajectories", default=None,
                    help="jsonl of searched nodes ({mission_id, context, goal, "
                         "action_type, block, outcome, steps_remaining})")
    ap.add_argument("--tiny", action="store_true", help="40 synthetic trajectories, CPU only")
    ap.add_argument("--dry-run", action="store_true",
                    help="print batch shapes/masks and exit (no torch)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <run>/stage2_state.json + value-head-stage2.pt")
    ap.add_argument("--steps", type=int, default=None, help="CE steps (default 200)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--value-lr", type=float, default=None)
    ap.add_argument("--policy-lr", type=float, default=None)
    ap.add_argument("--value-loss-weight", "--value-coef", dest="value_loss_weight",
                    type=float, default=head.VALUE_COEF)
    ap.add_argument("--stop-grad-value", action="store_true")
    ap.add_argument("--grad-clip-value", type=float, default=1.0)
    ap.add_argument("--grad-clip-policy", type=float, default=1.0)
    ap.add_argument("--include-failed", action="store_true",
                    help="keep failed trajectories as diagnostics (never CE-eligible)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save-name", default="value-head-stage2.pt")
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args(argv)

    run = load_run(args.run)
    vocab_list = run["vocab_list"]
    if args.steps is None:
        args.steps = 200
    if args.value_lr is None:
        # Never inherit stage-1's lr verbatim: tiny stage 1 uses 2e-2, a real
        # full-FT run 3e-4, and both are wrong here (DESIGN.md §9, L2).
        args.value_lr = 5e-3 if args.tiny else min(
            5e-3, float(run["config"].get("value_lr") or 5e-3))
    if args.policy_lr is None:
        # The policy lr is the leak that hurts.  Measured on
        # state/runs/tiny-sft-full (40 trajectories, 40-60 CE steps, full-FT
        # backbone, value MAE log before = 1.06):
        #   policy_lr 1e-2 -> 2.06   (the shared trunk rotates, L2 wins)
        #   policy_lr 1e-3 -> 0.50
        #   policy_lr 3e-4 -> 0.42
        #   policy_lr 1e-4 -> 0.40
        # so stage 2 defaults to <= 3e-4 even in tiny mode, where stage 1 used
        # 1e-2 to move the hashed skeleton fast enough in 200 steps.
        args.policy_lr = 3e-4 if args.tiny else min(
            3e-4, float(run["config"].get("policy_lr") or 1e-4) / 10.0)

    if args.tiny:
        rows, source = synthesize_tiny_trajectories(run, 40, args.seed), "tiny"
    elif args.trajectories:
        rows, source = read_trajectories(args.trajectories), args.trajectories
    else:
        print("value-head-training/stage2: pass --trajectories <jsonl> or --tiny",
              file=sys.stderr)
        return 2
    if not rows:
        print("value-head-training/stage2: no trajectories", file=sys.stderr)
        return 2
    # MCTS-p2m writes `outcome` as a word ("success"/"sorry"), a status dict, or
    # nothing at all; traj.py wants the dict form before it screens the pool.
    rows = [{**r, "outcome": normalise_outcome(r)} for r in rows]

    # Terminal-only signal: a node enters the CE posterior pool only if its
    # mission reached terminal success (traj.CE_SIGNAL).
    pool = traj.filter_ce_pool(rows, include_failed=args.include_failed,
                               diagnostic=args.include_failed)
    if not args.include_failed:
        rows = pool["kept"]
        if not rows:
            print("value-head-training/stage2: no terminal-success trajectory in "
                  f"{args.trajectories or 'the input'}; nothing to distil "
                  f"(signal={traj.CE_SIGNAL})", file=sys.stderr)
            return 2

    records = build_records(rows, vocab_list)
    usable = [r for r in records if r["posterior"] is not None]
    fully_masked = [r for r in records if r["n_kept"] == 0]
    records = [r for r in records if r["posterior"] is not None and r["n_kept"] > 0]
    if not records:
        print("value-head-training/stage2: every record is either fully masked or has "
              "no known policy class", file=sys.stderr)
        return 2

    if args.dry_run:
        chunk = records[: args.batch_size]
        shapes = batch_shapes(chunk)
        shapes["policy_target"][1] = len(vocab_list)
        print(json.dumps({
            "ok": True, "dry_run": True, "run": str(run["dir"]), "source": source,
            "signal": traj.CE_SIGNAL,
            "kept_pool": len(rows), "records": len(records),
            "dropped_no_label": len(usable) - len(records),
            "dropped_fully_masked": len(fully_masked),
            "batch_shapes": shapes,
            "example": {
                "mission_id": chunk[0]["mission_id"],
                "input_ids_head": chunk[0]["input_ids"][:8],
                "masked_spans": chunk[0]["spans"],
                "kept": chunk[0]["n_kept"], "masked": chunk[0]["n_tokens"] - chunk[0]["n_kept"],
                "value_target": chunk[0]["value_target"],
                "policy_label": chunk[0]["policy_label"],
                "posterior": chunk[0]["posterior"],
                "weight": round(chunk[0]["weight"], 4),
            },
            "lr": {"value": args.value_lr, "policy": args.policy_lr},
            "steps": args.steps,
        }, indent=1))
        return 0

    if not head.torch_available():
        print("value-head-training/stage2: PyTorch is not importable "
              f"({head.torch_error()}). Use --dry-run to inspect the batches, or "
              "install the CPU wheel. Aborting without training.", file=sys.stderr)
        return 2
    torch_ns = head.ensure()
    torch = torch_ns["torch"]
    torch.manual_seed(args.seed)
    if args.tiny:
        torch.set_num_threads(1)

    state_path = run["dir"] / "stage2_state.json"
    start_step = 0
    history: list[dict] = []
    model = None
    if args.resume and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        start_step = int(state.get("step", 0))
        history = list(state.get("history", []))[-50:]
        ckpt = run["dir"] / args.save_name
        if ckpt.is_file():
            model = torch_ns["load_checkpoint"](str(run["dir"]), name=args.save_name)
            print(f"[stage2] resumed from {ckpt.name} at step {start_step}")
        else:
            print(f"[stage2] --resume: no {ckpt.name}, starting from stage 1")
    if model is None:
        model = torch_ns["load_checkpoint"](str(run["dir"]))

    before = evaluate(model, records, vocab_list, torch_ns)
    value_params = [p for n, p in model.named_parameters() if n.startswith("value_")]
    policy_params = [p for n, p in model.named_parameters()
                     if not n.startswith("value_") and p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": value_params, "lr": args.value_lr},
        {"params": policy_params, "lr": args.policy_lr},
    ], weight_decay=0.0)

    t0 = time.time()
    for step in range(start_step + 1, start_step + args.steps + 1):
        idx = list(range(len(records)))
        random.Random(args.seed * 1000 + step).shuffle(idx)
        chunk = [records[i] for i in idx[: args.batch_size]]
        batch = encode_batch(chunk, vocab_list, torch_ns)
        out = model(batch["texts"], goals=batch["goals"], contexts=batch["contexts"])
        vloss = value_loss(model, out, batch["steps"], batch["weights"],
                          stop_grad=args.stop_grad_value)
        ploss = soft_policy_ce(out, batch["policy_target"], batch["keep"])
        loss = args.value_loss_weight * vloss + ploss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip_value > 0 and value_params:
            torch.nn.utils.clip_grad_norm_(value_params, args.grad_clip_value)
        if args.grad_clip_policy > 0 and policy_params:
            torch.nn.utils.clip_grad_norm_(policy_params, args.grad_clip_policy)
        opt.step()
        history.append({"step": step, "value_loss": round(float(vloss.detach()), 4),
                        "policy_ce": round(float(ploss.detach()), 4)})
        if step % args.log_every == 0 or step == start_step + args.steps:
            print(f"[stage2] step {step} value={float(vloss.detach()):.4f} "
                  f"policy_ce={float(ploss.detach()):.4f}", flush=True)

    model.eval()
    after = evaluate(model, records, vocab_list, torch_ns)
    ckpt = torch_ns["save_checkpoint"](model, str(run["dir"]), name=args.save_name)
    metrics = {
        "stage": 2, "source": source, "signal": traj.CE_SIGNAL,
        "trajectories": len(records), "unit": dataset.UNIT_STEPS,
        "steps": args.steps, "resumed_from_step": start_step,
        "value_loss_weight": args.value_loss_weight,
        "stop_grad_value": bool(args.stop_grad_value),
        "lr": {"value": args.value_lr, "policy": args.policy_lr},
        "grad_clip": {"value": args.grad_clip_value, "policy": args.grad_clip_policy},
        "mask": {
            "policy_version": traj.DEFAULT_MASK_POLICY["version"],
            "mask_kinds": list(traj.DEFAULT_MASK_POLICY["mask_kinds"]),
            "kept_ratio": round(sum(r["n_kept"] for r in records)
                                / max(1, sum(r["n_tokens"] for r in records)), 4),
            "dropped_fully_masked": len(fully_masked),
            "dropped_no_label": len(usable) - len(records),
        },
        "wall_seconds": round(time.time() - t0, 2),
        "before": before, "after": after,
        "delta_mae_log": round(after.get("mae_log", 0.0) - before.get("mae_log", 0.0), 4),
        "history": history[-10:],
        "checkpoint": Path(ckpt).name,
    }
    (run["dir"] / "metrics_stage2.json").write_text(json.dumps(metrics, indent=2),
                                                    encoding="utf-8")
    state_path.write_text(json.dumps({
        "step": start_step + args.steps,
        "history": history[-50:],
        "checkpoint": Path(ckpt).name,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2), encoding="utf-8")

    print(f"[stage2] signal={traj.CE_SIGNAL} trajectories={len(records)} "
          f"dropped={sum(pool['reason_counts'].values())} {pool['reason_counts']} "
          f"mask_kept_ratio={metrics['mask']['kept_ratio']}")
    print(f"[stage2] value MAE(log) {before.get('mae_log')} -> {after.get('mae_log')}  "
          f"policy agree(search) {before.get('policy_agree_with_search')} -> "
          f"{after.get('policy_agree_with_search')}")
    print(f"[stage2] gate decompose_rate {after.get('gate', {}).get('decompose_rate')} "
          f"(true {after.get('gate', {}).get('true_decompose_rate')})")
    print(f"[stage2] run dir: {run['dir']} ({metrics['wall_seconds']}s)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except head.TorchUnavailable as exc:
        print(f"value-head-training/stage2: {exc}", file=sys.stderr)
        sys.exit(2)
