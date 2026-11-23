#!/usr/bin/env python3
"""Score one (goal, context) pair and decide: prove or decompose?

    python3 train/predict.py --run runs/tiny-dual \\
        --goal "theorem foo : 1 + 1 = 2" --context "import Mathlib"

Prints one compact JSON object:

    {"estimate":..., "ci_low":..., "ci_high":..., "tail_saturated":false,
     "gate":{"action":"prove|decompose", "reason":..., "threshold":500, ...}}

`estimate` is in the checkpoint's unit: **remaining proof blocks for the
current mission** by default (`"unit": "mission"`), remaining single tactics only
for an old `--unit tactic` run.  The 500 threshold is a mission-level estimate:
above it the agent must decompose, it does not merely try harder.

`--prev-action` feeds the hysteresis: a run that already decomposed only goes
back to "prove" below 0.8 * threshold.  `--jsonl <file>` scores a batch of
{"goal","context"} records and prints one JSON array.

Exit codes: 0 ok, 2 no torch, 3 no run dir / checkpoint, 4 bad input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import head  # noqa: E402


def pick_checkpoint(run_dir: Path, name: str | None) -> Path:
    if name:
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path
    for candidate in ("value-head-stage2.pt", "value-head.pt"):
        if (run_dir / candidate).is_file():
            return run_dir / candidate
    raise FileNotFoundError(f"no value-head*.pt under {run_dir}")


def score(model, pairs: list[dict], threshold: float, prev: str) -> list[dict]:
    ns = head.ensure()
    texts = [f"{p.get('goal', '')}\n{p.get('context', '')}" for p in pairs]
    rows = ns["predict_texts"](model, texts,
                               goals=[str(p.get("goal", "")) for p in pairs],
                               contexts=[str(p.get("context", "")) for p in pairs])
    out = []
    for p, r in zip(pairs, rows):
        g = head.gate(r["estimate"], threshold=threshold, prev=prev,
                      ci_high=r["ci_high"], tail_saturated=r["tail_saturated"])
        out.append({**r, "gate": g})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="stage-1/2 run dir (default: $VHT_RUN)")
    ap.add_argument("--goal", default=None)
    ap.add_argument("--context", default="")
    ap.add_argument("--jsonl", default=None, help='file of {"goal","context"} records')
    ap.add_argument("--checkpoint", default=None, help="explicit checkpoint name in the run dir")
    ap.add_argument("--threshold", type=float, default=head.DEFAULT_THRESHOLD)
    ap.add_argument("--prev-action", choices=["prove", "decompose"], default="prove")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args(argv)

    if not head.torch_available():
        print("value-head-training/predict: PyTorch is not importable "
              f"({head.torch_error()}). Install torch in the training venv; "
              "the CLI cannot score without it.", file=sys.stderr)
        return 2

    run = args.run or __import__("os").environ.get("VHT_RUN")
    if not run:
        print("value-head-training/predict: pass --run <run dir> (or set VHT_RUN)", file=sys.stderr)
        return 3
    run_dir = Path(run)
    try:
        ckpt = pick_checkpoint(run_dir, args.checkpoint)
    except FileNotFoundError as exc:
        print(f"value-head-training/predict: {exc}", file=sys.stderr)
        return 3

    pairs: list[dict] = []
    if args.jsonl:
        with open(args.jsonl, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
    elif args.goal:
        pairs.append({"goal": args.goal, "context": args.context})
    else:
        print("value-head-training/predict: pass --goal or --jsonl", file=sys.stderr)
        return 4

    ns = head.ensure()
    # stage-2 checkpoints are loaded through the same helper: stage 1 wrote the
    # cfg (including lora_rank), so the adapter is reattached before load.
    model = ns["load_checkpoint"](str(run_dir), name=ckpt.name)
    scored = score(model, pairs, args.threshold, args.prev_action)
    indent = 2 if args.pretty else None

    if args.jsonl:
        print(json.dumps([{
            **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items() if k != "gate"},
            "gate": s["gate"],
            "goal": p.get("goal", "")[:200],
            "context": p.get("context", "")[:200],
        } for s, p in zip(scored, pairs)], indent=indent))
    else:
        s = scored[0]
        payload = {
            "ok": True,
            "estimate": round(s["estimate"], 2),
            "ci_low": round(s["ci_low"], 2),
            "ci_high": round(s["ci_high"], 2),
            "tail_saturated": s["tail_saturated"],
            "gate": s["gate"],
            "model": {"run": str(run_dir), "checkpoint": ckpt.name,
                      "encoding": model.cfg.get("encoding"),
                      "unit": model.cfg.get("unit", "mission"),
                      "block_steps": model.cfg.get("block_steps"),
                      "scheme": model.cfg.get("scheme"),
                      "base": model.cfg.get("base")},
        }
        print(json.dumps(payload, indent=indent))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except head.TorchUnavailable as exc:
        print(f"value-head-training/predict: {exc}", file=sys.stderr)
        sys.exit(2)
