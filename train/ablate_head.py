#!/usr/bin/env python3
"""64-bucket ablation: what the grid geometry and the decoder actually buy.

Five arms, one fixture, one trainer, one metric set:

    uniform64       64 equal-width bins over [0, uniform_span] (shipped baseline,
                    span 2048 by default: it cannot name >2048 steps at all)
    log64           64 log-spaced bins over [1, 32768], mode decode
    dual(mode)      log64 + bounded in-bucket residual, decoded by BINNED MODE
    dual(expected)  same head, decoded by GLOBAL EXPECTATION (the old d_hat)
    regress         direct smooth-L1 on log1p(d), no grid, no distribution

The fixture is `head.synthetic_corpus`: heavy tail (20% at 10k..32768), 15%
exact terminal zeros, and a text->steps map with bounded noise, so a correct
head reaches the noise floor and no better.  Differences between arms are then
encoding error, not data luck.

    python3 train/ablate_head.py --tiny          # ~10 s, CPU, 3 seeds
    python3 train/ablate_head.py --tiny --json   # machine-readable

Metrics (all on the held-out split, mean over seeds):

    mae_log      mean |log1p(d_hat) - log1p(d)|        primary, scale free
    mae_steps    mean |d_hat - d|                     interpretable
    zero_exact   P(d_hat == 0.0 | d == 0)             terminal exactness (B1)
    tail_recall  P(d_hat >= 10k | d >= 10k)           the gate's worst error
    tail_mae_log mean |log error| on the >10k subset
    cov80        fraction of truths inside the 80% interval
    mono         seeds (out of n) with no calibration violation, i.e.
                 "higher predicted bin => higher true mean"
    tail_rate    fraction of rows whose mode sits in the last bin (B2 flag)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import head

ARMS = (
    # (label, encoding, decode, extra cfg)
    ("uniform64", "uniform", "mode", {}),
    # same arm with the full 32k span: a fair test of "log spacing" vs
    # "linear spacing", i.e. not a strawman about the info ceiling.
    ("uniform64@32k", "uniform", "mode", {"uniform_span": 32768}),
    ("log64", "log", "mode", {}),
    ("dual(mode)", "dual", "mode", {}),
    ("dual(expected)", "dual", "expected", {}),
    ("regress", "regress", "mode", {}),
)


def build(arm: str, torch_ns, hidden: int = 48, value_hidden: int = 96,
          uniform_span: int = 2048, buckets: int = head.DEFAULT_BUCKETS,
          max_steps: int = head.DEFAULT_MAX_STEPS):
    for label, encoding, decode, extra in ARMS:
        if label == arm:
            cfg = {"encoding": encoding, "decode": decode, "buckets": buckets,
                   "max_steps": max_steps, "hidden": hidden, "value_hidden": value_hidden,
                   "vocab_size": 2, "uniform_span": uniform_span, **extra}
            return torch_ns["Model"](cfg)
    raise KeyError(arm)


def train_arm(arm: str, train_rows, torch_ns, *, seed: int, steps: int,
              batch_size: int = 16, lr: float = 3e-3, **cfg) -> object:
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = build(arm, torch_ns, **cfg)
    torch_ns["set_trainable"](model, backbone=True, heads=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for step in range(steps):
        rows = train_rows[(step * 13) % (len(train_rows) - batch_size):][:batch_size]
        texts = [r["text"] for r in rows]
        steps_t = torch.tensor([r["remaining_steps"] for r in rows], dtype=torch.float32)
        model.train()
        total, _ = torch_ns["combined_loss"](
            model, model(texts), {"texts": texts, "steps": steps_t}, value_coef=head.VALUE_COEF)
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
    return model


def score(model, rows, torch_ns, tail_at: float = head.TAIL_STEPS) -> dict:
    import torch

    preds_rows = torch_ns["predict_texts"](model, [r["text"] for r in rows])
    preds = [p["estimate"] for p in preds_rows]
    trues = [float(r["remaining_steps"]) for r in rows]
    centers = [float(c) for c in model.centers.tolist()]
    m = head.metrics(preds, trues, centers=centers, at=tail_at)
    cover = sum(1 for p, t in zip(preds_rows, trues) if p["ci_low"] <= t <= p["ci_high"])
    tail_rows = [p for p, t in zip(preds_rows, trues) if t >= tail_at]
    return {
        "mae_log": m["mae_log"],
        "mae_steps": m["mae_steps"],
        "zero_n": m["zero"]["n"],
        "zero_exact": m["zero"]["exact"],
        "zero_mae_steps": m["zero"]["mae_steps"],
        "tail_n": m["tail"]["n"],
        "tail_recall": m["tail"]["recall"],
        "tail_mae_log": m["tail"]["mae_log"],
        "cov80": round(cover / len(rows), 4),
        "monotone": m["calibration"]["monotone"],
        "gap_log": m["calibration"]["gap_log"],
        "tail_rate": round(sum(1 for p in tail_rows if p["tail_saturated"])
                           / max(1, len(tail_rows)), 4),
        "tail_flag_rate": round(sum(1 for p in preds_rows if p["tail_saturated"]) / len(rows), 4),
        "n": len(rows),
    }


def mean_rows(rows: list[dict]) -> dict:
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k != "n"]
    out = {"n": rows[0]["n"]}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = round(sum(vals) / len(vals), 4) if vals else None
    # a calibration violation in any seed is a violation, so report the count
    out["monotone"] = all(r["monotone"] for r in rows)
    out["mono_ok"] = "%d/%d" % (sum(1 for r in rows if r["monotone"]), len(rows))
    return out


def markdown(per_arm: dict[str, dict], meta: dict) -> str:
    head_row = ("| arm | MAE(log) | MAE(steps) | zero-exact | tail recall@10k "
                "| tail MAE(log) | flag@tail | cov80 | mono |")
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [f"n_train={meta['n_train']} n_test={meta['n_test']} "
             f"seeds={meta['seeds']} steps={meta['steps']} "
             f"tail_at={meta['tail_at']:.0f} buckets={meta['buckets']} "
             f"uniform_span={meta['uniform_span']}", "",
             head_row, sep]
    order = [a for a, *_ in ARMS if a in per_arm]
    for arm in order:
        r = per_arm[arm]
        lines.append("| {} | {:.4f} | {:.1f} | {} | {} | {} | {} | {:.3f} | {} |".format(
            arm, r["mae_log"], r["mae_steps"],
            "-" if r["zero_exact"] is None else f"{r['zero_exact']:.3f}",
            "-" if r["tail_recall"] is None else f"{r['tail_recall']:.3f}",
            "-" if r["tail_mae_log"] is None else f"{r['tail_mae_log']:.4f}",
            "-" if r.get("tail_rate") is None else f"{r['tail_rate']:.3f}",
            r["cov80"], r.get("mono_ok", "yes" if r["monotone"] else "NO")))
    return "\n".join(lines)


def verdict(per_arm: dict[str, dict], ref: str = "log64",
            cand: str = "dual(mode)") -> dict:
    """The fallback rule, in code so it cannot drift from DESIGN §11a.

    The gate is a threshold on a *tail* quantity with asymmetric cost, so bulk
    MAE(log) is a tie-breaker, not the decision.  Dual is kept unless it is
    measurably worse than plain log64 on one of the four decision-relevant
    columns.  A 5% bulk gain is explicitly *not* required, and a 1% bulk loss is
    explicitly *not* a reason to fall back.
    """
    if cand not in per_arm or ref not in per_arm:
        return {"decided": False, "why": f"need both {ref} and {cand}"}
    c, r = per_arm[cand], per_arm[ref]
    fails = []
    if c["tail_recall"] is not None and r["tail_recall"] is not None:
        if c["tail_recall"] < r["tail_recall"] - 0.02:
            fails.append(f"tail_recall {c['tail_recall']:.3f} < {r['tail_recall']:.3f} - 0.02")
    if c["tail_mae_log"] is not None and r["tail_mae_log"] is not None:
        if c["tail_mae_log"] > r["tail_mae_log"] + 0.02:
            fails.append(f"tail_mae_log {c['tail_mae_log']:.4f} > {r['tail_mae_log']:.4f} + 0.02")
    if c["cov80"] < r["cov80"] - 0.03:
        fails.append(f"cov80 {c['cov80']:.3f} < {r['cov80']:.3f} - 0.03")
    if (c["zero_exact"] or 0) < (r["zero_exact"] or 0):
        fails.append(f"zero_exact {c['zero_exact']} < {r['zero_exact']}")
    bulk = 1.0 - c["mae_log"] / max(1e-9, r["mae_log"])
    return {"decided": True, "keep_dual": not fails, "fails": fails,
            "bulk_gain": round(bulk, 4), "why": "ok" if not fails else "; ".join(fails)}



def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiny", action="store_true",
                    help="the standard ablation size (300/150 rows, 3 seeds, 120 steps)")
    ap.add_argument("--n", type=int, default=None, help="total synthetic rows")
    ap.add_argument("--n-test", type=int, default=None, help="held-out rows")
    ap.add_argument("--seeds", type=int, default=None, help="number of seeds (7, 11, 23, ...)")
    ap.add_argument("--steps", type=int, default=None, help="optimizer steps per arm")
    ap.add_argument("--buckets", type=int, default=head.DEFAULT_BUCKETS)
    ap.add_argument("--max-steps", type=int, default=head.DEFAULT_MAX_STEPS)
    ap.add_argument("--uniform-span", type=int, default=2048,
                    help="top of the uniform grid (the shipped head used 2048)")
    ap.add_argument("--tail-at", type=float, default=head.TAIL_STEPS)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--value-hidden", type=int, default=96)
    ap.add_argument("--arms", default=None, help="comma-separated subset of arm labels")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    ap.add_argument("--out", default=None, help="write the markdown table here")
    args = ap.parse_args(argv)

    n = args.n if args.n is not None else (300 if args.tiny or args.n is None else 300)
    n_test = args.n_test if args.n_test is not None else (150 if args.tiny or args.n_test is None else 150)
    n_seeds = args.seeds if args.seeds is not None else (3 if args.tiny else 1)
    steps = args.steps if args.steps is not None else (120 if args.tiny else 120)
    seeds = [7, 11, 23, 37, 41][:n_seeds]
    arms = [a.strip() for a in args.arms.split(",")] if args.arms else [a[0] for a in ARMS]

    if not head.torch_available():
        print(f"ablate: PyTorch is not importable ({head.torch_error()}). "
              "Nothing here works without torch; install the CPU wheel.", file=sys.stderr)
        return 2
    torch_ns = head.ensure()

    all_rows = head.synthetic_corpus(n=n + n_test, seed=7)
    train_rows, test_rows = all_rows[:n], all_rows[n:]
    cfg = {"hidden": args.hidden, "value_hidden": args.value_hidden,
           "uniform_span": args.uniform_span, "buckets": args.buckets,
           "max_steps": args.max_steps}

    per_arm: dict[str, dict] = {}
    raw: dict[str, list] = {}
    for arm in arms:
        rows = []
        for seed in seeds:
            model = train_arm(arm, train_rows, torch_ns, seed=seed, steps=steps, **cfg)
            rows.append(score(model, test_rows, torch_ns, tail_at=args.tail_at))
        per_arm[arm] = mean_rows(rows)
        raw[arm] = rows

    meta = {"n_train": len(train_rows), "n_test": len(test_rows), "seeds": seeds,
            "steps": steps, "tail_at": args.tail_at, "buckets": args.buckets,
            "uniform_span": args.uniform_span, "config": cfg,
            "tail_share": round(sum(1 for r in test_rows
                                    if r["remaining_steps"] >= args.tail_at) / len(test_rows), 3),
            "zero_share": round(sum(1 for r in test_rows
                                   if r["remaining_steps"] == 0) / len(test_rows), 3)}
    if args.json:
        print(json.dumps({"meta": meta, "mean": per_arm, "per_seed": raw}, indent=1))
        return 0

    table = markdown(per_arm, meta)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
        print(f"\n[ablate] table written to {args.out}")

    # The fallback rule, evaluated rather than asserted (DESIGN §11a).
    v = verdict(per_arm)
    if v.get("decided"):
        print(f"\n[ablate] dual(mode) vs log64: {v['why']} "
              f"(bulk MAE(log) gain {v['bulk_gain']:+.1%})")
        print("[ablate] {} -> {}".format(
            "fallback rule", "KEEP dual(mode)" if v["keep_dual"] else "FALL BACK to log64"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
