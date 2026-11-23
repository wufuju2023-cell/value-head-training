#!/usr/bin/env python3
"""Stage 1: joint SFT of the policy and the value head.

    L = VALUE_LOSS_WEIGHT * value_loss + policy_CE        (weight default 1e-3)

The value target is the number of **remaining proof steps** of a mission state;
the policy target is the class of the next tactic step (SFT examples) or the
mission action `block` / `decompose` (mission records).  One objective, two
heads, and the interference between them is measured, not assumed:

    every --log-every steps  ->  |g_value| and |g_policy| *on the shared params*
                                 (two `autograd.grad` probes, so the number is
                                 the actual contribution of each term, not the
                                 norm of a summed gradient)
    --warmup-steps N         ->  first N steps move only the heads
    --grad-clip-value/-policy->  one clip per parameter group, never a global one
    --stop-grad-value        ->  value head reads detach(h): no value -> policy leak

Schemes (DESIGN.md §9): `full` = full-parameter fine-tuning, the default and the
only one the current recipe needs.  `lora` is kept as the secondary path
(frozen base + low-rank adapter + heads) and is one flag away.

Data: `--data` is a stage-1 SFT directory/file from `train/build_sft.py`
(default `./data/sft/stage1`), a mission jsonl from
`train/dataset.py`, or `--tiny` for 50 synthetic records with no corpus at all.

`--tiny` uses a hashed-text backbone so the whole loop (data -> loss -> metrics
-> checkpoint) runs on one CPU in seconds.  If torch is not importable and
`--tiny` is set, the same data is fitted by a deterministic pure-python ridge
regression and the same metrics are printed (`"backend": "pure_python"`), so the
data path is checkable on a machine without torch.

    python3 train/train_stage1.py --tiny --out runs/tiny-sft
    python3 train/train_stage1.py --tiny --data ./data/sft/stage1 \\
        --limit 2000 --max-steps 200 --out runs/tiny-sft-2k
    python3 train/train_stage1.py --scheme full --base /path/to/model \\
        --data ./data/sft/stage1 --out runs/real-full --epochs 1

Run dir: config.json, metrics.json, value-head.pt, vocab.json, data_ref.json.
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

import dataset  # noqa: E402
import head  # noqa: E402


# ---------------------------------------------------------------------------
# Goal-full + context pooling (replaces the old 2048-token truncation)
# ---------------------------------------------------------------------------
#
# The old backbone tokenised `f"{goal}\n{context}"` with max_length=2048 and
# mean-pooled the last hidden state, so a long Lean context lost everything
# after token 2048 — usually the recent proof history, which is the part that
# carries the remaining-steps signal.
#
# The fix is structural, not a bigger constant: tokenise `goal` and `context`
# SEPARATELY, pool each with its own attention mask, then combine.  The default
# combine is the elementwise mean of the two pooled vectors, so the backbone
# output is still `hidden_size` and every existing checkpoint loads unchanged.
# `--pool-combine projection` adds a learned Linear(2H, H); it is off by
# default because it changes the checkpoint (new `pool_proj.*` tensors).

CONTEXT_POOLS = ("mean", "tail", "segments", "full")
POOL_COMBINES = ("mean", "projection")


def truncate_ids(ids: list[int], limit: int, side: str = "right") -> list[int]:
    """Keep the first (`right`) or last (`left`) `limit` ids."""
    if limit <= 0:
        return []
    if len(ids) <= limit:
        return list(ids)
    return list(ids[-limit:]) if side == "left" else list(ids[:limit])


def context_chunks(ids: list[int], mode: str, *, context_window: int = 3072,
                   segments: int = 4, max_length: int = 8192,
                   goal_len: int = 0) -> list[list[int]]:
    """Split context ids into the chunks that are each mean-pooled.

    tail      the LAST `context_window` tokens (recent proof history wins)
    segments  the last `segments * context_window` tokens, cut into chunks of
              at most `context_window` (so `segments` is a ceiling on compute,
              not a promise: a 6k-token context with a 3072 window is 2 chunks)
    mean      the whole context, capped to `max_length - goal_len` from the LEFT
              (oldest tokens go first)
    full      one chunk = the whole context; only used by the legacy path
    """
    if mode not in CONTEXT_POOLS:
        raise ValueError(f"context_pool must be one of {CONTEXT_POOLS}, got {mode!r}")
    if not ids:
        return []
    if mode == "tail":
        return [truncate_ids(ids, context_window, "left")]
    if mode == "segments":
        window = max(1, int(context_window))
        keep = truncate_ids(ids, max(1, int(segments)) * window, "left")
        return [keep[i:i + window] for i in range(0, len(keep), window)]
    if mode == "mean":
        budget = max_length - max(0, int(goal_len))
        return [truncate_ids(ids, budget, "left")] if budget > 0 else []
    return [list(ids)]


def fit_chunks(chunks: list[list[int]], budget: int) -> list[list[int]]:
    """Drop the OLDEST tokens until the chunks together fit `budget`."""
    if budget <= 0:
        return []
    total = sum(len(c) for c in chunks)
    if total <= budget:
        return [list(c) for c in chunks]
    drop = total - budget
    out = []
    for chunk in chunks:
        if drop <= 0:
            out.append(list(chunk))
            continue
        if len(chunk) <= drop:
            drop -= len(chunk)
            continue
        out.append(list(chunk[drop:]))
        drop = 0
    return [c for c in out if c]


_HF_CLASSES: dict[str, object] = {}


def hf_backbone_class(torch_ns):
    """Define (once) the nn.Module that wraps a HF model + tokenizer."""
    if "cls" in _HF_CLASSES:
        return _HF_CLASSES["cls"]
    nn = torch_ns["nn"]

    class HFBackbone(nn.Module):
        """text -> hidden, with goal/context tokenised and pooled separately."""

        def __init__(self, model, tokenizer, *, context_pool: str = "tail",
                     goal_max_tokens: int = 1024, context_window: int = 3072,
                     segments: int = 4, max_length: int = 8192,
                     pool_combine: str = "mean"):
            super().__init__()
            if context_pool not in CONTEXT_POOLS:
                raise ValueError(f"context_pool must be one of {CONTEXT_POOLS}")
            if pool_combine not in POOL_COMBINES:
                raise ValueError(f"pool_combine must be one of {POOL_COMBINES}")
            self.model = model
            self.tokenizer = tokenizer
            cfg = getattr(model.config, "text_config", model.config)
            self.hidden_size = int(getattr(cfg, "hidden_size"))
            self.context_pool = context_pool
            self.goal_max_tokens = int(goal_max_tokens)
            self.context_window = int(context_window)
            self.segments = max(1, int(segments))
            self.max_length = int(max_length)
            self.pool_combine = pool_combine
            self.pad_id = getattr(tokenizer, "pad_token_id", None)
            if self.pad_id is None:
                self.pad_id = getattr(tokenizer, "eos_token_id", None)
            self.pad_id = 0 if self.pad_id is None else int(self.pad_id)
            self.pool_proj = (nn.Linear(2 * self.hidden_size, self.hidden_size)
                              if pool_combine == "projection" else None)

        # -- tokenisation ---------------------------------------------------
        def ids_of(self, text: str) -> list[int]:
            """Raw ids, no special tokens: we pool, we do not generate."""
            enc = self.tokenizer(text, add_special_tokens=False)
            return list(enc["input_ids"])

        def split_text(self, text: str) -> tuple[str, str]:
            """`make_batch` joins with one newline; keep that exact convention."""
            goal, _, ctx = text.partition("\n")
            return goal, ctx

        # -- pooling --------------------------------------------------------
        def device(self):
            try:
                return next(self.model.parameters()).device
            except StopIteration:  # pragma: no cover - parameterless fake
                import torch
                return torch.device("cpu")

        def pool_ids(self, id_rows: list[list[int]]):
            """Masked mean of the last hidden state for a batch of id lists."""
            import torch
            dev = self.device()
            if not id_rows:
                return torch.zeros((0, self.hidden_size), device=dev)
            valid = [1.0 if row else 0.0 for row in id_rows]
            rows = [row if row else [self.pad_id] for row in id_rows]
            width = max(len(r) for r in rows)
            ids = torch.full((len(rows), width), int(self.pad_id), dtype=torch.long,
                             device=dev)
            mask = torch.zeros((len(rows), width), dtype=torch.long, device=dev)
            for i, row in enumerate(rows):
                ids[i, :len(row)] = torch.tensor(row, dtype=torch.long)
                mask[i, :len(row)] = 1
            out = self.model(input_ids=ids, attention_mask=mask,
                             output_hidden_states=True, use_cache=False)
            h = out.hidden_states[-1]
            m = mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            keep = torch.tensor(valid, dtype=pooled.dtype, device=pooled.device).unsqueeze(-1)
            return pooled * keep

        def encode_split(self, goals: list[str], contexts: list[str]):
            """Pool goal and context separately, then combine to `hidden_size`."""
            import torch
            seqs: list[list[int]] = []
            meta: list[tuple[int, bool]] = []
            for i, (goal, ctx) in enumerate(zip(goals, contexts)):
                g_ids = truncate_ids(self.ids_of(goal or ""), self.goal_max_tokens)
                c_ids = self.ids_of(ctx or "") if ctx else []
                c_chunks = context_chunks(
                    c_ids, self.context_pool,
                    context_window=self.context_window, segments=self.segments,
                    max_length=self.max_length, goal_len=len(g_ids))
                c_chunks = fit_chunks(c_chunks, self.max_length - len(g_ids))
                if g_ids:
                    seqs.append(g_ids)
                    meta.append((i, True))
                for chunk in c_chunks:
                    seqs.append(chunk)
                    meta.append((i, False))
            if not seqs:
                return torch.zeros((len(goals), self.hidden_size), device=self.device())
            pooled = self.pool_ids(seqs)
            n = len(goals)
            goal_vec = torch.zeros((n, self.hidden_size), dtype=pooled.dtype,
                                   device=pooled.device)
            ctx_vec = torch.zeros_like(goal_vec)
            ctx_n = torch.zeros(n, dtype=pooled.dtype, device=pooled.device)
            for k, (row, is_goal) in enumerate(meta):
                if is_goal:
                    goal_vec[row] = pooled[k]
                else:
                    ctx_vec[row] += pooled[k]
                    ctx_n[row] += 1.0
            ctx_vec = ctx_vec / ctx_n.clamp(min=1.0).unsqueeze(-1)
            has_ctx = (ctx_n > 0).unsqueeze(-1)
            if self.pool_proj is not None:
                combined = self.pool_proj(torch.cat([goal_vec, ctx_vec], dim=-1))
                combined = torch.where(has_ctx, combined, goal_vec)
            else:
                combined = torch.where(has_ctx, 0.5 * (goal_vec + ctx_vec), goal_vec)
            return torch.tanh(combined.float())

        def forward(self, texts=None, goals=None, contexts=None):
            # Fill in only the missing side, so a caller may pass texts plus
            # one explicit field without having the other silently rebuilt.
            if goals is None and texts is not None:
                goals = [self.split_text(t)[0] for t in texts]
            if contexts is None and texts is not None:
                contexts = [self.split_text(t)[1] for t in texts]
            if goals is None or contexts is None:
                raise ValueError("HFBackbone needs texts or (goals, contexts)")
            if len(goals) != len(contexts):
                raise ValueError(f"goals/contexts length mismatch: {len(goals)} vs {len(contexts)}")
            if self.context_pool == "full":
                import torch
                joined = [f"{g}\n{c}" for g, c in zip(goals, contexts)]
                rows = [truncate_ids(self.ids_of(t), self.max_length) for t in joined]
                return torch.tanh(self.pool_ids(rows).float())
            return self.encode_split(list(goals), list(contexts))

    _HF_CLASSES["cls"] = HFBackbone
    return HFBackbone


def make_hf_backbone(model, tokenizer, torch_ns, **pool_kwargs):
    """Wrap an already-loaded (model, tokenizer); the test seam."""
    cls = hf_backbone_class(torch_ns)
    return cls(model, tokenizer, **pool_kwargs)


def build_hf_backbone(base: str, torch_ns, **pool_kwargs):
    """Load a HF causal LM, freeze it, expose forward(texts|goals,contexts)."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on the machine
        raise head.TorchUnavailable(
            "a real `--base` needs `transformers` "
            f"({type(exc).__name__}: {exc}). Use --tiny for the CPU skeleton, "
            "or install the training extras in the GPU venv.")
    if not os.path.isdir(base):
        raise head.TorchUnavailable(f"--base is not a directory: {base}")
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True, torch_dtype="auto")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return make_hf_backbone(model, tok, torch_ns, **pool_kwargs)


def build_backbone(args, cfg, torch_ns):
    if args.tiny or not args.base or args.base in ("hashed", "tiny"):
        return torch_ns["HashedBackbone"](cfg["hidden"]), "hashed"
    backbone = build_hf_backbone(
        args.base, torch_ns,
        context_pool=args.context_pool, goal_max_tokens=args.goal_max_tokens,
        context_window=args.context_window, segments=args.segments,
        max_length=args.max_length, pool_combine=args.pool_combine)
    cfg["hidden"] = backbone.hidden_size
    return backbone, os.path.basename(os.path.normpath(args.base))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_records(args) -> tuple[list[dict], str]:
    """Training records: {goal, context, value_target, policy_label, ...}."""
    if args.tiny and not args.data:
        rows = dataset.build(bank_dir=None, mathlib=None, tiny=True,
                             unit=dataset.UNIT_STEPS,
                             block_steps=args.block_steps,
                             threshold=args.block_threshold)
        for r in rows:
            r["policy_label"] = r.get("next_action")
        return rows, "tiny"

    data = args.data
    if data:
        p = Path(data)
        if p.is_dir() or (p.is_file() and _looks_like_sft(p)):
            rows = dataset.load_sft(str(p), limit=args.limit, kinds=args.kind,
                                    stratify=args.stratify)
            for r in rows:
                r["policy_label"] = dataset.tactic_class(r.get("next_tactic") or "")
                r["policy_kind"] = "tactic_class"
            return rows, str(p)
        rows = dataset.read_jsonl(str(p))
        for r in rows:
            if r.get("unit") is None and "remaining_steps" in r:
                r["policy_label"] = r.get("next_action")
            else:
                r["policy_label"] = r.get("next_action")
        return rows, str(p)

    # default: the stage-1 SFT shards, falling back to the old bank build
    if os.path.isdir(dataset.SFT_DEFAULT):
        rows = dataset.load_sft(dataset.SFT_DEFAULT, limit=args.limit,
                                kinds=args.kind, stratify=args.stratify)
        for r in rows:
            r["policy_label"] = dataset.tactic_class(r.get("next_tactic") or "")
            r["policy_kind"] = "tactic_class"
        return rows, dataset.SFT_DEFAULT
    bank = args.bank_dir if args.bank_dir else (
        dataset.BANK_DEFAULT if os.path.isdir(dataset.BANK_DEFAULT) else None)
    rows = dataset.build(bank_dir=bank, mathlib=args.mathlib, tiny=False,
                         limit=args.limit, max_files=args.max_files,
                         mathlib_limit=args.mathlib_limit, seed=args.seed,
                         unit=dataset.UNIT_STEPS)
    for r in rows:
        r["policy_label"] = r.get("next_action")
    return rows, "built"


def _looks_like_sft(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    return "messages" in json.loads(line)
    except Exception:
        return False
    return False


def build_vocab(records: list[dict], unit: str = dataset.UNIT_STEPS) -> list[str]:
    """Policy classes: `block|decompose` for mission records, tactic classes for SFT."""
    labels = {policy_label_of(r) for r in records}
    labels.discard("")
    return ["<pad>"] + sorted(labels)


def split_records(records: list[dict], seed: int,
                  test_frac: float = 0.2) -> tuple[list[dict], list[dict]]:
    rows = list(records)
    random.Random(seed).shuffle(rows)
    n_test = max(1, int(len(rows) * test_frac)) if len(rows) >= 5 else 0
    return rows[n_test:], rows[:n_test]


def policy_label_of(record: dict) -> str:
    """Label the policy head must emit for a record (one source for vocab + batch)."""
    label = record.get("policy_label") or record.get("next_action")
    if not label and record.get("next_tactic"):
        label = dataset.tactic_class(str(record["next_tactic"]))
    return str(label or "")


def make_batch(rows: list[dict], vocab: dict[str, int], torch_ns,
               unit: str = dataset.UNIT_STEPS):
    import torch
    goals = [str(r.get("goal", "") or "") for r in rows]
    contexts = [str(r.get("context", "") or "") for r in rows]
    texts = [f"{g}\n{c}" for g, c in zip(goals, contexts)]
    steps = torch.tensor([dataset.value_target(r, unit) for r in rows], dtype=torch.float32)
    targets = torch.tensor(
        [vocab.get(policy_label_of(r), -100) if not r.get("terminal") else -100
         for r in rows], dtype=torch.long)
    return {"texts": texts, "goals": goals, "contexts": contexts,
            "steps": steps, "policy_targets": targets}


# ---------------------------------------------------------------------------
# Training: torch path
# ---------------------------------------------------------------------------

def configure_phase(model, scheme: str, phase: str, adapter_params):
    torch_ns = head.ensure()
    torch_ns["set_trainable"](model, backbone=False, heads=True)
    for group in (model.value_trunk, model.value_out, model.policy):
        for p in group.parameters():
            p.requires_grad = True
    if model.value_res is not None:
        for p in model.value_res.parameters():
            p.requires_grad = True
    if phase == "warmup":
        return
    if scheme == "full":
        for p in model.backbone.parameters():
            p.requires_grad = True
    else:
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in adapter_params:
            p.requires_grad = True


def group_grad_norm(params) -> float:
    import torch
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5


def shared_probe(model, out, losses: dict, shared_params) -> dict:
    """|g_value| and |g_policy| *on the shared params*, term by term."""
    import torch
    probe: dict[str, float] = {}
    if not shared_params:
        return {"g_value": 0.0, "g_policy": 0.0, "value_share": 0.0}
    for key, loss in losses.items():
        live = [p for p in shared_params if p.requires_grad]
        if not live:
            probe["g_" + key] = 0.0
            continue
        grads = torch.autograd.grad(loss, live, retain_graph=True, allow_unused=True)
        total = 0.0
        for g in grads:
            if g is not None:
                total += float(g.detach().pow(2).sum())
        probe["g_" + key] = round(total ** 0.5, 6)
    probe["live_shared"] = len([p for p in shared_params if p.requires_grad])
    gv, gp = probe.get("g_value", 0.0), probe.get("g_policy", 0.0)
    denom = gv * gv + gp * gp
    probe["value_share"] = round(gv * gv / denom, 6) if denom > 0 else 0.0
    return probe


def clip_groups(model, torch_ns, clip_value: float, clip_policy: float) -> dict:
    """One clip per parameter group (never a single global norm)."""
    torch = torch_ns["torch"]
    value_params = [p for n, p in model.named_parameters()
                    if n.startswith("value_") and p.grad is not None]
    policy_params = [p for n, p in model.named_parameters()
                     if not n.startswith("value_") and p.grad is not None]
    out = {"value_before": round(group_grad_norm(value_params), 4),
           "policy_before": round(group_grad_norm(policy_params), 4)}
    if clip_value > 0 and value_params:
        torch.nn.utils.clip_grad_norm_(value_params, clip_value)
    if clip_policy > 0 and policy_params:
        torch.nn.utils.clip_grad_norm_(policy_params, clip_policy)
    out["value_after"] = round(group_grad_norm(value_params), 4)
    out["policy_after"] = round(group_grad_norm(policy_params), 4)
    return out


def evaluate(model, rows: list[dict], vocab: dict[str, int],
             unit: str = dataset.UNIT_STEPS,
             tail_at: float = head.TAIL_STEPS) -> dict:
    if not rows:
        return {"n": 0}
    torch_ns = head.ensure()
    goals = [str(r.get("goal", "") or "") for r in rows]
    contexts = [str(r.get("context", "") or "") for r in rows]
    preds_rows = torch_ns["predict_texts"](
        model, [f"{g}\n{c}" for g, c in zip(goals, contexts)],
        goals=goals, contexts=contexts)
    preds = [r["estimate"] for r in preds_rows]
    trues = [dataset.value_target(r, unit) for r in rows]
    centers = [float(c) for c in model.centers.tolist()]
    m = head.metrics(preds, trues, centers=centers, at=tail_at)
    m["unit"] = unit
    keep = [i for i, r in enumerate(rows)
            if not r.get("terminal") and str(r.get("policy_label") or "") in vocab]
    if keep and model.cfg.get("vocab_size", 0) > 1:
        out = model([f"{rows[i].get('goal', '')}\n{rows[i].get('context', '')}" for i in keep],
                    goals=[str(rows[i].get("goal", "") or "") for i in keep],
                    contexts=[str(rows[i].get("context", "") or "") for i in keep])
        pred_id = out["policy_logits"].argmax(-1).tolist()
        m["policy_pass_at_1"] = round(
            sum(1 for k, i in enumerate(keep)
                if vocab[str(rows[i]["policy_label"])] == pred_id[k]) / len(keep), 4)
        m["policy_n"] = len(keep)
        m["policy_random"] = round(1.0 / max(1, len(vocab) - 1), 4)
    m["interval"] = {
        "coverage_80": round(sum(1 for r, t in zip(preds_rows, trues)
                                if r["ci_low"] <= t <= r["ci_high"]) / len(preds_rows), 4),
        "mean_width": round(sum(r["ci_high"] - r["ci_low"] for r in preds_rows) / len(preds_rows), 1),
        "tail_saturated": sum(1 for r in preds_rows if r["tail_saturated"]),
    }
    return m


def train_torch(args, records, vocab_list, train_rows, test_rows, run_dir,
                data_summary):
    torch_ns = head.ensure()
    torch = torch_ns["torch"]
    torch.manual_seed(args.seed)
    if args.tiny:
        torch.set_num_threads(1)
    vocab = {t: i for i, t in enumerate(vocab_list)}

    cfg = {
        "encoding": args.encoding, "buckets": args.buckets, "max_steps": args.grid_max,
        "hidden": args.hidden, "value_hidden": args.value_hidden,
        "vocab_size": len(vocab_list), "dropout": 0.0, "decode": args.decode,
        "scheme": args.scheme, "base": args.base or "hashed", "tiny": bool(args.tiny),
        "lora_rank": args.lora_rank if args.scheme == "lora" else 0,
        "unit": dataset.UNIT_STEPS,
        # pooling provenance: only the real-HF backbone reads these, but the run
        # dir must record how the state was built (a 2048-cap run and an 8192
        # tail run are not comparable).
        "context_pool": args.context_pool, "goal_max_tokens": args.goal_max_tokens,
        "context_window": args.context_window, "segments": args.segments,
        "max_length": args.max_length, "pool_combine": args.pool_combine,
    }
    backbone, backbone_name = build_backbone(args, cfg, torch_ns)
    model = torch_ns["Model"](cfg)
    model.backbone = backbone
    model.cfg["hidden"] = cfg["hidden"]
    model.cfg["base"] = backbone_name

    adapter_params: list = []
    if args.scheme == "lora":
        if backbone_name == "hashed":
            adapter_params = torch_ns["attach_lora"](model, rank=args.lora_rank)
        else:
            try:
                torch_ns["attach_lora_hf"](model.backbone.model, rank=args.lora_rank)
            except head.TorchUnavailable as exc:
                print(f"value-head-training/stage1: {exc}", file=sys.stderr)
                return 2
            adapter_params = [p for p in model.backbone.model.parameters() if p.requires_grad]
    configure_phase(model, args.scheme, "warmup", adapter_params)

    value_params = [p for n, p in model.named_parameters()
                    if n.startswith("value_") and p.requires_grad]
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith("value_") and p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": value_params, "lr": args.value_lr},
        {"params": other_params, "lr": args.policy_lr},
    ], weight_decay=0.0)

    # The shared params are the backbone (what both heads read), independent of
    # whether the current phase freezes them: `autograd.grad` reports the leak
    # even during warmup, when the backbone does not yet move.
    shared_params = [p for n, p in model.named_parameters()
                     if not n.startswith(("value_", "policy"))]
    history: list[dict] = []
    probes: list[dict] = []
    t0 = time.time()
    warm = max(0, args.warmup_steps)
    global_step = 0
    phase_now = "warmup"
    stop = False
    for epoch in range(1, args.epochs + 1):
        if stop:
            break
        configure_phase(model, args.scheme, phase_now, adapter_params)
        model.train()
        order = list(train_rows)
        random.Random(args.seed + epoch).shuffle(order)
        epoch_losses: list[dict] = []
        for start in range(0, len(order), args.batch_size):
            rows = order[start:start + args.batch_size]
            if global_step >= warm and phase_now == "warmup":
                # heads-only warm start is over: unfreeze the base (scheme full)
                # or the adapter (scheme lora) mid-epoch, not at the next epoch.
                configure_phase(model, args.scheme, "train", adapter_params)
                phase_now = "train"
                opt = torch.optim.AdamW([
                    {"params": [p for n, p in model.named_parameters()
                                if n.startswith("value_") and p.requires_grad],
                     "lr": args.value_lr},
                    {"params": [p for n, p in model.named_parameters()
                                if not n.startswith("value_") and p.requires_grad],
                     "lr": args.policy_lr},
                ], weight_decay=0.0)
            batch = make_batch(rows, vocab, torch_ns)
            out = model(batch["texts"], goals=batch["goals"], contexts=batch["contexts"])
            total, parts = torch_ns["combined_loss"](
                model, out, batch, value_coef=args.value_loss_weight,
                stop_grad_value=args.stop_grad_value,
                policy_weight=args.policy_weight)

            probe = None
            if global_step % max(1, args.log_every) == 0:
                # Per-term gradients on the shared params, measured before the
                # two losses are summed: `autograd.grad` does not touch `.grad`.
                vloss = model.value_loss(out, batch["steps"], weights=batch.get("weights"))
                ploss = (torch_ns["policy_ce"](out, batch["policy_targets"])
                         if bool((batch["policy_targets"] >= 0).any())
                         else torch.zeros((), dtype=vloss.dtype))
                losses = {"value": args.value_loss_weight * vloss,
                          "policy": args.policy_weight * ploss}
                probe = shared_probe(model, out, losses, shared_params)

            opt.zero_grad(set_to_none=True)
            total.backward()
            clipped = clip_groups(model, torch_ns, args.grad_clip_value,
                                  args.grad_clip_policy)
            opt.step()
            global_step += 1
            epoch_losses.append(parts)
            if probe is not None:
                probe.update({"step": global_step, **{k: clipped[k] for k in
                                                      ("value_before", "policy_before")}})
                probes.append(probe)
                marked = ("" if probe["live_shared"] else " [warmup: shared frozen]")
                print(f"[stage1] step {global_step} epoch {epoch} "
                      f"value={parts['value_loss']:.4f} policy={parts['policy_loss']:.4f} "
                      f"|g_value|={probe['g_value']:.6f} |g_policy|={probe['g_policy']:.6f} "
                      f"value_share={probe['value_share']:.5f}{marked} "
                      f"clip(v/p)=({clipped['value_before']:.2f}/{clipped['policy_before']:.2f})"
                      f"->({clipped['value_after']:.2f}/{clipped['policy_after']:.2f})",
                      flush=True)
            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break
        history.append({
            "epoch": epoch,
            "value_loss": round(sum(p["value_loss"] for p in epoch_losses) / max(1, len(epoch_losses)), 4),
            "policy_loss": round(sum(p["policy_loss"] for p in epoch_losses) / max(1, len(epoch_losses)), 4),
            "phase": "warmup" if global_step < warm else "train",
        })

    model.eval()
    test_metrics = evaluate(model, test_rows, vocab, dataset.UNIT_STEPS, args.tail_at)
    train_metrics = evaluate(model, train_rows, vocab, dataset.UNIT_STEPS, args.tail_at)
    ckpt = torch_ns["save_checkpoint"](model, str(run_dir))
    (run_dir / "vocab.json").write_text(json.dumps(vocab_list), encoding="utf-8")

    metrics = {
        "stage": 1, "backend": "torch", "scheme": args.scheme,
        "encoding": args.encoding, "base": backbone_name,
        "torch": torch.__version__,
        "params": torch_ns["count_params"](model),
        "value_loss_weight": args.value_loss_weight,
        "value_lr": args.value_lr, "policy_lr": args.policy_lr,
        "unit": dataset.UNIT_STEPS,
        "tail_at": args.tail_at,
        "stop_grad_value": bool(args.stop_grad_value),
        "warmup_steps": warm,
        "grad_clip": {"value": args.grad_clip_value, "policy": args.grad_clip_policy},
        "steps": global_step,
        "wall_seconds": round(time.time() - t0, 2),
        "data": data_summary,
        "test": test_metrics, "train": train_metrics,
        "grad_probes": probes[-5:],
        "history": history,
        "checkpoint": Path(ckpt).name,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# Training: no-torch deterministic path (tiny only)
# ---------------------------------------------------------------------------

def _features(text: str, dim: int = 64) -> list[float]:
    base = list(head.hash_vector(text, dim))
    goal, _, ctx = text.partition("\n")
    base += [math.log1p(len(goal.split())) / 8.0,
             math.log1p(len(ctx.split())) / 8.0,
             math.log1p(ctx.count("\n")) / 4.0,
             1.0]
    return base


def _solve(matrix: list[list[float]], rhs: list[list[float]]) -> list[list[float]]:
    """Gaussian elimination with partial pivoting (deterministic)."""
    n = len(matrix)
    k = len(rhs[0])
    a = [row[:] + r for row, r in zip(matrix, rhs)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            continue
        a[col], a[piv] = a[piv], a[col]
        pv = a[col][col]
        a[col] = [v / pv for v in a[col]]
        for r in range(n):
            if r == col:
                continue
            f = a[r][col]
            if f:
                a[r] = [vr - f * vc for vr, vc in zip(a[r], a[col])]
    return [row[n:n + k] for row in a]


def train_pure(args, records, vocab_list, train_rows, test_rows, run_dir,
               data_summary) -> dict:
    """Deterministic ridge regression: value + policy classes, pure python."""
    t0 = time.time()
    labels = [t for t in vocab_list if t != "<pad>"]
    label_id = {t: i for i, t in enumerate(labels)}
    dim = 68
    xtx = [[0.0] * dim for _ in range(dim)]
    xty = [[0.0] * (1 + len(labels)) for _ in range(dim)]
    wty = [0.0] * (1 + len(labels))
    for rec in train_rows:
        text = f"{rec.get('goal', '')}\n{rec.get('context', '')}"
        x = _features(text, dim - 4)
        target_log = math.log1p(max(0.0, dataset.value_target(rec)))
        label = str(rec.get("policy_label") or "")
        cls = label_id.get(label)
        row_y = [target_log] + [1.0 if i == cls else 0.0 for i in range(len(labels))] \
            if cls is not None else [target_log] + [0.0] * len(labels)
        for i in range(dim):
            xi = x[i]
            if xi == 0.0:
                continue
            for j in range(i, dim):
                xtx[i][j] += xi * x[j]
            for c in range(len(row_y)):
                xty[i][c] += xi * row_y[c]
        if cls is not None:
            wty[cls] += 1.0
    for i in range(dim):
        for j in range(i):
            xtx[i][j] = xtx[j][i]
    ridge = 1e-2
    for i in range(dim):
        xtx[i][i] += ridge
    weights = _solve(xtx, xty)          # dim x (1 + n_labels)
    n_policy = sum(wty)

    def predict(rec) -> tuple[float, str]:
        text = f"{rec.get('goal', '')}\n{rec.get('context', '')}"
        x = _features(text, dim - 4)
        logv = sum(x[i] * weights[i][0] for i in range(dim))
        est = math.expm1(min(20.0, max(0.0, logv)))
        scores = [sum(x[i] * weights[i][1 + c] for i in range(dim)) for c in range(len(labels))]
        best = labels[max(range(len(scores)), key=lambda c: scores[c])] if labels else ""
        return est, best

    def score(rows: list[dict]) -> dict:
        preds, trues, hits, n_pol = [], [], 0, 0
        for rec in rows:
            est, cls = predict(rec)
            preds.append(est)
            trues.append(dataset.value_target(rec))
            if rec.get("policy_label"):
                n_pol += 1
                hits += int(cls == rec["policy_label"])
        m = head.metrics(preds, trues)
        m["policy_pass_at_1"] = round(hits / n_pol, 4) if n_pol else None
        m["policy_n"] = n_pol
        m["policy_random"] = round(1.0 / max(1, len(labels)), 4) if labels else None
        return m

    test = score(test_rows)
    train = score(train_rows)
    metrics = {
        "stage": 1, "backend": "pure_python", "method": "ridge_hashed_features",
        "scheme": args.scheme, "torch": None,
        "value_loss_weight": args.value_loss_weight,
        "unit": dataset.UNIT_STEPS,
        "steps": 1,
        "wall_seconds": round(time.time() - t0, 2),
        "data": data_summary,
        "params": {"total": dim * (1 + len(labels)), "trainable": dim * (1 + len(labels))},
        "test": test, "train": train,
        "grad_probes": [],
        "policy_labels": len(labels),
        "checkpoint": None,
    }
    (run_dir / "vocab.json").write_text(json.dumps(vocab_list), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheme", choices=["full", "lora"], default="full",
                    help="full = full-parameter FT (default); lora = frozen base + adapter (secondary)")
    ap.add_argument("--base", default=None,
                    help="HF model dir; omit / 'hashed' for the tiny backbone")
    ap.add_argument("--data", default=None,
                    help=f"SFT dir/file or mission jsonl (default {dataset.SFT_DEFAULT})")
    ap.add_argument("--kind", action="append", default=None,
                    help="SFT variant filter: mission_start / prefix, repeatable")
    ap.add_argument("--out", default="runs/stage1")
    ap.add_argument("--tiny", action="store_true",
                    help="hashed backbone, CPU, synthetic data unless --data is given")
    ap.add_argument("--backend", choices=["auto", "torch", "pure"], default="auto",
                    help="auto = torch, falling back to pure python when --tiny and no torch")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="cap optimizer steps (0 = run --epochs epochs)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--value-lr", type=float, default=None,
                    help="default 2e-2 for --tiny, 3e-4 for a real base")
    ap.add_argument("--policy-lr", type=float, default=None,
                    help="default 1e-2 for --tiny, 1e-4 for a real base")
    ap.add_argument("--value-loss-weight", "--value-coef", dest="value_loss_weight",
                    type=float, default=head.VALUE_COEF,
                    help="lambda on the value loss (default 1e-3)")
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--warmup-steps", type=int, default=20,
                    help="steps where only the heads move (value-head warm start)")
    ap.add_argument("--grad-clip-value", type=float, default=1.0,
                    help="clip norm for the value parameter group (0 = off)")
    ap.add_argument("--grad-clip-policy", type=float, default=1.0,
                    help="clip norm for the policy/base parameter group (0 = off)")
    ap.add_argument("--stop-grad-value", action="store_true",
                    help="value head sees detach(h): no value -> policy leak")
    ap.add_argument("--encoding", choices=list(head.ENCODINGS), default="dual")
    ap.add_argument("--buckets", type=int, default=head.DEFAULT_BUCKETS)
    ap.add_argument("--grid-max", type=int, default=head.DEFAULT_MAX_STEPS,
                    help="top of the value grid in steps (>10k tail is inside it)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--value-hidden", type=int, default=128)
    ap.add_argument("--decode", choices=["mode", "expected"], default="mode")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--context-pool", choices=list(CONTEXT_POOLS), default="tail",
                    help="how the Lean context is pooled against the goal: "
                         "tail = last --context-window tokens (default), "
                         "segments = last segments*window tokens in window-sized "
                         "chunks, mean = whole context up to --max-length, "
                         "full = legacy joined truncation (A/B only)")
    ap.add_argument("--goal-max-tokens", type=int, default=1024,
                    help="goal tokens kept from the FRONT (default 1024)")
    ap.add_argument("--context-window", type=int, default=3072,
                    help="tokens per context chunk (default 3072)")
    ap.add_argument("--segments", type=int, default=4,
                    help="max context chunks under --context-pool segments")
    ap.add_argument("--max-length", type=int, default=8192,
                    help="hard token budget for one state, goal + context")
    ap.add_argument("--pool-combine", choices=list(POOL_COMBINES), default="mean",
                    help="mean = elementwise mean of the goal/context pools "
                         "(checkpoint-compatible, default); projection = learned "
                         "Linear(2H,H), which ADDS pool_proj.* to the checkpoint")
    ap.add_argument("--tail-at", type=float, default=None,
                    help="tail threshold for tail.recall (default 500 steps = the gate)")
    ap.add_argument("--block-steps", type=int, default=dataset.BLOCK_STEPS_DEFAULT)
    ap.add_argument("--block-threshold", type=float, default=dataset.BLOCK_THRESHOLD_DEFAULT)
    ap.add_argument("--bank-dir", default=None)
    ap.add_argument("--mathlib", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stratify", type=int, default=0,
                    help="keep at most N SFT records per log1p(target) bucket "
                         "(0 = take the first --limit lines)")
    ap.add_argument("--mathlib-limit", type=int, default=None)
    ap.add_argument("--max-files", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--log-every", type=int, default=10,
                    help="steps between |g_value| |g_policy| probes")
    args = ap.parse_args(argv)

    if args.epochs is None:
        args.epochs = 150 if args.tiny else 1
    if args.value_lr is None:
        args.value_lr = 2e-2 if args.tiny else 3e-4
    if args.policy_lr is None:
        args.policy_lr = 1e-2 if args.tiny else 1e-4
    if args.tail_at is None:
        args.tail_at = 500.0

    records, source = load_records(args)
    if not records:
        print("value-head-training/stage1: no records", file=sys.stderr)
        return 2
    sft = any(r.get("source") == "sft" for r in records)
    data_summary = dataset.summarize_sft(records) if sft else dataset.summarize(records)
    vocab_list = build_vocab(records)
    train_rows, test_rows = split_records(records, args.seed)
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)

    use_torch = args.backend == "torch" or (args.backend == "auto" and head.torch_available())
    if use_torch and not head.torch_available():
        print(f"value-head-training/stage1: PyTorch is not importable ({head.torch_error()}) "
              "and --backend torch was requested", file=sys.stderr)
        return 2

    if use_torch:
        metrics = train_torch(args, records, vocab_list, train_rows, test_rows, run_dir,
                              data_summary)
        if metrics is None:
            return 2
    elif args.tiny or args.backend == "pure":
        print("[stage1] torch unavailable: deterministic pure-python ridge fallback "
              "(tiny data path only)", file=sys.stderr)
        metrics = train_pure(args, records, vocab_list, train_rows, test_rows, run_dir,
                             data_summary)
    else:
        print("value-head-training/stage1: PyTorch is not importable "
              f"({head.torch_error()}). Install torch in the training venv, or run "
              "--tiny for the CPU skeleton. Aborting without training.", file=sys.stderr)
        return 2

    data_ref = {"path": source, "records": len(records),
                "note": "source path; shards are not copied into the run dir"}
    (run_dir / "data_ref.json").write_text(json.dumps(data_ref), encoding="utf-8")
    metrics["data_ref"] = data_ref
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps({**args.__dict__, **{
        "backbone": metrics.get("base", "hashed"),
        "resolved_vocab_size": len(vocab_list),
    }}, indent=2), encoding="utf-8")

    t = metrics["test"]
    print(f"[stage1] backend={metrics['backend']} scheme={args.scheme} "
          f"encoding={args.encoding} records={len(records)} steps={metrics['steps']}")
    print(f"[stage1] value MAE(log) = {t['mae_log']:.4f}  "
          f"MAE(steps) = {t['mae_steps']:.1f}  "
          f"baseline = {t['baseline_mae_log']:.4f}  "
          f"skill = {t.get('skill_vs_median')}  "
          f"bucket_acc = {t.get('bucket_acc')}  "
          f"pass@1 = {t.get('policy_pass_at_1')} (random {t.get('policy_random')})")
    print(f"[stage1] calibration monotone={t['calibration']['monotone']} "
          f"violations={t['calibration']['violations']} gap={t['calibration']['gap_log']}  "
          f"tail>={args.tail_at:g} recall={t['tail']['recall']} n={t['tail']['n']}")
    if metrics.get("grad_probes"):
        p = metrics["grad_probes"][-1]
        print(f"[stage1] last probe step {p['step']}: |g_value|={p['g_value']:.3f} "
              f"|g_policy|={p['g_policy']:.3f} value_share={p['value_share']:.5f}")
    print(f"[stage1] run dir: {run_dir}  ({metrics['wall_seconds']}s)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except head.TorchUnavailable as exc:
        print(f"value-head-training/stage1: {exc}", file=sys.stderr)
        sys.exit(2)
