#!/usr/bin/env python3
"""Plain-assert tests for goal-full + context pooling (no pytest, no HF model).

    python3 tests/tests_context_pool_plain.py

The bug this file pins down: `build_hf_backbone` used to tokenise
`f"{goal}\\n{context}"` with `max_length=2048` and mean-pool the result, so a
long Lean context lost every token after 2048 — usually the recent proof
history, which is the part that carries `remaining_steps`.

A fake tokenizer (one id per distinct word) and a fake causal LM (one fixed
embedding per id, `hidden_states[-1] == embeddings`) make the pooling testable
without `transformers` or a real checkpoint:

  * a signal at token 5000 must change the pooled vector under tail/segments/
    mean, and must NOT change it under `full` with the old 2048 cap;
  * padding must not move the pooled vector (masked mean);
  * the goal is truncated from the FRONT at `--goal-max-tokens`;
  * the output is `hidden_size`, and the default combine adds no parameters, so
    existing checkpoints still load.

Exit code 1 on the first failure.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import head  # noqa: E402

CHECKS = 0


def ok(name: str) -> None:
    global CHECKS
    CHECKS += 1
    print(f"ok {CHECKS:02d} - {name}")


def skip(name: str, why: str) -> None:
    print(f"skip - {name} ({why})")


# ---------------------------------------------------------------------------
# Fake tokenizer / backbone
# ---------------------------------------------------------------------------

class FakeTokenizer:
    """Whitespace tokenizer with one stable id per distinct word."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.pad_token_id = 0
        self.eos_token_id = 0

    def __call__(self, text, add_special_tokens: bool = False):
        ids = []
        for word in str(text).split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) + 1
            ids.append(self.vocab[word])
        return {"input_ids": ids}


def make_fake_model(torch, dim: int = 8, seed: int = 11):
    """Minimal causal LM: hidden_states[-1] is the token embedding table."""

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            torch.manual_seed(seed)
            self.emb = torch.nn.Embedding(100_000, dim)
            self.config = SimpleNamespace(hidden_size=dim)

        def forward(self, input_ids, attention_mask=None, output_hidden_states=True,
                    use_cache=False):
            h = self.emb(input_ids)
            return SimpleNamespace(hidden_states=(h,))

    return FakeModel()


def build(pool: str, tok: FakeTokenizer, torch, **kw):
    import train_stage1
    model = make_fake_model(torch, dim=8)
    bb = train_stage1.make_hf_backbone(
        model, tok, head.ensure(), context_pool=pool,
        goal_max_tokens=kw.get("goal_max_tokens", 1024),
        context_window=kw.get("context_window", 3072),
        segments=kw.get("segments", 4),
        max_length=kw.get("max_length", 8192),
        pool_combine=kw.get("pool_combine", "mean"))
    return bb


def filler(n: int, tag: str = "w") -> list[str]:
    return [f"{tag}{i}" for i in range(n)]


def diff(a, b) -> float:
    return float((a.detach() - b.detach()).abs().max())


# ---------------------------------------------------------------------------
# Pure geometry (no torch)
# ---------------------------------------------------------------------------

def test_chunk_geometry() -> None:
    import train_stage1 as ts
    ids = list(range(10_000))
    tail = ts.context_chunks(ids, "tail", context_window=3072)
    assert len(tail) == 1 and tail[0] == list(range(10_000 - 3072, 10_000))
    seg = ts.context_chunks(ids, "segments", context_window=3072, segments=4)
    assert len(seg) == 4, [len(c) for c in seg]
    assert seg[-1][-1] == 9999 and seg[0][0] == max(0, 10_000 - 4 * 3072)
    assert all(len(c) <= 3072 for c in seg)
    # a short context is one partial chunk, never padded up to the window
    short = ts.context_chunks([1, 2, 3], "segments", context_window=3072, segments=4)
    assert short == [[1, 2, 3]]
    mean = ts.context_chunks(ids, "mean", max_length=5000, goal_len=1000)
    assert mean == [list(range(10_000 - 4000, 10_000))]
    assert ts.context_chunks([], "tail", context_window=10) == []
    assert ts.truncate_ids([1, 2, 3, 4], 2) == [1, 2]
    assert ts.truncate_ids([1, 2, 3, 4], 2, "left") == [3, 4]
    ok("context_chunks: tail keeps the last window, segments covers segments*window, mean respects max_length")


def test_fit_chunks() -> None:
    import train_stage1 as ts
    chunks = [[1, 2, 3], [4, 5, 6], [7, 8]]
    assert ts.fit_chunks(chunks, 100) == chunks
    assert ts.fit_chunks(chunks, 5) == [[4, 5, 6], [7, 8]]
    # dropping is from the OLDEST token: keep the newest `budget` tokens
    got = ts.fit_chunks(chunks, 4)
    assert got == [[5, 6], [7, 8]], got
    assert ts.fit_chunks(chunks, 0) == []
    ok("fit_chunks: budget drops oldest tokens first, never the newest")


# ---------------------------------------------------------------------------
# The actual bug: token 5000 is invisible to `full` (2048 cap), visible to
# tail / segments / mean.
# ---------------------------------------------------------------------------

def test_signal_at_5000() -> None:
    if not head.torch_available():
        skip("signal at token 5000", head.torch_error())
        return
    import torch

    tok = FakeTokenizer()
    head_words = filler(6000)
    # same length, same positions; only token 5000 differs
    with_signal = list(head_words)
    with_signal[5000] = "SIGNALTOKEN"
    base_ctx = " ".join(head_words)
    sig_ctx = " ".join(with_signal)
    goal = "theorem t : 1 = 1"

    # full = the old behaviour: 2048-token cap, so 5000 is cut off
    full = build("full", tok, torch, max_length=2048)
    a = full([goal], contexts=[base_ctx])
    b = full([goal], contexts=[sig_ctx])
    assert diff(a, b) == 0.0, diff(a, b)
    ok("full (legacy 2048 cap): a signal at token 5000 does NOT change the pool")

    for mode, kw in (("tail", {"context_window": 3072}),
                     ("segments", {"context_window": 3072, "segments": 4}),
                     ("mean", {"max_length": 8192})):
        bb = build(mode, tok, torch, **kw)
        a = bb([goal], contexts=[base_ctx])
        b = bb([goal], contexts=[sig_ctx])
        d = diff(a, b)
        assert d > 1e-6, (mode, d)
        ok(f"{mode}: the same signal at token 5000 DOES change the pool (max|d|={d:.2e})")


def test_old_region_still_visible() -> None:
    """tail must still see the goal and the newest context, not only the middle."""
    if not head.torch_available():
        skip("goal + newest context visible", head.torch_error())
        return
    import torch
    tok = FakeTokenizer()
    goal = "theorem t : a + b = b + a"
    ctx = " ".join(filler(6000))
    bb = build("tail", tok, torch, context_window=3072)
    base = bb([goal], contexts=[ctx])
    goal2 = "theorem t : a + b = c + d"
    assert diff(base, bb([goal2], contexts=[ctx])) > 1e-6
    newest = ctx + " NEWESTTOKEN"
    assert diff(base, bb([goal], contexts=[newest])) > 1e-6
    ok("tail: goal and the newest context token both reach the pool")


# ---------------------------------------------------------------------------
# Padding invariance
# ---------------------------------------------------------------------------

def test_padding_invariance() -> None:
    if not head.torch_available():
        skip("padding invariance", head.torch_error())
        return
    import torch
    tok = FakeTokenizer()
    goal = "theorem t : True"
    short = " ".join(filler(5, "s"))
    long = " ".join(filler(500, "l"))
    for mode in ("tail", "segments", "mean", "full"):
        bb = build(mode, tok, torch, max_length=8192)
        alone = bb([goal], contexts=[short])
        batch = bb([goal, goal], contexts=[short, long])
        assert diff(alone, batch[:1]) < 1e-6, (mode, diff(alone, batch[:1]))
        assert batch.shape == (2, 8), batch.shape
        ok(f"{mode}: padding to the longest row does not move the short row's pool")


def test_empty_context() -> None:
    if not head.torch_available():
        skip("empty context", head.torch_error())
        return
    import torch
    tok = FakeTokenizer()
    bb = build("tail", tok, torch)
    out = bb(["theorem t : True", "theorem u : False"], contexts=["", "import Mathlib"])
    assert out.shape == (2, 8)
    assert bool(torch.isfinite(out).all())
    ok("empty context: falls back to the goal pool, no NaN, no crash")


# ---------------------------------------------------------------------------
# Goal truncation from the front
# ---------------------------------------------------------------------------

def test_goal_truncation() -> None:
    if not head.torch_available():
        skip("goal truncation", head.torch_error())
        return
    import torch
    tok = FakeTokenizer()
    ctx = "import Mathlib"
    words = filler(400, "g")
    inside = list(words)
    inside[10] = "EARLYSIGNAL"
    outside = list(words)
    outside[300] = "LATESIGNAL"
    bb = build("tail", tok, torch, goal_max_tokens=64, context_window=64)
    base = bb([" ".join(words)], contexts=[ctx])
    assert diff(base, bb([" ".join(inside)], contexts=[ctx])) > 1e-6
    assert diff(base, bb([" ".join(outside)], contexts=[ctx])) == 0.0
    ok("goal: --goal-max-tokens keeps the front of the statement and drops the tail")


# ---------------------------------------------------------------------------
# Output shape / checkpoint compatibility
# ---------------------------------------------------------------------------

def test_output_dim_and_params() -> None:
    if not head.torch_available():
        skip("output dim / checkpoint compatibility", head.torch_error())
        return
    import torch
    tok = FakeTokenizer()
    mean = build("tail", tok, torch)
    assert mean.hidden_size == 8
    names = dict(mean.named_parameters())
    assert not any(n.startswith("pool_proj") for n in names), list(names)
    proj = build("tail", tok, torch, pool_combine="projection")
    assert "pool_proj.weight" in dict(proj.named_parameters())
    out = proj(["theorem t : True"], contexts=["import Mathlib"])
    assert out.shape == (1, 8)
    ok("default combine is parameter-free and returns hidden_size; projection is opt-in")


def test_model_integration() -> None:
    """head.Model + predict_texts accept (goals, contexts) and keep the shapes."""
    if not head.torch_available():
        skip("head.Model integration", head.torch_error())
        return
    import torch
    import train_stage1
    tok = FakeTokenizer()
    bb = build("tail", tok, torch, context_window=128)
    ns = head.ensure()
    model = ns["Model"]({"encoding": "dual", "buckets": 64, "max_steps": 32768,
                         "hidden": 8, "value_hidden": 16, "vocab_size": 3,
                         "decode": "mode"})
    model.backbone = bb
    model.cfg["hidden"] = 8
    goals = ["theorem t : True", "theorem u : 1 = 2"]
    ctxs = [" ".join(filler(200)), ""]
    out = model([f"{g}\n{c}" for g, c in zip(goals, ctxs)], goals=goals, contexts=ctxs)
    assert out["value_logits"].shape == (2, 64)
    assert out["policy_logits"].shape == (2, 3)
    rows = ns["predict_texts"](model, [f"{g}\n{c}" for g, c in zip(goals, ctxs)],
                               goals=goals, contexts=ctxs)
    assert len(rows) == 2 and rows[0]["ci_low"] <= rows[0]["ci_high"]
    ok("head.Model/predict_texts: split goal+context flows through the value/policy heads")


def test_hashed_backbone_signature() -> None:
    """The tiny backbone must accept (texts) and (goals, contexts) identically."""
    if not head.torch_available():
        skip("hashed backbone signature", head.torch_error())
        return
    ns = head.ensure()
    bb = ns["HashedBackbone"](32)
    goals, ctxs = ["theorem t : True"], ["import Mathlib\nintro h"]
    a = bb([f"{goals[0]}\n{ctxs[0]}"])
    b = bb(goals=goals, contexts=ctxs)
    assert diff(a, b) == 0.0
    ok("HashedBackbone: (goals, contexts) joins to exactly the same text path")


def main() -> int:
    test_chunk_geometry()
    test_fit_chunks()
    test_signal_at_5000()
    test_old_region_still_visible()
    test_padding_invariance()
    test_empty_context()
    test_goal_truncation()
    test_output_dim_and_params()
    test_model_integration()
    test_hashed_backbone_signature()
    print(f"\n{CHECKS} checks passed (context pooling)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
