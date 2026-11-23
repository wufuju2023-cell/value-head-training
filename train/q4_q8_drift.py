#!/usr/bin/env python3
"""Q4 vs Q8 drift experiment for the value-head gate (small, real).

Same prompt, same temperature (0), same `max_tokens` (256) against two
llama.cpp endpoints that serve the SAME model at two quantisation levels:

    Q4_K  /data/models/qwen36-35b-abliterated-mtp/...Q4_K.gguf   (21.7 GB)
    Q8_0  /data/models/qwen36-35b-abliterated-mtp/...Q8_0.gguf   (37.8 GB)

The question is not "which is smarter" but **does the policy drift enough to
invalidate the value head's 500-step gate**:

  solved        did the generated proof compile (no error, no `sorry`)?
  tokens/steps  usage.completion_tokens and the tactic-line count of the proof
  d_hat         the value head scored on (goal, bank context + the model's own
                proof).  Absolute d_hat is SYNTHETIC when the checkpoint is a
                tiny/hashed run; only the Q4-vs-Q8 DIFFERENCE is meaningful.

Four subcommands (each writes jsonl; nothing here touches the GPU):

    select  bank shard -> missions.jsonl          (real Lean statements)
    run     missions -> one jsonl per endpoint     (needs llama-server)
    score   responses -> solved + d_hat            (lake env lean per mission)
    report  two scored files -> Markdown table

    python3 train/q4_q8_drift.py select --n 16 --tier T1 --tier T2 \\
        --out state/experiments/missions.jsonl
    python3 train/q4_q8_drift.py run --port 8097 --label q4 \\
        --missions state/experiments/missions.jsonl --out state/experiments/q4.jsonl
    python3 train/q4_q8_drift.py score --responses state/experiments/q4.jsonl \\
        --out state/experiments/q4_scored.jsonl
    python3 train/q4_q8_drift.py report --q4 state/experiments/q4_scored.jsonl \\
        --q8 state/experiments/q8_scored.jsonl --out state/experiments/q4_vs_q8_drift.md

The server itself must run under the GPU lease (`flock ... /data/gpu.lock`);
this script only speaks HTTP to 127.0.0.1.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import re
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dataset  # noqa: E402
import head  # noqa: E402

BANK_DEFAULT = Path("./data/bank_local")
LEAN_PROJECT_DEFAULT = Path("/data/projects/lean-4.28-reap/env-0df444a")
RUN_DIR_DEFAULT = HERE.parent / "state" / "experiments"

SYSTEM_PROMPT = (
    "You are a Lean 4 theorem prover working in Mathlib. "
    "Reply with the proof body only: a single `by` block, no prose, no markdown "
    "fences. Do not restate the theorem."
)


# ---------------------------------------------------------------------------
# select: bank -> missions
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


_LONG_NUMBER = re.compile(r"\d{12,}")


def _standalone_plausible(record: dict) -> bool:
    """Cheap pre-filter before spending a Lean compile on the reference proof.

    The bank's numeric-certificate lemmas (`..._cert015`, `..._seg1214`) are
    proved against definitions that live in their own module, so a standalone
    `import Mathlib` file cannot even state them.  They all carry long numeric
    literals; skip those instead of compiling thousands of doomed candidates.
    """
    if _LONG_NUMBER.search(record["statement_formal"]):
        return False
    if re.search(r"_(cert|seg)\d+", record["id"]):
        return False
    return True


def _stratum_of(record: dict) -> tuple[str, int] | None:
    """Difficulty stratum from the reference proof, never from model output."""
    if not _standalone_plausible(record):
        return None
    steps = dataset.split_tactic_steps(record["best_solution"]["proof"])
    n = len(steps)
    stmt_len = len(record["statement_formal"])
    if n == 1 and stmt_len < 220:
        return "easy", n
    if 2 <= n <= 8 and stmt_len < 420:
        return "moderate", n
    return None


def select_missions(bank: Path, tiers: list[str], per_stratum: int, seed: int,
                    project: Path | None = None,
                    workdir: Path | None = None,
                    verify_workers: int = 6,
                    max_attempts_per_stratum: int = 48) -> list[dict]:
    """Stratified sample: `per_stratum` easy + `per_stratum` moderate missions.

    "easy" = a 1-step reference proof and a short statement; "moderate" = a
    2..8-step reference proof.  The split is fixed BEFORE any model runs, so
    the success rate is not cherry-picked.  With `project`, a mission is kept
    only if its OWN reference proof compiles standalone against Mathlib — the
    same judge the model's proof will face.
    """
    rng = random.Random(seed)
    want = {"easy": per_stratum, "moderate": per_stratum}
    picked: list[dict] = []
    counts = {"easy": 0, "moderate": 0}
    attempts = {"easy": 0, "moderate": 0}
    seen: set[str] = set()
    for tier in tiers:
        path = bank / f"{tier}.jsonl"
        if not path.is_file():
            raise SystemExit(f"select: missing bank shard {path}")
        rows = [r for r in read_jsonl(path)
                if r.get("local_compilable") and r.get("statement_formal")
                and (r.get("best_solution") or {}).get("proof")]
        rng.shuffle(rows)
        for r in rows:
            if all(counts[k] >= want[k] or attempts[k] >= max_attempts_per_stratum
                   for k in want):
                break
            if r["id"] in seen:
                continue
            st = _stratum_of(r)
            if st is None:
                continue
            stratum, steps = st
            if counts[stratum] >= want[stratum]:
                continue
            if attempts[stratum] >= max_attempts_per_stratum:
                continue
            attempts[stratum] += 1
            mission = {
                "mission_id": f"drift-{tier}-{stratum[0]}-{r['id'].split('.')[-1]}",
                "tier": tier,
                "stratum": stratum,
                "bank_id": r["id"],
                "theorem_id": r.get("theorem_id"),
                "statement_formal": r["statement_formal"],
                "context_pre": r.get("context_pre") or "",
                "imports": r.get("imports") or ["Mathlib"],
                "reference_steps": steps,
                "reference_proof": r["best_solution"]["proof"],
                "remaining_steps": steps,      # mission start == full proof left
                "statement_nl": (r.get("statement_nl") or "")[:400],
                "url": r.get("url"),
                "label_source": "bank reference proof (tactic-step count)",
            }
            if project is not None:
                wd = (workdir or RUN_DIR_DEFAULT / "lean_select") / "cand"
                name = "ref_" + re.sub(r"[^A-Za-z0-9_]", "_", mission["mission_id"])
                res = compile_one(project, mission, mission["reference_proof"],
                                  name, wd)
                if not res["solved"]:
                    continue
                mission["reference_compiles"] = True
            seen.add(r["id"])
            counts[stratum] += 1
            picked.append(mission)
    picked.sort(key=lambda m: (m["stratum"], m["mission_id"]))
    for stratum in ("easy", "moderate"):
        got = counts[stratum]
        if got < want[stratum]:
            print(f"select: {stratum}: {got}/{want[stratum]} verified "
                  f"({attempts[stratum]} candidates tried)", file=sys.stderr)
    return picked


def build_prompt(mission: dict) -> list[dict]:
    ctx = (mission.get("context_pre") or "").strip()
    stmt = mission["statement_formal"].strip()
    user = (f"{ctx}\n\n{stmt}\n\n"
            "Prove the theorem above. Reply with the proof body only, "
            "starting with `by`.").strip()
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# run: one endpoint
# ---------------------------------------------------------------------------

def post_chat(port: int, messages: list[dict], max_tokens: int,
              temperature: float = 0.0, timeout: float = 600.0) -> dict:
    body = json.dumps({
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": 1234,
        "stream": False,
        # Qwen3.5/3.6 default to a thinking block; with a 256-token budget the
        # answer never appears (content == "").  Thinking off makes the budget
        # the answer budget, which is what the drift measurement needs.
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_health(port: int, timeout: float = 900.0) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return round(time.time() - t0, 1)
        except Exception:
            time.sleep(2.0)
    raise SystemExit(f"run: endpoint :{port} did not become healthy in {timeout}s")


def run_endpoint(missions: list[dict], port: int, label: str,
                 max_tokens: int, out: Path) -> list[dict]:
    wait_health(port)
    rows = []
    for i, m in enumerate(missions, 1):
        t0 = time.time()
        try:
            resp = post_chat(port, build_prompt(m), max_tokens)
            choice = (resp.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            text = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            usage = resp.get("usage") or {}
            timings = resp.get("timings") or {}
            err = None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            text, reasoning, usage, timings, err = "", "", {}, {}, f"{type(exc).__name__}: {exc}"
        row = {
            **{k: m[k] for k in ("mission_id", "tier", "stratum", "statement_formal",
                                 "context_pre", "reference_steps")},
            "endpoint": label,
            "port": port,
            "response": text,
            "reasoning_chars": len(reasoning),
            "reasoning_head": reasoning[:400],
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "wall_seconds": round(time.time() - t0, 2),
            "decode_tps": round(float(timings.get("predicted_per_second") or 0.0), 2),
            "error": err,
        }
        rows.append(row)
        print(f"[{label}] {i}/{len(missions)} {m['mission_id']} "
              f"tokens={row['completion_tokens']} {row['wall_seconds']}s", flush=True)
    write_jsonl(out, rows)
    return rows


# ---------------------------------------------------------------------------
# score: proof extraction + Lean compile + d_hat
# ---------------------------------------------------------------------------

def _extract_proof(text: str) -> str:
    t = (text or "").strip()
    fence = re.search(r"```(?:lean4?|)\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    m = re.search(r":=\s*(by\b.*)$", t, re.S)
    if m:
        t = m.group(1).strip()
    # prose before the proof: start at the first line that is a `by` block
    m = re.search(r"^[ \t]*by\b", t, re.M)
    if m:
        t = t[m.start():].lstrip()
    if not t.startswith("by"):
        # a bare tactic list is a valid `by` body; keep it indented so a
        # multi-line answer stays one block.
        t = "by\n" + textwrap.indent(t, "  ") if t else "by"
    return t.strip()


def rename_statement(statement: str, name: str) -> str:
    return re.sub(r"^\s*(theorem|lemma)\s+\S+", f"theorem {name}", statement.strip(),
                  count=1)


def count_steps(proof: str) -> int:
    lines = [ln.strip() for ln in proof.splitlines()]
    if lines and lines[0].startswith("by"):
        lines = lines[1:]
    return sum(1 for ln in lines if ln and not ln.startswith("--"))


def lean_file(mission: dict, proof: str, name: str) -> str:
    ctx = (mission.get("context_pre") or "").strip()
    stmt = rename_statement(mission["statement_formal"], name)
    return f"import Mathlib\n\n{ctx}\n\n{stmt} := {proof}\n"


def compile_one(project: Path, mission: dict, proof: str, name: str,
                workdir: Path, timeout: float = 300.0) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / f"{name}.lean"
    path.write_text(lean_file(mission, proof, name), encoding="utf-8")
    t0 = time.time()
    proc = subprocess.run(["lake", "env", "lean", str(path)], cwd=str(project),
                          capture_output=True, text=True, timeout=timeout)
    out = (proc.stdout or "") + (proc.stderr or "")
    has_sorry = bool(re.search(r"declaration uses ['`]sorry", out)) or \
        bool(re.search(r"\bsorry\b", proof))
    solved = proc.returncode == 0 and not has_sorry
    return {
        "solved": solved,
        "compile_ok": proc.returncode == 0,
        "used_sorry": has_sorry,
        "compile_seconds": round(time.time() - t0, 2),
        "error": "" if solved else out.strip()[:1200],
        "lean_file": str(path),
    }


def dhat_for(model, ns, mission: dict, proof: str, use_proof: bool) -> dict:
    ctx = (mission.get("context_pre") or "").strip()
    if use_proof:
        ctx = (ctx + "\n" + proof).strip()
    rows = ns["predict_texts"](model, [f"{mission['statement_formal']}\n{ctx}"],
                               goals=[mission["statement_formal"]], contexts=[ctx])
    r = rows[0]
    gate = head.gate(r["estimate"], ci_high=r["ci_high"],
                     tail_saturated=r["tail_saturated"])
    return {"estimate": round(r["estimate"], 2), "ci_low": round(r["ci_low"], 2),
            "ci_high": round(r["ci_high"], 2), "gate": gate["action"],
            "gate_reason": gate["reason"]}


def score_endpoint(responses: list[dict], project: Path, workdir: Path,
                   head_run: str | None, workers: int = 4) -> list[dict]:
    ns = None
    model = None
    if head_run and head.torch_available():
        ns = head.ensure()
        name = ("value-head-stage2.pt"
                if (Path(head_run) / "value-head-stage2.pt").is_file()
                else "value-head.pt")
        model = ns["load_checkpoint"](head_run, name=name)

    def one(row: dict) -> dict:
        proof = _extract_proof(row["response"])
        name = "drift_" + re.sub(r"[^A-Za-z0-9_]", "_", row["mission_id"])
        res = compile_one(project, row, proof, name, workdir)
        out = {**row, "proof": proof, "steps": count_steps(proof), **res}
        if model is not None:
            out["dhat_start"] = dhat_for(model, ns, row, proof, use_proof=False)
            out["dhat_after"] = dhat_for(model, ns, row, proof, use_proof=True)
        return out

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, responses))
    return rows


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def report(q4: list[dict], q8: list[dict], out: Path, meta: dict) -> str:
    by4 = {r["mission_id"]: r for r in q4}
    by8 = {r["mission_id"]: r for r in q8}
    ids = [r["mission_id"] for r in q4 if r["mission_id"] in by8]
    n = len(ids)

    lines = []
    lines.append("# Q4_K vs Q8_0 policy drift (value-head gate)\n")
    lines.append(f"Model: `{meta.get('model', 'Qwen3.6-35B-A3B abliterated MTP')}`, "
                 f"same prompt, `temperature=0`, `max_tokens={meta.get('max_tokens', 256)}`, "
                 f"`seed=1234`.\n")
    lines.append(f"Missions: **{n}** real bank statements "
                 f"({', '.join(sorted({r['tier'] for r in q4 if r['mission_id'] in by8}))}), "
                 "label `remaining_steps` = tactic steps of the bank reference proof.\n")
    lines.append("Judge: one `lake env lean` per mission on the model's own proof; "
                 "`solved` = exit 0 and no `sorry`.\n")

    if meta.get("head_run"):
        lines.append(f"d_hat: `{meta['head_run']}` "
                     f"(`{meta.get('head_kind', 'stage-2 checkpoint, hashed backbone, '
                                              'trained on 50 synthetic records')}`). "
                     "**Absolute d_hat is SYNTHETIC** (the checkpoint is a tiny/hashed "
                     "training run, not a head trained on this endpoint); only the "
                     "Q4-vs-Q8 difference is interpretable. States scored: "
                     "`(goal, bank context + that endpoint's own generated proof)`.\n")

    lines.append("\n## Per mission\n")
    lines.append("| mission | tier | ref steps | Q4 solved | Q4 tok | Q4 steps | "
                 "Q8 solved | Q8 tok | Q8 steps | d_hat Q4 | d_hat Q8 | gate Q4/Q8 |")
    lines.append("| --- | --- | ---: | :---: | ---: | ---: | :---: | ---: | ---: | "
                 "---: | ---: | :---: |")
    for mid in ids:
        a, b = by4[mid], by8[mid]
        da = a.get("dhat_after", {}).get("estimate")
        db = b.get("dhat_after", {}).get("estimate")
        ga = a.get("dhat_after", {}).get("gate", "-")
        gb = b.get("dhat_after", {}).get("gate", "-")
        mark = lambda r: "yes" if r["solved"] else "no"  # noqa: E731
        lines.append(
            f"| `{mid}` | {a['tier']} | {a['reference_steps']} | {mark(a)} | "
            f"{a['completion_tokens']} | {a['steps']} | {mark(b)} | "
            f"{b['completion_tokens']} | {b['steps']} | "
            f"{'-' if da is None else f'{da:.1f}'} | "
            f"{'-' if db is None else f'{db:.1f}'} | {ga}/{gb} |")

    s4 = sum(1 for i in ids if by4[i]["solved"])
    s8 = sum(1 for i in ids if by8[i]["solved"])
    tok4 = [by4[i]["completion_tokens"] for i in ids]
    tok8 = [by8[i]["completion_tokens"] for i in ids]
    st4 = [by4[i]["steps"] for i in ids]
    st8 = [by8[i]["steps"] for i in ids]
    dtok = [abs(a - b) for a, b in zip(tok4, tok8)]
    dst = [abs(a - b) for a, b in zip(st4, st8)]
    agree = sum(1 for i in ids if by4[i]["solved"] == by8[i]["solved"])

    lines.append("\n## Aggregate\n")
    lines.append("| metric | Q4_K | Q8_0 |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| solved | {s4}/{n} ({s4 / max(1, n):.0%}) | {s8}/{n} ({s8 / max(1, n):.0%}) |")
    lines.append(f"| mean generated tokens | {_mean(tok4):.0f} | {_mean(tok8):.0f} |")
    lines.append(f"| median generated tokens | {_median(tok4):.0f} | {_median(tok8):.0f} |")
    lines.append(f"| mean tactic steps | {_mean(st4):.1f} | {_mean(st8):.1f} |")
    lines.append(f"| median tactic steps | {_median(st4):.0f} | {_median(st8):.0f} |")
    if meta.get("decode_tps"):
        lines.append(f"| measured decode t/s | {meta['decode_tps'].get('q4', '-')} | "
                     f"{meta['decode_tps'].get('q8', '-')} |")
    lines.append("")
    lines.append(f"- solved/not agreement: **{agree}/{n}**")
    lines.append(f"- token drift per mission: mean |Δ| = {_mean(dtok):.0f}, "
                 f"median |Δ| = {_median(dtok):.0f}")
    lines.append(f"- step drift per mission: mean |Δ| = {_mean(dst):.1f}, "
                 f"median |Δ| = {_median(dst):.0f}")

    if meta.get("head_run"):
        d4 = [by4[i]["dhat_after"]["estimate"] for i in ids if "dhat_after" in by4[i]]
        d8 = [by8[i]["dhat_after"]["estimate"] for i in ids if "dhat_after" in by8[i]]
        if d4 and d8:
            dlog = [abs(math.log1p(a) - math.log1p(b)) for a, b in zip(d4, d8)]
            flips = sum(1 for i in ids
                        if by4[i].get("dhat_after", {}).get("gate")
                        != by8[i].get("dhat_after", {}).get("gate"))
            lines.append(f"- d_hat drift (SYNTHETIC head): median |Δlog1p| = "
                         f"{_median(dlog):.3f}, max = {max(dlog):.3f}, "
                         f"gate flips = {flips}/{n}")

    def failure_class(row: dict) -> str:
        if row["solved"]:
            return "solved"
        err = row.get("error") or ""
        if row.get("completion_tokens", 0) >= meta.get("max_tokens", 256):
            return "truncated at max_tokens"
        if "No goals to be solved" in err:
            return "closed then extra tactic"
        if "unknown identifier" in err or "Unknown constant" in err:
            return "unknown identifier"
        if "timeout at" in err:
            return "heartbeat timeout"
        if "unsolved goals" in err:
            return "unsolved goals"
        return "other"

    lines.append("\n### Failure classes\n")
    lines.append("| class | Q4 | Q8 |")
    lines.append("| --- | ---: | ---: |")
    classes = ["solved", "closed then extra tactic", "truncated at max_tokens",
               "unknown identifier", "unsolved goals", "heartbeat timeout", "other"]
    for cls in classes:
        c4 = sum(1 for i in ids if failure_class(by4[i]) == cls)
        c8 = sum(1 for i in ids if failure_class(by8[i]) == cls)
        if c4 or c8:
            lines.append(f"| {cls} | {c4} | {c8} |")

    strata = sorted({by4[i].get("stratum", "-") for i in ids})
    if len(strata) > 1 or strata != ["-"]:
        lines.append("\n### By stratum\n")
        lines.append("| stratum | n | Q4 solved | Q8 solved | Q4 tok | Q8 tok | "
                     "mean abs Δtok | mean abs Δsteps |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for st in strata:
            sub = [i for i in ids if by4[i].get("stratum", "-") == st]
            a4 = sum(1 for i in sub if by4[i]["solved"])
            a8 = sum(1 for i in sub if by8[i]["solved"])
            t4 = [by4[i]["completion_tokens"] for i in sub]
            t8 = [by8[i]["completion_tokens"] for i in sub]
            d = [abs(x - y) for x, y in zip(t4, t8)]
            ds = [abs(by4[i]["steps"] - by8[i]["steps"]) for i in sub]
            lines.append(f"| {st} | {len(sub)} | {a4}/{len(sub)} | {a8}/{len(sub)} | "
                         f"{_mean(t4):.0f} | {_mean(t8):.0f} | {_mean(d):.0f} | "
                         f"{_mean(ds):.1f} |")

    lines.append("\n## Reproduce\n")
    lines.append("Server flags are identical for both endpoints; only the GGUF and "
                 "`--n-cpu-moe` differ, and only one server runs at a time, under "
                 "the GPU lease:\n")
    lines.append("```text")
    lines.append("flock -x -w 3600 /data/gpu.lock bash /tmp/vht-exp/one_endpoint.sh q4")
    lines.append("llama-server -m ...Q4_K.gguf -c 262144 -np 1 -ngl 99 -fa on \\")
    lines.append("  -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 --n-cpu-moe 14 \\")
    lines.append("  --load-mode none --no-mmproj --jinja --spec-type draft-mtp \\")
    lines.append("  --spec-draft-n-max 2 --host 127.0.0.1 --port 8097")
    lines.append("")
    lines.append("# Q8_0: same, --n-cpu-moe 27 --port 8098")
    lines.append("```\n")
    lines.append("Requests: `temperature=0`, `seed=1234`, `max_tokens=256`, "
                 "`chat_template_kwargs={\"enable_thinking\": false}`. Thinking is "
                 "off on purpose: with it on, all 12 Q4 answers were empty at 256 "
                 "tokens (the budget went to the reasoning block).\n")

    lines.append("\n## Recommendation\n")
    rec = meta.get("recommendation")
    if not rec:
        drift_ratio = _mean(dtok) / max(1.0, _mean(tok4))
        flips = 0
        if meta.get("head_run"):
            flips = sum(1 for i in ids
                        if by4[i].get("dhat_after", {}).get("gate")
                        != by8[i].get("dhat_after", {}).get("gate"))
        solved_delta = s4 - s8
        if flips == 0 and abs(solved_delta) <= 1 and drift_ratio <= 0.25:
            rec = (f"**Q4 is safe as the primary policy.** Solved set agrees on "
                   f"{agree}/{n} missions, mean token drift is {drift_ratio:.0%} "
                   f"of the Q4 answer length, and the gate decision does not flip. "
                   f"Keep the 500-step threshold; keep Q8 as the audit endpoint for "
                   f"checkpoint promotion and for any mission the gate calls close "
                   f"to the boundary.")
        elif flips > 0 or solved_delta < -1:
            rec = (f"**Q4 is not safe as-is.** Solved delta Q4-Q8 = {solved_delta:+d} "
                   f"over {n} missions, gate flips = {flips}, mean token drift = "
                   f"{drift_ratio:.0%}. Either run Q8 for gate-critical decisions or "
                   f"recalibrate the threshold on Q4-generated states (re-fit the "
                   f"head on Q4 prefixes, then re-run this experiment).")
        else:
            rec = (f"**Conditional.** Solved delta Q4-Q8 = {solved_delta:+d}, gate "
                   f"flips = {flips}, mean token drift = {drift_ratio:.0%}. Q4 is "
                   f"usable for rollouts, but the gate should be validated on Q4 "
                   f"prefixes before promotion; do not lower the 500-step threshold.")
    if rec:
        lines.append(rec.strip() + "\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("select")
    p.add_argument("--bank", default=str(BANK_DEFAULT))
    p.add_argument("--tier", action="append", default=None)
    p.add_argument("--per-stratum", type=int, default=8,
                   help="missions per difficulty stratum (easy / moderate)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--project", default=None,
                   help="Lean project used to verify the reference proof "
                        f"(default {LEAN_PROJECT_DEFAULT}); omit to skip verification")
    p.add_argument("--workdir", default=str(RUN_DIR_DEFAULT / "lean_select"))
    p.add_argument("--out", default=str(RUN_DIR_DEFAULT / "missions.jsonl"))

    p = sub.add_parser("run")
    p.add_argument("--missions", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--out", required=True)

    p = sub.add_parser("score")
    p.add_argument("--responses", required=True)
    p.add_argument("--project", default=str(LEAN_PROJECT_DEFAULT))
    p.add_argument("--workdir", default=str(RUN_DIR_DEFAULT / "lean"))
    p.add_argument("--head-run", default=None,
                   help="value-head run dir; d_hat is SYNTHETIC unless this head "
                        "was trained on the endpoint being measured")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)

    p = sub.add_parser("report")
    p.add_argument("--q4", required=True)
    p.add_argument("--q8", required=True)
    p.add_argument("--head-run", default=None)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--decode-tps-q4", type=float, default=None)
    p.add_argument("--decode-tps-q8", type=float, default=None)
    p.add_argument("--recommendation", default=None)
    p.add_argument("--out", default=str(RUN_DIR_DEFAULT / "q4_vs_q8_drift.md"))

    args = ap.parse_args(argv)
    if args.cmd == "select":
        tiers = args.tier or ["T1", "T2"]
        rows = select_missions(Path(args.bank), tiers, args.per_stratum, args.seed,
                               project=Path(args.project) if args.project else None,
                               workdir=Path(args.workdir))
        write_jsonl(Path(args.out), rows)
        print(json.dumps({"missions": len(rows), "tiers": tiers,
                          "strata": {k: sum(1 for r in rows if r["stratum"] == k)
                                     for k in ("easy", "moderate")},
                          "out": args.out,
                          "steps": [r["reference_steps"] for r in rows]}))
        return 0
    if args.cmd == "run":
        rows = run_endpoint(read_jsonl(Path(args.missions)), args.port, args.label,
                            args.max_tokens, Path(args.out))
        print(json.dumps({"endpoint": args.label, "rows": len(rows), "out": args.out}))
        return 0
    if args.cmd == "score":
        head_run = args.head_run
        rows = score_endpoint(read_jsonl(Path(args.responses)), Path(args.project),
                              Path(args.workdir), head_run, args.workers)
        write_jsonl(Path(args.out), rows)
        print(json.dumps({"scored": len(rows), "solved": sum(1 for r in rows if r["solved"]),
                          "head_run": head_run, "out": args.out}))
        return 0
    q4 = read_jsonl(Path(args.q4))
    q8 = read_jsonl(Path(args.q8))
    tps = {}
    for label, rows in (("q4", q4), ("q8", q8)):
        vals = [r["decode_tps"] for r in rows if r.get("decode_tps")]
        if vals:
            tps[label] = round(_median(vals), 1)
    meta = {"max_tokens": args.max_tokens, "head_run": args.head_run,
            "decode_tps": tps, "recommendation": args.recommendation}
    text = report(q4, q8, Path(args.out), meta)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
