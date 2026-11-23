import { Plugin } from "@opencode/plugin"
import { spawnSync } from "node:child_process"
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { dirname, isAbsolute, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"

import { DEFAULT_THRESHOLD, decideGate } from "./lib/gate.mjs"

/**
 * value-head-training (staged)
 *
 * Drives the two-stage value-head pipeline that lives in `train/`:
 *
 *   stage 1  SFT on prove2me + Mathlib: init the head while the policy moves
 *   stage 2  MCTS + CE: distil the search posterior into policy + value
 *   gate     estimate > 500 steps => decompose into missions instead of proving
 *
 * The plugin never trains inline: `vht_plan` / `vht_stage2` materialise a run
 * dir and hand back the exact command, and `execute: true` runs it via
 * execFileSync with an argv array (no shell).  Every tool returns a compact
 * JSON string.  No network calls, no opencode.json edits.
 */

type Options = {
  root?: string
  runsDir?: string
  python?: string
  timeoutMs?: number
}

// Resolved from the plugin file itself, so a copied install (plugins/…)
// trains against its own train/ directory instead of the staged source tree.
const DEFAULT_ROOT = dirname(fileURLToPath(import.meta.url))
const ACTIVE_LIMIT = 6000

function json(value: unknown): { content: string } {
  return { content: JSON.stringify(value) }
}

function fail(error: unknown, code = "ERROR"): { content: string } {
  const message = error instanceof Error ? error.message : String(error)
  return json({ ok: false, code, error: message })
}

function str(value: unknown): string {
  return typeof value === "string" ? value : ""
}

function num(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined
}

function tail(text: string, n = 800): string {
  const trimmed = text.trim()
  return trimmed.length <= n ? trimmed : `…${trimmed.slice(trimmed.length - n)}`
}

function slug(text: string): string {
  const base = text
    .replace(/[^A-Za-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40)
  return base || "model"
}

function timestamp(): string {
  return new Date().toISOString().replace(/[:.]/g, "").replace("T", "-").slice(0, 15)
}

export default Plugin.define({
  id: "value-head-training",
  async setup(ctx) {
    const options = (ctx.options ?? {}) as Options
    const root = resolve(options.root ?? process.env.VHT_ROOT ?? DEFAULT_ROOT)
    const runsDir = resolve(options.runsDir ?? join(root, "state", "runs"))
    const python = options.python ?? process.env.VHT_PYTHON ?? "python3"
    const timeoutMs = Math.max(5_000, Math.min(Number(options.timeoutMs ?? 600_000), 3_600_000))
    const trainDir = join(root, "train")
    const gateStatePath = join(root, "state", "gate-state.json")
    const active = new Map<string, { run: string; estimate: number; action: string }>()

    const resolveRun = (value: string): string => (isAbsolute(value) ? value : join(runsDir, value))

    const runPython = (script: string, args: string[], timeout = 120_000) => {
      const file = join(trainDir, script)
      if (!existsSync(file)) throw new Error(`missing script: ${file}`)
      const proc = spawnSync(python, [file, ...args], {
        encoding: "utf8",
        timeout,
        maxBuffer: 4 * 1024 * 1024,
        env: { ...process.env, PYTHONPATH: `${trainDir}:${process.env.PYTHONPATH ?? ""}` },
      })
      const stdout = String(proc.stdout ?? "")
      const stderr = String(proc.stderr ?? "")
      if (proc.error) throw new Error(`spawn ${python}: ${proc.error.message}`)
      if (proc.status !== 0) {
        return { ok: false as const, status: proc.status ?? -1, stdout, stderr }
      }
      return { ok: true as const, status: 0, stdout, stderr }
    }

    const parseLastJson = (stdout: string): unknown => {
      const lines = stdout.trim().split("\n").filter((l) => l.trim().startsWith("{"))
      if (!lines.length) throw new Error("script produced no JSON line")
      return JSON.parse(lines[lines.length - 1]!)
    }

    const readGateState = (): Record<string, string> => {
      try {
        return JSON.parse(readFileSync(gateStatePath, "utf8")) as Record<string, string>
      } catch {
        return {}
      }
    }

    const writeGateState = (state: Record<string, string>) => {
      mkdirSync(dirname(gateStatePath), { recursive: true })
      const keys = Object.keys(state).slice(-500)
      const trimmed: Record<string, string> = {}
      for (const k of keys) trimmed[k] = state[k]!
      writeFileSync(gateStatePath, JSON.stringify(trimmed), "utf8")
    }

    const readJson = (path: string): any => {
      try {
        return JSON.parse(readFileSync(path, "utf8"))
      } catch {
        return null
      }
    }

    // -------------------------------------------------------------------
    // Tools
    // -------------------------------------------------------------------

    await ctx.tool.transform((editor) => {
      editor.namespace({
        name: "vht",
        description:
          "value-head-training: plan/inspect the two-stage value head (SFT init, then MCTS+CE), " +
          "score a (goal, context), and apply the >500-step decomposition gate.",
      })

      editor.add({
        name: "plan",
        description:
          "Write a stage-1 run dir (config.json + optional dataset) for a base model and scheme, and return the " +
          "exact command to launch stage 1. Schemes: lora (adapter only) or full (full-parameter). --tiny uses a " +
          "hashed-text backbone so the plan is runnable on CPU without any HF weights.",
        input: {
          type: "object",
          properties: {
            base: { type: "string", description: 'HF model dir, or "hashed"/"tiny" for the CPU stand-in.' },
            scheme: { type: "string", enum: ["lora", "full"] },
            data: { type: "string", description: "Existing dataset jsonl; built into the run dir when omitted." },
            out: { type: "string", description: "Run dir (relative names land under state/runs)." },
            encoding: { type: "string", enum: ["uniform", "log", "regress", "dual"], description: "Value encoding (default dual)." },
            buckets: { type: "integer", description: "Bucket count for the bucket encodings (default 64)." },
            tiny: { type: "boolean", description: "Plan the CPU skeleton (50 synthetic records)." },
            prepare: { type: "boolean", description: "Also build the dataset now (default true when data is omitted and tiny)." },
          },
          required: ["base", "scheme"],
          additionalProperties: false,
        },
        options: { namespace: "vht", permission: "vht_plan", codemode: false },
        execute: async (input, context) => {
          try {
            const args = (input ?? {}) as Record<string, unknown>
            const scheme = str(args.scheme) === "full" ? "full" : "lora"
            const base = str(args.base) || "hashed"
            const tiny = args.tiny === true || base === "hashed" || base === "tiny"
            const encoding = ["uniform", "log", "regress", "dual"].includes(str(args.encoding))
              ? str(args.encoding)
              : "dual"
            const buckets = Math.max(4, Math.min(num(args.buckets) ?? 64, 4096))
            const runName = str(args.out) || `${timestamp()}-${slug(base)}-${scheme}`
            const run = resolveRun(runName)
            mkdirSync(run, { recursive: true })

            const dataPath = str(args.data) || join(run, "data.jsonl")
            const prepare = args.prepare === true || (args.prepare === undefined && !str(args.data))
            let dataSummary: unknown = null
            if (prepare && !existsSync(dataPath)) {
              const datasetArgs = tiny
                ? ["--tiny", "--out", dataPath]
                : ["--out", dataPath]
              const res = runPython("dataset.py", datasetArgs, 300_000)
              if (!res.ok) {
                return json({
                  ok: false,
                  code: "DATASET_FAILED",
                  error: tail(res.stderr || res.stdout),
                  run,
                })
              }
              dataSummary = parseLastJson(res.stdout)
            } else if (existsSync(dataPath)) {
              dataSummary = { path: dataPath, exists: true }
            }

            const config = {
              stage: 1,
              base,
              scheme,
              encoding,
              buckets,
              tiny,
              data: dataPath,
              out: run,
              created_at: new Date().toISOString(),
            }
            writeFileSync(join(run, "plan.json"), JSON.stringify(config, null, 2), "utf8")

            const argv = [
              join(trainDir, "train_stage1.py"),
              "--scheme", scheme,
              "--base", base,
              "--out", run,
              "--encoding", encoding,
              "--buckets", String(buckets),
              ...(tiny ? ["--tiny"] : []),
              ...(existsSync(dataPath) ? ["--data", dataPath] : []),
            ]
            const command = [python, ...argv].map((a) => (a.includes(" ") ? JSON.stringify(a) : a)).join(" ")
            active.set(context.sessionID, { run, estimate: 0, action: "plan" })
            return json({
              ok: true,
              run,
              scheme,
              encoding,
              tiny,
              config_path: join(run, "plan.json"),
              data: dataSummary,
              command,
              next: "run the command, then vht_status({run})",
            })
          } catch (error) {
            return fail(error)
          }
        },
      })

      editor.add({
        name: "predict",
        description:
          "Score a (goal, context) with a trained run dir: value estimate (steps remaining), 80% interval and the " +
          "gate decision. Returns compact JSON.",
        input: {
          type: "object",
          properties: {
            goal: { type: "string" },
            context: { type: "string", description: "Tactic state / proof prefix / preamble (may be empty)." },
            model: { type: "string", description: "Run dir or run name under state/runs." },
            threshold: { type: "number", description: "Gate threshold in steps (default 500)." },
            prev_action: { type: "string", enum: ["prove", "decompose"], description: "Gate state for hysteresis." },
            checkpoint: { type: "string", description: "Explicit checkpoint filename inside the run dir." },
          },
          required: ["goal", "model"],
          additionalProperties: false,
        },
        options: { namespace: "vht", permission: "vht_predict", codemode: false },
        execute: async (input, context) => {
          try {
            const args = (input ?? {}) as Record<string, unknown>
            const run = resolveRun(str(args.model))
            if (!existsSync(join(run, "config.json")) && !existsSync(join(run, "plan.json"))) {
              return json({ ok: false, code: "NO_MODEL", error: `no run dir at ${run}`, run })
            }
            const threshold = num(args.threshold) ?? DEFAULT_THRESHOLD
            const res = runPython("predict.py", [
              "--run", run,
              "--goal", str(args.goal),
              "--context", str(args.context),
              "--threshold", String(threshold),
              "--prev-action", str(args.prev_action) === "decompose" ? "decompose" : "prove",
              ...(str(args.checkpoint) ? ["--checkpoint", str(args.checkpoint)] : []),
            ], 120_000)
            if (!res.ok) {
              return json({
                ok: false,
                code: res.stderr.includes("PyTorch is not importable") ? "NO_TORCH" : "PREDICT_FAILED",
                error: tail(res.stderr || res.stdout),
                run,
              })
            }
            const parsed = parseLastJson(res.stdout) as Record<string, unknown>
            active.set(context.sessionID, {
              run,
              estimate: Number(parsed.estimate ?? 0),
              action: String((parsed.gate as any)?.action ?? "prove"),
            })
            return json({ run, ...parsed })
          } catch (error) {
            return fail(error, "PREDICT_FAILED")
          }
        },
      })

      editor.add({
        name: "gate",
        description:
          "Pure gate logic (no model needed): estimate > threshold => decompose, with hysteresis at 0.8*threshold and a " +
          "tail-saturation override. Pass `key` to persist the decision so the next call keeps its hysteresis.",
        input: {
          type: "object",
          properties: {
            estimate: { type: "number", description: "Value estimate in remaining tactic steps." },
            threshold: { type: "number", description: "Default 500." },
            prev: { type: "string", enum: ["prove", "decompose"], description: "Previous decision (default from key state)." },
            key: { type: "string", description: "Stable problem id; stores the decision for hysteresis." },
            ci_high: { type: "number", description: "Upper end of the estimate interval, if known." },
            tail_saturated: { type: "boolean", description: "Head ran out of grid => unknown tail => decompose." },
          },
          required: ["estimate"],
          additionalProperties: false,
        },
        options: { namespace: "vht", permission: "vht_gate", codemode: false },
        execute: async (input) => {
          try {
            const args = (input ?? {}) as Record<string, unknown>
            const estimate = Number(args.estimate)
            if (!Number.isFinite(estimate)) return json({ ok: false, code: "BAD_ESTIMATE", error: "estimate must be a number" })
            const key = str(args.key) || undefined
            const state = readGateState()
            const prev = str(args.prev) === "prove" || str(args.prev) === "decompose"
              ? (str(args.prev) as "prove" | "decompose")
              : ((key ? state[key] : undefined) === "decompose" ? "decompose" : "prove")
            const result = decideGate(estimate, {
              threshold: num(args.threshold) ?? DEFAULT_THRESHOLD,
              prev,
              ciHigh: num(args.ci_high) ?? null,
              tailSaturated: args.tail_saturated === true,
            })
            if (key) {
              state[key] = result.action
              writeGateState(state)
            }
            return json({ ok: true, key: key ?? null, ...result })
          } catch (error) {
            return fail(error, "GATE_FAILED")
          }
        },
      })

      editor.add({
        name: "stage2",
        description:
          "Stage 2 (MCTS + CE): consume searched trajectories ({goal, context, steps_remaining, visits:[{tactic,visits}]}) " +
          "and distil them into the policy + value head. Returns the command, or runs it when execute=true.",
        input: {
          type: "object",
          properties: {
            run: { type: "string", description: "Stage-1 run dir (or name under state/runs)." },
            trajectories: { type: "string", description: "jsonl of searched nodes; omit with tiny=true." },
            tiny: { type: "boolean", description: "40 synthetic trajectories, CPU only." },
            execute: { type: "boolean", description: "Actually run it (default false: return the command)." },
            steps: { type: "integer", description: "CE steps (default 200 tiny / 60 real)." },
          },
          required: ["run"],
          additionalProperties: false,
        },
        options: { namespace: "vht", permission: "vht_stage2", codemode: false },
        execute: async (input) => {
          try {
            const args = (input ?? {}) as Record<string, unknown>
            const run = resolveRun(str(args.run))
            if (!existsSync(join(run, "value-head.pt"))) {
              return json({ ok: false, code: "NO_MODEL", error: `no stage-1 checkpoint under ${run}`, run })
            }
            const tiny = args.tiny === true
            const trajectories = str(args.trajectories)
            if (!tiny && !trajectories) {
              return json({ ok: false, code: "NO_TRAJECTORIES", error: "pass trajectories or tiny=true", run })
            }
            const argv = [
              join(trainDir, "train_stage2.py"),
              "--run", run,
              ...(tiny ? ["--tiny"] : ["--trajectories", trajectories]),
              ...(num(args.steps) ? ["--steps", String(num(args.steps))] : []),
            ]
            const command = [python, ...argv].map((a) => (a.includes(" ") ? JSON.stringify(a) : a)).join(" ")
            if (args.execute !== true) {
              return json({ ok: true, run, executed: false, command, next: "vht_status({run}) after it finishes" })
            }
            const res = runPython("train_stage2.py", argv.slice(1), timeoutMs)
            if (!res.ok) {
              return json({
                ok: false,
                code: res.stderr.includes("PyTorch is not importable") ? "NO_TORCH" : "STAGE2_FAILED",
                error: tail(res.stderr || res.stdout, 1500),
                run,
                command,
              })
            }
            const metrics = readJson(join(run, "metrics_stage2.json"))
            return json({
              ok: true,
              run,
              executed: true,
              delta_mae_log: metrics?.delta_mae_log ?? null,
              before: metrics?.before?.mae_log ?? null,
              after: metrics?.after?.mae_log ?? null,
              log: tail(res.stdout, 600),
            })
          } catch (error) {
            return fail(error, "STAGE2_FAILED")
          }
        },
      })

      editor.add({
        name: "status",
        description:
          "Read a run dir: plan, stage-1 metrics (value MAE in log space, tail recall >10k, calibration, policy pass@1) " +
          "and stage-2 deltas. Compact JSON.",
        input: {
          type: "object",
          properties: { run: { type: "string", description: "Run dir or name under state/runs." } },
          required: ["run"],
          additionalProperties: false,
        },
        options: { namespace: "vht", permission: "vht_status", codemode: false },
        execute: async (input) => {
          try {
            const run = resolveRun(str((input as Record<string, unknown>)?.run))
            if (!existsSync(run)) return json({ ok: false, code: "NO_RUN", error: `no run dir at ${run}`, run })
            const plan = readJson(join(run, "plan.json"))
            const config = readJson(join(run, "config.json"))
            const m1 = readJson(join(run, "metrics.json"))
            const m2 = readJson(join(run, "metrics_stage2.json"))
            const test = m1?.test ?? {}
            return json({
              ok: true,
              run,
              has: {
                plan: !!plan,
                config: !!config,
                checkpoint: existsSync(join(run, "value-head.pt")),
                stage2: existsSync(join(run, "value-head-stage2.pt")),
              },
              stage1: m1
                ? {
                    scheme: m1.scheme,
                    encoding: m1.encoding,
                    base: m1.base,
                    steps: m1.steps,
                    mae_log: test.mae_log ?? null,
                    mae_steps: test.mae_steps ?? null,
                    baseline_mae_log: test.baseline_mae_log ?? null,
                    bucket_acc: test.bucket_acc ?? null,
                    policy_pass_at_1: test.policy_pass_at_1 ?? null,
                    tail: test.tail ?? null,
                    coverage_80: test.interval?.coverage_80 ?? null,
                    monotone: test.calibration?.monotone ?? null,
                    params: m1.params ?? null,
                  }
                : null,
              stage2: m2
                ? {
                    trajectories: m2.trajectories,
                    delta_mae_log: m2.delta_mae_log,
                    before: m2.before?.mae_log ?? null,
                    after: m2.after?.mae_log ?? null,
                    policy_agree: m2.after?.policy_agree_with_search ?? null,
                    gate_decompose_rate: m2.after?.gate?.decompose_rate ?? null,
                  }
                : null,
            })
          } catch (error) {
            return fail(error, "STATUS_FAILED")
          }
        },
      })
    })

    // -------------------------------------------------------------------
    // Active-run hint
    // -------------------------------------------------------------------

    await ctx.session.hook("context", (event) => {
      if (!event.tools) return
      if (["compaction", "title", "summary"].includes(event.agent)) return
      if (!("vht_predict" in event.tools)) return
      const current = active.get(event.sessionID)
      if (!current) return
      const text = [
        "## value-head-training active (vht)",
        `run: ${current.run} · last estimate: ${Math.round(current.estimate)} steps · gate: ${current.action}`,
        "Threshold is 500 remaining tactic steps: at or above it, decompose the goal into missions instead of attempting the full proof; leave the decompose state only below 400 (hysteresis). A tail-saturated estimate always decomposes.",
        "Stage 1 is SFT (value init + policy), stage 2 is MCTS + CE over searched trajectories. Never claim a solved proof from the value head: it only decides prove vs decompose.",
      ].join("\n")
      event.system.push({ type: "text", text: text.slice(0, ACTIVE_LIMIT) })
    })

    console.log(`[value-head-training] loaded · root=${root} · runs=${runsDir} · python=${python}`)
  },
})
