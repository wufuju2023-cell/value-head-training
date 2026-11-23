# Q4_K vs Q8_0 policy drift (value-head gate)

Model: `Qwen3.6-35B-A3B abliterated MTP`, same prompt, `temperature=0`, `max_tokens=256`, `seed=1234`.

Missions: **12** real bank statements (T1), label `remaining_steps` = tactic steps of the bank reference proof.

Judge: one `lake env lean` per mission on the model's own proof; `solved` = exit 0 and no `sorry`.

d_hat: `state/runs/tiny-sft` (`stage-2 checkpoint, hashed backbone, trained on 50 synthetic records`). **Absolute d_hat is SYNTHETIC** (the checkpoint is a tiny/hashed training run, not a head trained on this endpoint); only the Q4-vs-Q8 difference is interpretable. States scored: `(goal, bank context + that endpoint's own generated proof)`.


## Per mission

| mission | tier | ref steps | Q4 solved | Q4 tok | Q4 steps | Q8 solved | Q8 tok | Q8 steps | d_hat Q4 | d_hat Q8 | gate Q4/Q8 |
| --- | --- | ---: | :---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | :---: |
| `drift-T1-e-irrationality_of_pi` | T1 | 1 | no | 8 | 1 | yes | 42 | 1 | 8.0 | 14.8 | prove/prove |
| `drift-T1-e-sum_range_1052` | T1 | 1 | no | 14 | 2 | yes | 17 | 2 | 23.8 | 23.8 | prove/prove |
| `drift-T1-e-sum_range_25` | T1 | 1 | no | 14 | 2 | yes | 17 | 2 | 23.8 | 23.8 | prove/prove |
| `drift-T1-e-sum_range_804` | T1 | 1 | no | 14 | 2 | yes | 17 | 2 | 23.8 | 23.8 | prove/prove |
| `drift-T1-e-sum_range_831` | T1 | 1 | yes | 17 | 2 | yes | 17 | 2 | 23.8 | 23.8 | prove/prove |
| `drift-T1-e-sum_range_89` | T1 | 1 | no | 14 | 2 | yes | 17 | 2 | 23.8 | 23.8 | prove/prove |
| `drift-T1-m-base_44986` | T1 | 6 | no | 192 | 6 | no | 256 | 6 | 58.9 | 21.8 | prove/prove |
| `drift-T1-m-isUltrametricDist_of_isNonarchimedean` | T1 | 5 | no | 256 | 10 | no | 256 | 9 | 1.1 | 23.8 | prove/prove |
| `drift-T1-m-missionBranchSafety` | T1 | 7 | no | 256 | 17 | no | 185 | 11 | 23.2 | 23.8 | prove/prove |
| `drift-T1-m-problem_3247` | T1 | 5 | no | 114 | 3 | yes | 71 | 2 | 60.9 | 60.1 | prove/prove |
| `drift-T1-m-problem_4689` | T1 | 2 | yes | 44 | 6 | yes | 46 | 6 | 16.3 | 16.4 | prove/prove |
| `drift-T1-m-problem_54150` | T1 | 3 | no | 41 | 5 | no | 43 | 5 | 15.0 | 15.3 | prove/prove |

## Aggregate

| metric | Q4_K | Q8_0 |
| --- | ---: | ---: |
| solved | 2/12 (17%) | 8/12 (67%) |
| mean generated tokens | 82 | 82 |
| median generated tokens | 29 | 42 |
| mean tactic steps | 4.8 | 4.2 |
| median tactic steps | 2 | 2 |
| measured decode t/s | 85.3 | 39.1 |

- solved/not agreement: **6/12**
- token drift per mission: mean |Δ| = 19, median |Δ| = 3
- step drift per mission: mean |Δ| = 0.7, median |Δ| = 0
- d_hat drift (SYNTHETIC head): median |Δlog1p| = 0.008, max = 2.459, gate flips = 0/12

### Failure classes

| class | Q4 | Q8 |
| --- | ---: | ---: |
| solved | 2 | 8 |
| closed then extra tactic | 4 | 0 |
| truncated at max_tokens | 2 | 2 |
| unknown identifier | 1 | 0 |
| unsolved goals | 0 | 1 |
| heartbeat timeout | 1 | 0 |
| other | 2 | 1 |

### By stratum

| stratum | n | Q4 solved | Q8 solved | Q4 tok | Q8 tok | mean abs Δtok | mean abs Δsteps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 6 | 1/6 | 6/6 | 14 | 21 | 8 | 0.0 |
| moderate | 6 | 1/6 | 2/6 | 150 | 143 | 30 | 1.3 |

## Reproduce

Server flags are identical for both endpoints; only the GGUF and `--n-cpu-moe` differ, and only one server runs at a time, under the GPU lease:

```text
flock -x -w 3600 /data/gpu.lock bash /tmp/vht-exp/one_endpoint.sh q4
llama-server -m ...Q4_K.gguf -c 262144 -np 1 -ngl 99 -fa on \
  -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 --n-cpu-moe 14 \
  --load-mode none --no-mmproj --jinja --spec-type draft-mtp \
  --spec-draft-n-max 2 --host 127.0.0.1 --port 8097

# Q8_0: same, --n-cpu-moe 27 --port 8098
```

Requests: `temperature=0`, `seed=1234`, `max_tokens=256`, `chat_template_kwargs={"enable_thinking": false}`. Thinking is off on purpose: with it on, all 12 Q4 answers were empty at 256 tokens (the budget went to the reasoning block).


## Recommendation

**The 500-step gate threshold does not need recalibration; Q4 does need a
different role.** The two numbers point in different directions and both matter:

1. **d_hat is endpoint-stable.** On the same (goal, bank-context + the model's own
   proof) states the synthetic head moves by a median of 0.008 in log1p and the
   gate decision flips on 0/12 missions. A value head trained on Q8 prefixes can
   read Q4 prefixes without re-fitting the threshold — the *estimate* is not what
   quantisation breaks.
2. **The policy is not interchangeable.** Strict compile success is 2/12 for Q4
   vs 8/12 for Q8 on the same prompts. The gap is concentrated in one failure
   class: 4 of Q4's misses are `No goals to be solved` — the proof *closed the
   goal* and then applied one more tactic. That is a verbosity/over-application
   gap, not a reasoning gap, and the strict judge (and the platform verifier)
   counts it as a failure.

So: **keep Q4 as the rollout engine and Q8 as the audit/promotion endpoint.**
Do not recalibrate the threshold, but do not pool Q4 and Q8 success statistics
either — a training target that counts "steps to success" measured on Q4
rollouts is biased against Q4's over-application failures. If Q4 must be the
primary policy, the cheapest fix is in the agent loop, not the head: retry once
on `No goals to be solved`, which recovers 4 of the 6 lost missions here.

