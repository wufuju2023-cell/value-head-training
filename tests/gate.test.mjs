import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { test } from "node:test"

import { DEFAULT_THRESHOLD, decideGate, suggestedChildren } from "../lib/gate.mjs"

const table = JSON.parse(readFileSync(new URL("./gate_cases.json", import.meta.url), "utf8"))

for (const c of table.cases) {
  test(`gate: ${c.name}`, () => {
    const estimate = typeof c.estimate === "string" ? Number(c.estimate) : c.estimate
    const got = decideGate(estimate, {
      ...(c.threshold !== undefined ? { threshold: c.threshold } : {}),
      ...(c.prev ? { prev: c.prev } : {}),
      ...(c.ci_high !== undefined ? { ciHigh: c.ci_high } : {}),
      ...(c.tail_saturated === true ? { tailSaturated: true } : {}),
    })
    assert.equal(got.action, c.expect.action, `action for ${c.name}`)
    assert.equal(got.reason, c.expect.reason, `reason for ${c.name}`)
    if (c.expect.suggested_children !== undefined) {
      assert.equal(got.suggested_children, c.expect.suggested_children, `children for ${c.name}`)
    }
    if (c.expect.threshold !== undefined) {
      assert.equal(got.threshold, c.expect.threshold, `threshold for ${c.name}`)
    }
  })
}

test("gate: default threshold is 500 and the exit band is 400", () => {
  assert.equal(DEFAULT_THRESHOLD, 500)
  assert.equal(decideGate(499).action, "prove")
  assert.equal(decideGate(500).action, "decompose")
  assert.equal(decideGate(401, { prev: "decompose" }).action, "decompose")
  assert.equal(decideGate(400, { prev: "decompose" }).action, "prove")
})

test("gate: hysteresis never flips inside (400, 500)", () => {
  for (let e = 401; e < 500; e += 7) {
    assert.equal(decideGate(e, { prev: "decompose" }).action, "decompose", `hold at ${e}`)
    assert.equal(decideGate(e, { prev: "prove" }).action, "prove", `stay at ${e}`)
  }
})

test("gate: children hint clamps to [2, 8]", () => {
  assert.equal(suggestedChildren(1, 500), 2)
  assert.equal(suggestedChildren(500, 500), 2)
  assert.equal(suggestedChildren(3000, 500), 6)
  assert.equal(suggestedChildren(12000, 500), 8)
  assert.equal(suggestedChildren(100, 0), 2)
})

test("gate: tail saturation always decomposes, even with a tiny estimate", () => {
  const got = decideGate(3, { tailSaturated: true })
  assert.equal(got.action, "decompose")
  assert.equal(got.reason, "tail_saturated")
})
