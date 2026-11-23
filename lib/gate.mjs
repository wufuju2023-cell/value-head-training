/**
 * The >500-step decomposition gate — pure JS twin of train/gate.py.
 *
 * Keeping the rule in one small module means the plugin tool `vht_gate`, the
 * python CLI and the tests all agree, and `tests/gate_cases.json` is asserted
 * by both suites.
 *
 * Rules, in order:
 *   1. estimate < 0                 -> prove      (no estimate: do not block)
 *   2. tail_saturated               -> decompose  (the head ran out of grid;
 *                                                   "500-ish" is unknown, not small)
 *   3. prev = decompose, est <= exit-> prove      (hysteresis release)
 *   4. prev = decompose             -> decompose  (hysteresis hold)
 *   5. est >= threshold             -> decompose  (above_threshold)
 *   6. ci_high >= 4 * threshold     -> decompose  (ci_upper_far_above: uncertain tail)
 *   7. otherwise                    -> prove
 *
 * `suggested_children = clamp(2, 8, ceil(estimate / threshold))` is a planning
 * hint only: how many missions a first split should produce.
 */

export const DEFAULT_THRESHOLD = 500
export const HYSTERESIS = 0.8
export const FAR_TAIL = 4
export const MIN_CHILDREN = 2
export const MAX_CHILDREN = 8

/** @param {number} estimate @param {number} threshold */
export function suggestedChildren(estimate, threshold) {
  if (!(threshold > 0)) return MIN_CHILDREN
  const raw = Math.ceil(estimate / threshold)
  return Math.max(MIN_CHILDREN, Math.min(MAX_CHILDREN, raw))
}

/**
 * @param {number} estimate
 * @param {{threshold?: number, prev?: "prove"|"decompose", ciHigh?: number|null, tailSaturated?: boolean}} [opts]
 */
export function decideGate(estimate, opts = {}) {
  const raw = Number(opts.threshold)
  const threshold = Number.isFinite(raw) && raw > 0 ? raw : DEFAULT_THRESHOLD
  const enter = threshold
  const exit = HYSTERESIS * threshold
  const prev = opts.prev === "decompose" ? "decompose" : "prove"
  const est = Number(estimate)
  const tailSaturated = opts.tailSaturated === true
  const ciHigh = typeof opts.ciHigh === "number" && Number.isFinite(opts.ciHigh) ? opts.ciHigh : null

  const out = (action, reason) => ({
    action,
    reason,
    estimate: Math.round(est * 1000) / 1000,
    threshold,
    enter,
    exit: Math.round(exit * 1000) / 1000,
    prev,
    tail_saturated: tailSaturated,
    suggested_children: action === "decompose" ? suggestedChildren(est, threshold) : 0,
  })

  if (!Number.isFinite(est) || est < 0) return out("prove", "no_estimate")
  if (tailSaturated) return out("decompose", "tail_saturated")
  if (prev === "decompose") {
    if (est <= exit) return out("prove", "hysteresis_exit")
    return out("decompose", "hysteresis_hold")
  }
  if (est >= enter) return out("decompose", "above_threshold")
  if (ciHigh !== null && ciHigh >= FAR_TAIL * enter) return out("decompose", "ci_upper_far_above")
  return out("prove", "below_threshold")
}
