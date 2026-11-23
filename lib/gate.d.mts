export type GateAction = "prove" | "decompose"

export type GateResult = {
  action: GateAction
  reason: string
  estimate: number
  threshold: number
  enter: number
  exit: number
  prev: GateAction
  tail_saturated: boolean
  suggested_children: number
}

export declare const DEFAULT_THRESHOLD: number
export declare const HYSTERESIS: number
export declare const FAR_TAIL: number
export declare const MIN_CHILDREN: number
export declare const MAX_CHILDREN: number

export declare function suggestedChildren(estimate: number, threshold: number): number

export declare function decideGate(
  estimate: number,
  opts?: {
    threshold?: number
    prev?: GateAction
    ciHigh?: number | null
    tailSaturated?: boolean
  },
): GateResult
