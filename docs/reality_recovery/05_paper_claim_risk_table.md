# Paper Claim Risk Table

Freeze date: `2026-05-19`

This table maps paper claims to risks: supporting evidence, residual risk, and whether the claim is blocked, partial, or supported.

## 1. Risk Map

| Claim area | Paper claim IDs | Status | Supporting evidence | Residual risk | Risk IDs |
| --- | --- | --- | --- | --- | --- |
| Bounded software-HIL system | C1 | `supported` | T6/T12 recovery + deterministic replay + T24 revalidation | Wording drift into hardware/runtime claim | R8 |
| Frozen-set benchmark win | C2, C3 | `supported` | T24 formal revalidation; 4 scenarios x 5 modes x 2 repeats | Over-generalization beyond frozen set | R5, R9 |
| Mechanism explanation | C4 | `partial` | T36 diagnosis + T38 trace (4798 rows, single seed) | Single-seed overread; no causal intervention; no multi-seed confirmation | R10 |
| Clean training smoke | C5 | `supported` | T39 clean env + T40 one real training smoke | Only one bounded smoke; not full reproducibility | R11 |
| Statcalib interface | C9 | `supported` | T26 feasibility gate + T30 interface contract + 6 focused tests | Interface-only; no integrated benchmark evidence | R24 |
| Broad training reproducibility | C6 | `blocked` | Only one CPU-only smoke exists | No cross-host/OS/GPU matrix | R11 |
| True TFLite runtime | C7 | `blocked` | Code paths exist; stub fallback functional | tensorflow/tflite_runtime not installed | R12 |
| Real-board validation | C8 | `blocked` | Readiness checklist + execution plan exist; code structurally complete | No device connected; no board logs | R13, R14 |
| Statcalib integrated comparator | C10 | `blocked` | Interface contract only | Not wired into slow loop; no benchmark evidence | R24 |
| Paper-grade expanded benchmark | C11 | `blocked` | Frozen-set only; no expansion beyond 4 scenarios | Missing broader scenarios, ablation pack, multi-seed mechanism | R5, R9, R10 |

## 2. T44 Closure Assessment

| Question | Answer |
| --- | --- |
| Does T44 close any blocked claims? | No. T44 freezes truth and makes gaps explicit; it does not create missing evidence. |
| Does T44 close any partial claims? | No. C4 remains `partial` after T44. |
| Does T44 close any supported claims? | N/A. Supported claims are already supported. T44 adds governance clarity. |
| Does T44 close the roadmap sufficiency question? | Yes, by making it explicit that T44 alone is insufficient and that T45-T47 are mainline blockers. |

## 3. Task Coverage Assessment

| Claim area | Closed by T44? | Closed by T45? | Closed by T46? | Closed by T47? | Closed by T48/T49? | Still needs |
| --- | --- | --- | --- | --- | --- | --- |
| Software-HIL wording governance | Yes | N/A | N/A | N/A | N/A | None if paper stays bounded |
| Frozen-set wording governance | Yes | N/A | N/A | N/A | N/A | None if paper stays narrow |
| Benchmark expansion | No | Partially (protocol lock) | No | No | No | Execution + evidence |
| Multi-seed mechanism | No | No | Partially (plan + trace) | No | No | Execution + evidence |
| Ablation result pack | No | No | No | Partially (ledger) | No | Execution + evidence |
| Training reproducibility | No | No | No | No | No | Dedicated task (T50) |
| TFLite runtime | No | No | No | No | Partially (T48) | Environment + execution |
| Real-board smoke | No | No | No | No | Partially (T49) | Hardware + execution |
| Statcalib integration | No | No | No | No | No | Dedicated task beyond current roadmap |

## 4. Mainline vs Booster vs Extension

### 4.1 Mainline Paper-Readiness Blockers

These block a strong method paper if unresolved:

1. **T45**: paper-grade benchmark expansion protocol lock and gap audit
2. **T46**: multi-seed mechanism/intervention plan and trace pack
3. **T47**: paper ablation result-pack and material ledger

### 4.2 Strong-Quality Boosters

These materially raise paper quality but are not hard blockers for a minimum evidence-bounded paper:

1. **T48**: true `.tflite` runtime smoke gate
2. **T49**: real-board smoke execution gate
3. **T50**: training reproducibility and material-regeneration pack

### 4.3 Paper Re-Open Gate

4. **T51**: paper positioning re-gate after evidence hardening
5. **T52**: manuscript expansion gate for the next bounded prose wave

### 4.4 Reference-Only Extension Lane

Items from `docs/reference/延伸改进思路.md`:
- Future extension reference
- Not current mainline truth
- Not prerequisites for the minimum evidence-bounded paper thesis
- May become later tasks only after separate scope lock

## 5. Verdict

The current risk table confirms:

1. T44 freezes the claim/risk boundary but does not close blocked claims.
2. T45-T47 are the correct mainline next tasks for evidence hardening.
3. T48-T49 are deployment-boundary boosters.
4. T50 is a reproducibility booster.
5. T51-T52 are paper-re-open gates that should only fire after mainline blockers show evidence progress.
6. Items from `延伸改进思路.md` remain extension-only.
