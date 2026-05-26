# Paper Claim Risk Table

Freeze date: `2026-05-26`

This table maps paper claims to risks: supporting evidence, residual risk, and whether the claim is blocked, partial, or supported.

## 1. Risk Map

| Claim area | Paper claim IDs | Status | Supporting evidence | Residual risk | Risk IDs |
| --- | --- | --- | --- | --- | --- |
| Bounded software-HIL system | C1 | `supported` | T6/T12 recovery + deterministic replay + T24 revalidation | Wording drift into hardware/runtime claim | R8 |
| Frozen-set benchmark win | C2, C3 | `supported` | T24 formal revalidation; 4 scenarios x 5 modes x 2 repeats | Over-generalization beyond frozen set | R5, R9 |
| Mechanism explanation | C4 | `partial` | T36/T38/T54/T55 diagnosis chain | Single-seed overread; no causal intervention closure; no full multi-seed mechanism closure | R10 |
| Bounded FR6 mechanism/intervention figure pack | FR6 | `supported` | T54/T55/T56 evidence packaged by T58 into a final figure, figure-data snapshot, caption, and manifest | Easy to over-read as causal closure or validated mitigation even though `C4` stays partial | R10 |
| Frozen-set feature/teacher ablation table | FR7 / T6 | `supported` | T57 re-execution: 4 scenarios x 6 modes x 2 repeats under locked T24 feature-ablation lane | Easy to over-read as causal attribution or proof that every teacher channel is necessary | R10 |
| Clean training smoke | C5 | `supported` | T39 clean env + T40 one real training smoke | Only one bounded smoke; not full reproducibility | R11 |
| Statcalib interface | C9 | `supported` | T26 feasibility gate + T30 interface contract + focused tests | Interface-only; no integrated benchmark evidence | R24 |
| Broad training reproducibility | C6 | `blocked` | Only one CPU-only smoke exists | No cross-host/OS/GPU matrix | R11 |
| True TFLite runtime | C7 | `blocked` | Code paths exist; stub fallback functional | `tensorflow` / `tflite_runtime` not installed | R12 |
| Real-board validation | C8 | `blocked` | Readiness checklist + execution plan exist | No device connected; no board logs | R13, R14 |
| Statcalib integrated comparator | C10 | `blocked` | Interface contract only | Not wired into slow loop; no benchmark evidence | R24 |
| Paper-grade expanded benchmark | C11 | `blocked` | Frozen-set software-HIL evidence plus T57 FR7 | Still missing broader scenarios, TFLite, real-board, and stronger mechanism closure | R5, R9, R10, R12, R13 |

## 2. T57/T58 Closure Assessment

| Question | Answer |
| --- | --- |
| Does T57 close FR7 as a historical-only gap? | Yes. FR7 now has a formal frozen-set result table under the locked T24 feature-ablation lane. |
| Does T58 close FR6 as a missing paper-material asset? | Yes. FR6 now has a bounded six-seed figure pack with final assets and provenance. |
| Does T57 close the broader mechanism-evidence gap? | No. FR7 closes the table gap, not the causal interpretation gap. |
| Does T58 close the broader mechanism-evidence gap? | No. T58 closes the figure-pack gap, not the causal interpretation gap. |
| Does T57 upgrade C4 from `partial` to `supported`? | No. T56 mechanism hedge boundaries still apply. |
| Does T58 upgrade C4 from `partial` to `supported`? | No. T58 packages the evidence; it does not change the hedge boundary. |
| Does T57/T58 upgrade C11 to paper-grade expanded benchmark evidence? | No. The evidence remains bounded to the frozen-set software-HIL lane and the locked six-seed mechanism lane. |

## 3. Task Coverage Assessment

| Claim area | Closed by T44? | Closed by T45? | Closed by T46? | Closed by T47? | Closed by T57? | Closed by T58? | Still needs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Software-HIL wording governance | Yes | N/A | N/A | N/A | N/A | N/A | None if paper stays bounded |
| Frozen-set wording governance | Yes | N/A | N/A | N/A | N/A | N/A | None if paper stays narrow |
| Benchmark expansion | No | Partially (protocol lock) | No | No | No | No | Broader scenarios and a new approved benchmark lane |
| Multi-seed mechanism figure pack | No | No | Partially (plan + trace) | No | No | Yes | Interpretation must remain bounded by T56/T58 |
| Ablation result pack | No | No | No | Partially (ledger) | Yes | N/A | Interpretation must remain bounded by T56/T57 |
| Training reproducibility | No | No | No | No | No | No | Dedicated task |
| TFLite runtime | No | No | No | No | No | No | Environment + execution |
| Real-board smoke | No | No | No | No | No | No | Hardware + execution |
| Statcalib integration | No | No | No | No | No | No | Dedicated task beyond current roadmap |

## 4. Mainline vs Booster vs Extension

### 4.1 Mainline Blockers Still Open

1. FR8 / statcalib integrated comparator result table
2. Any attempt to stretch bounded frozen-set evidence into an expanded benchmark claim

### 4.2 Strong-Quality Boosters

1. True `.tflite` runtime smoke
2. Real-board smoke execution
3. Training reproducibility and material-regeneration pack

### 4.3 Mainline Evidence Closed by T57/T58

1. FR7 no longer depends on historical pre-T24 ablation evidence
2. FR6 no longer depends on narrative-only references to T54/T55; it now has a bounded figure pack
3. The paper can now cite a bounded feature/teacher ablation table under the locked T24 lane
4. The paper can now cite a bounded six-seed mechanism/intervention figure under the locked T54/T55/T56 lane
5. The remaining FR6/FR7 risk is interpretation, not missing execution

## 5. Verdict

The current risk table confirms:

1. T57 closes the FR7 execution gap.
2. T58 closes the FR6 figure-pack gap.
3. T57/T58 do not close R10 or upgrade the mechanism story to causal proof.
4. T57/T58 reduce two paper-material blockers without reopening benchmark scope.
5. The next bounded paper-material gaps are FR8 plus the deployment/training boundary items outside T57/T58.
