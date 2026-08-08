# Reviewer response: relationship to Puviani non-Markovian feedback

- Task: `T7.3.5`
- Verdict: `PASS_PUVIANI_RELATIONSHIP_SEPARATED_WITH_SURPASS_PROHIBITED`
- Package readiness: `draft_with_placeholders`
- Gates/mutations: `24/24` / `24/24`

## Point-by-point response

No. More precisely, the manuscript neither presents an official reproduction of Puviani et al. nor claims to outperform their NMF controller. The conceptual overlap is the use of measurement history; the decision problems are different. Puviani et al. optimize fifteen physical sBs control parameters in a single-mode cavity protocol and evaluate six-state logical-channel lifetime. Our current multimode software lane maps observed syndromes to logical actions and evaluates per-round LER, while the independent single-mode RTL lane implements a bounded MAP/event transaction path.

We imported and audited the public GQF source at commit c9ab1ef2b3ff6fa6d6d24cd95fbd06e2872e016d. The paper-exact qualification passed 0/15 criteria: official checkpoints, twenty-agent seeds, the selection ledger and the six-state evaluator are unavailable, and all exact Standard/MF/NMF T_X, T_Y, T_Z, T_ch and F_avg fields remain null. The only official-code-derived execution is a patched, reduced Standard-path diagnostic at cutoff 8: six states, three seeds, 36 trajectories, 378 environment steps and 756 rows. It contains no MF/NMF training or lifetime comparison. The matched Route-A comparison therefore follows an ineligible negative branch and all thirteen result fields remain null.

For completeness, a separate project-native finite-horizon study found cutoff-12 logical-Z area-equivalent lifetimes of 2.747662, 6.534671 and 6.740785 cycles for Standard, MF and NMF, with NMF-minus-MF 0.206114 and paired 95% CI [0.084161, 0.328067]. This is not an official replay. Moreover, at cutoff 16 the latest-only history-reset ablation reaches 8.271987 cycles, above NMF's 7.708351, so we do not claim a universal memory mechanism.

The current primary evidence is deliberately more limited. The multimode method ties the strongest static-mixture exact MLD baseline at p_L=0.111979 over 79,872 development rounds, yielding 0% relative improvement and a NO-GO algorithm verdict. The tail policy establishes a scoped fail-closed safety contract, but calibration-step fallback is 0.958546 and its worst window ties the locked baseline at 181 errors. Historical teacher-to-student retention/compression remains an ablation: zero of sixteen learned families is same-task eligible and the student is absent from the current RTL.

The distinct positive result is an exact pre-board digital-system contribution: seventeen formal gates, twenty-one killed mutants, one million CXXRTL cycles with zero full-vector mismatch, six-cycle latency and initiation interval one, together with atomic A/B publication, CRC/version checking and last-known-good fail-closed recovery. It is not an NMF controller, a physical lifetime result, a board measurement or a fastest-FPGA claim. Any future Puviani-surpass statement requires a protocol-matched, six-state, no-postselection lifetime experiment with identical observation/action/environment and training, wall-clock and compute budgets, plus a positive simultaneous paired 95% lower confidence bound.

## Evidence taxonomy

| Lane | Status | Numeric boundary |
| --- | --- | --- |
| Official GQF paper-exact | `NO_GO_SOURCE_INCOMPLETE` | `0/15` gates; all exact lifetime fields null |
| Official-code reduced diagnostic | `REDUCED_STANDARD_PATH_DIAGNOSTIC_NOT_PAPER_REPRODUCTION` | 756 rows; no MF/NMF or lifetime fit |
| Same-GQF matched comparison | `INELIGIBLE_NEGATIVE_BRANCH_NO_MATCHED_RUN` | 13/13 metrics null |
| Project-native directional study | `PASS` | NMF-MF=0.206114, 95% CI [0.084161, 0.328067] |
| Current multimode algorithm | `NO_GO_MULTIMODE_CAUSAL_HEADROOM` | 0% improvement over strongest static exact MLD |
| Current exact RTL | `GO_RTL_ONLY` | 6 cycles, II=1, 1,000,000 cycles, zero mismatch; pre-board only |

## Manuscript checklist

- State conceptual overlap with Puviani without claiming official reproduction.
- Report 0/15 exact qualification and keep all exact lifetime fields null.
- Label T2.3.7 project-native and retain the cutoff-16 history-reset counterexample.
- Keep Phase-6D 0% strongest-baseline headroom and high fallback costs visible.
- Label teacher/student as historical ablation absent from current RTL.
- Restrict the positive claim to deterministic atomic fail-closed pre-board RTL.
- Keep Phase 9 planned and require a protocol-matched paired-CI gate for any surpass claim.

## Missing author input

- `ACTUAL_REVIEWER_ID_AND_VERBATIM_WORDING`

## 中文核对

共同使用历史信息不构成同任务复现。官方 exact 为 0/15，所有 Standard/MF/NMF lifetime 字段与 matched comparison 字段均为 null；project-native 十周期方向性结果不能填补官方资产。当前算法结论为 strongest-static 0% NO-GO，正贡献仅限 exact single-mode 六周期、II=1、atomic/fail-closed 的预板数字系统。Phase 9 是未来程序，不是既成结果。
