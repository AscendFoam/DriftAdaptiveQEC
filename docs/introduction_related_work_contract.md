# T7.2.1 Introduction / Related Work 证据合同

- 状态：`PASS_EVIDENCE_BOUNDED_INTRODUCTION_RELATED_WORK`
- 主论点：For repeated approximate-GKP correction, decoder adaptation is an evidence-gated execution contract: MAP owns LER, typed event/fallback logic owns safety, and a versioned six-cycle fast path owns deterministic execution.
- 正文源：`docs/paper_notes/Contract_Centric_Regime_Aware_GKP_note_draft.tex`
- Introduction：6 个 prose 段，section SHA-256 `768e1ff5da168e2288a4e12c2323fcf0d4640285d7f1dfc3a6651216f847191a`
- Related Work：5 个 mechanism group，section SHA-256 `763235d18083ab0918271e21e8f75605011b5af8fa4d0072240d49000d0dab40`
- 引用：30 个已解析 key；机器合同 18/18 gates，语义篡改 18/18 检出。

## 写作结构与边界

Introduction 采用 general-to-specific + open-with-challenge：从 finite-energy GKP 与真实反馈约束，依次收敛到 analog/history、drift/calibration、timing-boundary，再提出 execution-contract 问题和当前 restricted 结论。Related Work 按机制分组，不按截图中的类别做总榜。

必须保留的负证据：static joint MAP 平均 LER 更低、Window MAP 是强反例、tail 只通过 locked-EWMA non-inferiority、V5 在 headroom 门提前停止、NMF exact reproduction 不合格、learned same-task eligibility=0、external FPGA same-task comparator=0、实板字段仍为 null。

## Claim / citation / evidence 行

| Row | Section | Lane | Evidence | Citation / project source | Boundary |
| --- | --- | --- | --- | --- | --- |
| IRW-001 | Introduction | `physical_gkp_context` | `LITERATURE_PRIMARY` | gkp2001, grimsmo2021, hastrup2023, jafarzadeh2025, lachance2024, sivak2023 | External theory and experiment motivate the task; they are not project hardware evidence. |
| IRW-002 | Introduction | `single_mode_decoder` | `LITERATURE_PRIMARY` | fukui2018, noh2020, noh2022, wan2020, berent2024, lin2023 | No cross-code CI/ML/MAP/CPD leaderboard is permitted. |
| IRW-003 | Introduction | `drift_calibration` | `LITERATURE_PRIMARY` | spitz2018, wagner2021, chen2022, dgr2023, sivak2024, stein2026 | The project does not claim the first adaptive or calibration-aware QEC decoder. |
| IRW-004 | Introduction | `contract_system` | `PROJECT_NATIVE_PREBOARD` | T7.1.1:CONTRACT_SYSTEM_INTEGRATION, T6.9.3:V4_FINAL_GATE | Restricted simulator/pre-board integration only; no measured board or global decoder claim. |
| IRW-005 | Introduction | `single_mode_decoder` | `PROJECT_NATIVE_MATCHED` | T7.1.1:SMOOTH_LOCKED_EWMA_ADVANTAGE, T7.1.1:STATIC_GKP_SUPERIORITY | No superiority over static or Window MAP and no all-family drift advantage. |
| IRW-006 | Introduction | `v5_negative` | `NEGATIVE` | T6.10.1:CAUSAL_HEADROOM, T6.15.5:V5_EARLY_STOP | V5 has no formal, fixed-point, CXXRTL, P&R, or measured-hardware result. |
| IRW-007 | Analog, history-aware, and structured GKP decoding | `surface_gkp_gate_ci_ml` | `LITERATURE_PLUS_OFFICIAL_REPRODUCTION` | noh2020, noh2022, raveendran2022, berent2024, lin2023 | Gate failure, threshold, and multimode LER cannot be subtracted from single-mode Route-A LER. |
| IRW-008 | Calibration and drift-adaptive decoding | `drift_calibration` | `SYNTHESIS_INFERENCE` | spitz2018, wagner2021, chen2022, dgr2023, sivak2024, stein2026 | Novelty is joint systems/evidence integration, not absolute algorithmic first use. |
| IRW-009 | Learned, non-Markovian, and autonomous feedback | `direct_nn_rl_nmf_controller` | `LITERATURE_PRIMARY` | bausch2024, wang2022, sivak2026, puviani2025, lachance2024 | Controller gain, decoder LER, and autonomous lifetime are never merged. |
| IRW-010 | Learned, non-Markovian, and autonomous feedback | `direct_nn_rl_nmf_controller` | `NEGATIVE` | puviani2025 | No claim of reproducing or surpassing Puviani NMF is allowed. |
| IRW-011 | Deterministic and FPGA QEC decoders | `fpga_implementation` | `LITERATURE_PRIMARY` | lilliput2022, helios2023, collision2025, ziad2024, maurer2025, yang2026, caune2024 | Core, per-round, source-to-action, and closed-loop latency are not interchangeable. |
| IRW-012 | Deterministic and FPGA QEC decoders | `fpga_implementation` | `NEGATIVE` | T6.19.2:SAME_TASK_COMPARATOR_ZERO, T7.1.1:FPGA_SPEED_ADVANTAGE | Six cycles/II=1 and post-route values are estimates, not measured board superiority. |
| IRW-013 | Position of the present work | `positioning` | `PROJECT_SYNTHESIS` | T7.1.1:MANUSCRIPT_DECISION, T6.19.3:AUX_INTEGRITY | Auxiliary positive results remain task-local and cannot rescue V4/V5 promotion. |
| IRW-014 | Position of the present work | `positioning` | `PROJECT_SYNTHESIS` | T7.1.1:MANDATORY_NEGATIVES, T7.1.3:MAIN_RESULTS, T7.1.4:FAILURE_LEDGER | The manuscript is restricted pre-board, not a cross-protocol or experimental GKP ranking. |

## 反简化检查

- 不是只检查章节标题：合同同时验证段落结构、五个机制组、30 个必需 citation key、14 条 claim-evidence 行和六条 comparison lane。
- 不是关键词 smoke：18 个 gate 各有定向语义篡改，删除 static/Window 负结果、把 V5 改成继续实现、升级 NMF/learned/FPGA 或加入绝对首次主张都会被拒绝。
- 引用从独立 BibTeX 文件解析；任何正文 citation key 缺失都会 fail closed。
- 文献值、official-code reproduction、project-native simulation、P&R estimate 与 board measurement 不得互换。

## 论文 claim 影响

该任务把旧 CNN-centric 叙事收敛为 contract-centric、regime-aware 的安全双回路，但没有升级性能 verdict。当前可写的是 restricted simulator/pre-board integration；不可写的是 static/Window superiority、broad tail gain、Puviani NMF surpass、真实 break-even、fastest FPGA 或 measured board result。
