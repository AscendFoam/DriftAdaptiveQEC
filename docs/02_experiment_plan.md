# DriftAdaptiveQEC 实验规划与后续开发计划

**最后更新：** 2026-06-15
**当前阶段：** `Phase 2: Controlled Development`  
**当前决策状态：** `Go`  
**当前唯一任务：** 以 `docs/04_task_board.md` 和 `docs/07_handoff.md` 为准；当前为 `T90`。`T74`、`T75`、`T76`、`T77`、`T78`、`T79`、`T80`、`T81`、`T82`、`T83`、`T84`、`T85`、`T86`、`T87`、`T88`、`T89` 已完成并收口；`T79` 给出了 `GO_FOR_BOUNDED_PROSE_REOPEN`，`T80` 已完成 ready sections 的有界 prose reopen，`T81` 已完成 `Summary of Contributions` 与三章 methods 的受控校准，`T82` 已完成 supporting-material 收口与 appendix/supplement 边界整合，`T83` 已完成全文一致性 sweep 并给出 `GO_FOR_BOUNDED_FINAL_POLISH_ONLY`，`T84` 已完成有界 final polish 与读者化装配，`T85` 已完成 submission-readiness preflight 与 residual wording-lag 清扫并关闭 `R36`，`T86` 已完成 bounded submission-pack assembly 与显式 exclusion route 收口并经 Captain 以 `PASS` 接受，`T87` 已完成主线作者终检与 pre-submission QA 收口并经 Captain 以 `PASS` 接受，`T88` 已完成 bounded manual finish 执行与 surface freeze 收口并经 Captain 以 `PASS` 接受，`T89` 已完成 frozen-mainline handoff / source-of-truth / change-control 收口并经 Captain 以 `PASS` 接受；因此主线当前转入 `T90` 的训练链 clean-CPU 同机 repeated-run consistency 强化阶段。在暂时缺少 `Linux + FPGA` 硬件宿主的前提下，real-board 执行相关任务继续降为最低优先级 backlog。

## 文档角色

本文档现在承担两个职责：

1. **Part I：项目从开始至今的规划与证据演进**  
   只保留高层时间线、P0-P4 / T 系列关键转折、仍有效的结论，以及已经被后续任务替换或降级的旧结论。该部分自 `2026-06-11` 起接管 `docs/progress_summary/CNN_FPGA_GKP_阶段结论.md` 的当前阶段结论职责。
2. **Part II：后续开发计划**  
   吸收 `docs/follow-up_plan/README.md` 的功能，作为后续开发、论文准备、任务候选池和计划维护的唯一入口。

本文档不是结果证明文件。任何结果 claim 必须回到对应的 task package、review、run root、artifact、summary helper 或治理文档中验证。当前任务状态仍以 `docs/04_task_board.md` 和 `docs/07_handoff.md` 为权威来源。

## 2026-06-15 Captain Update (T89 closeout)

- `T89` 已由 Captain 判定为 `PASS`。
- `T89` 真实完成了 `paper_frozen_mainline_handoff_packet.md`、`paper_frozen_mainline_source_of_truth_map.md`、`paper_postfreeze_change_control.md` 与 `paper_blocked_surface_reentry_conditions.md`，并同步登记到 `docs/paper_materials/README.md` 与 `docs/paper_notes/README.md`。
- `T89` 的 non-blocking notes 全部按 operational reminder 接受处理，不新增 `deferred/rejected` warning，也不新增风险项。
- 当前唯一任务切换为 `T90: 训练链 clean-CPU 同机 repeated-run 一致性证据包`。
- `T90` 只允许在已冻结 mainline 与已隔离 theory 分支的前提下，沿 `R11` 补一个 code-backed、same-host、clean CPU-only 的 repeated-run consistency pack；它不是主线 prose reopen，也不是 `.tflite` portability、real-board execution、expanded benchmark 或 submission-ready completed。

## 2026-06-15 Captain Update (T88 closeout)

- `T88` 已由 Captain 判定为 `PASS`。
- `T88` 真实完成了 manual-finish execution log、surface freeze manifest、author edit decision register、blocked disclaimer table、frozen-mainline handoff gate 与最小 `% T88-MANUAL` note refresh；其 non-blocking notes 全部按 operational reminder 接受处理，不新增 `deferred/rejected` warning。
- 当前唯一任务切换为 `T89: 主线 frozen-mainline handoff 包与 post-freeze change-control 收口`。
- `T89` 只允许做 docs-only、mainline-only、freeze-preserving 的 handoff packet、source-of-truth map、post-freeze change-control 与 blocked-surface re-entry 条件固化；它不是 submission-ready completed，也不得混入独立 theory 分支内容。

## 2026-06-14 Captain Update (T87 closeout)

- `T87` 已由 Captain 判定为 `PASS`。
- `T87` 真实完成了 author-final QA checklist、pre-submission regression gate、wording red-flag register、manual-finish queue 与最小 QA note refresh；其 non-blocking notes 全部按 operational reminder 接受处理，不新增 `deferred/rejected` warning。
- 当前唯一任务切换为 `T88: 主线 bounded manual finish 执行与 surface freeze 收口包`。
- `T88` 只允许做 docs-only、mainline-only、manual-finish execution、surface freeze 与 blocked-disclaimer 固化；它不是 submission-ready completed，也不得混入独立 theory 分支内容。

## 2026-06-14 Captain Update (T86 closeout)

- `T86` 已由 Captain 判定为 `PASS`。
- `T86` 的 non-blocking notes 全部按 operational reminder 接受处理，不新增 `deferred/rejected` warning，也不新增风险项。
- 当前唯一任务切换为 `T87: 主线作者终检与 pre-submission QA 收口包`。
- `T87` 只允许做 docs-only、mainline-only、QA-only 的作者终检 / 投稿前回归 gate；它不是 submission-ready completed，也不得把独立 theory 分支内容拉回 main。

## 2026-06-14 Captain Update (T85 closeout)

- `T85` 已由 Captain 判定为 `PASS`。
- `T85` 真实完成了 residual wording-lag 清扫、submission-readiness preflight、blocker matrix 与残余状态核对；其 non-blocking notes 全部按 operational reminder 接受处理，不新增 `deferred/rejected` warning。
- `R36` 已由 `T85` 收口关闭。
- 当前唯一任务切换为 `T86: 主线 bounded submission-pack assembly 与显式 exclusion route 收口`。
- `T86` 只允许做 docs-only、mainline-only、assembly-only 的装配与排除项收口；它不是 submission-ready pack 完成态，也不得混入独立 theory 分支内容。

---

# Part I：项目从开始至今的规划与证据演进

## 1. 项目核心问题

本项目研究的是：**在 dual-loop runtime 约束下，用 teacher-anchored residual/control calibration 路线实现 drift-adaptive 的 GKP 解码**。

当前仍然有效的核心技术合同是：

- fast loop 执行低延迟线性/残差修正，运行时真实消费的是 `(K, b)` 或等价 residual/control term；
- slow loop 从 histogram、teacher estimate、compact statistics 或 calibration module 中产出 bounded update；
- `ParamMapper` 的主线语义、benchmark 口径、baseline 集合和 evidence level 不得在同一任务中被静默改写；
- 项目目标不是“CNN 全面替代经典解码器”，也不是“真实 FPGA 板级系统已经完成验证”。

当前更稳妥的项目/论文定位仍是：

> A deployment-bounded, teacher-anchored residual calibration framework for drift-adaptive GKP decoding under dual-loop runtime constraints.

## 2. 当前阶段与证据边界快照

| 证据层 | 当前状态 | 权威锚点 | 不得外推到 |
| --- | --- | --- | --- |
| Recovery decision | `Phase 2: Controlled Development` / `Go` | `T13`、`docs/04_task_board.md` | 任意无界扩范围开发 |
| P0/P3/P4 recovery smoke | 最小可复验入口已恢复 | `docs/recovery_bootstrap/P0_smoke_bootstrap.md`、`docs/recovery_bootstrap/P3_software_hil_bootstrap.md`、`docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md` | 完整训练链、真实 `.tflite`、真板 |
| P3 software HIL | bounded path 已按 `mock + model_artifact + artifact_npz + inproc` 收口 | `T12` | true `.tflite` HIL 或 real-board HIL |
| P4 frozen-set formal software revalidation | 已完成，且 `T24` 仍是历史主锚点 | `T24`、`T25`、`runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743` | paper-grade expanded benchmark、真实 runtime、真板 |
| Mechanism evidence | 已有 multi-seed trace 与 bounded intervention，但仍非因果闭环 | `T46`、`T54`、`T55`、`T56`、`T57`、`T58` | 简单 causal proof |
| Training/material regeneration | 已有 bounded pack 与 CPU-only rerun | `T31`、`T39`、`T40`、`T50` | full reproducibility 或跨主机保证 |
| True `.tflite` runtime | current-host isolated path 已确认 | `T48` | 默认环境恢复、HIL closure、deployment closure |
| Real-board gate | current-host verdict 仍为 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE` | `T49`、`T71`、`T72` | 真板执行成功 |
| `statcalib` | bounded mock-backed software-HIL extension lane，且明确 `no-promotion` | `T64`-`T70` | 成熟主线 comparator 或 `T24` 替代表 |
| Paper materials / authoring | `T74` 的 paper-ready material pack、`T75` 的 bounded Results authoring pack、`T76` 的 rendered QA / assembly、`T77` 的 note-draft results sync / traceability hardening、`T78` 的 note 非结果层校准 / hierarchy / layout closeout、`T79` 的 reopen gate、`T80` 的 ready-sections prose reopen、`T81` 的 contribution/methods calibration、`T82` 的 supporting-material closeout、`T83` 的全文一致性 sweep / closeout gate、`T84` 的 bounded reader-facing final polish / assembly、`T85` 的 submission-readiness preflight / residual wording-lag sweep、`T86` 的 bounded submission-pack assembly / exclusion route 收口、`T87` 的作者终检 / pre-submission QA 收口、`T88` 的 bounded manual finish / surface freeze 收口均已完成；当前进入 `T89` 的 frozen-mainline handoff / post-freeze change-control 收口 | `T74`、`T75`、`T76`、`T77`、`T78`、`T79`、`T80`、`T81`、`T82`、`T83`、`T84`、`T85`、`T86`、`T87`、`T88`、`T89` | submission-ready pack 完成态、无证据升级的方法章扩写、deployment/board 叙事放大或 paper claim 升级 |
| Sidecar 扩展 | 可并行设计，但不能自动进入主线事实 | `PSE0`、`PSE1`、`docs/sidecar/README.md` | 主线 benchmark 或论文 claim |

## 3. 高层时间线

| 时间 | 阶段 / 任务 | 关键转折 | 当前保留方式 |
| --- | --- | --- | --- |
| 2026-03-17 | P0 | `full_qec` vs `simplified` baseline gap 被确认 | 作为物理/简化模型差异的历史起点，数值只按原 run 引用 |
| 2026-03-19 | P1/P2 | `static_theta_v2` 模型、量化资产和行为级自适应链路形成 | 代码与 artifact 作为历史材料保留；训练复现以 `T31/T39/T40/T50` 为准 |
| 2026-03-28 | P3 software HIL | software HIL 路径首次打通 | 后续经 `T3/T4/T6/T12` 降级并收口为 `mock + artifact_npz + inproc` bounded path |
| 2026-04-01 至 2026-04-17 | P4 早期主线 | 路线从 absolute parameter regression 转向 teacher-guided residual-b；No TeacherParams 等离线现象一度看起来更强 | 历史候选现象保留，但正式结论以后续 `T24/T57` 和 review 边界为准 |
| 2026-04-27 至 2026-04-29 | Gated v5/v8/v9 | Gated v5 一度成为强 candidate，v8/v9 边际收益有限 | 作为机制和 sidecar 素材保留，不再鼓励无界超参微调 |
| 2026-05-05 至 2026-05-08 | T0-T13 Recovery | 完成治理、边界审计、P0/P3/P4 smoke、manifest 与 exit review | 项目由 Recovery 进入 Controlled Development |
| 2026-05-10 | T23-T25 | P4 formal protocol lock 与 `T24` frozen-set formal software revalidation 完成 | `T24` 成为历史权威 frozen-set software-HIL anchor |
| 2026-05-16 至 2026-05-24 | T31/T36/T38/T46/T54-T58 | 训练依赖、seed=20260429、机制 trace/intervention、paper-facing material lane 逐步补强 | 机制解释更细，但仍不能写成完整因果闭环 |
| 2026-05-26 至 2026-06-10 | T59-T70 | `statcalib` 从 smoke、isolation、fairness、FR8 benchmark、tie-break 到 closure pack | 保留为 extension lane，`T70` 明确 `no-promotion` |
| 2026-06-10 | T48/T49/T50 | `.tflite` isolated runtime、real-board gate、training/material pack 三类边界补强 | 均为 bounded evidence，不升级为部署闭环 |
| 2026-06-11 | T71/T72 | real-board gate pack 从 role-aware/regeneration 进入 provenance hardening | `T72` 完成 provenance hardening；仍无真板执行成功 |
| 2026-06-11 | T73 / 优先级调整 | 因当前暂无 `Linux + FPGA` 硬件宿主，主线从 real-board 前移改为论文材料优先 | `T73` 完成后，real-board 路线继续降为最低优先级 backlog |
| 2026-06-12 | T74/T75/T76/T77/T78/T79/T80/T81/T82 | 主线从台账刷新推进到 paper-ready 材料包、bounded Results authoring、rendered QA / Results assembly、note-draft 结果层同步、非结果层校准 / hierarchy / layout closeout、reopen gate、section-bounded prose reopen、contribution/methods calibration，并继续进入 supporting-material closeout / appendix-supplement boundary integration | `T74`、`T75`、`T76`、`T77`、`T78`、`T79`、`T80`、`T81` 已完成；`T82` 为当前唯一任务；full-manuscript reopen 仍未开启 |
| 2026-06-13 | T82/T83 | `T82` 完成 supporting-boundary 四层收口；主线从“局部段落与 supporting route 收口”推进到“全文一致性 sweep 与 closeout gate” | `T82` 已完成；`T83` 为当前唯一任务；full-manuscript closeout 仍未开启 |
| 2026-06-14 | T83/T84/T85/T86/T87/T88 | `T83` 完成全文一致性 sweep 并给出 `GO_FOR_BOUNDED_FINAL_POLISH_ONLY`；`T84` 完成有界 final polish / reader-facing assembly；`T85` 完成 submission-readiness preflight 并关闭 residual wording-lag 风险 `R36`；`T86` 完成 bounded submission-pack assembly 与显式 exclusion route 收口并经 Captain 以 `PASS` 接受；`T87` 完成作者终检 / pre-submission QA 收口并给出 `GO_FOR_BOUNDED_AUTHOR_MANUAL_FINISH_ONLY`；`T88` 完成 bounded manual finish 执行与 surface freeze 收口并给出 `GO_FOR_FROZEN_MAINLINE_HANDOFF_ONLY` | `T83`、`T84`、`T85`、`T86`、`T87`、`T88` 已完成；submission-ready pack 完成态与 deployment closure 仍未开启 |
| 2026-06-15 | T89 / T90 / 优先级调整 | 主线先从“继续手工终修”切换为“冻结主线答案的 handoff / source-of-truth / change-control 固化”，随后在 `T89 -> PASS` 后转入 non-hardware evidence-hardening 的训练 repeated-run consistency lane | `T89` 已完成并由 Captain 接受为 `PASS`；当前唯一任务切换为 `T90`；real-board 路线继续维持最低优先级 backlog |

## 4. 当前仍有效的结论

1. 项目不是空壳；`physics/`、`cnn_fpga/`、`benchmark/`、`docs/` 中有完整的历史代码、配置、artifact 与治理材料。
2. `Go` 的含义是允许继续 bounded development，不是允许无任务包地扩 benchmark、runtime、真板或论文 claim。
3. fast/slow dual-loop 与 teacher-anchored residual/control calibration 仍是当前最稳的主线。
4. `ParamMapper`、P4 runner 语义、baseline 集合、scenario matrix 与 evidence level 必须继续显式冻结；修改必须任务化。
5. `T24` 仍是最权威的 frozen-set software-HIL 结果锚点。
6. `T48` 只确认 current-host isolated true `.tflite` runtime，不确认默认环境、HIL 集成或部署闭环。
7. `T49/T71/T72` 只属于 real-board gate/provenance 读侧材料，不是真板执行成功。
8. `T64`-`T70` 的 `statcalib` 仍是 extension lane；`T70` 的 `no-promotion` gate 必须随引用保留。
9. 机制诊断已经比早期更强，但 `T55/T56` 也削弱了简单的单因果叙事；论文写作必须保留 hedge。
10. `T74` 已完成 stable-ID 的 paper-ready simulation/material pack，`T75` 已完成 bounded Results authoring，`T76` 已完成真实 rendered preview、人工可读性 QA 与 Results-section assembly，`T77` 已完成 note-draft 的结果层同步与 `T76` traceability hardening，但这些仍不等于 full-manuscript reopen。
11. `T78` 已完成 note 非结果层校准、`statcalib` 层级降权、section-scope 审计与排版 warning 收口；`T79` 已完成 reopen gate，并给出 `GO_FOR_BOUNDED_PROSE_REOPEN`；`T80` 已完成 ready narrative / result-facing sections 的有界重写；`T81` 已完成 `Summary of Contributions` 与三章 methods 的受控校准；`T82` 已完成 supporting-material 与 appendix/supplement 边界整合；`T83` 已完成全文一致性 sweep 与 closeout gate，并给出 `GO_FOR_BOUNDED_FINAL_POLISH_ONLY`；`T84` 已完成有界 final polish 与读者化装配；`T85` 已完成 submission-readiness preflight 与 residual wording-lag 清扫并关闭 `R36`；`T86` 已完成 bounded submission-pack assembly 与显式 exclusion route 收口并经 Captain 以 `PASS` 接受；`T87` 已完成主线作者终检与 pre-submission QA 收口并给出 `GO_FOR_BOUNDED_AUTHOR_MANUAL_FINISH_ONLY`；`T88` 已完成 bounded manual finish 执行与 surface freeze 收口并给出 `GO_FOR_FROZEN_MAINLINE_HANDOFF_ONLY`；当前进入 `T89`，继续做 frozen-mainline handoff / post-freeze change-control 收口，而不是直接宣布 submission-ready pack 完成态或 deployment closure。
12. `runs/` 与 `artifacts/` 是历史证据材料，不应被整体改写成新的事实来源。

## 5. 已被替换或降级的旧结论

| 旧说法 / 旧入口 | 当前处理 |
| --- | --- |
| “当前唯一任务：待定义” | 已由 `docs/04_task_board.md` / `docs/07_handoff.md` 的 current unique task 机制 supersede |
| `docs/follow-up_plan/README.md` 是后续计划唯一入口 | 已由本文档 Part II 替代；该文件只保留为退役索引 |
| `docs/progress_summary/CNN_FPGA_GKP_阶段结论.md` 是当前阶段结论入口 | 已由本文档 Part I 替代；该文件只保留为退役索引 |
| P3 中出现 `real_board` mode 就代表真板 HIL 已近似完成 | 已降级为 placeholder / gate / readiness / provenance 层证据；真板执行仍未发生 |
| `.tflite` artifact、`.tflite.json` stub、TFLite runtime、HIL runtime 可以混写 | 已拆成 artifact type、stub fallback、isolated true runtime、HIL/board integration 四层 |
| No TeacherParams 离线更好，因此可直接作为主线 | formal HIL 和后续 bounded 结果已表明它不能被写成稳定更优主线 |
| 继续追 Gated v10/v11/v12 超参可能是主路径 | 已降级为低优先级；后续应转成机制诊断、protocol lock 或 sidecar |
| `statcalib` 可自然并入 `T24` frozen table | 已被 `T26/T30/T64-T70` 改写为 separate extension lane，并由 `T70` 明确 `no-promotion` |
| paper-ready prose / authoring 可以直接恢复 full manuscript | 当前只成立到 `T74` material pack、`T75` bounded Results authoring 与 `T76` rendered QA / assembly；在 `T77` note-draft results sync / traceability hardening 完成前，仍不能把 full-manuscript reopen 当成既成事实 |
| real-board gate pack 已足够 future-host 复用 | `T72` 收紧了 provenance，但在 future-host 最小 config 标签精确性与真实硬件宿主缺失前，仍不能写成 fully clean / fully reusable |

## 6. 当前工作方式

后续所有开发继续遵守：

- 每轮只推进一个 current unique task。
- Worker 只改 Allowed files，不自动领取下一任务。
- Reviewer 默认只读，优先查 overclaim、mock/stub/placeholder、benchmark 公平性、环境省略和可复现性。
- 新任务必须有 `Allowed files`、`Forbidden scope`、`Verification`、`Docs to update`。
- 不把计划、参考建议、draft prose、sidecar output 或 historical artifact 写成完成事实。

---

# Part II：后续开发计划

## 7. 计划维护规则

1. 从 `2026-06-11` 起，后续计划只维护本文档 Part II。
2. `docs/follow-up_plan/README.md` 退役为索引说明，不再作为活跃计划入口。
3. 任何来自 `docs/reference/`、`docs/deep_research_reports/`、`docs/legacy_context/` 或旧 follow-up 文档的建议，必须先在这里归纳，再拆成独立任务包。
4. 计划本身不能证明结果；结果必须引用 task/review/run/artifact。
5. 当前任务状态变化时，优先同步 `docs/04_task_board.md` 和 `docs/07_handoff.md`；本文档只记录稳定路线与候选池。

## 8. 当前主线任务

当前唯一任务仍以 `docs/04_task_board.md` 和 `docs/07_handoff.md` 为准。

本次整理时的状态说明：

- `T90: 训练链 clean-CPU 同机 repeated-run 一致性证据包`
- 任务包：`docs/tasks/Phase2/T90_training_same_host_repeated_run_consistency_pack.md`
- 状态：`Current Unique Task`

当前主线优先级边界：

- `T74` 已完成 paper-ready simulation/material pack，冻结了 stable-ID 结果表、figure pack、caption pack、insertion map 与 traceability。
- `T75` 已完成 bounded main-text Results authoring、最终成图资产、caption/placement lock、appendix bridge 与 do-not-write guardrail。
- `T76` 已完成真实 rendered preview、人工可读性审查，以及 manuscript-facing Results-section assembly。
- `T77` 已完成 main 分支上的 docs-only note results sync / traceability 任务，把当前经过 `T74/T75/T76` 收口的结果层材料同步到 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`，并修补了 `T76` 的 preview source-map / stable-ID 粒度问题。
- `T78` 已完成当前 note 的非结果层校准、`statcalib` 视觉层级降权、section-scope 审计与 LaTeX 排版 warning 收口，并关闭了对应的 note 质量收口风险。
- `T79` 已完成当前材料栈的受控 reopen/readiness gate，并给出唯一结论 `GO_FOR_BOUNDED_PROSE_REOPEN`。
- `T80` 已完成当前 note 的 8 个 ready narrative / result-facing sections 的 section-bounded prose reopen。
- `T81` 已完成 `Summary of Contributions` 与三章 methods 的受控校准。
- `T82` 已完成，已把 `FR8/statcalib`、training/material、isolated true `.tflite`、real-board `NO_GO` 等 supporting-boundary 材料压成一条 manuscript-facing closeout route。
- `T83` 已完成，已对当前 note 做全文一致性 sweep、受控 wording 收口，并产出唯一 closeout gate / blocker register，结论为 `GO_FOR_BOUNDED_FINAL_POLISH_ONLY`。
- `T84` 已完成，目标层面的 strongest supported truth 是：当前 note 已完成有界 final polish、内部术语读者化翻译、Results/appendix/supplement reader-facing 装配。
- `T85` 已完成，目标层面的 strongest supported truth 是：当前 note 中唯一残余 wording-lag 已清扫，submission-readiness preflight 与 blocker matrix 已建立，且 `R36` 已关闭。
- `T86` 已完成，目标层面的 strongest supported truth 是：当前 mainline note / paper-material / blocker / exclusion 信息已经被装配成一套 submission-facing 但仍显式有边界的 package；这不等于 submission-ready pack 完成态。
- `T87` 已完成，目标层面的 strongest supported truth 是：当前 mainline note/material 已通过作者终检，并且只被允许进入 bounded manual finish，而不是 submission-ready completed。
- `T88` 已完成，目标层面的 strongest supported truth 是：当前 mainline note/material 的 bounded manual finish 已真实执行，并且唯一允许的后续动作已收紧为 frozen-mainline handoff，而不是 submission-ready completed。
- `T89` 已完成，目标层面的 strongest supported truth 是：当前 frozen mainline 已有单一 handoff packet、source-of-truth map、post-freeze change-control 与 blocked-surface re-entry 规则；这仍不是 submission-ready completed。
- `T90` 是当前唯一任务，目标是在不改写 mainline note、不碰 theory 分支、不升级部署/板级语义的前提下，把 `T50` 的“单次 bounded train+eval rerun”推进到“same-host clean CPU-only repeated-run consistency pack”。
- 在暂时缺少 `Linux + FPGA` 硬件宿主的前提下，`T37` 及其他 real-board execution 任务继续 `blocked + lowest-priority backlog`，不抢占当前主线。

## 9. 后续路线总览

后续开发按“论文材料先行、硬件路径后置”的顺序分层推进，不一次性展开全部方向：

1. 训练链 clean-CPU 同机 repeated-run consistency 证据强化
2. 待 `T90` honest closeout 后，再由 Captain 决定是否继续向 cross-host reproducibility / `.tflite` portability 等独立 evidence lane 推进
3. 主线可信度、training / `.tflite` / transfer-pack 边界补强
4. 机制诊断与 bounded ablation
5. paper-grade benchmark expansion 的 protocol lock 与小步执行
6. sidecar / 工程仿真补强
7. 板级语义与真板路径（当前最低优先级 backlog）

## 10. 论文材料与写作路线

当前更稳妥的写作节奏是：

1. `T74` 已完成 paper-ready simulation/material pack，稳定了主线结果表、caption、insertion map 与 traceability。
2. `T75` 已完成 bounded main-text Results authoring、最终成图资产与 do-not-write guardrail。
3. `T76` 已完成真实 rendered preview、人工可读性 QA、必要的 presentation-level 修正与 Results-section assembly。
4. `T78` 已完成 note-draft 的非结果层校准、`statcalib` 层级降权、section-scope 审计与必要的 LaTeX 排版 warning 收口。
5. `T79` 已完成 gate，`T80` 已完成 ready sections 的 section-bounded prose reopen，`T81` 已完成 contribution/methods calibration，`T82` 已完成 supporting-material closeout，`T83` 已完成全文一致性 closeout gate，`T84` 已完成有界 final polish 与读者化装配，`T85` 已完成 submission-readiness preflight 与 residual wording-lag 清扫，`T86` 已完成 bounded submission-pack assembly 与显式 exclusion route 收口，`T87` 已完成作者终检 / pre-submission QA 收口，`T88` 已完成 bounded manual finish 执行与 surface freeze 收口；当前执行 `T89` 的 frozen-mainline handoff / post-freeze change-control 收口。即便如此，也仍不是 submission-ready pack 完成态，更不是部署故事升级。
6. training / `.tflite` / real-board 现阶段只补 boundary table、portability table 或 supporting material，不冒进写成 deployment / board closure。

当前可保留的标题方向：

1. `Runtime-Consistent Teacher-Guided Residual Decoding for Drift-Adaptive GKP Codes`
2. `Deployment-Bounded Residual Calibration for Adaptive GKP Decoding`
3. `Histogram-Conditioned Teacher-Anchored Calibration for GKP Fast-Path Decoding`

当前可保留的贡献点：

1. dual-loop runtime-consistent GKP adaptive decoding framework；
2. teacher-anchored residual/control calibration formulation；
3. 分层证据链：frozen-set mock-backed software-HIL、extension-lane `statcalib` closure、isolated true `.tflite` runtime gate、real-board read-only gate/provenance boundary。

当前明确避免的写法：

- “first neural decoder for GKP”
- “CNN 全面优于所有经典解码器”
- “完整真实 FPGA 系统已经验证”
- “`.tflite` / HIL / real-board 已形成统一闭环”
- “`statcalib` 已成为成熟主线 comparator”

## 11. 主线可信度与复现边界

目标：让当前已存在的主线结论更容易复查、迁移和引用。

可任务化方向：

1. 继续维护 `T24` frozen-set benchmark 的权威地位，不改写历史表格。
2. 将每张结果表、每个图资产继续绑定到 task、run root、config、summary helper 与 review。
3. 为训练材料、模型 artifact、`.tflite` artifact、runtime gate 和 real-board gate 建立更清晰的 manifest。
4. 对未来任何 benchmark rerun 先写 protocol lock，再执行。
5. 保持 claim/evidence ledger、result/figure ledger、authoring manifest 与 preview manifest 之间的一致性。

验收口径：

- 新文档必须能说明“这条结论来自哪个 task 和哪个 evidence level”。
- 不把 recovery smoke、development smoke、mock-backed software-HIL、true `.tflite` runtime、real-board gate 写成同一种证据。

## 12. Paper-Grade Benchmark Expansion

这一部分仍有价值，但只能通过新任务包推进。

可任务化方向：

1. 扩 classical baseline：fixed teacher、window variance、EKF、UKF、RLS residual、oracle-style upper/lower bound。
2. `statcalib` / prior-update baseline：只作为 extension lane 或 future-selection task，不自动进入主线冻结表。
3. learned baseline：CNN-only、teacher-guided residual-b、residual-(K,b)、compact-statistics variant。
4. scenario 扩展：random-walk drift、sinusoidal drift、burst/reset drift、unseen drift holdout。
5. 统计协议：训练 seed 与评测 seed 分离、公共随机流复用、置信区间或停止准则预先声明。
6. 指标扩展：除 `LER` 外，记录 update lag、commit/rollback、slow-loop violation、latency p50/p95/p99、overflow/saturation。

边界：

- `T45` 只锁定了 policy/protocol 分类，没有执行 broader benchmark。
- 未来 expanded benchmark 必须保留 `T24` frozen table 作为历史 anchor。
- 未经新任务执行前，不得写成 paper-grade expanded benchmark 已完成。

## 13. 机制诊断与 Ablation

后续仍需要解释为什么 teacher-anchored residual/control calibration 有效，但不能让解释跑在证据前面。

可保留问题：

1. histogram 中哪些统计量最能预示 drift-induced failure：均值偏移、轴向方差、偏度、边缘峰值、时间差分、anisotropy。
2. `residual-b` 为什么在一些场景足够，在另一些 drift family 下不够。
3. teacher-only、CNN-only、residual-b、residual-K、residual-(K,b)、statcalib-only 的分层 ablation。
4. context window、histogram delta、teacher prediction、teacher params、teacher deltas 的输入通道贡献。
5. update cadence、commit cadence、rollback/fallback 与稳定性的 trade-off。

验收口径：

- 机制诊断先用小样本、frozen scenario 或 focused trace，不直接启动正式长跑。
- 诊断结论不能替代 formal benchmark，也不能把相关性写成因果证明。
- `T55/T56` 之后，简单的 “high committed-b is harmful” 叙事不能再无条件保留。

## 14. Runtime 与 `.tflite` 边界

当前事实：

- `T48` 已确认 current-host isolated true `.tflite` runtime 的窄路径。
- 默认环境、跨主机、部署链、HIL 链路和真板链仍未闭合。

后续方向：

1. 建立 `.tflite` runtime bootstrap，记录 Python、TensorFlow/LiteRT、artifact hash、source-vs-tflite 对照与 latency。
2. 将 `.tflite` 证据拆成 isolated current-host verification、default-env compatibility、cross-host/deployment portability、HIL/board integration 四层 gate。
3. true runtime smoke 必须显式拒绝 `.tflite.json`、`tflite_stub_service` 或 fallback predictor 通过。
4. 在 software-HIL 内引入 `.tflite` slow-loop path 前，先做最小 deterministic smoke。

边界：

- `T48` 不等于默认环境恢复。
- `.tflite` runtime 不等于 HIL closure。
- HIL closure 不等于 real-board validation。

## 15. 板级语义与真板路径

当前事实：

- `board_backend.py` 仍不能写成真实板级完成。
- `T49/T71/T72` 证明的是 read-only gate / regeneration / transfer-pack provenance 边界。
- current-host verdict 仍是 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`。
- `T72` 已完成 provenance hardening；它处理的是 transfer-pack provenance，而不是真板执行。
- 当前还缺少可用的 `Linux + FPGA` 硬件宿主，因此真板执行路线不仅证据未闭合，也处于资源受限状态。

后续方向：

1. 在硬件条件变化前，只保留 host/device/bitstream/AXI/DMA/repo-path truth 的只读 gate、manifest 与 provenance 维护。
2. 建立板级异常事件 taxonomy：DMA stale read、partial write、commit timeout、bank mismatch、device path unavailable。
3. 只有在设备路径、bitstream/RTL/DMA contract、地址表、权限条件和 `Linux + FPGA` 宿主都满足后，才可重新考虑真板 smoke execution task。

边界：

- 任何 write-side MMIO/DMA/register action 必须另有明确授权和任务包。
- `T37` 在 real-board gate/provenance 条件满足且硬件宿主到位前继续 `blocked + lowest-priority backlog`。

## 16. Sidecar / 工程仿真扩展

sidecar 现在被允许作为并行探索层存在，但不能自动晋升主线。

主线可保留的 sidecar 方向：

1. temporal histogram stack + tiny TCN residual-b head
2. adaptive syndrome-only teacher + confidence-gated fallback
3. piecewise-affine / gain-scheduled FPGA parameter bank
4. atomic commit / rollback and transfer-boundary controller checks

治理边界：

- `PSE1` 之后，sidecar lanes 由 `docs/sidecar/` 作为 main 控制台统一治理。
- 默认可在 main 代码基础上通过新增-only helper、standalone module、task-scoped config 推进。
- 需要重计算或多会话隔离时，再使用短生命周期 worktree / clean clone。
- sidecar 结果仍只能写入 `runs/sidecar/<lane_id>/...`，不得写入或改写主线历史 run root。
- 任何 sidecar 输出进入主线前必须通过 Captain promotion gate。
- S4/Mamba、surface-GKP、QLDPC-GKP、transformer/full decoder 等方向只保留为 research-only 或 future-work，除非后续另开任务定义。

## 17. 图表与材料清单

| 图 / 表 / 材料 | 用途 | 当前边界 |
| --- | --- | --- |
| 系统架构图 | dual-loop fast/slow path、histogram、teacher/calibration、commit | 可画概念图，但必须标注 mock/software-HIL |
| 证据等级表 | 区分 smoke、formal software-HIL、`.tflite`、real-board gate | 必须与治理文档一致 |
| 主结果表 | frozen `T24` anchor 与后续 extension lane 分开展示 | 不混表、不改写历史 |
| main-text Results authoring pack | `T75` 的 prose、caption/placement、appendix bridge、do-not-write guardrail | 只代表 bounded authoring，不代表 full manuscript |
| rendered figure QA / assembly pack | `T76` 的 preview、visual QA、callout 与 section assembly | 只代表质量控制与装配，不代表证据升级 |
| paper note results sync pack | `T77` 的 note-draft 结果层同步、traceability hardening 与可选编译检查 | 只代表结果层受控同步，不代表 full-manuscript reopen |
| ablation 表 | teacher/context/features/histogram/residual-b | 未完成项保持 pending 或 appendix/supplement |
| runtime 表 | latency、commit、rollback、overflow、saturation | 不写成真板指标 |
| real-board gate 表 | device path、bitstream、AXI/DMA、repo path truth | 当前是 NO_GO/provenance，不是执行结果 |
| claim/evidence ledger | claim -> task/review/run/artifact | prose reopen 前必须刷新 |
| result/figure ledger | figure/table -> script/config/run root/review | prose reopen 前必须刷新 |

## 18. 投稿路线

如果保持 mock-backed software-HIL + 清晰 runtime boundary + paper-grade writing，当前更稳的目标是：

- IEEE Quantum Week / QCE
- IEEE Transactions on Quantum Engineering / TQE
- EPJ Quantum Technology
- Quantum Science and Technology / QST
- ACM Transactions on Quantum Computing / TQC

FCCM、ACM FPGA、DATE、ICCAD 等硬件向 venue 只有在补齐以下证据后才适合作为主目标：

1. 真实板卡或等价硬件路径；
2. 资源、时延、吞吐、接口 serialization 成本；
3. bitstream / RTL / DMA / AXI / register contract；
4. 与 software-HIL 的误差对照。

当前不能把这些 venue 写成“马上适合”的主目标。

## 19. 后续任务候选池

以下不是当前任务，只是可拆包候选。任何候选转为执行前都必须写成独立 task package。

| 优先层 | 候选任务 | 主要输出 | 验证 |
| --- | --- | --- | --- |
| Current | `T90` 训练链 clean-CPU 同机 repeated-run 一致性证据包 | code-backed repeat-consistency pack、3 次同配置 train+eval rerun、pairwise metric/model consistency table、README 登记 | clean CPU-only repeated reruns + helper/tests + evidence-pack review |
| Immediate-next | 按 `T90` closeout 决定的唯一后续任务 | 继续推进 cross-host reproducibility / `.tflite` portability 等独立 evidence lane，或保持 frozen-mainline + evidence-only 状态 | captain / reviewer gate |
| P1 | training/material reproducibility follow-up（跨主机/更强 portability） | repeated-run / cross-host / CPU-vs-GPU 边界表 | bounded train/eval smoke |
| P1 | `.tflite` runtime portability audit | default env / isolated env / cross-host 差异表 | bounded runtime smoke |
| P1 | `.tflite` isolated-env bootstrap hardening | interpreter/package/artifact/source manifest | true-runtime smoke rejects stub/fallback |
| P2 | mechanism diagnosis pack | histogram/residual-b/update cadence 诊断 | focused trace / small sample |
| P2 | paper-grade expanded benchmark execution protocol | scenarios、baselines、metrics、seeds、stopping | protocol review first |
| P2 | GPT-Pro extension-route triage | adopted/deferred/rejected sidecar list | docs-only protocol review |
| P2 | temporal TCN / adaptive teacher / parameter-bank sidecar design | bounded experiment spec + shared inputs | no long-run execution without new task |
| P4-lowest | real-board smoke execution | real device smoke | only after gate conditions satisfy and hardware host becomes available |
| P3 | broader paper draft reopen after frozen-handoff closeout | 在 `T89` 之后才考虑的更大范围 prose 扩写 | only after `T89` closes honestly and a new gate approves broader scope |

## 20. 红线

后续计划继续严格遵守：

1. 不要把 `T24` 写成 paper-grade expanded benchmark。
2. 不要把 `T48` 写成 default-env closure、HIL closure 或 deployment closure。
3. 不要把 `T49/T71/T72` 写成 real-board execution success。
4. 不要把 `T64`-`T70` 写成 mature `statcalib` comparator promotion。
5. 不要把 `T74/T75/T76/T77` 这样的 paper-material / authoring / QA / note-sync 任务写成新实验结果。
6. 不要把 sidecar 输出自动写成主线事实。
7. 不要在没有新任务包的前提下重开 teacher-representation 长跑、formal benchmark 长跑或真板 smoke。
