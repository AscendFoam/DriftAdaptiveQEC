# Route-A 高水平论文最终证据门

T6.9.3 对 T6.7.4、T6.8.7、T6.9.1 与 T6.9.2 的当前证据做逐 claim 收口。结论不是完整高水平论文 GO，而是：

> `NO_GO_FULL_HIGH_LEVEL_PAPER_RESTRICTED_PREBOARD_DRAFT_ONLY`

这表示可以继续整理一篇明确标注 simulator/pre-board 边界的受限系统稿，但不允许进入 Phase 7 的完整主图/正文冻结，也不能用叙事把失败的性能或实板主张补成通过。

## 逐主张结论

| 主张 | 当前状态 | 允许写法与必要边界 |
| --- | --- | --- |
| contract-centric integrated system | `SUPPORTED_RESTRICTED_PREBOARD` | MAP fast path、regime-aware contract/FSM 与百万周期 pre-board correctness；不得写整体最佳或已实板闭环 |
| smooth locked-EWMA outcome | `SUPPORTED_PAIRED_OUTCOME` | aggregate paired LER 相对锁定 EWMA 的 95% CI 下界大于 0；Holm 只确认 periodic，且 Route-A 不是全局最优 deployable method |
| static GKP superiority | `FALSIFIED` | frozen same-model smooth benchmark 中 static joint MAP average LER 更低，必须作为正文负结果保留 |
| static K4 hard-action equivalence | `SUPPORTED_PREBOARD_NARROW` | 仅对冻结 covariance/prior 的完整 `1,048,576` 点域，0 hard disagreement；非 universal exact、非实测资源优势 |
| tail safety/improvement | `SAFETY_NONINFERIORITY_ONLY` | 只支持 catastrophic/nominal 门下相对 locked EWMA 的安全 non-inferiority；0 个 family 确认 average improvement |
| general drift external comparison | `PERFORMANCE_OUTCOME_BUDGET_FAIL` | Route-A 相对 pinned BOCD wrapper 的 paired LER 较低，但外部 worst update `13,004.1 us > 5,000 us`，不能写 matched-budget 或 general SOTA |
| Puviani NMF surpass | `PROHIBITED_SOURCE_INCOMPLETE` | official GQF paper-exact `0/15`，matched metric 非空数为 0；不能写 lifetime comparison 或 surpass |
| deterministic FPGA architecture | `SUPPORTED_PR_ESTIMATE` | no-student profile 三 seed P&R Fmax 最低 `39.137 MHz`，六周期 @27 MHz 的 clock model 为 `222.222 ns`；只能写 estimate |
| measured board correctness/latency | `BLOCKED_ALL_FIELDS_NULL` | 六项 physical prerequisite 缺失，42 个 measured fields 全为 `null` |
| FPGA speed advantage | `PROHIBITED_NO_SAME_TASK_BOARD_COMPARATOR` | 无真板 measurement，T6.8.6 same-task external comparator 为 0 |
| CNN/HMM role | `CNN_ABLATION_HMM_SOFTWARE_ONLY` | CNN/teacher/student 只作可替换消融；HMM 是 software slow loop，不得写成 FPGA 主创新 |

## 可继续与不可继续的论文路径

当前只允许 `RESTRICTED_PREBOARD_SYSTEM_DRAFT`，可使用：locked-EWMA smooth paired outcome、tail safety non-inferiority 及代价、六周期 pre-board 架构、static 负结果、BOCD performance outcome 加 budget fail、official GQF negative reproduction audit。

以下六条阻断完整 cross-lane 高水平论文：static superiority 已证否；tail broad improvement 未建立；一般 drift matched-budget/SOTA 未建立；Puviani exact/lifetime 未建立；真板 correctness/latency 未测；FPGA same-task speed advantage 未建立。它们不能相互补偿，也不能合成 global score。

## Phase 7 图表门

- 可受限准备：系统架构、smooth locked-EWMA paired outcome、三 seed pre-board P&R。
- 必须保留为主文负面/限制：static/Window 对照、tail 高 fallback/false-update 成本、BOCD budget fail、GQF 0/15 exact。
- 保持阻塞：板级 latency/deadline/power 与 same-task speed 图。

Phase 7 完整主图和正文冻结保持关闭。恢复条件是先完成 T6.9.2 并重跑 T6.9.3；性能失败项还需要新的、预注册的方法和实验，单纯改措辞不能恢复。

## 完整性与反简化验证

- 11 条原子最终主张，每条都有允许/禁止措辞、current result、父证据、remaining gate 与 revocation conditions；
- 四个父报告、11-row Source Data CSV 与门控实现均有实时 SHA-256/字节数绑定；
- 17/17 integrity gates 通过；
- 17/17 target-specific semantic mutations 被拒绝；
- focused tests：6/6 通过。
- Phase 6A 相邻证据门、GQF/static/external/FPGA lane 与任务板治理联合回归：126/126 通过。

复核命令：

```powershell
python -m cnn_fpga.benchmark.route_a_board_measurement_gate
python -m cnn_fpga.benchmark.route_a_final_evidence_gate
python -m cnn_fpga.benchmark.route_a_final_evidence_gate --verify docs/t6_9_3_route_a_final_evidence_gate.json
python -m pytest tests/test_route_a_final_evidence_gate.py -q
```
