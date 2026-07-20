# T6.16.1 两张异构方法图的一手来源审计

- verdict：`PASS_SECONDARY_METHOD_SOURCE_AUDIT_NO_GLOBAL_RANKING`
- gate：`15/15`
- 来源：`11`；具体方法/实现：`12`；截图 claim：`24`
- 用途：Phase 6C 非主要比较的 source registry；不是 global leaderboard，也不回写 Phase 6B。

## 审计后的核心结论

1. Noh 的两-GKP CNOT 中，ML 相对 CI 的 failure reduction 分别为 9 dB `31.782%`、12 dB `58.458%`、13 dB `67.192%`；因此“约 50%”不是通用值。
2. `9.9 dB` 是 finite-squeezing-only 条件下、带 analog/history-aware outer decoding 的完整 surface–GKP finite-size threshold；不是 Table-I 门级 ML 阈值，也不是本项目 single-mode threshold。
3. Direct NN、model-free RL 与 Puviani model-based Feedback-GRAPE NMF 是三类不同 decision object。Wang 的“decoding rate +50%”缺少可移植分母，规范化值为 null。
4. NN/FPGA 只能报具体实现边界：Overwater d=5 为 `87.6 ns` post-implementation core estimate；Yang d=3 为 `124 ns` core、`550 ns` end-of-readout-to-feedback real closed loop。不存在 `10--100 us` 类别范围。
5. AQEC 是 physical protocol，不是 syndrome decoder；classical decoder latency 记 N/A 而非 0。实验 lifetime gain 为 `1.14(18)`/`1.14(16)`，即约 14%，不是 universal 20%。
6. Lin structured surface–GKP 的同任务 threshold 是 CPD `0.602` 对 analog-MWPM `0.599`；数值保留 paper sigma，不换算 dB。generic、linear 与 polynomial complexity 必须按 lattice structure 分列。
7. 项目 T5 仅有 six-cycle、`222.222 ns`、II `37.037 ns` 的 preboard core estimate；external same-task FPGA comparator 为 `0`，所以 faster/SOTA 禁止。V5 仍为 `NO_GO_V5_EARLY_HEADROOM_STOP`。
8. 项目 T3.2.8 common-wall-clock simulator 的 autonomous/measurement lifetime ratio 范围为 `0.805901--0.942271`，是负/不占优的 project-native model result，不能借用 AQEC 论文的 1.14。

## 分 lane 使用规则

| lane | 可以比较 | 禁止混排 |
| --- | --- | --- |
| single-mode decoder | 同 syndrome/action/observability/budget 的 LER 与 tail | surface-code threshold、AQEC lifetime、controller gain |
| surface-GKP gate/outer code | 同 CNOT circuit 的 failure；同 family finite-size threshold | single-mode repeated-memory LER |
| multimode structured CPD | 同 lattice/noise/size 的 correctness、threshold、scaling | 把 CPD 当 single-mode 新 comparator |
| controller/RL/NMF | 同 physical protocol、history/action、training/compute budget | direct decoder inference |
| AQEC wall-clock | 同 apparatus/model、wall-clock、duty/event budget的 lifetime | zero-latency decoder claim |
| FPGA implementation | 同 code/input/action/problem size/precision/boundary/evidence | 跨 code family 的纳秒总榜 |

## 产物与证据状态

- machine registry：`docs/t6_16_1_secondary_method_source_audit.json`
- Source Data：`docs/t6_16_1_secondary_method_source_data.csv`（`82` rows）
- evidence grade 仅允许 `LITERATURE_ONLY/OFFICIAL_CODE_REPRODUCTION/PROJECT_NATIVE_MATCHED/INELIGIBLE/BLOCKED/NEGATIVE`。
- Puviani exact reproduction：`COMPLETE_GQF_PAPER_EXACT_ATTEMPT_NO_GO_SOURCE_INCOMPLETE`；matched gate：`COMPLETE_T6_8_5_INELIGIBLE_NEGATIVE_BRANCH`；`surpass Puviani NMF` 仍为 `PROHIBITED`。
- null 表示没有一手 locator 或不适用；不得用 high/medium/low、0、类别均值或相邻论文插补。
