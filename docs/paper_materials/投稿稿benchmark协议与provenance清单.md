# 投稿稿 benchmark 协议与 provenance 清单

本文档服务 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它把当前投稿稿可使用的 benchmark、消融、机制、statcalib 和硬件占位符材料按“来源 artifact -> 协议 / review 锚点 -> 可写 claim -> 禁止外推”固定下来，避免后续画图或润色时把 mock-backed 软件证据写成硬件、部署或 paper-grade expanded benchmark 结论。

本文档不是新实验，不改变 `T90` 当前唯一任务，也不升级 `T24`、`FR6/FR7`、`FR8/statcalib`、`.tflite` 或 real-board 的证据等级。

## 一句话边界

当前投稿稿可以写成一篇以 dual-loop runtime 约束下的 affine calibration 为核心的 bounded software-HIL 论文草案；它还不能写成完成硬件验证、完成部署闭环、完成 expanded paper-grade benchmark 或证明机制因果闭环的论文。

## 1. T24 frozen software-HIL benchmark

| 项 | 当前锚点 |
| --- | --- |
| Run root | `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/` |
| 直接数据 | `comparison.csv`, `summary.json`, `report.md`, `teacher_scalar_diagnostics.csv` |
| 协议 | `docs/protocols/benchmark/P4_benchmark_formal_protocol.md` |
| Review | `docs/review/T24_review.md`, `docs/review/T25_p4_formal_evidence_gate_review.md` |
| Matrix | 4 scenarios x 5 modes x paired seeds x repeats=2 |
| 可写结论 | `hybrid_residual_b` 在锁定的四场景、五模式、paired-seed 软件 HIL frozen set 中表现最好 |
| 必须保留的边界 | mock-backed software HIL only；不是 `.tflite` runtime；不是 real-board；不是 paper-grade expanded benchmark；teacher diagnostics 和 correction saturation 仍有机制解释缺口 |

投稿稿中建议写法：

- 可以写：`the locked software-HIL benchmark was re-run on the recovered mock-backed path with full coverage`.
- 可以写：`hybrid residual calibration ranked first across the four locked drift scenarios`.
- 不要写：`hardware validation`, `deployment-ready`, `SOTA decoder`, `complete benchmark`, `real-board HIL`, `paper-grade expanded benchmark`.

## 2. FR7 feature and teacher ablation

| 项 | 当前锚点 |
| --- | --- |
| Run root | `runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000/` |
| 直接数据 | `summary_pack/table.csv`, `summary_pack/summary.json`, `provenance_manifest.json` |
| Review | `docs/review/T57_review.md` |
| 可写结论 | feature / teacher ablation 支撑 bounded feature-sensitivity 叙事 |
| 必须保留的边界 | 不支持“teacher params 必要”的强 claim；不支持机制因果闭环；不替代 T24 frozen ranked table |

投稿稿中建议写法：

- 可以写：`feature-ablation results constrain the mechanism story`.
- 可以写：`the ablation weakens a simple teacher-parameter-necessity interpretation`.
- 不要写：`teacher parameters are necessary`, `teacher branch causally explains the win`, `ablation closes the mechanism`.

## 3. FR6 multi-seed mechanism / intervention figure pack

| 项 | 当前锚点 |
| --- | --- |
| Figure root | `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/` |
| 直接数据 | `figure_data.csv`, `figure_manifest.json`, `caption.md`, rendered `svg/png` |
| Evidence pack | `docs/evidence_packs/mechanism_ablation/fr6_multi_seed_mechanism_intervention_figure_pack.md` |
| Review | `docs/review/T58_review.md` |
| 可写结论 | cross-seed 机制材料可作为 descriptive mechanism evidence |
| 必须保留的边界 | T58 review 有 provenance warning；seed-category logic 是任务内重建；不支持 intervention success 或 causal closure |

投稿稿中建议写法：

- 可以写：`cross-seed mechanism and intervention summaries remain descriptive`.
- 可以写：`the available intervention evidence motivates a hedged interpretation`.
- 不要写：`causal mechanism proven`, `intervention succeeded`, `failure mode solved`.

## 4. FR8 / statcalib extension lane

| 项 | 当前锚点 |
| --- | --- |
| Closure pack | `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md` |
| Review | `docs/review/T70_review.md` |
| Helper scope | `cnn_fpga/benchmark/build_fr8_statcalib_bounded_closure_pack.py` task-scoped consolidator |
| 可写结论 | affine contract 可容纳 non-neural statistical-calibration estimator，作为 supplement-side extension lane |
| 必须保留的边界 | `no_promotion_keep_extension_lane_only`; no unique threshold; no mature comparator; no `.tflite`; no real-board; no T24 table replacement |

投稿稿中建议写法：

- 可以写：`statistical calibration is retained as a separately labeled extension lane`.
- 可以写：`the result supports the generality of the affine contract, not a promoted comparator`.
- 不要写：`statcalib beats the main method`, `statcalib is mature`, `statcalib should replace T24`, `unique threshold found`.

## 5. Hardware / deployment placeholders

| 项 | 当前锚点 |
| --- | --- |
| `.tflite` runtime | `T48` isolated current-host true runtime evidence only |
| Real-board gate | `T49/T71/T72` read-only gate / regeneration / provenance packs |
| 当前 verdict | `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE` for current-host real-board path |
| 可写结论 | 只能作为 future measurement surface 或 placeholder protocol |
| 必须保留的边界 | 不支持 board execution success、hardware validation、deployment closure、source-vs-board agreement、latency/resource measured claim |

投稿稿中建议写法：

- 可以写：`hardware validation remains a future measurement surface`.
- 可以写：`the current manuscript keeps board-level results as explicit placeholders`.
- 不要写：`FPGA-validated`, `real-board HIL completed`, `deployment closure`, `measured FPGA latency`, `source-vs-board agreement`.

## 6. 当前稿件还缺的 source-data / supplement 材料

1. `T24` 主结果的机器可读 source-data bundle：每个 scenario/mode/repeat 的 row、seed policy、metric definition、config hash、artifact path。
2. 主结果不确定性：当前只有 `repeats=2` 的 mean/std；若要高质量投稿，至少需要更清楚的 paired uncertainty 或后续预声明 CI/更多 repeats 任务。
3. `FR7` ablation supplement：每个 mode 的 artifact provenance、feature set、teacher/no-teacher 解释边界。
4. `FR6` mechanism source-data：figure input rows、seed category derivation、T54/T55/T56 回链；最好补 frozen seed-category 表，减少任务内重建逻辑的脆弱性。
5. `FR8` statcalib supplement：closure pack、no-promotion gate、tie/no-unique-threshold 解释。
6. 硬件占位符替换条件：board logs、bitstream hash、DMA/MMIO evidence、fixed-point/source-vs-board vectors、latency/resource reports。缺这些之前，硬件图表只能保持 placeholder。
7. 参考文献与源数据映射：最终投稿前需要把 TeX 中的手写 bibliography 迁移到核对过的 BibTeX，并把图表 source data 路径与 citation/supporting 文件分开。

## 7. 审稿人视角的可信度判断

当前实验可信度适合支撑“bounded software-HIL + strict provenance”的稿件雏形，理由是：

- T24 有锁定协议、完整 coverage、review gate 和明确 forbidden scope；
- T57 / T58 / T70 都有单独 review，且边界没有被写成更强 claim；
- 当前稿件主动保留 `.tflite`、real-board、statcalib、mechanism causality 的限制。

当前实验还不足以支撑“强期刊正式投稿的最终结果层”，主要缺口是：

- 主结果 repeats 少，不足以给出强 statistical robustness；
- 机制材料是 descriptive，不是 causal proof；
- hardware / deployment 仍是占位符；
- expanded benchmark、stronger theoretical/classical baseline、holdout drift family 仍需后续预声明任务；
- source-data bundle 和 figure-generation provenance 还没有完全冻结。

因此，当前最稳妥的写作策略是：把投稿稿推进为一份结构清楚、claim 边界严格、source-data 缺口显式的 manuscript draft；在真正投稿前，再按本文清单补至少一轮 source-data / uncertainty / figure / reference freeze。
