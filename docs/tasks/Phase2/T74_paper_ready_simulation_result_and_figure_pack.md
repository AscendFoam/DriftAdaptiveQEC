# T74：论文可直接复用的仿真结果与图表打包

## 状态

- 由 Captain 在 `2026-06-11` 基于 `T73` 之后的主线需要提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：高强度 paper-material / figure-pack / traceability 打包任务

## 为什么现在做这个任务

在当前暂时没有 `Linux + FPGA` 硬件宿主的前提下，real-board execution 不是近期高收益主线。更紧迫的问题是：现有主线仿真证据虽然已经分散存在于 `T24/T50/T57/T58/T70/T72` 等任务里，但还没有被整理成一套论文可直接复用的结果表、图表、caption 与材料追溯包。

如果 `T73` 只完成了 claim/result/risk 三台账刷新，而没有进一步把“具体哪些图表能直接进文稿、每张图表的 caption 怎么写、每张图表的素材来自哪个 run/review/evidence pack”整理出来，那么后续 paper reopen 仍然会卡在材料层，而不是卡在文字层。

因此，`T74` 的目标不是新跑实验，也不是恢复 prose，而是把已经存在的主线仿真证据压缩成一套可直接服务论文写作的 simulation result / figure package。

## 前置条件

只有在 `T73` 已完成且其主线 claim/result/risk 三台账已形成统一入口后，`T74` 才可执行。若 `T73` 尚未完成，Worker 不得在 `T74` 中自建一套 competing ledger。

## 目标

在不运行任何新实验、不修改任何源码/配置/历史结果的前提下，完成以下七件事：

1. 形成一份主线仿真结果总包文档，给出论文主结果表与补充表的推荐取材方案。
2. 形成一份图表与 caption 总包文档，明确哪些图表已经 ready、哪些仍是 partial、哪些必须留在 supplement。
3. 在 task-scoped figure asset 目录下生成一套 paper-facing traceability 资产：manifest、source map、table snapshot、caption source map。
4. 把 `docs/paper_materials/README.md` 更新为“主台账 + 仿真结果包 + 图表包”的最新入口。
5. 给出一份 `submission-material gap checklist`，明确在不碰真板的情况下，论文材料还缺什么。
6. 形成一份 `main-text / appendix / supplement` 插入映射，明确每张主候选表/图应该放在哪一层。
7. 形成一份任务级 `submission bundle manifest`，把所有产物通过稳定 ID 串成一套可审计材料包。

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T74_paper_ready_simulation_result_and_figure_pack.md`
- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_simulation_result_table_pack.md`
- `docs/paper_materials/paper_figure_caption_pack.md`
- `docs/paper_materials/paper_submission_material_gap_checklist.md`
- `docs/paper_materials/paper_maintext_insertion_map.md`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/README.md`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/figure_manifest.json`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/result_source_map.csv`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/caption_source_map.csv`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/table_snapshot.csv`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/submission_bundle_manifest.json`
- `docs/review/T74_review.md`
- `docs/for_human/T74_explanation.md`
- `docs/worker_summary/T74_worker_summary.md`

## Docs To Update

Worker 必须更新：

- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_simulation_result_table_pack.md`
- `docs/paper_materials/paper_figure_caption_pack.md`
- `docs/paper_materials/paper_submission_material_gap_checklist.md`
- `docs/paper_materials/paper_maintext_insertion_map.md`
- `docs/review/T74_review.md`
- `docs/for_human/T74_explanation.md`
- `docs/worker_summary/T74_worker_summary.md`

并且必须新建 `docs/figure_assets/T74_paper_ready_simulation_result_pack/` 下的 traceability 资产。

## Forbidden Scope

Worker 不得：

- 修改 `docs/02_experiment_plan.md`
- 修改任何治理文档 `docs/00_*` 到 `docs/08_*`
- 修改任何 `cnn_fpga/`、`physics/`、`benchmark/`、`tests/` 源码或测试
- 修改任何 `runs/`、`artifacts/`、`docs/evidence_packs/` 历史事实文件
- 运行任何 benchmark、训练、`.tflite` smoke、real-board gate/regeneration/smoke
- 新开任何 theory branch、sidecar lane、benchmark expansion、paper prose 正文扩写
- 静默提升任何证据等级，尤其不得把 `T24` 写成 paper-grade expanded benchmark、把 `T48` 写成 deployment closure、把 `T70` 写成 mature statcalib comparator、把 `T72` 写成 real-board execution success

## 必须复用的输入

Worker 必须复用以下输入，而不是重写历史事实：

- 治理入口：
  - `README.md`
  - `docs/00_project_snapshot.md`
  - `docs/02_experiment_plan.md`
  - `docs/03_hil_p4_boundary_audit.md`
  - `docs/04_task_board.md`
  - `docs/07_handoff.md`
  - `docs/08_risks_and_open_questions.md`
- `T73` 刷新的主线 paper-facing 台账：
  - `docs/paper_materials/paper_claim_evidence_ledger.md`
  - `docs/paper_materials/paper_result_figure_ledger.md`
  - `docs/paper_materials/paper_claim_risk_table.md`
  - `docs/paper_materials/paper_ablation_result_pack.md`
- 关键主线 evidence/review：
  - `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv`
  - `docs/evidence_packs/mechanism_ablation/fr6_multi_seed_mechanism_intervention_figure_pack.md`
  - `docs/evidence_packs/mechanism_ablation/fr7_feature_teacher_ablation_reexecution.md`
  - `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md`
  - `docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md`
  - `docs/evidence_packs/deployment_boundary/t48_true_tflite_runtime_gate.md`
  - `docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md`
  - `docs/review/T24_review.md`
  - `docs/review/T25_p4_formal_evidence_gate_review.md`
  - `docs/review/T48_review.md`
  - `docs/review/T50_review.md`
  - `docs/review/T57_review.md`
  - `docs/review/T58_review.md`
  - `docs/review/T70_review.md`
  - `docs/review/T72_review.md`

## 固定边界

- 这是主线 paper-material 打包任务，不是实验任务
- 只允许“整理已有仿真证据”“产出 paper-facing materials”“补 traceability manifest”
- 不允许“新增 run”“重算结果”“替换 figure 结论”
- 任何结果/图表/caption 如果不能由现有证据直接支撑，必须写成 `partial`、`blocked`、`supplement-only` 或等价降级状态

## ID 与一致性约束

- 所有主候选表/图必须使用稳定 ID，推荐格式：
  - 表：`T74-TBL-01`、`T74-TBL-02` ...
  - 图：`T74-FIG-01`、`T74-FIG-02` ...
  - 边界/补充材料项：`T74-SUP-01`、`T74-SUP-02` ...
- 同一个 ID 必须在以下文件中保持一致：
  - `paper_simulation_result_table_pack.md`
  - `paper_figure_caption_pack.md`
  - `paper_maintext_insertion_map.md`
  - `figure_manifest.json`
  - `result_source_map.csv`
  - `caption_source_map.csv`
  - `table_snapshot.csv`
  - `submission_bundle_manifest.json`
- 不允许在正文包、traceability 资产和 gap checklist 中分别使用不同命名去指代同一项材料。

## 任务要求

### A. 产出主结果表总包

`docs/paper_materials/paper_simulation_result_table_pack.md` 至少要包含：

1. 论文主结果表候选：
   - `T24` frozen-set formal software revalidation 主表
   - `FR7` feature/teacher ablation 表
2. supplement / appendix 表候选：
   - `FR6` mechanism/intervention descriptive table
   - `FR8` statcalib extension-lane closure table
   - training / `.tflite` / real-board boundary tables
3. 每张表必须写明：
   - 表 ID
   - 建议放置位置：`main text` / `appendix` / `supplement only`
   - 来源：task / review / run root / evidence pack
   - 安全表述
   - 禁止表述

### B. 产出图表与 caption 总包

`docs/paper_materials/paper_figure_caption_pack.md` 至少要包含：

1. 每张图的唯一 ID
2. 推荐图题 / caption 草案
3. 图的核心 message
4. 直接证据来源
5. 当前状态：`ready` / `partial` / `blocked`
6. 若为 `partial` 或 `blocked`，必须说明缺口在哪里

最低必须覆盖：

- 冻结主结果对比图或主结果表替代说明
- `FR6` multi-seed mechanism/intervention figure
- `FR7` feature/teacher ablation figure 或表格替代说明
- `FR8` statcalib extension-lane figure 或 supplement-only 表述
- deployment boundary figure/table 的 caption 边界

### C.1 形成 main-text / appendix / supplement 插入映射

`docs/paper_materials/paper_maintext_insertion_map.md` 至少要包含：

1. 一个按 `main text`、`appendix`、`supplement only` 分层的材料清单。
2. 每个条目的稳定 ID、推荐标题、对应文稿位置、依赖证据、边界说明。
3. 至少覆盖以下六类材料中的各一项：
   - `T24` 主结果
   - `FR7` 特征/teacher 消融
   - `FR6` 机制/干预图或表
   - `FR8` statcalib extension-lane
   - training/material 边界
   - deployment boundary
4. 若某项暂时不能进入 `main text`，必须明确写出降级原因，而不是只写“待定”。

### C.2 生成 task-scoped traceability 资产

`docs/figure_assets/T74_paper_ready_simulation_result_pack/` 下必须生成：

1. `figure_manifest.json`
   - 列出所有 figure/table IDs、状态、来源文件、是否主文可用
2. `result_source_map.csv`
   - `result_id,source_task,source_review,source_run_or_pack,boundary_note`
3. `caption_source_map.csv`
   - `figure_or_table_id,caption_source,evidence_source,risk_or_boundary`
4. `table_snapshot.csv`
   - 给出论文主结果表与补充表的推荐快照索引，不要求复制全部原始数值，但必须能明确回链到原始数据来源
5. `README.md`
   - 解释该目录是什么、不是什麽、如何回链到原始证据
6. `submission_bundle_manifest.json`
   - 列出本任务全部交付件、稳定 ID 集合、每个交付件覆盖的材料范围、以及是否可直接进入 `main text`

### D. 形成 submission-material gap checklist

`docs/paper_materials/paper_submission_material_gap_checklist.md` 必须明确：

1. 在不等待真板的前提下，当前论文材料还缺哪些 simulation-side 组件
2. 哪些组件已经 `ready`
3. 哪些组件仍需 `partial` 补强
4. 哪些组件明确不能写
5. 哪些组件要等硬件条件变化后才能重新考虑

这份 checklist 不能把“缺真板”写成“当前论文不可写”；而要把它写成“当前主线先走 simulation/material-complete 路线”。

## 预期输出

Worker 必须产出：

- `docs/paper_materials/paper_simulation_result_table_pack.md`
- `docs/paper_materials/paper_figure_caption_pack.md`
- `docs/paper_materials/paper_submission_material_gap_checklist.md`
- `docs/paper_materials/paper_maintext_insertion_map.md`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/README.md`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/figure_manifest.json`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/result_source_map.csv`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/caption_source_map.csv`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/table_snapshot.csv`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/submission_bundle_manifest.json`
- 更新后的 `docs/paper_materials/README.md`
- `docs/review/T74_review.md`
- `docs/for_human/T74_explanation.md`
- `docs/worker_summary/T74_worker_summary.md`

## 验证

Worker 必须实际执行并报告：

1. `paper_simulation_result_table_pack.md` 中每张表是否都能回指到具体 task/review/run/evidence pack
2. `paper_figure_caption_pack.md` 中每张图的 caption 是否都能回指到具体 evidence path
3. `figure_manifest.json` / `result_source_map.csv` / `caption_source_map.csv` / `table_snapshot.csv` 之间的 ID 是否一致
4. `paper_maintext_insertion_map.md` 与 `submission_bundle_manifest.json` 是否使用同一套稳定 ID，并与上述四个 traceability 文件一致
5. `git diff --name-only -- runs`
6. `git diff --name-only -- artifacts`
7. `git diff --name-only -- cnn_fpga physics benchmark tests`
8. `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md`

## 完成标准

只有同时满足以下条件，`T74` 才可视为完成：

1. 已形成一套论文可直接复用的主线仿真结果/图表/caption/material pack
2. 所有主结果表与图表都有 traceable source map
3. `paper_maintext_insertion_map.md` 与 `submission_bundle_manifest.json` 已形成，并与 traceability 资产使用同一套稳定 ID
4. 没有把任何 blocked / partial / extension-lane / no-promotion / gate-only 证据静默升级
5. 没有改动治理文档、源码、测试、`runs/`、`artifacts/`
