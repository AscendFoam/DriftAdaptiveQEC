# T75：主文 Results 段落与最终成图 authoring pack

## 状态

- 由 Captain 在 `2026-06-12` 基于 `T74` 完成后的主线需要提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：高强度 paper-facing authoring / final-figure / bounded prose 打包任务

## 为什么现在做这个任务

`T74` 已经把当前主线 simulation/material 证据整理成一套可追溯、可插入、可审计的 stable-ID 材料包，但它仍然停留在“作者入口层”，还不是“作者可以直接落笔和定图层”。

更具体地说，`T74` 已经回答了：

- 哪些表和图可以进入主文、附录、补充材料；
- 每个 stable ID 的证据来源和安全边界是什么；
- 当前哪些项是 `ready`、`partial`、`blocked`。

但它还没有最终回答：

- 主文 Results 段落到底应该怎么写，才能既清楚又不越界；
- 主文和附录的最终图到底长什么样；
- caption、标题、placement、替代表述和“绝对不能写的话”如何一次锁定。

因此，`T75` 的目标不是新跑实验，也不是直接恢复 full-manuscript，而是把 `T74` 的 stable-ID 材料包压缩成一套真正可用于论文主结果写作与成图的 authoring pack。

## 前置条件

只有在 `T74` 已完成且其 stable-ID result/caption/insertion/traceability 包已经形成统一入口后，`T75` 才可执行。若 `T74` 尚未完成，Worker 不得在 `T75` 中重新定义 main-text 路线。

## 目标

在不运行任何新实验、不修改任何源码/配置/历史结果的前提下，完成以下六件事：

1. 形成一份 bounded 的主文 Results authoring pack，明确主文段落如何围绕 `T74` stable IDs 组织。
2. 形成一份 caption / placement lock 文档，锁定主文和附录图表的推荐标题、caption 核心句、位置和替代表述。
3. 形成一份 appendix bridge pack，说明主文结果如何自然过渡到附录与补充材料。
4. 形成一份 do-not-write guardrail 清单，明确哪些话当前绝对不能写。
5. 在 task-scoped figure asset 目录下产出至少三份 publication-facing、可直接引用的最终成图资产。
6. 用 manifest/source map 把 `T75` 产出的成图和 prose 再回链到 `T74` stable IDs 与原始证据路径。

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T75_maintext_results_prose_and_final_figure_authoring_pack.md`
- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_maintext_results_authoring_pack.md`
- `docs/paper_materials/paper_caption_lock_and_placement_notes.md`
- `docs/paper_materials/paper_appendix_bridge_pack.md`
- `docs/paper_materials/paper_authoring_do_not_write_list.md`
- `docs/figure_assets/T75_maintext_results_authoring_pack/README.md`
- `docs/figure_assets/T75_maintext_results_authoring_pack/authoring_manifest.json`
- `docs/figure_assets/T75_maintext_results_authoring_pack/asset_source_map.csv`
- `docs/figure_assets/T75_maintext_results_authoring_pack/t75_fig_m01_t24_frozen_summary.svg`
- `docs/figure_assets/T75_maintext_results_authoring_pack/t75_fig_m02_fr6_multi_seed_mechanism.svg`
- `docs/figure_assets/T75_maintext_results_authoring_pack/t75_fig_a01_boundary_schematic.svg`
- `docs/review/T75_review.md`
- `docs/for_human/T75_explanation.md`
- `docs/worker_summary/T75_worker_summary.md`

## Docs To Update

Worker 必须更新：

- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_maintext_results_authoring_pack.md`
- `docs/paper_materials/paper_caption_lock_and_placement_notes.md`
- `docs/paper_materials/paper_appendix_bridge_pack.md`
- `docs/paper_materials/paper_authoring_do_not_write_list.md`
- `docs/review/T75_review.md`
- `docs/for_human/T75_explanation.md`
- `docs/worker_summary/T75_worker_summary.md`

并且必须新建 `docs/figure_assets/T75_maintext_results_authoring_pack/` 下的 authoring 资产。

## Forbidden Scope

Worker 不得：

- 修改 `docs/02_experiment_plan.md`
- 修改任何治理文档 `docs/00_*` 到 `docs/08_*`
- 修改任何 `cnn_fpga/`、`physics/`、`benchmark/`、`tests/` 源码或测试
- 修改任何 `runs/`、`artifacts/`、`docs/evidence_packs/` 历史事实文件
- 运行任何 benchmark、训练、`.tflite` smoke、real-board gate/regeneration/smoke
- 修改任何 theory branch 材料、`docs/paper_notes/` 中的 full-manuscript 草稿或 sidecar lane 文档
- 直接恢复 full-manuscript 扩写、摘要/引言/related work 全文撰写或投稿包总装
- 静默提升任何证据等级，尤其不得把 `T24` 写成 paper-grade expanded benchmark、把 `T48` 写成 deployment closure、把 `T72` 写成 real-board execution success、把 `T70` 写成 mature statcalib comparator

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
- `T74` 的主线 paper-facing 材料包：
  - `docs/paper_materials/paper_simulation_result_table_pack.md`
  - `docs/paper_materials/paper_figure_caption_pack.md`
  - `docs/paper_materials/paper_maintext_insertion_map.md`
  - `docs/paper_materials/paper_submission_material_gap_checklist.md`
  - `docs/figure_assets/T74_paper_ready_simulation_result_pack/figure_manifest.json`
  - `docs/figure_assets/T74_paper_ready_simulation_result_pack/result_source_map.csv`
  - `docs/figure_assets/T74_paper_ready_simulation_result_pack/caption_source_map.csv`
  - `docs/figure_assets/T74_paper_ready_simulation_result_pack/table_snapshot.csv`
  - `docs/figure_assets/T74_paper_ready_simulation_result_pack/submission_bundle_manifest.json`
- 关键主线 evidence/review：
  - `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv`
  - `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/fr6_multi_seed_mechanism_intervention.svg`
  - `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/figure_data.csv`
  - `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/caption.md`
  - `docs/evidence_packs/mechanism_ablation/fr6_multi_seed_mechanism_intervention_figure_pack.md`
  - `docs/evidence_packs/mechanism_ablation/fr7_feature_teacher_ablation_reexecution.md`
  - `docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md`
  - `docs/evidence_packs/deployment_boundary/t48_true_tflite_runtime_gate.md`
  - `docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md`
  - `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md`
  - `docs/review/T24_review.md`
  - `docs/review/T48_review.md`
  - `docs/review/T50_review.md`
  - `docs/review/T57_review.md`
  - `docs/review/T58_review.md`
  - `docs/review/T70_review.md`
  - `docs/review/T72_review.md`
  - `docs/review/T74_review.md`

## 固定边界

- 这是主线 paper-facing authoring 任务，不是实验任务
- 只允许“基于现有证据写主文 Results 段落”“产出 task-local 最终成图资产”“锁定 caption 和 placement”
- 不允许“新增 run”“重算结果”“替换主线结论”“补新的 paper claim”
- `T74-FIG-04` 的 blocked 状态必须保留；`T75` 不得试图把它画成“统一 portability / deployment closure 图”
- 任何主文 prose 或图表如果不能由现有证据直接支撑，必须写成 `partial`、`blocked`、`supplement-only` 或等价降级状态

## ID 与一致性约束

- `T75` 必须显式回链到 `T74` stable IDs，而不是重新发明一套脱离 `T74` 的材料命名。
- 允许为 task-local 成图资产新增 `T75` 资产 ID，推荐格式：
  - `T75-FIG-M01`：主文 figure
  - `T75-FIG-M02`：主文 figure
  - `T75-FIG-A01`：附录/边界 figure
- 每个 `T75` 资产 ID 必须在 `authoring_manifest.json` 中明确映射到一个或多个上游 `T74-*` stable IDs。
- 同一个 `T75` 资产 ID 必须在以下文件中保持一致：
  - `paper_maintext_results_authoring_pack.md`
  - `paper_caption_lock_and_placement_notes.md`
  - `paper_appendix_bridge_pack.md`
  - `authoring_manifest.json`
  - `asset_source_map.csv`

## 任务要求

### A. 产出主文 Results authoring pack

`docs/paper_materials/paper_maintext_results_authoring_pack.md` 至少要包含：

1. 一个明确的主文结果路线：
   - 主文主表/主图入口必须围绕 `T74-TBL-01`、`T74-FIG-02`
   - `T75-FIG-M01` 若被成功 authoring，应明确其与 `T74-TBL-01` 的关系
   - 若 `T75-FIG-M01` 仍不够诚实或不够稳，则必须明确 `T74-TBL-01` 仍是 authoritative substitute
2. 至少三段 bounded Results prose：
   - 主结果段
   - 机制/解释段
   - 边界/部署限制段
3. 每段 prose 必须写明：
   - 上游 `T74-*` stable IDs
   - 可写表述
   - 不可写表述
4. 该文档以中文 authoring pack 为主；如有必要，可附极短英文句胚，但不得扩展成 full-manuscript。

### B. 产出 caption / placement lock

`docs/paper_materials/paper_caption_lock_and_placement_notes.md` 至少要包含：

1. 每个 `T75` 成图资产的标题草案
2. 每个资产的一句主 caption 核心句
3. 建议放置位置：`main text` / `appendix`
4. 推荐尺寸、横竖版式或图例布局说明
5. 如果某个上游 `T74` 项不适合画成最终图，必须明确保留表格替代方案

### C. 产出 appendix bridge pack

`docs/paper_materials/paper_appendix_bridge_pack.md` 至少要包含：

1. 主文到附录的过渡逻辑
2. `T74-TBL-02` 到 `T74-TBL-05`、`T74-FIG-03` 的附录放置建议
3. `T74-TBL-06`、`T74-TBL-07`、`T74-SUP-*` 的 supplement-only 保留理由
4. 哪些内容必须留在附录/补充材料，不能挤进主文

### D. 产出 do-not-write guardrail

`docs/paper_materials/paper_authoring_do_not_write_list.md` 至少要覆盖：

1. `T48` 不得写成 default-env / HIL / deployment closure
2. `T49/T71/T72` 不得写成 real-board execution success
3. `FR8` 不得写成 promoted mature comparator 或唯一阈值
4. `T74-FIG-04` 仍是 blocked，不得画成真实 closure 图
5. 不得把 `FR6/FR7` 写成 causal closure

### E. 生成 task-scoped 最终成图资产

`docs/figure_assets/T75_maintext_results_authoring_pack/` 下必须生成：

1. `t75_fig_m01_t24_frozen_summary.svg`
   - 基于 `T24` frozen-set 主结果的 task-local 最终成图资产
   - 必须回链到 `T74-TBL-01`
2. `t75_fig_m02_fr6_multi_seed_mechanism.svg`
   - 基于 `FR6` 现有六 seed 图与数据的 task-local 主文 figure 资产
   - 必须回链到 `T74-FIG-02` / `T74-TBL-03`
3. `t75_fig_a01_boundary_schematic.svg`
   - 一个附录/边界说明图
   - 必须回链到 `T74-FIG-03`、`T74-TBL-05`、`T74-TBL-06` 或相关 `T74-SUP-*`
4. `authoring_manifest.json`
   - 列出所有 `T75` 资产 ID、对应上游 `T74` IDs、placement、status、输出文件和边界说明
5. `asset_source_map.csv`
   - `t75_asset_id,upstream_t74_id,source_path,role,boundary_note`
6. `README.md`
   - 解释该目录是什么、不是什麽、如何回链到 `T74` stable IDs 与原始证据

成图要求：

- 必须是 publication-facing 资产，而不是占位框或纯文字 note
- 三张图之间应保持统一视觉风格、字号层级和图例/注释习惯
- 若某图无法诚实完成，必须在 prose/caption lock 中明确退回表格替代方案，而不是偷偷删掉

## 预期输出

Worker 必须产出：

- `docs/paper_materials/paper_maintext_results_authoring_pack.md`
- `docs/paper_materials/paper_caption_lock_and_placement_notes.md`
- `docs/paper_materials/paper_appendix_bridge_pack.md`
- `docs/paper_materials/paper_authoring_do_not_write_list.md`
- `docs/figure_assets/T75_maintext_results_authoring_pack/README.md`
- `docs/figure_assets/T75_maintext_results_authoring_pack/authoring_manifest.json`
- `docs/figure_assets/T75_maintext_results_authoring_pack/asset_source_map.csv`
- `docs/figure_assets/T75_maintext_results_authoring_pack/t75_fig_m01_t24_frozen_summary.svg`
- `docs/figure_assets/T75_maintext_results_authoring_pack/t75_fig_m02_fr6_multi_seed_mechanism.svg`
- `docs/figure_assets/T75_maintext_results_authoring_pack/t75_fig_a01_boundary_schematic.svg`
- 更新后的 `docs/paper_materials/README.md`
- `docs/review/T75_review.md`
- `docs/for_human/T75_explanation.md`
- `docs/worker_summary/T75_worker_summary.md`

## 验证

Worker 必须实际执行并报告：

1. `authoring_manifest.json` 是否能解析，并且每个 `T75` 资产都映射到上游 `T74-*` stable IDs
2. `asset_source_map.csv` 中的 `t75_asset_id` 是否与 manifest 一致
3. 三个 `.svg` 文件是否真实存在，且文件内容包含 `<svg`
4. `paper_maintext_results_authoring_pack.md` 中每个主段落是否都引用至少一个上游 `T74-*` stable ID，并明确写出 safe / forbidden wording
5. `paper_authoring_do_not_write_list.md` 是否覆盖 `T48`、`T72`、`FR8`、`T74-FIG-04`
6. `git diff --name-only -- runs`
7. `git diff --name-only -- artifacts`
8. `git diff --name-only -- cnn_fpga physics benchmark tests`
9. `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md`

## 完成标准

只有同时满足以下条件，`T75` 才可视为完成：

1. 已形成一套 bounded 的主文 Results prose + final figure authoring pack
2. 所有 `T75` 成图资产都能 trace back 到 `T74` stable IDs 和原始证据路径
3. caption / placement / appendix bridge / do-not-write guardrail 已形成统一入口
4. 没有把任何 blocked / partial / extension-lane / no-promotion / gate-only 证据静默升级
5. 没有改动治理文档、源码、测试、`runs/`、`artifacts/`
