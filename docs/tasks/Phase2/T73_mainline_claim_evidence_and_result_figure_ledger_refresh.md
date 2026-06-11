# T73：主线 claim/evidence 与 result/figure/risk 三台账刷新

## 状态

- 由 Captain 在 `2026-06-11` 基于 `T72` closeout 提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：高强度 docs-only 主线台账一致性刷新任务

## 为什么现在做这个任务

到 `T72` 为止，主线已经补齐了几类关键但彼此分散的证据：

1. `T48`：isolated current-host true `.tflite` runtime gate
2. `T50`：training reproducibility and material-regeneration pack
3. `T57/T58`：FR7/FR6 paper-facing ablation 与 figure pack
4. `T70`：FR8 statcalib bounded closure pack
5. `T72`：real-board transfer-pack provenance hardening

但当前 paper-facing 主台账仍存在明显滞后：

- `docs/paper_materials/paper_claim_evidence_ledger.md` 仍停留在较早状态，尚未吸收 `T48/T50/T70/T72`
- 仓库缺少一个当前主线可直接引用的 `result/figure ledger`
- 仓库缺少一个当前主线可直接引用的 `paper claim risk table`
- `docs/paper_materials/README.md` 也还没有把这些 post-recovery / post-T72 的主台账组织起来

因此，`T73` 的目标不是继续写论文 prose，也不是再开新实验，而是把主线 paper-facing 三套台账刷新到 post-`T72` 的一致状态，给后续任何 paper re-open 或 claim 审查提供统一入口。

## 目标

在不运行任何新实验、不修改任何源码/配置/历史结果的前提下，完成以下四件事：

1. 刷新 `paper_claim_evidence_ledger.md`
2. 新建 `paper_result_figure_ledger.md`
3. 新建 `paper_claim_risk_table.md`
4. 让 `paper_ablation_result_pack.md` 与 `docs/paper_materials/README.md` 同步反映新的主线材料结构和边界

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T73_mainline_claim_evidence_and_result_figure_ledger_refresh.md`
- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_claim_evidence_ledger.md`
- `docs/paper_materials/paper_ablation_result_pack.md`
- `docs/paper_materials/paper_result_figure_ledger.md`
- `docs/paper_materials/paper_claim_risk_table.md`
- `docs/review/T73_review.md`
- `docs/for_human/T73_explanation.md`
- `docs/worker_summary/T73_worker_summary.md`

## Docs To Update

Worker 必须更新：

- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_claim_evidence_ledger.md`
- `docs/paper_materials/paper_result_figure_ledger.md`
- `docs/paper_materials/paper_claim_risk_table.md`
- `docs/review/T73_review.md`
- `docs/for_human/T73_explanation.md`
- `docs/worker_summary/T73_worker_summary.md`

若 `paper_ablation_result_pack.md` 中存在与 post-`T72` 主台账冲突的状态描述，也必须同步修正。

## Forbidden Scope

Worker 不得：

- 修改 `docs/02_experiment_plan.md`
- 修改任何治理文档 `docs/00_*` 到 `docs/08_*`
- 修改任何 `cnn_fpga/`、`physics/`、`benchmark/`、`tests/` 源码或测试
- 修改任何 `runs/`、`artifacts/`、`docs/evidence_packs/` 历史事实文件
- 运行任何 benchmark、训练、`.tflite` smoke、real-board gate/regeneration/smoke
- 新开任何 sidecar lane、theory branch 内容整合或 paper prose 扩写
- 静默提升任何证据等级，尤其不得把 `T48` 写成 default-env / deployment closure、把 `T70` 写成 mature statcalib comparator、把 `T72` 写成 real-board execution success

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
- 当前 paper materials：
  - `docs/paper_materials/README.md`
  - `docs/paper_materials/paper_claim_evidence_ledger.md`
  - `docs/paper_materials/paper_ablation_result_pack.md`
- 历史 recovery 只读参考：
  - `docs/legacy_context/reality_recovery_2026-05/01_claim_evidence_table.md`
  - `docs/legacy_context/reality_recovery_2026-05/04_figure_and_result_ledger.md`
  - `docs/legacy_context/reality_recovery_2026-05/05_paper_claim_risk_table.md`
- 关键主线 evidence/review：
  - `docs/evidence_packs/mechanism_ablation/fr6_multi_seed_mechanism_intervention_figure_pack.md`
  - `docs/evidence_packs/mechanism_ablation/fr7_feature_teacher_ablation_reexecution.md`
  - `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md`
  - `docs/evidence_packs/deployment_boundary/t48_true_tflite_runtime_gate.md`
  - `docs/evidence_packs/deployment_boundary/t49_real_board_smoke_execution_gate.md`
  - `docs/evidence_packs/deployment_boundary/t71_real_board_gate_regeneration_pack.md`
  - `docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md`
  - `docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md`
  - `docs/review/T48_review.md`
  - `docs/review/T50_review.md`
  - `docs/review/T57_review.md`
  - `docs/review/T58_review.md`
  - `docs/review/T70_review.md`
  - `docs/review/T72_review.md`

## 固定边界

- 这是主线 docs-only 台账任务，不是实验任务
- 所有状态只能来自已有 review / evidence pack / run root / artifact / governance risk
- 允许“刷新表述”“补齐索引”“新建主台账”，不允许“发明结果”
- 任何 claim/status 如果不能由现有证据直接支撑，必须写成 `partial`、`blocked`、`deferred` 或等价降级状态

## 任务要求

### A. Claim/Evidence 台账必须刷新到 post-T72 状态

`docs/paper_materials/paper_claim_evidence_ledger.md` 至少要完成以下刷新：

1. `.tflite` 条目必须吸收 `T48`
   - 明确是 isolated current-host true runtime 已确认
   - 不能写成 default env / cross-host / deployment closure
2. training/material 条目必须吸收 `T50`
   - 明确 bounded train+eval rerun 与 canonical material pack 已存在
   - 不能写成 full reproducibility closure
3. `statcalib` 条目必须吸收 `T70`
   - 明确 FR8 已形成 bounded closure pack
   - 但仍必须保留 `no-promotion` / extension-lane / no mature comparator 的边界
4. real-board 条目必须吸收 `T72`
   - 明确 checked-in、read-only、role-aware、可 replay/regeneration 的 gate/transfer-pack 已更严谨
   - 仍不能写成 execution success、hardware validated 或 `T37` 解锁

若原 claim ID 已不足以承载现状，可拆分或缩窄，但必须显式写明 `superseded` / `split` / `narrowed`，不得静默复用旧 ID。

### B. 新建 result/figure ledger

`docs/paper_materials/paper_result_figure_ledger.md` 至少要包含：

1. 每个 paper-facing figure/table/result pack 的唯一 ID
2. 对应 claim 或问题
3. 对应 evidence pack / review / run root / figure asset / table.csv
4. 当前状态：`ready` / `partial` / `blocked`
5. 不能外推的边界

最低必须覆盖：

- T24 frozen-set formal software revalidation table
- FR6 mechanism/intervention figure pack
- FR7 feature/teacher ablation table
- FR8 statcalib bounded closure pack
- training/material reproducibility boundary table
- deployment boundary table（true `.tflite` / gate / real-board non-claim）

### C. 新建 paper claim risk table

`docs/paper_materials/paper_claim_risk_table.md` 必须把当前 paper-facing claim area 与活动风险重新映射：

- 机制解释
- training reproducibility
- `.tflite` runtime / deployment
- real-board / HIL / host-transfer
- statcalib comparator
- expanded benchmark / paper-grade completion

要求：

1. 明确记录 `R31` 已收口
2. 明确记录新的残余风险（若 `T72 review` closeout 产生新的 carry-forward risk，则要按治理文档口径写入）
3. 每一项都要给出“安全写法”和“禁止写法”

### D. README 与材料索引必须同步

`docs/paper_materials/README.md` 必须反映：

- 当前目录中的主台账入口
- 哪些文件是 claim ledger / result ledger / risk table / ablation pack
- 哪些只是草稿或背景材料

若 `paper_ablation_result_pack.md` 的状态描述仍停留在 `pre-T70` 或 `pre-T72` 口径，必须做最小同步修正，但不得把它扩写成新的 paper prose。

## 预期输出

Worker 必须产出：

- `docs/paper_materials/paper_claim_evidence_ledger.md` 的刷新版
- `docs/paper_materials/paper_result_figure_ledger.md`
- `docs/paper_materials/paper_claim_risk_table.md`
- 必要时更新后的 `docs/paper_materials/paper_ablation_result_pack.md`
- 更新后的 `docs/paper_materials/README.md`
- `docs/review/T73_review.md`
- `docs/for_human/T73_explanation.md`
- `docs/worker_summary/T73_worker_summary.md`

## 验证

Worker 必须实际执行并报告：

1. claim ledger 中每个 `supported / partial / blocked` 条目是否都能回指到具体 evidence path
2. result/figure ledger 中每个 `ready / partial / blocked` 条目是否都能回指到具体 figure/table/run/review
3. risk table 中每条 paper-facing 风险是否都能回指到具体 `R*` 或 review warning
4. `git diff --name-only -- runs`
5. `git diff --name-only -- artifacts`
6. `git diff --name-only -- cnn_fpga physics benchmark tests`
7. `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md`

## 完成标准

只有同时满足以下条件，`T73` 才可视为完成：

1. post-`T72` 的主线 paper-facing 三台账已经形成统一入口
2. `.tflite`、training/material、statcalib、real-board 四类近期变化都已被正确吸收
3. 没有把任何 blocked / partial / extension-lane / no-promotion 证据静默升级
4. 没有改动治理文档、源码、测试、`runs/`、`artifacts/`
