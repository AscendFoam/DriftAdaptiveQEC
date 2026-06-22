# T90：训练链 clean-CPU 同机 repeated-run 一致性证据包

## 状态

- 由 Captain 于 `2026-06-15` 基于 `T89 -> PASS` closeout 提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：有界训练复现强化任务，包含 task-scoped helper、focused tests、3 次 clean CPU-only 同配置真实 train+eval rerun，以及一份 code-backed same-host repeat-consistency pack

## 为什么现在做这个任务

`T89` 已经把当前 mainline note/material 收紧成 frozen-mainline handoff、source-of-truth、post-freeze change-control 与 blocked-surface re-entry 条件。也就是说，当前主线的 paper-facing prose 已经故意冻结，不应继续在 `main` 上做无界扩写。

与此同时，当前最现实、最有价值、又不依赖 `Linux + FPGA` 硬件宿主的下一条证据强化路线，是 `R11`：

1. `T31` 证明了 CPU-only 训练 lane 在配置与依赖层面可行。
2. `T39` 证明了可以在 `DLEnv` 之外建立干净的 `Python 3.12` CPU-only 环境。
3. `T40` 证明了该环境能跑一次真实训练 smoke。
4. `T50` 进一步把这条链推进到“一次 bounded real train+eval rerun + 一份训练材料再生 pack”。

但 `T50` 仍然只是一组单次 rerun 证据。当前还缺一件更强的东西：

1. 在**同一宿主、同一解释器、同一配置、同一 seed** 下，连续多次 rerun 的一致性到底如何。
2. 哪些字段应该稳定一致，哪些字段允许只因时间戳/路径而变化。
3. 当前 clean CPU-only lane 最多能诚实支持到什么 reproducibility 说法，仍然不能支持到什么层级。

因此，`T90` 的目标不是跨主机 portability，也不是 full reproducibility closure，而是把 `T50` 的 “one bounded rerun” 推进成 “same-host repeated-run consistency pack”。

## 前置条件

只有以下条件全部满足时，`T90` 才可执行：

1. `T89` 已完成并通过 Captain `PASS`。
2. 执行前，当前主线已提交到 `main`，并且 Worker 所在执行目录是**clean committed main checkout**。
3. `T39/T50` 的 clean CPU-only 解释器仍可用，或能按既有文档在不修改仓库跟踪文件的前提下重建出同等环境。
4. 以下文件已存在：
   - `docs/evidence_packs/training_reproducibility/training_chain_portable_dependency_lock_plan.md`
   - `docs/evidence_packs/training_reproducibility/training_chain_cpu_cleanenv_bootstrap.md`
   - `docs/evidence_packs/training_reproducibility/training_chain_cpu_cleanenv_train_smoke.md`
   - `docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md`
   - `docs/review/T31_review.md`
   - `docs/review/T39_review.md`
   - `docs/review/T40_review.md`
   - `docs/review/T50_review.md`
   - `cnn_fpga/config/task_tmp/T50_static_theta_repro_pack.yaml`

如果这些前提不满足，Worker 不得在 `T90` 中补造上游事实，而必须如实汇报 blocker。

## 目标

在不改写 mainline note、不碰 theory 分支、不修改 benchmark/HIL/`.tflite`/real-board 边界、不改写 canonical historical artifacts 的前提下，完成以下工作：

1. 执行 **3 次** clean CPU-only、同配置、同 seed 的真实 train rerun。
2. 对这 3 次 train 产出的 model artifact 各执行 **1 次**真实 eval rerun。
3. 新增一个 task-scoped helper，把 3 组 train/eval/model 输出收成同一份 repeat-consistency pack。
4. 明确回答：当前 same-host clean CPU-only lane 是否呈现出 byte-identical / tensor-identical / metric-identical / metric-near-identical / honest-drift 之中的哪一类；如果有漂移，漂移发生在什么层级。
5. 把 strongest supported truth 收口到：
   - 当前 clean CPU-only same-host lane 至少有一份 repeated-run consistency audit；
   - 但仍不等于 cross-host、GPU/CUDA、Linux、`.tflite`、real-board 或 full training reproducibility closure。

## 当前推荐唯一结论方向

本任务的目标不是“强行证明完全一致”，而是**诚实测量并固定当前 same-host repeated-run 的 strongest supported truth**。

也就是说：

1. 如果 3 次 rerun 完全一致，应如实写成 current-host / current-interpreter / current-config / fixed-seed 下的 bounded consistency strengthened。
2. 如果只做到数值近似一致，也应如实写成 metric-near-identical 或 tensor-near-identical，而不是夸大成 deterministic closure。
3. 如果出现实质漂移，也不构成任务失败，只要 Worker 真实记录并把 supported/unsupported claims 收紧。

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T90_training_same_host_repeated_run_consistency_pack.md`
- `docs/evidence_packs/training_reproducibility/README.md`
- `docs/evidence_packs/training_reproducibility/training_same_host_repeated_run_consistency_pack.md`
- `docs/review/T90_review.md`
- `docs/for_human/T90_explanation.md`
- `docs/worker_summary/T90_worker_summary.md`
- `cnn_fpga/model/README.md`
- `cnn_fpga/model/build_training_repeat_consistency_pack.py`
- `tests/test_training_repeat_consistency_pack.py`
- `cnn_fpga/config/task_tmp/T90_static_theta_repeat_consistency.yaml`
- `artifacts/t90_training_repeat_consistency/`

说明：

- `artifacts/t90_training_repeat_consistency/` 是本任务唯一允许写入的 artifact 输出根目录。
- 若 Worker 需要重建 clean CPU-only 环境，可在仓库外或 `.venvs/` 下操作，但不得修改仓库跟踪文件。

## Docs To Update

Worker 必须更新：

- `docs/evidence_packs/training_reproducibility/README.md`
- `docs/evidence_packs/training_reproducibility/training_same_host_repeated_run_consistency_pack.md`
- `docs/review/T90_review.md`
- `docs/for_human/T90_explanation.md`
- `docs/worker_summary/T90_worker_summary.md`
- `cnn_fpga/model/README.md`

Worker 不得更新治理文档；Captain 会在 review 后统一更新。

## Forbidden Scope

Worker 不得：

- 修改 `docs/00_*` 到 `docs/08_*` 治理文档
- 修改 `docs/paper_notes/*`、`docs/paper_materials/*` 或任何 frozen-mainline handoff / change-control 文档
- 修改任何 theory 分支材料或把 theory 内容回写到 main
- 修改任何 `runs/` 下文件
- 修改 canonical historical artifact 目录：
  - `artifacts/datasets/static_theta_v2/`
  - `artifacts/models/static_theta_v2/`
  - `artifacts/reports/static_theta_v2/`
  - `artifacts/datasets/runtime_b_residual_v1/`
  - `artifacts/models/runtime_b_residual_v1/`
  - `artifacts/reports/runtime_b_residual_v1/`
- 修改 `requirements-recovery.txt`
- 修改 `requirements-train-cpu-win-py312.txt`
- 修改 `cnn_fpga/model/train.py`
- 修改 `cnn_fpga/model/evaluate.py`
- 修改 `cnn_fpga/model/export.py`
- 修改任何 benchmark、HIL、runtime、decoder 主线语义文件
- 运行 benchmark、HIL、`.tflite` runtime、real-board 或 sidecar 实验
- 把 same-host repeated-run evidence 写成 full training reproducibility、cross-host portability、GPU/CUDA portability、Linux portability、`.tflite` correctness、deployment closure 或真板正确性证明

## 必须复用的输入

Worker 必须复用以下既有输入，而不是重写历史事实：

- `docs/evidence_packs/training_reproducibility/training_chain_portable_dependency_lock_plan.md`
- `docs/evidence_packs/training_reproducibility/training_chain_cpu_cleanenv_bootstrap.md`
- `docs/evidence_packs/training_reproducibility/training_chain_cpu_cleanenv_train_smoke.md`
- `docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md`
- `docs/evidence_packs/training_reproducibility/README.md`
- `docs/review/T31_review.md`
- `docs/review/T39_review.md`
- `docs/review/T40_review.md`
- `docs/review/T50_review.md`
- `requirements-train-cpu-win-py312.txt`
- `artifacts/datasets/static_theta_v2/manifest.json`
- `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
- `artifacts/reports/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_train_report.json`
- `artifacts/datasets/runtime_b_residual_v1/manifest.json`
- `artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz`
- `artifacts/reports/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d_train_report.json`
- `cnn_fpga/config/task_tmp/T50_static_theta_repro_pack.yaml`
- `cnn_fpga/model/build_training_reproducibility_pack.py`

## 固定边界

- 主线分支：current main only
- 环境边界：clean CPU-only lane only
- 宿主边界：same host only
- 输出边界：只允许写 `artifacts/t90_training_repeat_consistency/`
- 证据边界：training repeated-run consistency only
- 非目标边界：不是 `.tflite` portability task，不是真板 task，不是 benchmark task，不是 mainline prose task，不是 theory-mergeback task

## 任务要求

### A. Repeat-consistency helper

新增一个 task-scoped helper：

- `cnn_fpga/model/build_training_repeat_consistency_pack.py`

它至少要完成以下工作：

1. 读取本轮 3 次 rerun 的：
   - train report
   - eval report
   - model artifact
2. 校验这 3 组输出是否同时满足：
   - same host / same interpreter lane
   - same dataset family
   - same config family
   - same seed
   - backend = `numpy`
   - device = `cpu`
   - 输出都位于 `artifacts/t90_training_repeat_consistency/`
3. 给出每次 rerun 的 run-level summary，至少包含：
   - train/val/test 关键指标
   - model path / report path
   - seed / backend / device
4. 给出 pairwise consistency summary，至少包含：
   - train/val/test 指标差值
   - model artifact file hash 或 tensor-level comparison summary
   - 哪些字段完全一致，哪些字段只因 timestamp/path 不同
5. 给出统一 pack，显式包含：
   - run inventory
   - pairwise comparison table
   - strongest supported same-host consistency claim
   - unsupported claims
   - interpretation boundary

### B. Focused tests

新增：

- `tests/test_training_repeat_consistency_pack.py`

测试至少覆盖：

1. synthetic 3-run inputs 下 helper 能构建预期 pack
2. 若 run outputs 混入 task root 外路径，helper 明确拒绝
3. 若 seed / backend / device / dataset family 不一致，helper 明确拒绝
4. 若缺少任一 train/eval/model pair，helper 明确拒绝

### C. Bounded clean CPU repeated reruns

Worker 必须在 `T39/T50` 的 clean CPU-only 环境中执行 **3 次**真实 train rerun：

1. 新增派生 config：
   - `cnn_fpga/config/task_tmp/T90_static_theta_repeat_consistency.yaml`
2. 基于：
   - `cnn_fpga/config/task_tmp/T50_static_theta_repro_pack.yaml`
3. 只允许覆盖：
   - `paths.model_dir = artifacts/t90_training_repeat_consistency/models/static_theta_v2`
   - `paths.report_dir = artifacts/t90_training_repeat_consistency/reports/static_theta_v2`
   - 如有必要，可显式补写与 `T50` 相同的 bounded 参数，但不得扩大规模
4. 固定训练规模应与 `T50` 保持同层级：
   - `max_train_samples = 2048`
   - `max_val_samples = 512`
   - `epochs = 5`
   - `patience = 3`
   - seed 维持 `20260319`

说明：

- 本任务的重点不是把训练做大，而是让 3 次 rerun 具有可比性。
- 若 Worker 发现需要再缩小规模才能保证 task bounded，也可等价缩小，但必须在文档中解释，并保持 3 次 rerun 配置完全一致。

### D. Bounded eval reruns

Worker 必须对 3 次 train 生成的 model artifact 各执行 **1 次**真实评估复跑：

1. 使用现有 `cnn_fpga.model.evaluate`
2. 输出必须仍落在：
   - `artifacts/t90_training_repeat_consistency/reports/static_theta_v2/`
3. 目标是形成 3 组 `train_report + eval_report + model artifact` 可比三元组

### E. 最终文档必须回答的问题

`docs/evidence_packs/training_reproducibility/training_same_host_repeated_run_consistency_pack.md` 至少要回答：

1. 3 次 rerun 是否真的发生在同一 clean CPU-only lane
2. 3 次 rerun 的训练与评估指标差异有多大
3. model artifact 是 byte-identical、tensor-identical、近似一致，还是存在诚实漂移
4. 哪些字段应该稳定，哪些字段只会因时间戳/路径不同而变化
5. 现在可以诚实地支持哪些 same-host reproducibility claims
6. 现在仍然不能支持哪些 claims

文档中必须包含一个紧凑表格，至少区分：

- `run_inventory`
- `pairwise_metric_deltas`
- `model_consistency`
- `supported_claims`
- `unsupported_claims`

## 预期输出

Worker 必须产出：

- `docs/evidence_packs/training_reproducibility/training_same_host_repeated_run_consistency_pack.md`
- `docs/evidence_packs/training_reproducibility/README.md` 的登记更新
- `docs/review/T90_review.md`
- `docs/for_human/T90_explanation.md`
- `docs/worker_summary/T90_worker_summary.md`
- `cnn_fpga/model/build_training_repeat_consistency_pack.py`
- `tests/test_training_repeat_consistency_pack.py`
- `cnn_fpga/config/task_tmp/T90_static_theta_repeat_consistency.yaml`
- `artifacts/t90_training_repeat_consistency/` 下的 3 组 rerun outputs 与 pack JSON

## 验证

Worker 必须实际执行并报告：

1. `python -m py_compile cnn_fpga/model/build_training_repeat_consistency_pack.py`
2. `python -m unittest tests.test_training_repeat_consistency_pack`
3. clean CPU-only 环境下的 3 次真实训练 rerun
4. clean CPU-only 环境下与之对应的 3 次真实评估 rerun
5. helper 的一次真实执行
6. 边界检查：
   - `git diff --name-only -- runs`
   - `git diff --name-only -- artifacts/models/static_theta_v2 artifacts/reports/static_theta_v2 artifacts/models/runtime_b_residual_v1 artifacts/reports/runtime_b_residual_v1`
   - `git diff --name-only -- requirements-recovery.txt requirements-train-cpu-win-py312.txt`

Worker 还必须显式报告：

1. clean CPU-only rerun 使用的解释器路径
2. 执行时所基于的 `main` 提交哈希
3. 3 组 rerun 生成的 model/report 路径
4. pairwise 指标差值摘要
5. model artifact 一致性判断摘要
6. 当前支持的 same-host reproducibility claims
7. 当前不支持的 claims

## Review No-Go Triggers

Reviewer 在以下任一情况应返回 `BLOCK`：

1. Worker 不是从 clean committed `main` 执行，且未如实停止
2. Worker 改写任何 canonical historical artifact 目录
3. Worker 少跑 train/eval rerun，或没有形成 3 组完整三元组，却写成已完成
4. Worker 把 `T90` 写成 full training reproducibility、cross-host portability、GPU/CUDA portability、Linux portability、`.tflite` correctness 或真板正确性证明
5. helper 没有做 run-to-run consistency comparison，只是机械罗列文件
6. Worker 越界修改 benchmark/HIL/runtime/decoder 主线代码

## Captain 备注

- 这是 `T89` 之后最合适的下一步，因为当前 mainline prose 已冻结，而 `T51/T52` 更大范围 paper reopen 仍然过早。
- 该任务比简单文档任务更强，要求 helper、tests、3 次真实训练复跑、3 次真实评估复跑、以及一份可审计的一致性证据包一起收口。
- 该任务即使结果不完全一致，也可以 honest pass；关键要求是**真实执行、真实比较、真实收紧结论**。
