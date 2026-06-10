# T48：真实 `.tflite` runtime smoke gate 与 preserved-artifact 一致性包

## 状态

- 由 Captain 于 `2026-06-10` 在 `T50` closeout 后提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：有界 `.tflite` runtime gate 任务，包含环境真值核验、preserved artifact 选择、真实 `.tflite` eval/validate execution、task-scoped helper、focused tests 和显式 gate verdict

## 为什么现在做这个任务

`T18` 只完成了 `.tflite` 路径的 manifest/boundary 文档化，结论仍然是：

1. 代码入口存在；
2. `tflite_stub_v1` 是明确回退路径；
3. 当前机器上还没有“真实 `.tflite` runtime 已可用”的证据。

`T50` 之后，主线又多了两类关键前提：

1. 我们现在有一份 code-backed 训练材料与 preserved reference 证据包：
   - `docs/training_reproducibility_and_material_regeneration_pack.md`
   - `artifacts/t50_training_repro_pack/training_reproducibility_pack.json`
2. 这份 pack 已明确枚举出 canonical `static_theta_v2` 下真实存在的：
   - float `.npz`
   - int8 `.npz`
   - 多个 `.tflite`
   - 多个 export/eval/validate-export 报告

因此，当前主线最小且更有信息量的下一步，已经不再是继续补训练材料，而是对“真实 `.tflite` runtime 到底能不能在当前机器上落地”做一次有界 gate。

这个 gate 任务必须同时回答四个问题：

1. 当前机器能否提供真实 TFLite 解释器，而不是 stub？
2. preserved historical `.tflite` artifacts 中，是否存在可直接执行的 float / int8 候选？
3. 如果真实 runtime 可用，float / int8 `.tflite` 是否能在 test split 上真实评估？
4. `.tflite` 预测与源 artifact 预测之间的偏差，在一个有界样本集上到底是什么量级？

## 目标

产出一份真实 `.tflite` runtime gate 包，明确收口到下列其中一种结论：

1. `GO_TRUE_TFLITE_RUNTIME_FLOAT_AND_INT8`
2. `GO_TRUE_TFLITE_RUNTIME_FLOAT_ONLY`
3. `NO_GO_TRUE_TFLITE_RUNTIME_UNAVAILABLE`
4. `NO_GO_PRESERVED_TFLITE_ARTIFACT_INVALID_OR_STUB_ONLY`

说明：

- 这里的 `GO / NO_GO` 是任务内部的 gate verdict，不是 reviewer verdict。
- 如果 worker 以真实证据得出 `NO_GO`，该任务仍可完成并通过 review；关键是必须真实、可复核、不能冒充成功。

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T48_true_tflite_runtime_smoke_gate.md`
- `docs/t48_true_tflite_runtime_gate.md`
- `docs/review/T48_review.md`
- `docs/for_human/T48_explanation.md`
- `docs/worker_summary/T48_worker_summary.md`
- `cnn_fpga/model/build_t48_true_tflite_runtime_gate.py`
- `tests/test_t48_true_tflite_runtime_gate.py`
- `cnn_fpga/config/task_tmp/T48_static_theta_tflite_gate.yaml`
- `requirements-tflite-win-py311.txt`
- `artifacts/t48_true_tflite_runtime_gate/`

说明：

- `artifacts/t48_true_tflite_runtime_gate/` 是本任务唯一允许写入的 artifact 输出根目录。
- `.venvs/t48_*` 等临时环境目录若需要创建，可作为未跟踪本地环境存在，但不得作为仓库跟踪文件加入提交。

## Docs To Update

Worker 必须更新：

- `docs/t48_true_tflite_runtime_gate.md`
- `docs/review/T48_review.md`
- `docs/for_human/T48_explanation.md`
- `docs/worker_summary/T48_worker_summary.md`

Worker 不得更新治理文档；Captain 会在 review 后统一更新。

## Forbidden Scope

Worker 不得：

- 修改 `docs/02_experiment_plan.md`
- 修改任何治理文档 `docs/00_*` 到 `docs/08_*`
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
- 修改 `cnn_fpga/model/export.py`
- 修改 `cnn_fpga/model/evaluate_tflite.py`
- 修改 `cnn_fpga/model/validate_export.py`
- 修改 `cnn_fpga/runtime/inference_service.py`
- 修改任何 benchmark、HIL、runtime、decoder 主线语义文件
- 运行 benchmark、HIL、real-board 或 sidecar 实验
- 使用 `.tflite.json` stub manifest 冒充真实 `.tflite` runtime 成功
- 把本任务写成 `.tflite` 路径已与 HIL/board 完整闭环的证明

## 必须复用的输入

Worker 必须复用以下既有输入，而不是重写历史事实：

- `docs/TFLite_runtime_bootstrap.md`
- `docs/tasks/Phase2/T18_tflite_manifest_and_smoke_plan.md`
- `docs/training_reproducibility_and_material_regeneration_pack.md`
- `artifacts/t50_training_repro_pack/training_reproducibility_pack.json`
- `docs/review/T18_review.md`
- `docs/review/T50_review.md`
- `cnn_fpga/model/evaluate_tflite.py`
- `cnn_fpga/model/validate_export.py`
- `cnn_fpga/runtime/inference_service.py`
- `cnn_fpga/config/experiment_static_theta_v2.yaml`
- canonical `static_theta_v2` 下真实存在的：
  - float `.npz`
  - int8 `.npz`
  - float `.tflite`
  - int8 `.tflite`
  - export / eval / validate-export 报告

## 固定边界

- 主线分支：当前 mainline experiment branch only
- 证据边界：`.tflite` runtime truth only
- 数据边界：`static_theta_v2` canonical materials only
- 输出边界：只允许写 `artifacts/t48_true_tflite_runtime_gate/`
- 非目标边界：不是训练任务，不是真板任务，不是 benchmark 任务，不是 paper prose 任务，不是 sidecar 任务

## 任务要求

### A. 真实 runtime 环境真值核验

Worker 必须明确记录一个“真实 `.tflite` 解释器环境”：

1. 优先方式：
   - 创建或复用一个隔离环境，且其解释器能真实 import：
     - `tensorflow`
     - 或 `tflite_runtime`
2. 需要产出：
   - 解释器路径
   - Python 版本
   - 实际 import 成功的包名与版本
3. 若真实 runtime 环境始终无法建立：
   - 不得伪造成功
   - 必须把失败事实写入 gate 文档与 summary JSON
   - 可给出 `NO_GO_TRUE_TFLITE_RUNTIME_UNAVAILABLE`

`requirements-tflite-win-py311.txt` 必须作为本任务的最小环境清单输出，记录本轮真实 runtime gate 使用的最小包集合。

### B. preserved artifact 选择

Worker 必须基于 `T50` pack 和 canonical `static_theta_v2` 目录，确定一组可复核的 preserved artifact 组合：

1. 一组 float 对照：
   - source float artifact `.npz`
   - true `.tflite` artifact
2. 一组 int8 对照：
   - source int8 artifact `.npz`
   - true `.tflite` artifact

要求：

1. 选择规则必须写入文档和 helper 输出。
2. 必须显式拒绝 `.tflite.json` stub。
3. 若只有 float 可真实执行，而 int8 不能，则必须明确写成 `GO_TRUE_TFLITE_RUNTIME_FLOAT_ONLY`，不能含糊写成“全部通过”。

### C. 派生 gate 配置

新增一个 task-scoped config：

- `cnn_fpga/config/task_tmp/T48_static_theta_tflite_gate.yaml`

基于：

- `cnn_fpga/config/experiment_static_theta_v2.yaml`

仅允许覆盖：

- `paths.report_dir = artifacts/t48_true_tflite_runtime_gate/reports/static_theta_v2`
- 必要时覆盖 `paths.model_dir` 到 `artifacts/t48_true_tflite_runtime_gate/models/static_theta_v2`
- `evaluation.target_split`

说明：

- 默认 `target_split = test`
- 本任务不要求重新训练，也不要求重新导出 canonical artifact；若 preserved `.tflite` 本身可执行，则优先复用 preserved artifact

### D. task-scoped helper

新增一个 task-scoped helper：

- `cnn_fpga/model/build_t48_true_tflite_runtime_gate.py`

它至少要完成以下工作：

1. 读取 `T50` pack
2. 读取本轮环境探测结果
3. 校验选中的 float / int8 `.tflite` 候选不是 `.json` stub
4. 读取本轮 `evaluate_tflite` 与 `validate_export` 的报告
5. 生成统一 gate JSON，总结：
   - 环境真值
   - preserved artifact 选择
   - float gate 结果
   - int8 gate 结果
   - source-vs-tflite 偏差摘要
   - 最终 gate verdict

### E. focused tests

新增：

- `tests/test_t48_true_tflite_runtime_gate.py`

测试至少覆盖：

1. helper 能拒绝 `.tflite.json` stub 候选
2. helper 在 float 成功 / int8 缺失时给出 `GO_TRUE_TFLITE_RUNTIME_FLOAT_ONLY`
3. helper 在无真实 runtime import 证据时给出 `NO_GO_TRUE_TFLITE_RUNTIME_UNAVAILABLE`
4. helper 在缺关键报告文件时明确拒绝

### F. 真实 `.tflite` 执行面

若真实 runtime 环境可用，Worker 必须实际执行以下最小有界执行：

1. 一次 float `.tflite` 真实评估：
   - `python -m cnn_fpga.model.evaluate_tflite --config ... --split test --tflite-path <float_tflite>`
2. 一次 float artifact-vs-`.tflite` 一致性验证：
   - `python -m cnn_fpga.model.validate_export --config ... --artifact-path <float_npz> --tflite-path <float_tflite> --split test --max-samples 128`
3. 若 int8 preserved pair 存在且可执行，再做：
   - 一次 int8 `.tflite` 真实评估
   - 一次 int8 artifact-vs-`.tflite` 一致性验证

说明：

- `max-samples=128` 是建议值，可在 64~256 内等价调整，但必须在文档中说明。
- 本任务重点是“真实 runtime truth + bounded consistency”，不是追求大规模 `.tflite` benchmark。

### G. 最终文档必须回答的问题

`docs/t48_true_tflite_runtime_gate.md` 至少要回答：

1. 当前机器上是否真的有可用的 `.tflite` 解释器环境
2. 实际使用了哪个解释器、哪个包（`tensorflow` 或 `tflite_runtime`）
3. 选中了哪些 preserved float / int8 artifact 对
4. float `.tflite` 是否真实执行成功
5. int8 `.tflite` 是否真实执行成功
6. artifact-vs-`.tflite` 的最大绝对误差 / 平均绝对误差是什么
7. 最终 gate verdict 是什么
8. 当前仍然不能支持哪些 `.tflite` / deployment claims

文档中必须包含一个紧凑表格，至少区分：

- `environment_truth`
- `artifact_selection`
- `float_runtime_result`
- `int8_runtime_result`
- `supported_claims`
- `unsupported_claims`

## 预期输出

Worker 必须产出：

- `docs/t48_true_tflite_runtime_gate.md`
- `docs/review/T48_review.md`
- `docs/for_human/T48_explanation.md`
- `docs/worker_summary/T48_worker_summary.md`
- `cnn_fpga/model/build_t48_true_tflite_runtime_gate.py`
- `tests/test_t48_true_tflite_runtime_gate.py`
- `cnn_fpga/config/task_tmp/T48_static_theta_tflite_gate.yaml`
- `requirements-tflite-win-py311.txt`
- `artifacts/t48_true_tflite_runtime_gate/`

## 验证

Worker 必须实际执行并报告：

1. `python -m py_compile cnn_fpga/model/build_t48_true_tflite_runtime_gate.py`
2. `python -m unittest tests.test_t48_true_tflite_runtime_gate`
3. 真实 runtime 环境的 import probe
4. 一次 float `.tflite` `evaluate_tflite`（若环境可用）
5. 一次 float `validate_export`（若环境可用）
6. 一次 int8 `.tflite` `evaluate_tflite`（若存在可执行 preserved pair）
7. 一次 int8 `validate_export`（若存在可执行 preserved pair）
8. helper 的一次真实执行
9. 边界检查：
   - `git diff --name-only -- runs`
   - `git diff --name-only -- artifacts/models/static_theta_v2 artifacts/reports/static_theta_v2 artifacts/models/runtime_b_residual_v1 artifacts/reports/runtime_b_residual_v1`
   - `git diff --name-only -- requirements-recovery.txt requirements-train-cpu-win-py312.txt`

Worker 还必须显式报告：

1. runtime gate 使用的解释器路径
2. import 成功的包名与版本
3. float / int8 选择到的 preserved artifact 路径
4. float / int8 gate 是否成功
5. 最终 gate verdict
6. 当前支持的 `.tflite` claims
7. 当前不支持的 `.tflite` / deployment claims

## Review No-Go Triggers

Reviewer 在以下任一情况应返回 `BLOCK`：

1. worker 使用 `.tflite.json` stub manifest，却写成真实 `.tflite` runtime 成功
2. worker 未给出真实解释器环境真值，却写成 `.tflite` runtime 已恢复
3. worker 越界修改 canonical historical artifact 目录
4. worker 越界修改 `export.py`、`evaluate_tflite.py`、`validate_export.py`、`inference_service.py` 或任何 benchmark/HIL/runtime 主线语义文件
5. worker 未产出显式 gate verdict
6. worker 把本任务写成 benchmark/HIL promotion、真板验证或 deployment closure

## Captain 备注

- `T48` 之所以放在 `T50` 之后，是因为 `T50` 已经把 canonical 训练材料、preserved references 与 `.tflite` 候选存在性收清楚；现在可以更有针对性地问“真实 runtime 到底是否成立”。
- 这是一个比 `T18` 强得多的 gate：不是只写文档，而是要求环境真值、preserved artifact 选择、真实 `evaluate_tflite` / `validate_export`、helper、tests 和显式 gate verdict 一起收口。
- 若最终结论是 `NO_GO`，只要证据真实、边界诚实，该任务仍然是有效完成，而不是失败。
## Worker Output

- 最终解释器环境：`D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t48_tf221\Scripts\python.exe`
- Python / runtime：`3.11.15` + `tensorflow==2.21.0`
- preserved true `.tflite` load probe（tf2.21）：
  - `artifacts/t48_true_tflite_runtime_gate/preserved_tflite_load_probe_tf221.json`
  - 结果：`6 / 6` true `.tflite` `allocate_tensors()` 成功
- float 真执行：
  - eval：`artifacts/t48_true_tflite_runtime_gate/reports/static_theta_v2/eval_tflite_test_20260610_211759.json`
  - validate：`artifacts/t48_true_tflite_runtime_gate/reports/static_theta_v2/validate_export_tiny_cnn_20260319_151717_b87c6c227b57_20260610_211815.json`
- int8 真执行：
  - eval：`artifacts/t48_true_tflite_runtime_gate/reports/static_theta_v2/eval_tflite_test_20260610_211830.json`
  - validate：`artifacts/t48_true_tflite_runtime_gate/reports/static_theta_v2/validate_export_tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_20260610_211845.json`
- helper 输出：
  - `artifacts/t48_true_tflite_runtime_gate/t48_true_tflite_runtime_gate.json`
- 最终 gate verdict：
  - `GO_TRUE_TFLITE_RUNTIME_FLOAT_AND_INT8`
