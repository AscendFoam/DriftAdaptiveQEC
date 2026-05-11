# T27 Teacher Diagnostics Path Audit

## Verdict

`PASS_WITH_WARNINGS`

现有只读证据足以定位 `R10` 的主因，并证明 `R20` 不走同一条死路径；但 `R10` 仍未修复，且 `comparison.csv` 的 `0.0` 默认值会掩盖“未生成”与“真实为零”的区别。

## Scope And Inputs

本审计只做代码与既有产物的只读检查，未运行 benchmark、训练、`.tflite`、硬件或 cleanup 命令。

核心输入：

- `docs/review/T25_p4_formal_evidence_gate_review.md`
- `docs/review/T24_review.md`
- `docs/review/T15_frozen_smoke_review.md`
- `docs/P4_benchmark_formal_protocol.md`
- `docs/tasks/Phase2/T24_p4_formal_software_revalidation.md`
- `docs/tasks/Phase2/T25_p4_formal_evidence_gate_review.md`
- `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv`
- `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/teacher_scalar_diagnostics.csv`
- `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/report.md`
- `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/*/*/repeat_*/hil_summary.json`

## Files And Functions Inspected

1. Teacher feature construction
   - `cnn_fpga/runtime/feature_builder.py`
   - `scalar_feature_names()`
   - `build_feature_sample()`

2. Runtime diagnostics generation
   - `cnn_fpga/runtime/slow_loop_runtime.py`
   - `_teacher_branch_diagnostics()`
   - `_predict_from_artifact()`

3. Model explain path
   - `cnn_fpga/model/tiny_cnn.py`
   - `explain_from_loaded_artifact()`

4. HIL aggregation
   - `cnn_fpga/benchmark/run_hil_suite.py`
   - `_aggregate_teacher_branch_diagnostics()`

5. P4 benchmark aggregation and CSV writing
   - `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
   - per-repeat row extraction
   - scenario aggregation
   - `_write_per_scalar_csv()` call site

6. Correction saturation source
   - `cnn_fpga/runtime/fast_loop_emulator.py`
   - per-window diagnostics construction
   - `summary()`

7. T24 strong-baseline config
   - `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`

## Teacher Diagnostics Data Flow

### 1. Generation source

`hybrid_residual_b` 在 `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml` 中启用了：

- `include_teacher_prediction: true`
- `include_teacher_params: true`
- `include_teacher_deltas: true`

但没有把这些 teacher 特征切到 `scalar_branch` 布局；因此走的是默认 `broadcast` 布局。

`cnn_fpga/runtime/feature_builder.py` 显示：

- 只有 `teacher_prediction_layout == "scalar_branch"`、`teacher_params_layout == "scalar_branch"`、`teacher_deltas_layout == "scalar_branch"` 时，teacher 特征才会进入 `scalar_features`
- `broadcast` 布局会把 teacher 值展开成额外 histogram channel，而不是 `scalar_features`

这直接决定了后续 explain 路径是否能产出 scalar 诊断。

### 2. Explain path becoming empty

`cnn_fpga/model/tiny_cnn.py::explain_from_loaded_artifact()` 的关键条件是：

- 先读取 `scalar_feature_dim`
- 若 `scalar_feature_dim <= 0`，函数立即返回，只保留：
  - `prediction`
  - `label_names`
  - `scalar_feature_dim`
  - `scalar_fusion_mode`
  - `scalar_norm_clip`

它不会生成下面这些字段：

- `scalar_features_raw`
- `prediction_without_teacher`
- `teacher_contribution`
- `teacher_contribution_l2`
- `per_scalar_contribution`
- `teacher_gate_*`
- `per_scalar_gate_effect`

因此，对于当前 T24 的 `hybrid_residual_b` broadcast-teacher 路径，teacher 诊断数据在模型 explain 阶段就没有生成。

### 3. Runtime fallback masks absence as zero

`cnn_fpga/runtime/slow_loop_runtime.py::_teacher_branch_diagnostics()` 会把 explain 结果写入 metadata，但它对缺失字段使用默认值：

- `teacher_contribution_vector`: 缺失时回退到 `predicted_vector`
- `teacher_contribution_l2`: 缺失时写成 `0.0`
- `teacher_gate_mean/std/min/max`: 缺失时保持 `None`

这里没有计算独立的 `teacher_scalar_abs_mean`，也没有自行生成 `per_scalar`。

结果是：

- “无 teacher explain 数据” 被写成 `teacher_contribution_l2 = 0.0`
- 而不是显式区分为 “not generated”

### 4. HIL aggregation faithfully preserves emptiness

`cnn_fpga/benchmark/run_hil_suite.py::_aggregate_teacher_branch_diagnostics()` 的行为与上游一致：

- 只有 `scalar_features_raw` 存在且非空，才会累计 `teacher_scalar_abs_mean`
- 只有 `artifact_explanation.per_scalar_contribution` 或 `artifact_explanation.per_scalar_gate_effect` 存在，才会累计 `per_scalar`
- `teacher_gate_*` 只有存在时才累计

因此当 explain 没有生成这些字段时，HIL 汇总结果自然是：

- `teacher_scalar_abs_mean: null`
- `teacher_gate_mean: null`
- `teacher_gate_std: null`
- `per_scalar: {}`

### 5. P4 writer is not the root cause, but it further obscures it

`cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py` 会从 HIL summary 取值，并把缺失/`null` 再次强制写成 `0.0`：

- `teacher_contribution_l2_mean`
- `teacher_scalar_abs_mean`
- `teacher_gate_mean`
- `teacher_gate_std`

同时 `teacher_per_scalar` 缺失时写成 `{}`，所以 `teacher_scalar_diagnostics.csv` 只能输出表头。

结论：

- `teacher_scalar_diagnostics.csv` header-only 不是 writer 单点 bug
- 根因在更上游：当前 T24 `hybrid_residual_b` 路径没有生成 scalar teacher 诊断
- 但 writer 的 `or 0.0` 默认值确实掩盖了“未生成”这一事实

## Evidence From Existing T24 Outputs

### A. T24 `teacher_scalar_diagnostics.csv`

现有文件只有表头，没有数据行。说明 `comparison_rows[*]["teacher_per_scalar"]` 全为空字典。

### B. T24 hybrid `hil_summary.json`

示例：`runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/static/hybrid/repeat_00/hil_summary.json`

其中：

- `artifact_path` 非空，说明确实走了 artifact/inference 路径
- `teacher_branch_diagnostics.teacher_contribution_l2_mean = 0.0`
- `teacher_branch_diagnostics.teacher_scalar_abs_mean = null`
- `teacher_branch_diagnostics.teacher_gate_mean = null`
- `teacher_branch_diagnostics.per_scalar = {}`

这与“artifact 存在，但 scalar explain 未生成”完全一致。

### C. T24 non-hybrid `hil_summary.json`

示例：`runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/static/consta/repeat_00/hil_summary.json`

其中：

- `artifact_path = null`
- `teacher_branch_diagnostics.*` 基本都为 `null`/空

因此非 hybrid baseline 的 teacher 诊断本来就是 mode-not-applicable，不应与 hybrid 的“应有解释但未生成”混为一类。

## Classification

### Teacher diagnostics (`R10`)

在当前 T24 冻结证据里，需要拆成两层：

1. 对非 `hybrid_residual_b` 模式：
   - 分类：`mode/scenario not applicable`

2. 对 `hybrid_residual_b` 模式：
   - 分类：`data not generated`
   - 次级说明：聚合与 CSV 写出路径会把缺失值进一步写成 `0.0` 或空表，造成“像是真零”的表象

更具体地说，当前 explain 机制只对 `scalar_feature_dim > 0` 的 side-branch 产出 teacher 诊断；而 T24 strong-baseline hybrid 配置走的是 broadcast teacher 通道，不满足这个前提。

## Correction Saturation Path Note (`R20`)

`correction_saturation_rate_mean` 不与 teacher diagnostics 共享同一条空路径。

证据链：

- `cnn_fpga/runtime/fast_loop_emulator.py` 在每个 window 诊断里直接计算 `correction_saturation_ratio`
- `summary()` 直接用 `_correction_sat_count / n_cycles` 生成 `correction_saturation_rate`
- `run_p4_multiscenario_benchmark.py` 只是把 HIL summary 中这个数值转抄到 `comparison.csv`

因此 `R20` 的现有分类更接近：

- `independent path`
- 在当前 T24 参数区间和样本下表现为 `genuine zero under current parameter regime`

但它还不是“永远不会触发”的证明，只能说明当前冻结 formal 软件重验证没有打到 correction saturation。

## Risk Mapping

### R10

状态：`remains open but narrowed`

已缩窄为：

- 不是 `teacher_scalar_diagnostics.csv` writer 单点故障
- 不是 `run_hil_suite.py` 聚合逻辑单点丢失
- 主因是当前 hybrid baseline 使用 broadcast teacher 布局，而 explain 仅对 scalar side-branch 产出 teacher 诊断
- 另一个治理风险是 downstream CSV 把缺失值强制写成 `0.0`

### R20

状态：`remains open but materially narrowed`

已缩窄为：

- 不属于 teacher diagnostics 的共享死路径
- 当前 `0.0` 有独立生成路径支撑，不能再简单归类为“指标没写出来”
- 仍缺少 stress/edge 条件来证明该指标在更激进参数区间下是否可触发

## Minimal Next Task Recommendation

建议后续新开一个有界 repair 任务，不在本任务内执行：

1. 明确选择一种机制并固定语义：
   - 要么只在 `scalar_branch` teacher 配置下宣称支持 teacher scalar diagnostics
   - 要么为 `broadcast` teacher 路径新增独立 explain/ablation 诊断

2. 在 writer 层保留“未生成”语义：
   - 不要再把 `null` 统一压成 `0.0`
   - 让 `comparison.csv` 可以区分 `not generated` 与 `true zero`

3. 用最小 smoke 重新验证：
   - 一条 `broadcast` hybrid
   - 一条显式 `scalar_branch` hybrid（若仓库允许）
   - 只验证 teacher diagnostics 路径，不扩 benchmark 口径

## Final Audit Statement

本次审计认为：

- `R10` 的主因已经定位到“当前 hybrid teacher 特征布局与 explain 机制不匹配，导致 teacher 诊断未生成”
- `R20` 不共享该问题路径，当前更像独立生成且在现参数区间真实为零

因此本任务可 `PASS_WITH_WARNINGS` 收口，但不代表问题已修复，也不代表 T24/T25 可以把 teacher 机理证据写成已补齐。
