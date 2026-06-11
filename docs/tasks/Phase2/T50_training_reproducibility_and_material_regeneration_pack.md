# T50：训练复现与材料再生证据包

## 状态

- 由 Captain 于 `2026-06-10` 提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：有界训练复现/材料台账任务，包含 task-scoped helper、focused tests、一次 clean-CPU 真实训练复跑和一次评估复跑

## 为什么现在做这个任务

`T31 / T39 / T40` 已经把训练链的可移植性边界缩窄到了一个更清楚的位置：

1. `T31` 证明当前训练链按配置语义存在一个 CPU-only 可行 lane。
2. `T39` 证明可以在 `DLEnv` 之外创建干净的 `Python 3.12` CPU-only 环境，并通过 dataset-builder dry-run / runtime-dataset-builder dry-run / train help。
3. `T40` 证明这个 clean CPU-only 环境可以执行一次真实的 `tiny_cnn` 训练 smoke，并把输出隔离到 `artifacts/t40_train_smoke/`。

但主线仓库现在仍缺一件更大的东西：

1. 没有一个统一、代码驱动的训练材料台账，把主线训练 artifacts、train/eval reports、quant/int8 派生物、`.tflite` 派生物和当前 P3/P4/HIL 入口之间的关系固定下来。
2. 没有一个主线可引用的“训练复现与材料再生证据包”，说明：
   - 哪些训练 artifacts 是 preserved historical materials
   - 哪些训练链在 clean CPU-only lane 已经得到真实执行证据
   - 哪些派生物只是存在但还没有当前机器上的强验证
3. `T48` / `T49` 仍依赖更强的 `.tflite` / 真板前提；直接跳过去不会是最小诚实下一步。

因此，当前主线最小而又有实质性的下一步，不是继续开 FR8 结果任务，也不是立刻跳到真板或 `.tflite`，而是补一份更强的训练复现与材料再生证据包。

## 目标

产出一个有界的训练复现与材料再生证据包，至少回答以下问题：

1. 当前主线训练材料链中，哪些 canonical dataset/model/report artifacts 真实存在并能被代码侧重建成统一 ledger。
2. `static_theta_v2` 主训练链能否在 `T39` 的 clean CPU-only 环境中再次完成一次真实、隔离、可记录的 bounded rerun。
3. bounded rerun 之后，是否能再对新 artifact 执行一次真实评估复跑，并把 train/eval report 一起纳入证据包。
4. 当前主线 P3/P4/HIL 入口对哪些 preserved model artifacts 形成直接依赖。
5. 当前训练材料链支持哪些说法，不支持哪些说法。

## 当前推荐唯一结论方向

本任务的目标不是证明“完整训练复现已经闭环”，而是把结论收口到下列更诚实的层级：

1. clean CPU-only lane 已不只是 dry-run/import-level，可完成一次更强的真实 train+eval bounded rerun
2. canonical `static_theta_v2` 与 `runtime_b_residual_v1` 材料链可以被统一梳理为一个 code-backed pack
3. 当前仍不支持：
   - full training reproducibility
   - GPU/CUDA portability
   - Linux portability
   - `.tflite` runtime correctness on this machine
   - real-board validation

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T50_training_reproducibility_and_material_regeneration_pack.md`
- `docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md`
- `docs/review/T50_review.md`
- `docs/for_human/T50_explanation.md`
- `docs/worker_summary/T50_worker_summary.md`
- `cnn_fpga/model/build_training_reproducibility_pack.py`
- `tests/test_training_reproducibility_pack.py`
- `cnn_fpga/config/task_tmp/T50_static_theta_repro_pack.yaml`
- `artifacts/t50_training_repro_pack/`

说明：

- `artifacts/t50_training_repro_pack/` 是本任务唯一允许写入的 artifact 输出根目录。
- `.venvs/t39_train_cpu_py312/` 若已存在，可复用；若缺失，可按 `T39` 既有步骤重建，但不得改写仓库跟踪文件。

## Docs To Update

Worker 必须更新：

- `docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md`
- `docs/review/T50_review.md`
- `docs/for_human/T50_explanation.md`
- `docs/worker_summary/T50_worker_summary.md`

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
- 修改 `cnn_fpga/model/train.py`
- 修改 `cnn_fpga/model/evaluate.py`
- 修改 `cnn_fpga/model/export.py`
- 修改任一 benchmark、HIL、runtime、decoder 主线语义文件
- 运行 benchmark、HIL、`.tflite` runtime、real-board 或 sidecar 实验
- 使用 `DLEnv` 代替 clean CPU-only lane
- 把 bounded rerun 写成 full training reproducibility、GPU/CUDA portability、Linux portability、`.tflite` 正确性或真板正确性证明

## 必须复用的输入

Worker 必须复用以下既有输入，而不是重写历史事实：

- `docs/evidence_packs/training_reproducibility/training_chain_portable_dependency_lock_plan.md`
- `docs/evidence_packs/training_reproducibility/training_chain_cpu_cleanenv_bootstrap.md`
- `docs/evidence_packs/training_reproducibility/training_chain_cpu_cleanenv_train_smoke.md`
- `docs/review/T31_review.md`
- `docs/review/T39_review.md`
- `docs/review/T40_review.md`
- `requirements-train-cpu-win-py312.txt`
- `artifacts/datasets/static_theta_v2/manifest.json`
- `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
- `artifacts/reports/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_train_report.json`
- `artifacts/datasets/runtime_b_residual_v1/manifest.json`
- `artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz`
- `artifacts/reports/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d_train_report.json`
- `cnn_fpga/config/experiment_static_theta_v2.yaml`
- `cnn_fpga/config/experiment_runtime_b_residual.yaml`
- `cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
- `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`
- `cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml`

## 固定边界

- 主线分支：当前 mainline experiment branch only
- 训练边界：clean CPU-only lane only
- 输出边界：只允许写 `artifacts/t50_training_repro_pack/`
- 证据边界：训练材料链与 reproducibility pack only
- 非目标边界：不是 `.tflite` runtime task，不是真板 task，不是 benchmark task，不是 theory-branch task

## 任务要求

### A. 训练材料链 helper

新增一个 task-scoped helper：

- `cnn_fpga/model/build_training_reproducibility_pack.py`

它至少要完成以下工作：

1. 只读读取 canonical `static_theta_v2` 材料链：
   - dataset manifest
   - float model artifact
   - train report
   - 若存在，则纳入 int8 / export / eval 派生材料的 presence-summary
2. 只读读取 canonical `runtime_b_residual_v1` 材料链：
   - dataset manifest
   - float model artifact
   - train report
3. 检查当前 mainline `P3/P4/HIL` 入口仍然引用 preserved historical model path / model_dir，而不是丢失或漂移到不存在的路径
4. 读取本轮 `T50` bounded rerun 的 train report 与 eval report
5. 给出一个统一 pack，显式包含：
   - canonical material presence
   - canonical-vs-rerun relation
   - supported claims
   - unsupported claims
   - 当前 clean CPU-only reproducibility boundary

### B. Focused tests

新增：

- `tests/test_training_reproducibility_pack.py`

测试至少覆盖：

1. 当前 preserved canonical inputs 下 helper 能构建预期 pack
2. 若 canonical `static_theta_v2` float model path 缺失，helper 明确拒绝
3. 若 mainline config reference 漂移到缺失路径，helper 明确拒绝

### C. Bounded clean CPU rerun

Worker 必须在 `T39` clean CPU-only 环境中执行一次比 `T40` 更强但仍然 bounded 的真实 rerun：

1. 新增派生 config：
   - `cnn_fpga/config/task_tmp/T50_static_theta_repro_pack.yaml`
2. 基于：
   - `cnn_fpga/config/experiment_static_theta_v2.yaml`
3. 仅允许覆盖：
   - `paths.model_dir = artifacts/t50_training_repro_pack/models/static_theta_v2`
   - `paths.report_dir = artifacts/t50_training_repro_pack/reports/static_theta_v2`
   - `training.max_train_samples`
   - `training.max_val_samples`
   - `training.tiny_cnn.epochs`
   - `training.tiny_cnn.patience`
4. 建议规模：
   - `max_train_samples = 2048`
   - `max_val_samples = 512`
   - `epochs = 5`
   - `patience = 3`

说明：

- 这是建议值；若 worker 发现更小值已足够达到稳定产物，也可等价缩小，但必须在文档中解释。
- 不允许把 canonical historical output 目录当输出路径。

### D. Bounded eval rerun

Worker 必须对 `T50` 新生成的 model artifact 再执行一次真实评估复跑：

1. 使用现有 `cnn_fpga.model.evaluate`
2. 输出必须仍落在：
   - `artifacts/t50_training_repro_pack/reports/static_theta_v2/`
3. 目标是把 `train_report + eval_report` 一并纳入训练材料再生 pack

### E. 最终文档必须回答的问题

`docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md` 至少要回答：

1. canonical `static_theta_v2` 材料链是否完整
2. canonical `runtime_b_residual_v1` 材料链是否完整
3. 当前 mainline P3/P4/HIL 入口依赖哪些 preserved model artifacts
4. `T50` clean CPU-only rerun 真实执行到了什么程度
5. 现在可以诚实地支持哪些 reproducibility/material claims
6. 现在仍然不能支持哪些 claims

文档中必须包含一个紧凑表格，至少区分：

- `canonical_materials`
- `bounded_rerun_materials`
- `supported_claims`
- `unsupported_claims`

## 预期输出

Worker 必须产出：

- `docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md`
- `docs/review/T50_review.md`
- `docs/for_human/T50_explanation.md`
- `docs/worker_summary/T50_worker_summary.md`
- `cnn_fpga/model/build_training_reproducibility_pack.py`
- `tests/test_training_reproducibility_pack.py`
- `cnn_fpga/config/task_tmp/T50_static_theta_repro_pack.yaml`
- `artifacts/t50_training_repro_pack/` 下的 bounded rerun outputs

## 验证

Worker 必须实际执行并报告：

1. `python -m py_compile cnn_fpga/model/build_training_reproducibility_pack.py`
2. `python -m unittest tests.test_training_reproducibility_pack`
3. clean CPU-only 环境下的一次真实训练 rerun
4. clean CPU-only 环境下的一次真实评估 rerun
5. helper 的一次真实执行
6. 边界检查：
   - `git diff --name-only -- runs`
   - `git diff --name-only -- artifacts/models/static_theta_v2 artifacts/reports/static_theta_v2 artifacts/models/runtime_b_residual_v1 artifacts/reports/runtime_b_residual_v1`
   - `git diff --name-only -- requirements-recovery.txt requirements-train-cpu-win-py312.txt`

Worker 还必须显式报告：

1. clean CPU-only rerun 使用的解释器路径
2. rerun 生成的 model/report 路径
3. canonical material chain 中确认存在的关键 artifact 列表
4. 当前 mainline P3/P4/HIL 入口依赖的 preserved model references
5. 当前支持的 reproducibility/material claims
6. 当前不支持的 reproducibility/material claims

## Review No-Go Triggers

Reviewer 在以下任一情况应返回 `BLOCK`：

1. worker 改写任何 canonical historical artifact 目录
2. worker 使用 `DLEnv` 或其它未说明环境替代 clean CPU-only lane
3. worker 未执行真实 train rerun 或真实 eval rerun，却写成已完成
4. worker 把 `T50` 写成 full training reproducibility、GPU/CUDA portability、Linux portability、`.tflite` correctness 或真板正确性证明
5. helper 没有检查 mainline config references 与 canonical material chain 的关系
6. worker 越界修改 benchmark/HIL/runtime/decoder 主线代码

## Captain 备注

- 这是当前主线最小而有实质性的下一步，因为 `.tflite` / 真板前提仍未满足，而 FR8 主线已在 `T70` 形成 closure pack。
- 该任务比简单文档任务更强，要求 helper、tests、真实训练复跑、真实评估复跑、材料链台账一起收口。
- 该任务仍然必须保持边界诚实：它强化的是训练复现与材料再生证据，不是部署边界。
## Worker Output

### Files changed

- `cnn_fpga/model/build_training_reproducibility_pack.py`
- `tests/test_training_reproducibility_pack.py`
- `cnn_fpga/config/task_tmp/T50_static_theta_repro_pack.yaml`
- `artifacts/t50_training_repro_pack/`
- `docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md`
- `docs/review/T50_review.md`
- `docs/for_human/T50_explanation.md`
- `docs/worker_summary/T50_worker_summary.md`
- `docs/tasks/Phase2/T50_training_reproducibility_and_material_regeneration_pack.md`

### Exact verification commands executed

1. `D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t39_train_cpu_py312\Scripts\python.exe -m py_compile cnn_fpga/model/build_training_reproducibility_pack.py`
2. `D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t39_train_cpu_py312\Scripts\python.exe -m unittest tests.test_training_reproducibility_pack`
3. `D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t39_train_cpu_py312\Scripts\python.exe -m cnn_fpga.model.train --config cnn_fpga/config/task_tmp/T50_static_theta_repro_pack.yaml`
4. `D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t39_train_cpu_py312\Scripts\python.exe -m cnn_fpga.model.evaluate --config cnn_fpga/config/task_tmp/T50_static_theta_repro_pack.yaml --split test --model-path artifacts/t50_training_repro_pack/models/static_theta_v2/tiny_cnn_20260610_195014_7126933acb7c.npz`
5. `D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t39_train_cpu_py312\Scripts\python.exe -m cnn_fpga.model.build_training_reproducibility_pack --rerun-train-report artifacts/t50_training_repro_pack/reports/static_theta_v2/tiny_cnn_20260610_195014_7126933acb7c_train_report.json --rerun-eval-report artifacts/t50_training_repro_pack/reports/static_theta_v2/eval_test_20260610_195030.json`
6. `git diff --name-only -- runs`
7. `git diff --name-only -- artifacts/models/static_theta_v2 artifacts/reports/static_theta_v2 artifacts/models/runtime_b_residual_v1 artifacts/reports/runtime_b_residual_v1`
8. `git diff --name-only -- requirements-recovery.txt requirements-train-cpu-win-py312.txt`

### Interpreter path

- `D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t39_train_cpu_py312\Scripts\python.exe`
- Python version: `3.12.7`

### Rerun outputs

- model: `artifacts/t50_training_repro_pack/models/static_theta_v2/tiny_cnn_20260610_195014_7126933acb7c.npz`
- train report: `artifacts/t50_training_repro_pack/reports/static_theta_v2/tiny_cnn_20260610_195014_7126933acb7c_train_report.json`
- eval report: `artifacts/t50_training_repro_pack/reports/static_theta_v2/eval_test_20260610_195030.json`
- pack: `artifacts/t50_training_repro_pack/training_reproducibility_pack.json`

### Key canonical artifacts found

- `artifacts/datasets/static_theta_v2/manifest.json`
- `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
- `artifacts/reports/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_train_report.json`
- `artifacts/datasets/runtime_b_residual_v1/manifest.json`
- `artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz`
- `artifacts/reports/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d_train_report.json`

### Mainline preserved model references

1. `experiment_runtime_b_residual.yaml` 仍指向 `artifacts/models/runtime_b_residual_v1`
2. `hardware_hil_recovery_smoke.yaml` 仍从 `artifacts/models/static_theta_v2` 取 latest float
3. `p4_multiscenario_recovery_smoke.yaml` 仍显式指向 `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
4. `p4_multiscenario_statcalib_extension_lane.yaml` 的 `hybrid_residual_b` override 仍指向 `artifacts/models/runtime_b_residual_v1`

### Supported reproducibility/material claims

1. canonical `static_theta_v2` 材料链存在且可被代码统一枚举
2. canonical `runtime_b_residual_v1` 材料链存在且仍支撑主线引用
3. clean CPU-only lane 已完成一次 bounded real train+eval rerun
4. 当前 `P3/P4/HIL` preserved model references 没有漂移到缺失路径

### Unsupported reproducibility/material claims

1. full training reproducibility
2. GPU/CUDA portability
3. Linux portability
4. `.tflite` runtime correctness
5. real-board validation
6. benchmark/HIL promotion 结论

### Remaining risks / interpretation limits

1. `R11` 仍未关闭；本轮只把 clean CPU-only 证据推进到 one-train + one-eval bounded rerun。
2. canonical 历史 report 不含 `training_backend` / `training_device` 字段。
3. bounded rerun 指标明显弱于 canonical，这反映的是缩界配置，不是 canonical quality regression。
