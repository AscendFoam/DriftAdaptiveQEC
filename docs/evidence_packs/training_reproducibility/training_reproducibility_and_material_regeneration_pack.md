# T50 训练复现与材料再生证据包

## 任务边界

`T50` 只做一件事：把当前主线中已经存在的训练材料链、当前 `P3/P4/HIL` 对 preserved historical model 的依赖关系，以及一次新的 clean CPU-only bounded train+eval rerun，收成一个 code-backed 证据包。

这不是 full training reproducibility 任务，也不是 `.tflite`、真板、benchmark 或 sidecar 任务。

## 执行环境与产物

- clean CPU-only 解释器：`D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t39_train_cpu_py312\Scripts\python.exe`
- Python 版本：`3.12.7`
- 派生配置：`cnn_fpga/config/task_tmp/T50_static_theta_repro_pack.yaml`
- bounded train 输出：
  - model: `artifacts/t50_training_repro_pack/models/static_theta_v2/tiny_cnn_20260610_195014_7126933acb7c.npz`
  - report: `artifacts/t50_training_repro_pack/reports/static_theta_v2/tiny_cnn_20260610_195014_7126933acb7c_train_report.json`
- bounded eval 输出：
  - report: `artifacts/t50_training_repro_pack/reports/static_theta_v2/eval_test_20260610_195030.json`
- 最终 pack：
  - `artifacts/t50_training_repro_pack/training_reproducibility_pack.json`

本轮为加速而采用任务包建议的 bounded 规模：

- `max_train_samples = 2048`
- `max_val_samples = 512`
- `epochs = 5`
- `patience = 3`

这样做的目的不是逼近 canonical 训练质量，而是在不碰 canonical 历史目录的前提下，用 clean CPU-only lane 生成一组真实、隔离、可审计的新训练与评估证据。

## 核心回答

### 1. canonical `static_theta_v2` 材料链是否完整

是。以下关键材料都存在且已被 helper 读取：

- dataset manifest：`artifacts/datasets/static_theta_v2/manifest.json`
- float model artifact：`artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
- train report：`artifacts/reports/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_train_report.json`

此外，helper 还对 preserved derived materials 做了 presence summary：

- int8 model artifact：`1`
- `.tflite` model artifact：`6`
- `.tflite.json` sidecar：`2`
- float eval report：`2`
- `.tflite` eval report：`2`
- quant report：`1`
- export report：`8`
- validate-export report：`2`

这些 derived materials 在本轮只被枚举存在性，不被写成“当前机器已重新强验证”的事实。

### 2. canonical `runtime_b_residual_v1` 材料链是否完整

是。以下关键材料都存在且已被 helper 读取：

- dataset manifest：`artifacts/datasets/runtime_b_residual_v1/manifest.json`
- float model artifact：`artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz`
- train report：`artifacts/reports/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d_train_report.json`

`T50` 对这条链的重点不是重新训练它，而是确认主线 runtime residual / statcalib extension lane 仍然引用这条 preserved chain，而不是漂移到缺失路径。

### 3. 当前 mainline `P3/P4/HIL` 入口依赖哪些 preserved model references

helper 对以下引用做了显式校验：

| surface | config | 当前依赖 | 结论 |
| --- | --- | --- | --- |
| P3 runtime residual training / inference | `cnn_fpga/config/experiment_runtime_b_residual.yaml` | `paths.dataset_dir = artifacts/datasets/runtime_b_residual_v1`；`paths.model_dir = artifacts/models/runtime_b_residual_v1`；`paths.report_dir = artifacts/reports/runtime_b_residual_v1`；`slow_loop.model_artifact.use_latest_model_dir_artifact = true` | 仍然指向 preserved `runtime_b_residual_v1` 链 |
| HIL recovery smoke | `cnn_fpga/config/hardware_hil_recovery_smoke.yaml` | `paths.model_dir = artifacts/models/static_theta_v2`；`use_latest_model_dir_artifact = true` | 仍然从 preserved `static_theta_v2` model_dir 取 latest float |
| P4 recovery smoke | `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml` | 显式 `slow_loop.model_artifact.path` 和 `slow_loop.inference_service.model_path` 都指向 `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz` | 仍然指向 preserved `static_theta_v2` historical float model |
| P4 statcalib extension lane | `cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml` | `hybrid_residual_b` override 的 `paths.model_dir = artifacts/models/runtime_b_residual_v1`，`artifact_selector = latest_float` | 仍然指向 preserved `runtime_b_residual_v1` model_dir |

`T50` 的 helper 会在这些引用漂移到缺失路径时直接拒绝生成 pack。

### 4. `T50` clean CPU-only rerun 实际执行到了什么程度

执行了两步真实复跑：

1. 一次 bounded `tiny_cnn` 训练复跑
   - config: `cnn_fpga/config/task_tmp/T50_static_theta_repro_pack.yaml`
   - dataset：仍然是 canonical `static_theta_v2`
   - backend：`numpy`
   - device：`cpu`
   - train samples：`2048`
   - val samples：`512`
   - epochs：`5`
   - patience：`3`

2. 一次针对新模型的真实 test-split 评估复跑
   - split：`test`
   - model path：`artifacts/t50_training_repro_pack/models/static_theta_v2/tiny_cnn_20260610_195014_7126933acb7c.npz`

关键指标如下：

| category | file | key result |
| --- | --- | --- |
| bounded train rerun | `tiny_cnn_20260610_195014_7126933acb7c_train_report.json` | `n_train=2048`，`n_val=512`，`backend=numpy`，`device=cpu`，`train mse=6.1902`，`val mse=6.3958`，`val r2_mean=0.8570` |
| bounded eval rerun | `eval_test_20260610_195030.json` | `n_samples=1639`，`test mse=6.0881`，`test mae=1.0395`，`test r2_mean=0.8600` |

这里要特别强调：bounded rerun 的目标是证明 clean CPU-only lane 能真实完成 train+eval，并形成隔离证据，不是追求与 canonical 28-epoch 历史模型同等指标。

### 5. 现在可以诚实支持哪些 reproducibility / material claims

| category | subject | 结论 |
| --- | --- | --- |
| canonical_materials | `static_theta_v2` | 当前仓库中存在完整的 canonical dataset/float-model/train-report 链，且能被代码统一枚举 |
| canonical_materials | `runtime_b_residual_v1` | 当前仓库中存在完整的 canonical dataset/float-model/train-report 链，且仍支撑主线 runtime residual 引用 |
| bounded_rerun_materials | clean CPU-only rerun | 当前 clean Windows/Python 3.12 CPU-only lane 已经能完成一次更强于 T40 的 bounded real train+eval rerun |
| supported_claims | mainline reference integrity | 当前 `P3/P4/HIL` 主线入口仍然引用 preserved historical model path / model_dir，没有漂移到缺失路径 |
| supported_claims | material-regeneration pack | 仓库现在有一份 code-backed training reproducibility / material-regeneration pack，可把 canonical chain、mainline reference、bounded rerun 收到同一 ledger 中 |

### 6. 现在仍然不能支持哪些 claims

| category | subject | 结论 |
| --- | --- | --- |
| unsupported_claims | full reproducibility | 本轮不证明 full training reproducibility，也不证明 repeated-run / cross-host / cross-OS reproducibility |
| unsupported_claims | GPU/CUDA / Linux | 本轮不证明 GPU/CUDA portability，也不证明 Linux portability |
| unsupported_claims | `.tflite` runtime | 本轮没有重新验证 `.tflite` runtime correctness |
| unsupported_claims | real-board | 本轮没有任何真板验证 |
| unsupported_claims | benchmark / HIL promotion | 本轮训练材料证据包不应被转写成 benchmark/HIL superiority 或 deployment promotion 结论 |

## canonical-vs-rerun relation

`artifacts/t50_training_repro_pack/training_reproducibility_pack.json` 明确记录了以下关系：

- rerun dataset 仍然是 `static_theta_v2`
- rerun model 与 eval model 一致
- rerun model / report 全部落在 `artifacts/t50_training_repro_pack/` 下
- rerun model 与 canonical historical float model 不同，是本轮新生成、隔离保存的 artifact
- rerun train backend = `numpy`
- rerun train device = `cpu`

换句话说，`T50` 现在支撑的是：

- “主线 canonical 材料链仍在”
- “主线引用仍没漂”
- “clean CPU-only lane 现在能真实再跑一次 bounded train+eval”

而不是：

- “canonical 训练质量已被当前机器完整重现”
- “训练链已经跨平台稳定复现”
- “部署链已经跟着一起被验证”

## 剩余风险

1. canonical 历史 train report 生成时尚未写入 `training_backend` / `training_device` 字段，所以当前 pack 对 canonical 链的“环境语义”只能依赖历史文档与本轮 rerun，而不能从旧 report 中直接恢复完整执行环境。
2. bounded rerun 指标明显弱于 canonical 历史模型，这是由 `2048/512/5/3` 的有界配置带来的预期现象；不能把这组指标差异误读为 canonical artifacts 失效。
3. `R11` 仍未关闭。当前只是在 clean CPU-only lane 上从 `T40` 的 one-train smoke，推进到更强的 one-train + one-eval bounded rerun 和统一材料账本。
