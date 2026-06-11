# T50 Worker Summary

## 改了什么

本轮只改了 `T50` 允许路径：

1. 新增 task-scoped helper：`cnn_fpga/model/build_training_reproducibility_pack.py`
2. 新增 focused tests：`tests/test_training_reproducibility_pack.py`
3. 新增派生配置：`cnn_fpga/config/task_tmp/T50_static_theta_repro_pack.yaml`
4. 新增 bounded rerun 输出目录内容：`artifacts/t50_training_repro_pack/`
5. 新增主报告：`docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md`
6. 新增 review 草稿：`docs/review/T50_review.md`
7. 新增人类说明：`docs/for_human/T50_explanation.md`
8. 新增本文件：`docs/worker_summary/T50_worker_summary.md`
9. 在任务包 `docs/tasks/Phase2/T50_training_reproducibility_and_material_regeneration_pack.md` 末尾追加本轮 Worker Output

helper 负责：

- 读取 canonical `static_theta_v2` 材料链
- 读取 canonical `runtime_b_residual_v1` 材料链
- 校验当前 mainline `P3/P4/HIL` preserved model references
- 读取本轮 clean CPU-only bounded rerun 的 train/eval report
- 输出统一 pack：`artifacts/t50_training_repro_pack/training_reproducibility_pack.json`

## 如何验证

实际执行的命令：

1. `D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t39_train_cpu_py312\Scripts\python.exe -m py_compile cnn_fpga/model/build_training_reproducibility_pack.py`
2. `D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t39_train_cpu_py312\Scripts\python.exe -m unittest tests.test_training_reproducibility_pack`
3. `D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t39_train_cpu_py312\Scripts\python.exe -m cnn_fpga.model.train --config cnn_fpga/config/task_tmp/T50_static_theta_repro_pack.yaml`
4. `D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t39_train_cpu_py312\Scripts\python.exe -m cnn_fpga.model.evaluate --config cnn_fpga/config/task_tmp/T50_static_theta_repro_pack.yaml --split test --model-path artifacts/t50_training_repro_pack/models/static_theta_v2/tiny_cnn_20260610_195014_7126933acb7c.npz`
5. `D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t39_train_cpu_py312\Scripts\python.exe -m cnn_fpga.model.build_training_reproducibility_pack --rerun-train-report artifacts/t50_training_repro_pack/reports/static_theta_v2/tiny_cnn_20260610_195014_7126933acb7c_train_report.json --rerun-eval-report artifacts/t50_training_repro_pack/reports/static_theta_v2/eval_test_20260610_195030.json`
6. `git diff --name-only -- runs`
7. `git diff --name-only -- artifacts/models/static_theta_v2 artifacts/reports/static_theta_v2 artifacts/models/runtime_b_residual_v1 artifacts/reports/runtime_b_residual_v1`
8. `git diff --name-only -- requirements-recovery.txt requirements-train-cpu-win-py312.txt`

验证结果：

- `py_compile` 通过
- unittest 通过：`Ran 3 tests`, `OK`
- clean CPU-only real train rerun 成功
- clean CPU-only real eval rerun 成功
- helper 实跑成功并写出 pack JSON
- 三个边界 diff 检查均为空

## 关键结果

- clean CPU-only 解释器：`D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t39_train_cpu_py312\Scripts\python.exe`
- Python 版本：`3.12.7`

bounded rerun 产物：

- model：`artifacts/t50_training_repro_pack/models/static_theta_v2/tiny_cnn_20260610_195014_7126933acb7c.npz`
- train report：`artifacts/t50_training_repro_pack/reports/static_theta_v2/tiny_cnn_20260610_195014_7126933acb7c_train_report.json`
- eval report：`artifacts/t50_training_repro_pack/reports/static_theta_v2/eval_test_20260610_195030.json`
- pack：`artifacts/t50_training_repro_pack/training_reproducibility_pack.json`

确认存在的 canonical key artifacts：

- `artifacts/datasets/static_theta_v2/manifest.json`
- `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
- `artifacts/reports/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_train_report.json`
- `artifacts/datasets/runtime_b_residual_v1/manifest.json`
- `artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz`
- `artifacts/reports/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d_train_report.json`

当前 mainline preserved model references：

1. `experiment_runtime_b_residual.yaml` 仍指向 `artifacts/models/runtime_b_residual_v1`
2. `hardware_hil_recovery_smoke.yaml` 仍从 `artifacts/models/static_theta_v2` 取 latest float
3. `p4_multiscenario_recovery_smoke.yaml` 仍显式指向 `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
4. `p4_multiscenario_statcalib_extension_lane.yaml` 的 `hybrid_residual_b` override 仍指向 `artifacts/models/runtime_b_residual_v1`

当前支持的 claims：

1. canonical `static_theta_v2` 材料链存在且可被代码统一枚举
2. canonical `runtime_b_residual_v1` 材料链存在且仍支撑主线引用
3. clean CPU-only lane 已完成一次 bounded real train+eval rerun
4. `P3/P4/HIL` preserved model references 当前没有漂移到缺失路径

当前不支持的 claims：

1. full training reproducibility
2. GPU/CUDA portability
3. Linux portability
4. `.tflite` runtime correctness
5. real-board validation
6. benchmark/HIL promotion 结论

## 剩余风险

1. `R11` 仍未关闭；`T50` 只是把 clean CPU-only 证据从 one-train smoke 推进到 one-train + one-eval bounded rerun，并补上统一材料账本。
2. canonical 历史 report 不含 `training_backend` / `training_device` 字段，所以 canonical 执行环境语义仍需结合历史文档理解。
3. bounded rerun 指标明显弱于 canonical，是缩界配置带来的预期结果，不应用来做 canonical quality parity 叙事。
