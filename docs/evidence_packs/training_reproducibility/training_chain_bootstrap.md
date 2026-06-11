# Training Chain Bootstrap

## 1. 目的

本文件只服务训练链的独立接力说明，不替代 `requirements-recovery.txt`。

目标是固定三件事：

1. 当前训练链推荐解释器；
2. 当前训练入口、典型配置与最小 smoke 检查；
3. 当前没有覆盖到的依赖边界与真实性边界。

## 2. 当前推荐解释器

截至 `2026-05-09`，当前机器上训练链推荐解释器为：

- `C:\ProgramData\anaconda3\envs\DLEnv\python.exe`

当前只确认了“这台机器上可用”的事实，不把它写成跨机器保证。

已确认的最小环境事实：

- `numpy = 2.2.4`
- `PyYAML = 6.0.2`
- `torch = 2.8.0.dev20250405+cu128`
- `torch.cuda.is_available() = True`

## 3. 当前训练入口

训练入口：

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m cnn_fpga.model.train --config <config_path>
```

已验证该入口可正常显示 `--help`：

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m cnn_fpga.model.train --help
```

CLI 参数当前只固定为：

- `--config`
- `--train-split`
- `--val-split`

## 4. 训练后端边界

`cnn_fpga/model/train.py` 当前支持两类训练路径：

1. `linear_regression_baseline`
   - 依赖 `numpy`
   - 不依赖 `torch`
2. `tiny_cnn`
   - 默认可走 `numpy` backend
   - 也支持 `torch` backend
   - 若选择 `torch` backend，则依赖 `torch`
   - 若选择 `device=auto`，当前机器会优先落到 `cuda` 可用时的 GPU 路径

因此，训练链当前更准确的口径应是：

- `train.py` 不是强制依赖 `torch` 的单一路径；
- 但当前主训练实验通常依赖 `DLEnv + torch`；
- 不能把这条链路混入只覆盖 recovery smoke 的 `requirements-recovery.txt`。

## 5. 典型训练配置

当前仓库中可直接作为训练链参考的典型配置包括：

### 5.1 P1 静态主模型

- `cnn_fpga/config/experiment_static_theta_v2.yaml`
- 训练类型：`tiny_cnn`
- 数据目录：`artifacts/datasets/static_theta_v2`
- 模型目录：`artifacts/models/static_theta_v2`
- 报告目录：`artifacts/reports/static_theta_v2`

示例命令：

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m cnn_fpga.model.train --config cnn_fpga/config/experiment_static_theta_v2.yaml
```

### 5.2 P4 residual-b / gated teacher 主线候选

- `cnn_fpga/config/experiment_runtime_b_residual.yaml`
- `cnn_fpga/config/experiment_runtime_b_residual_norm_gated_teacher_v5.yaml`

说明：

- `experiment_runtime_b_residual.yaml` 固定了 runtime-consistent residual-b 数据语义；
- `..._v5.yaml` 在其基础上保留 gated scalar teacher 融合，是当前文档中反复引用的主线候选之一。

示例命令：

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m cnn_fpga.model.train --config cnn_fpga/config/experiment_runtime_b_residual_norm_gated_teacher_v5.yaml
```

## 6. 当前最小 smoke 检查

本轮没有启动完整训练，只固定以下最小可读验证：

1. `DLEnv` 可导入：
   - `numpy`
   - `yaml`
   - `torch`
2. `python -m cnn_fpga.model.train --help` 可正常返回

这足以证明：

- 当前训练入口在本机上可被解释器加载；
- 当前训练候选环境与 recovery smoke 环境已经清晰分离。

这不证明：

- 全部训练配置都已逐一复验；
- 任意机器都能直接复用同一环境；
- `.tflite` 导出 / runtime 已随训练链一并恢复。

## 7. 当前未覆盖项

本文件当前故意不承诺：

1. 跨机器可复现的完整训练依赖锁定；
2. `requirements-train.txt` 已经完整枚举所有可选依赖；
3. `.tflite` export / runtime 环境；
4. `real_board` HIL 相关依赖；
5. 任何训练长跑已经重新验收。

## 8. 与其他 bootstrap 的关系

当前三条环境说明应保持分离：

1. `requirements-recovery.txt`
   - 只覆盖 `P0/P3/P4 recovery smoke`
2. `docs/evidence_packs/training_reproducibility/training_chain_bootstrap.md`
   - 只覆盖训练链入口与环境边界
3. 后续 `T18`
   - 再单独处理 `.tflite` export/runtime manifest 与边界 smoke

## 9. 推荐表述

后续文档若引用训练链，建议统一写法为：

`训练链当前推荐在本机的 DLEnv 解释器下运行；该说明只覆盖训练入口与最小导入检查，不等于跨机器完整依赖保证，也不等于 .tflite 或真板路径已恢复。`
