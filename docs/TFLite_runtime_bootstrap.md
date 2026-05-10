# TFLite Runtime Bootstrap

## 1. 目的

本文件只服务 `.tflite` export/runtime 路径的独立接力，不替代 `docs/training_chain_bootstrap.md`，也不替代 `requirements-recovery.txt`。

目标是固定四件事：

1. 真实 `.tflite` export/runtime 需要什么；
2. `tflite_stub_v1` 是什么边界；
3. 现有最小 smoke 命令是什么；
4. 当前环境为什么仍然不能把真实 TFLite 写成已恢复事实。

## 2. 当前边界判断

`.tflite` 路径目前有两种明确语义：

1. 真实 `.tflite`
   - `cnn_fpga/model/export.py` 会优先尝试 TensorFlow 导出。
   - `cnn_fpga/runtime/inference_service.py` 会在 `TFLiteHistogramPredictor` 中走真实 TFLite 解释器。
2. `tflite_stub_v1`
   - 导出失败时，`export.py` 回退生成 `.tflite.json` manifest。
   - runtime 读取该 manifest 后，`inference_service.py` 以 `source="tflite_stub_service"` 运行。
   - 这条路径可用于管线验证，但不能写成真实 TFLite 已部署。

## 3. 当前机器上的可用事实

截至 `2026-05-10`，本机验证结果为：

- `tensorflow`：未安装
- `tflite_runtime`：未安装

因此当前机器上：

- 真实 `.tflite` export/runtime 不能被视为已可用；
- 只能明确说明 `tflite_stub_v1` 回退路径存在。

## 4. 相关入口

### 4.1 导出入口

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m cnn_fpga.model.export --config <config_path>
```

行为：

1. 优先尝试真实 `.tflite` 导出；
2. 若缺 TensorFlow / 导出失败，则回退写出 `.tflite.json` stub manifest。

### 4.2 真实 TFLite 评估入口

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m cnn_fpga.model.evaluate_tflite --config <config_path> --tflite-path <path>
```

该入口要求可用的 TFLite 解释器环境；当前机器上尚未满足。

### 4.3 导出一致性验证入口

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m cnn_fpga.model.validate_export --config <config_path> --artifact-path <artifact.npz> --tflite-path <path>
```

该入口用于比较源 artifact 与导出产物的数值一致性。

## 5. 真实 `.tflite` 的依赖边界

真实 `.tflite` 路径至少依赖：

- `tensorflow` 或 `tflite_runtime`
- 对应的模型 artifact
- 可用的数据集 split

如果使用 `subprocess` 推理服务，还需要：

- `slow_loop.inference_service.mode=tflite`
- `slow_loop.inference_service.backend=tflite`
- `python_executable` 指向带有 TFLite 解释器的环境

## 6. `tflite_stub_v1` 的边界

`tflite_stub_v1` 的作用是：

1. 给导出链提供可追溯、可保存的回退清单；
2. 允许 runtime 通过 `tflite_stub_service` 继续跑通接口层；
3. 不要求真实 TensorFlow / TFLite 运行时存在。

它不代表：

- 真实 `.tflite` 解释器可用；
- 导出语义已与真实 TFLite 逐字一致；
- `.tflite` 已完成跨机器部署验收。

## 7. 当前最小 smoke

本轮已完成的最小只读验证：

1. `cnn_fpga/model/export.py --help`
2. `cnn_fpga/model/evaluate_tflite.py --help`
3. `cnn_fpga/model/validate_export.py --help`
4. 环境探测确认：
   - `tensorflow = False`
   - `tflite_runtime = False`

因此当前最准确的 smoke 结论是：

- 命令入口存在；
- 真实 runtime 依赖未满足；
- stub 路径仍是当前可接力的唯一可运行边界。

## 8. 未覆盖项

本文件当前故意不承诺：

1. 真实 `.tflite` 已在当前机器上复验通过；
2. `tflite_stub_v1` 等同于真实 TFLite；
3. `.tflite` 路径已与 HIL benchmark 完整闭环；
4. 真板 backend 已完成。

## 9. 与其他 bootstrap 的关系

1. `requirements-recovery.txt`
   - 只覆盖 `P0/P3/P4 recovery smoke`
2. `docs/training_chain_bootstrap.md`
   - 只覆盖训练链入口与环境边界
3. `docs/TFLite_runtime_bootstrap.md`
   - 只覆盖 `.tflite` export/runtime 与 stub 边界
4. 后续 `T19/T20`
   - 再分别处理 repo cleanup 与 real-board readiness

## 10. 推荐表述

后续文档若引用 `.tflite`，建议统一写法为：

`当前仅确认 `.tflite` 导出/runtime 代码路径存在；真实 TensorFlow / TFLite 运行时在本机尚未可用，因此现阶段只能把 tflite_stub_v1 视为可追溯回退路径，不能视为真实 TFLite 部署已完成。`
