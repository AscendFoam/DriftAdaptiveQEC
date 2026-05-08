# DriftAdaptiveQEC

`DriftAdaptiveQEC` 是一个围绕 “CNN + FPGA 快慢回路协同近似 GKP 解码” 的研究型工程仓库。当前代码已经覆盖物理仿真、数据集生成、Tiny-CNN 训练、量化/导出、软件侧 HIL 与 P4 多场景 benchmark。仓库现已完成第一轮恢复期治理收尾，进入“受控继续开发”阶段。

## 当前状态

- 研究背景与阶段结论主要见：
  - `docs/CNN_FPGA_GKP_工程化实验方案.md`
  - `docs/CNN_FPGA_GKP_阶段结论.md`
  - `docs/CNN_FPGA_GKP_后续仿真与工程补强计划.md`
  - `docs/CNN_FPGA_GKP_paper_inspired分支实验设计草案.md`
- 自 `2026-05-05` 起，项目治理以以下文件为准：
  - `docs/00_project_snapshot.md`
  - `docs/01_legacy_audit.md`
  - `docs/02_experiment_plan.md`
  - `docs/04_task_board.md`
  - `docs/07_handoff.md`
  - `docs/08_risks_and_open_questions.md`
- 当前阶段：`Phase 2: Controlled Development`
- 当前决策状态：`Go`
  - 原因：截至 `2026-05-08`，治理文件、recovery-scoped manifest、最小 P0/P3/P4 路径与 bounded software HIL 的确定性表述都已收口到可稳定接力状态。
  - 边界：这个 `Go` 只代表“允许继续开发”，不代表 `real_board`、真实 `.tflite` runtime 或正式多场景 P4 benchmark 已恢复。

## 仓库结构

- `physics/`: GKP 物理层与逻辑错误追踪
- `cnn_fpga/`: 数据、模型、解码器、运行时、HIL、benchmark 主模块
- `benchmark/`: P0 基础对比脚本
- `docs/`: 方案、阶段结论、恢复治理文件
- `runs/`, `artifacts/`: 运行产物与历史证据

## 当前已确认的入口

- P0 基线脚本：
  - `python benchmark/compare_full_vs_simplified_ler.py --n-rounds 10 --repeats 2 --no-plot --output-dir runs/smoke_test`
- 训练入口：
  - `python -m cnn_fpga.model.train --config cnn_fpga/config/experiment_static_theta_v2.yaml`
- HIL 入口：
  - `python -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil.yaml`
- P4 入口：
  - `python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_hybrid_b_long.yaml`

这些入口代表“代码中存在”；其中当前已经重新验收通过的 bounded recovery 路径，请以 `docs/P0_smoke_bootstrap.md`、`docs/P3_software_hil_bootstrap.md` 与 `docs/P4_benchmark_recovery_bootstrap.md` 为准。当前唯一任务状态见 `docs/04_task_board.md` 与 `docs/07_handoff.md`。

## 环境说明

当前仓库根目录现已补一份 recovery 期最小依赖说明文件：

- `requirements-recovery.txt`

它只覆盖当前已复验的 `P0/P3/P4 recovery smoke` 路径，不等于完整训练链、`.tflite` runtime 或 `real_board` HIL 全环境。

如果只是先把当前 recovery 路径装到一个新解释器里，可执行：

```powershell
python -m pip install -r requirements-recovery.txt
```

截至 `2026-05-06`，本机已确认的解释器分工如下：

- `C:\Python313\python.exe`
  - 有 `yaml`
  - 无 `numpy / torch / tensorflow`
  - 不适合作为项目运行解释器
- `C:\ProgramData\anaconda3\python.exe`
  - 有 `numpy + yaml`
  - 无 `torch / tensorflow`
  - 当前恢复期推荐的最小 smoke 解释器
- `C:\ProgramData\anaconda3\envs\DLEnv\python.exe`
  - 有 `numpy + yaml + torch`
  - `torch.cuda.is_available() = True`
  - 适合作为后续训练环境候选
  - 这是 legacy 开发常用环境
  - 但当前不作为恢复期最小 smoke 解释器
- `C:\ProgramData\anaconda3\envs\QuantumEnv\python.exe`
  - 有 `numpy + yaml + torch`
  - 可作为训练/实验候选环境
- `C:\ProgramData\anaconda3\envs\TF1_14\python.exe`
  - 有 `tensorflow`
  - 缺 `yaml`
  - 当前不适合作为完整仓库入口环境

另一个关键事实是：仓库和工作区内未找到文档中多次提到的 `.venvs/tf311`，所以它目前不能被当成现成可用前提。

当前还没有“一次覆盖整个仓库所有路径”的统一根级环境文件；训练链、`.tflite` runtime 与真板 HIL 仍需继续按恢复期边界单独说明。

因此，在继续任何新功能或新 benchmark 之前，请先确认：

1. 依赖矩阵确认
2. 当前任务是否有清晰 task package
3. 验证结果是否会回写到治理文档

## 当前推荐最小入口

恢复期当前推荐的最小 smoke 命令为：

```powershell
& 'C:\ProgramData\anaconda3\python.exe' benchmark/compare_full_vs_simplified_ler.py --n-rounds 10 --repeats 2 --no-plot --output-dir runs/smoke_test_anaconda
```

截至 `2026-05-06`，该命令已在当前机器上跑通，输出位于：

- `runs/smoke_test_anaconda/n10_r2_s0.250_ler_curve_compare.csv`
- `runs/smoke_test_anaconda/n10_r2_s0.250_summary.json`

最小 smoke 的复用说明已整理到：

- `docs/P0_smoke_bootstrap.md`

如果目标是 `P3/P4 recovery smoke`，请继续参照：

- `docs/P3_software_hil_bootstrap.md`
- `docs/P4_benchmark_recovery_bootstrap.md`

## 复用建议

- 如果目标只是恢复期最小 smoke，优先用 `C:\ProgramData\anaconda3\python.exe`
- 如果目标是后续 torch 训练或更重的模型实验，优先切到 `C:\ProgramData\anaconda3\envs\DLEnv\python.exe`
- `DLEnv` 是 legacy 开发常用环境，但不应反向取代恢复期最小 smoke 口径
- `requirements-recovery.txt` 只承诺 `numpy + PyYAML` 这一层的 recovery smoke 依赖，不承诺训练、`.tflite` 或真板环境已经恢复

## 工作方式

- 项目状态以仓库文件为准，不以聊天上下文为准
- 当前已退出恢复期，但继续开发仍必须是有界任务，且要保持验证与文档一致性
- 不把 `mock`、`placeholder`、未来计划或未复验结果写成“已完成事实”

具体协作规范见 `AGENTS.md` 与 `CLAUDE.md`。
