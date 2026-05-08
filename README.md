# DriftAdaptiveQEC

`DriftAdaptiveQEC` 是一个围绕 “CNN + FPGA 快慢回路协同近似 GKP 解码” 的研究型工程仓库。当前代码已经覆盖物理仿真、数据集生成、Tiny-CNN 训练、量化/导出、软件侧 HIL 与 P4 多场景 benchmark，但仓库此前缺少统一治理层，因此本仓库现已进入“恢复可信度优先”的整理阶段。

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
- 当前阶段：`Phase 1: Recovery`
- 当前决策状态：`Repair`
  - 原因：核心代码与实验资产明显有价值，但默认环境、入口说明、治理文件和可复现性仍未恢复到可稳定接力的状态。
  - 截至 `2026-05-08`，最小 software HIL 与最小 P4 benchmark recovery path 都已复验通过，但当前仍只到 recovery smoke 级别，不等于正式全量 benchmark 已恢复。

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

这些入口目前只代表“代码中存在并被文档引用”，不代表已经在当前默认环境中重新验收通过。恢复期的当前唯一任务见 `docs/04_task_board.md` 与 `docs/07_handoff.md`。

## 环境说明

当前仓库内没有已确认可直接复现的根级依赖说明文件（如 `requirements.txt`、`pyproject.toml` 或 `environment.yml`）。

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

因此，在继续任何新功能或新 benchmark 之前，请先完成：

1. 依赖矩阵确认
2. 最小可运行入口恢复
3. smoke 级验证回写到治理文档

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

## 复用建议

- 如果目标只是恢复期最小 smoke，优先用 `C:\ProgramData\anaconda3\python.exe`
- 如果目标是后续 torch 训练或更重的模型实验，优先切到 `C:\ProgramData\anaconda3\envs\DLEnv\python.exe`
- `DLEnv` 是 legacy 开发常用环境，但不应反向取代恢复期最小 smoke 口径

## 工作方式

- 项目状态以仓库文件为准，不以聊天上下文为准
- 恢复期默认不新增功能，优先修复环境、入口、验证与文档一致性
- 不把 `mock`、`placeholder`、未来计划或未复验结果写成“已完成事实”

具体协作规范见 `AGENTS.md` 与 `CLAUDE.md`。
