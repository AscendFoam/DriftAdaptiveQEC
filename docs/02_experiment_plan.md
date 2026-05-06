# 恢复期实验与工程计划

## 1. 定位

本文件不是替代现有研究方案文档，而是把当前仓库带回“可继续开发”的最小执行计划。

完整研究背景仍以以下文档为准：

- `docs/CNN_FPGA_GKP_工程化实验方案.md`
- `docs/CNN_FPGA_GKP_阶段结论.md`
- `docs/CNN_FPGA_GKP_后续仿真与工程补强计划.md`
- `docs/CNN_FPGA_GKP_paper_inspired分支实验设计草案.md`

本文件只解决一个问题：

- 接下来如何先恢复仓库可信度，再恢复实验推进能力

## 2. 当前恢复期 MVP

恢复期 MVP 不是“再做一个更强模型”，而是：

1. 明确当前支持的运行环境
2. 跑通一个最小 P0/P3/P4 入口中的至少一个 smoke path
3. 把 mock、software HIL、real board placeholder 的边界写清
4. 为后续继续 teacher-representation 或 runtime 补强建立可信接力面

## 3. 当前阶段明确不做什么

恢复期完成前，不做以下事情：

1. 不新增 teacher-representation 长跑
2. 不切换正式 benchmark 主线语义
3. 不把 board backend 扩写成“已真板完成”
4. 不重写历史阶段结论为宣传稿
5. 不做大规模架构重构

## 4. 输入输出

### 输入

- `physics/` 现有物理与逻辑错误代码
- `cnn_fpga/` 现有数据、模型、runtime、HIL 与 benchmark 代码
- `docs/` 中已有阶段方案和结论
- `runs/` 与 `artifacts/` 中的历史结果证据

### 输出

- 统一治理文件
- 依赖/环境说明
- 最小可运行入口说明
- 至少一个 smoke 级验证结果
- 明确的下一唯一任务

## 5. 目录与核心模块

恢复期重点涉及：

- `benchmark/compare_full_vs_simplified_ler.py`
- `cnn_fpga/model/train.py`
- `cnn_fpga/benchmark/run_hil_suite.py`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- `cnn_fpga/hwio/board_backend.py`
- `docs/` 下治理文件

## 6. 恢复期执行流程

### M0: 治理恢复

1. 冻结旧状态
2. 完成 legacy audit
3. 建立 README / AGENTS / CLAUDE / task board / handoff / risks

### M1: 入口与依赖恢复

1. 确认项目需要的 Python 环境与核心依赖
2. 识别当前支持的解释器路径
3. 跑通一个最小 smoke 命令，或把阻塞记录成明确可修问题

### M2: 最小 benchmark 可信度恢复

1. 验证至少一条 P0 或 P3 软件 HIL 最小路径
2. 明确哪些 benchmark 只在特定环境下运行
3. 为后续 P4 或 teacher 分支恢复建立最小测试/脚本

## 7. 评价指标

恢复期指标不以 LER 排名为主，而以“可验证性指标”为主：

1. 能否定位并使用正确运行环境
2. 能否跑通最小入口
3. 文档是否与代码一致
4. 是否明确区分 mock、真实 `.tflite`、board placeholder
5. 是否形成可继续接力的唯一任务板

## 8. 失败判据

出现以下任一情况，恢复期不得宣称完成：

1. 仍无明确环境说明
2. 最小入口仍不可运行且阻塞未被清晰记录
3. 治理文件与代码现实不一致
4. 真板能力被表述为已完成
5. 当前唯一任务不明确

## 9. 三个里程碑

### 里程碑 1：Stabilization

目标：

- 建立治理层，固定项目真实现状

验收：

- `README.md`
- `AGENTS.md`
- `CLAUDE.md`
- `docs/00_project_snapshot.md`
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

### 里程碑 2：Bootstrap

目标：

- 恢复依赖矩阵和最小可运行入口

验收：

- 明确支持的 Python/依赖组合
- 至少一个 smoke 命令可成功执行，或阻塞被清晰记录
- handoff 中给出下一唯一任务

### 里程碑 3：Recovery-Ready

目标：

- 让项目重新进入受控开发

验收：

- MVP 范围清楚
- 最小路径可跑
- mock/stub/placeholder 边界清楚
- 后续任务可按单任务循环继续推进
