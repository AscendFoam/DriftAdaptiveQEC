# 项目快照

## 1. 快照目的

本文件用于把 `2026-05-05` 恢复期开始时的项目现状固定下来，作为后续整理和继续开发的基准面。

## 2. 基本信息

- 快照日期：`2026-05-05`
- 当前分支：`main`
- 恢复策略：按 `docs/reference/AI_coding_workflow.md` 第 4 节“已开发一部分项目的工作流”执行
- 当前决策建议：`Repair`

## 3. 项目身份

该项目围绕 “CNN + FPGA 快慢回路进行近似 GKP 码解码” 展开，当前主代码与实验材料主要分布在：

- `physics/`
- `cnn_fpga/`
- `benchmark/`
- `docs/`

从文档与代码交叉读取后的判断是：

- 这不是空壳仓库
- 也不是只停留在 idea 或设计稿
- 它已经积累了较完整的 P0-P4 代码路径、实验配置和结果目录
- 但仓库缺少统一治理层，默认环境也尚未恢复到“开箱可跑”

## 4. 恢复前观察到的仓库现实

### 4.1 治理文件缺失

恢复前，仓库根目录中没有以下最小治理文件：

- `README.md`
- `AGENTS.md`
- `CLAUDE.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

### 4.2 代码主干存在

已确认存在以下主模块：

- `cnn_fpga/data/`
- `cnn_fpga/model/`
- `cnn_fpga/decoder/`
- `cnn_fpga/runtime/`
- `cnn_fpga/hwio/`
- `cnn_fpga/benchmark/`
- `physics/`

### 4.3 结果与生成物很多

仓库内已有大量：

- `runs/`
- `artifacts/`
- 自动生成配置
- `__pycache__/` / `.pyc`

这说明项目确实跑过很多轮实验，但也意味着仓库噪声较大，恢复期需要明确区分“源码、治理文件、实验产物、缓存文件”。

## 5. 已做的最小验证

### 5.1 成功

命令：

```powershell
python --version
```

结果：

- 默认解释器为 `Python 3.13.7`

命令：

```powershell
python -c "import cnn_fpga, physics; print(cnn_fpga.__version__)"
```

结果：

- 可导入本地包
- `cnn_fpga.__version__ = 0.1.0`

### 5.2 失败

命令：

```powershell
python benchmark/compare_full_vs_simplified_ler.py --n-rounds 10 --repeats 2 --no-plot --output-dir runs/smoke_test
```

结果：

- 失败
- 报错：`ModuleNotFoundError: No module named 'numpy'`

### 5.3 当前解释

这说明：

1. 默认 Python 解释器不是完全空的，但也不是项目可运行环境
2. 当前最优先阻塞项不是算法实现，而是依赖矩阵与最小入口恢复

## 6. 当前已确认的关键边界

### 6.1 软件 HIL 与真板 HIL 边界

- `cnn_fpga/benchmark/run_hil_suite.py` 已支持 mock/backend 抽象
- `cnn_fpga/hwio/board_backend.py` 仍然是 placeholder 风格的真板后端骨架
- 文档中也已明确：当前更准确状态是 `P3-软件 HIL 完成，P3-真板 HIL 待完成`

### 6.2 benchmark 代码主线存在

已确认存在：

- `benchmark/compare_full_vs_simplified_ler.py`
- `cnn_fpga/benchmark/run_hil_suite.py`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`

这说明恢复期不需要从零设计实验入口，而要优先确认它们在当前环境下是否还能真实跑通。

## 7. 本轮恢复参考的背景文档

- `docs/CNN_FPGA_GKP_工程化实验方案.md`
- `docs/CNN_FPGA_GKP_阶段结论.md`
- `docs/CNN_FPGA_GKP_后续仿真与工程补强计划.md`
- `docs/CNN_FPGA_GKP_paper_inspired分支实验设计草案.md`
- `docs/reference/AI_coding_workflow.md`

## 8. 快照结论

当前项目的核心问题不是“没有内容”，而是“已有大量内容，但缺少可靠的治理层和默认可运行入口”。因此恢复期第一目标应为：

- 先恢复可验证性
- 再恢复可复现性
- 最后才恢复新功能推进
