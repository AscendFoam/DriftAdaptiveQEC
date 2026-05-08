# AGENTS

本仓库采用恢复期治理模式，目标不是继续堆新功能，而是先把已有成果整理成“可验证、可复现、可接力”的状态。

## 当前阶段

- 当前阶段：`Phase 1: Recovery`
- 决策状态：`Repair`
- 当前唯一任务以 `docs/04_task_board.md` 和 `docs/07_handoff.md` 为准

## 开始任何工作前必须阅读

1. `README.md`
2. `docs/00_project_snapshot.md`
3. `docs/01_legacy_audit.md`
4. `docs/02_experiment_plan.md`
5. `docs/04_task_board.md`
6. `docs/07_handoff.md`
7. `docs/08_risks_and_open_questions.md`

如果任务涉及研究背景或阶段结论，再补读：

- `docs/CNN_FPGA_GKP_工程化实验方案.md`
- `docs/CNN_FPGA_GKP_阶段结论.md`
- `docs/CNN_FPGA_GKP_后续仿真与工程补强计划.md`
- `docs/CNN_FPGA_GKP_paper_inspired分支实验设计草案.md`

## 角色约束

### Captain

- 负责拆任务、控范围、收口文档
- 不直接顺手扩功能
- 每轮必须明确：
  - 当前唯一任务
  - Allowed files
  - Forbidden scope
  - Verification
  - Docs to update

### Worker

- 只完成当前任务包
- 只改 Allowed files
- 不自动领取下一任务
- 完成后必须汇报：
  - 改了什么
  - 怎么验证
  - 剩余风险

### Reviewer

- 默认只读
- 优先查：
  - 文档与代码是否一致
  - 是否把 mock/placeholder 写成完成态
  - benchmark 是否公平
  - 环境假设是否被偷偷省略
  - 结果是否可复现

## 当前仓库的特殊硬规则

1. 不得把 `P3-软件 HIL` 写成 `P3-真板 HIL 已完成`
2. 不得把 `board_backend.py` 的 placeholder 语义写成真实板级完成
3. 不得静默改动正式 benchmark 口径、baseline 集合、ParamMapper 主线语义
4. 不得把 `runs/`、`artifacts/` 中的历史结果改写为新的“事实来源”
5. 不得在恢复期顺手启动新的 teacher-representation 长跑
6. 不得跳过验证就更新阶段结论类文档

## 恢复期默认允许的任务类型

- 确认依赖与入口
- 恢复最小 smoke path
- 标记 mock/stub/placeholder 边界
- 补治理文件
- 补最小测试或最小 bootstrap 说明
- 修复阻塞运行的最小问题

## 恢复期默认禁止的任务类型

- 大规模重构
- 新模型主线切换
- 新论文分支大规模展开
- 真板联调语义扩写成既成事实
- 在无验证前提下重写阶段结论文档
