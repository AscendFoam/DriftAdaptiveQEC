# Repo Noise Governance

## 1. 目的

`T5` 的目标不是立刻把仓库“清干净”，而是先把当前仓库中的缓存、生成物与历史运行噪声分类清楚，固定恢复期可执行的治理口径，避免后续 `T6/T7` 的最小复验继续和历史产物混在一起。

本文件只定义恢复期规则，不宣称仓库已经完成物理清理。

## 2. 当前噪声清点

基于 `2026-05-07` 的只读清点，当前可确认的事实如下：

- `.gitignore` 已经忽略：
  - `__pycache__/`
  - `runs/`
  - `artifacts/`
- 但 Git 历史中仍已有大量已跟踪噪声，这说明当前问题主要是“历史已跟踪文件”，而不是“缺少忽略规则”。
- 已跟踪缓存/字节码文件：`116`
  - 分布于 `9` 个 `__pycache__` 目录
- 当前工作区 `.pyc` 文件总数：`133`
- 已跟踪 `runs/` 文件数：`1841`
- 已跟踪 `artifacts/` 文件数：`110`

代表性内容：

- `runs/`
  - 包含 `drift_suite`、`hardware_emulation`、`hil_mode_benchmark`、`hil_suite`、`p4_*` 等历史运行输出
- `artifacts/`
  - 包含数据集 `.npz`
  - 模型 `.npz`
  - `.tflite`
  - `.tflite.json`
  - 训练、导出、评估报告
- 当前恢复期最小 software HIL 路径仍依赖：
  - `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`

## 3. 噪声分类与治理口径

| 类别 | 路径模式 | 当前状态 | Phase 0 是否保留 | 当前治理口径 | 后续动作 |
| --- | --- | --- | --- | --- | --- |
| A | `**/__pycache__/`, `*.pyc` | 已忽略，但历史上已有已跟踪文件 | 是 | `T5` 不做批量删除或 untrack；后续禁止再把新缓存文件纳入版本库 | 未来单开有界 cleanup 任务，将其清理出版本库 |
| B | `runs/` | 已忽略，但历史运行结果大量已跟踪 | 是 | 视为历史证据，不视为新的事实来源；`T6/T7` 必须引用具体 run dir，而不是笼统引用 `runs/` | 后续区分“保留摘要”与“归档移出” |
| C | `artifacts/` | 已忽略，但历史模型/数据/报告已跟踪 | 是 | 暂保留，因为当前恢复期最小路径仍依赖部分 `.npz` artifact；不可默认把整个目录当作源码一部分 | 后续拆分为“bootstrap 必需”和“历史归档”两类 |
| D | 本地临时文档文件，例如 `*.drawio.dtmp` | 容易制造工作树噪声 | 否 | 立即通过 `.gitignore` 忽略，不进入治理结论文档 | 保持忽略即可 |

## 4. 恢复期立即执行的规则

1. `T5` 只做治理，不做破坏性清理。
2. 在专门 cleanup 任务出现前，不执行：
   - `git rm --cached` 批量移除 `__pycache__/`
   - `git rm --cached` 批量移除 `runs/`
   - `git rm --cached` 批量移除 `artifacts/`
   - 批量删除历史 `.pyc`
3. 后续 `T6/T7` 若产生新的运行结果，允许本地落盘，但必须：
   - 在文档里写清解释器、命令、run dir、backend、artifact type
   - 不把历史结果目录直接写成“当前事实来源”
4. 当前恢复期最小路径依赖的历史 artifact，可以继续被引用，但要显式写清具体文件路径。
5. `board_backend.py` 仍是 placeholder real-board backend；`.tflite` 真路径与 stub 路径仍按 `T3/T4` 的既有口径区分，不能被噪声治理任务改写语义。

## 5. T5 实际落地动作

本轮实际执行了以下非破坏性动作：

1. 新增本治理文档，固定噪声分类与阶段口径。
2. 新增 `T5` 任务包，限制本轮作用域只在治理文档侧。
3. 给 `.gitignore` 增补 `*.drawio.dtmp`，避免 Draw.io 临时文件继续制造工作树噪声。
4. 同步更新 task board、decision log、handoff、legacy audit 与风险文档。

本轮没有执行：

- `runs/` 清理
- `artifacts/` 清理
- `__pycache__/` / `.pyc` untrack
- 任意 benchmark 口径改写

## 6. 后续清理建议

物理 cleanup 应在后续单独任务中进行，并至少满足以下条件：

1. 先完成 `T6/T7`，避免在最小复验前误伤当前可复用入口。
2. cleanup 任务必须提供明确 manifest：
   - 哪些 `__pycache__/` / `.pyc` 会被移出版本库
   - 哪些 `runs/` 只保留摘要
   - 哪些 `artifacts/` 被认定为 bootstrap 必需
3. cleanup 前必须明确回滚方式和验收标准。

## 7. 对后续任务的约束

- `T6` 应继承 `T4` 的最小 software HIL bootstrap，而不是扩写成真板能力。
- `T7` 若进入 P4 最小复验，仍必须显式写清 backend 与 inference artifact type。
- 在 `T8` 决策前，不应因为仓库中仍有历史 `runs/` / `artifacts/` 就把项目误判为“已经整理完成”。
