# T39 任务说明与 Review 解释

## 1. 通俗解释：T39 在做什么

### 1.1 用日常比喻来理解

如果 T31 是"打开厨房门，写了一份说明书，说清需要什么工具"，那么 T39 就是"真的去租了一间新厨房，买了说明书中写的最小工具集，然后试了试炉子能不能点着"。

T31 只看了配置文件和源码，得出结论："按道理，训练链不需要 torch，只用 numpy + PyYAML 应该就行。"

T39 把这个"按道理"变成了"实际试过了"：

- 在一台完全新的 Python 3.12 环境里（不是 DLEnv）
- 只装了 `numpy` 和 `PyYAML`
- 跑了三条最小验证命令
- 三条都通过了

但 T39 **没有**真的做菜（跑训练）。它只是验证了"炉子能点着、锅能放上去"。

### 1.2 为什么这一步在 T31 之后

T31 留下的状态是：

- 计划上说明了 CPU-only 路径可能成立
- 但没有实际创建过干净环境
- 没有实际验证过依赖是否足够

Milestone 2I review 的最弱项是 clean-environment reproducibility。T31 回答了"理论上需不需要 torch"，T39 回答了"实践中不装 torch 行不行"。

---

## 2. 详细解释：任务的实现

### 2.1 任务目标

T39 的目标是在 T31 的计划基础上，创建一个 CPU-only 的干净 Python 3.12 环境，安装最小依赖，并只用 dry-run/import 级别的命令验证训练链入口。

关键限制：

- 不跑训练、不跑 benchmark、不跑 `.tflite`、不碰硬件、不做 cleanup
- 不改源码、配置、`requirements-recovery.txt`
- 不把 DLEnv 的事实当作可移植保证
- 如果环境创建失败（比如网络问题），应该精确报告 blocker，不能退而改 DLEnv

### 2.2 任务流程

**第一步：创建干净环境**

Worker 用 base Anaconda（Python 3.12.7）创建了一个全新的 `venv`：

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m venv .venvs/t39_train_cpu_py312
```

这个环境：
- 不是 DLEnv 的副本
- 没有 torch
- 没有 CUDA
- 被 `.gitignore` 忽略，不会被提交

**第二步：安装最小依赖**

```powershell
& '.venvs\t39_train_cpu_py312\Scripts\python.exe' -m pip install numpy PyYAML
```

结果：
- `numpy==2.4.5`（比 DLEnv 的 2.2.4 更新）
- `PyYAML==6.0.3`（比 DLEnv 的 6.0.2 更新）

注意：第一次安装尝试因为沙箱网络限制失败了，Worker 正确地没有退回 DLEnv，而是在获得批准后重试同一命令。

**第三步：Dry-run/import 级别验证**

在干净环境中运行三条命令：

1. `dataset_builder --dry-run` → 打印数据集构建计划，不写文件
2. `runtime_dataset_builder --dry-run` → 打印运行时数据集构建计划，不写文件
3. `train --help` → 打印 CLI 帮助信息

三条全部通过。

**第四步：产出依赖规范**

创建了 `requirements-train-cpu-win-py312.txt`：

```
numpy==2.4.5
PyYAML==6.0.3
```

这个文件：
- 只有两个包，精确版本锁定
- 没有 `file:///C:/...` 本地引用
- 头部注释明确写了验证范围和非验证范围
- 与 `requirements-recovery.txt` 完全独立

### 2.3 文档变化

T39 产出了 5 个文件：

1. **`requirements-train-cpu-win-py312.txt`**（新建）
   - CPU-only 草案依赖规范，Windows + Python 3.12 专用

2. **`docs/training_chain_cpu_cleanenv_bootstrap.md`**（新建）
   - 完整的 clean-env bootstrap 记录：环境创建命令、安装命令、验证命令、结果、验证范围、非声明

3. **`docs/tasks/Phase2/T39_training_chain_cpu_cleanenv_draft_lock.md`**（修改）
   - 追加了 Worker Output 和 Verification Record

4. **`docs/review/T39_review.md`**（新建）
   - Worker 自检 + reviewer adversarial review

5. **`docs/for_human/T39_explanation.md`**（新建）
   - 面向人类的通俗说明

### 2.4 对后续开发的意义

T39 的核心贡献是把 T31 的"计划层面"推进到了"实际验证层面"：

1. **R11 进一步缩窄**：T31 证明了 CPU-only 路径"配置上可行"，T39 证明了"实践中可运行"
2. **依赖分离**：现在仓库有两份独立的依赖规范：
   - `requirements-recovery.txt`：只覆盖 P0/P3/P4 recovery smoke（numpy + PyYAML）
   - `requirements-train-cpu-win-py312.txt`：只覆盖训练链 dry-run/import 级别（numpy + PyYAML，但版本不同、范围不同）
3. **DLEnv 不再是唯一选项**：训练链至少有一个 CPU-only 候选路径，可以在没有 GPU 的环境下使用
4. **下一步入口**：如果需要进一步关闭 R11，可以在同一个 clean env 中跑一次真实训练

与项目其他文档的关系：

- `docs/training_chain_bootstrap.md`（T17）：记录了 DLEnv 为推荐环境；T39 的 bootstrap 文档补充了 CPU-only clean env 证据
- `docs/training_chain_portable_dependency_lock_plan.md`（T31）：T39 执行了 T31 Section 7 推荐的 bootstrap procedure
- `docs/08_risks_and_open_questions.md`：R11 应随 T39 更新为"进一步缩窄"
- Milestone 2J（Reproducibility And Deployment Boundary）：T39 是这个 milestone 的第二个交付项

---

## 3. 为什么我给出了 PASS 的 review 结果

### 3.1 任务完成度

T39 要求的 5 个产出全部存在。不只是占位——每个文件都有实质内容。

### 3.2 独立验证全部通过

我没有只看 Worker 的报告，而是自己用干净环境重新跑了三条验证命令：

- `dataset_builder --dry-run`：通过了，只打印计划，没有写文件
- `runtime_dataset_builder --dry-run`：通过了，只打印计划，没有写文件
- `train --help`：通过了，打印 CLI 帮助

环境元数据也独立验证了：Python 3.12.7、numpy 2.4.5、PyYAML 6.0.3，没有 torch。

### 3.3 没有越界

只有 5 个文件变更，全部在 Allowed Files 列表内。`runs/`、`artifacts/`、源码、配置、`requirements-recovery.txt` 均无变更。`.venvs/` 在 `.gitignore` 中，不会被提交。

### 3.4 干净环境确实干净

关键检查点：

- 创建自 base Anaconda（不是 DLEnv 的副本）
- 只有 numpy + PyYAML + pip，没有任何 torch/CUDA 包
- Worker 没有在安装失败时退回 DLEnv

### 3.5 诚实性

文档中没有过度声明：

- 明确写了"这不是完整训练可复现性证明"
- 8 项显式非声明
- R11 描述为"缩窄但未关闭"
- 依赖规范头部注释区分了已验证和未验证范围

### 3.6 发现的非阻塞问题

- **N1**：依赖版本是安装时的当前版本，不是兼容性矩阵分析的结果。对于草案级别这是合理的。
- **N2**：没有记录 `pip freeze` 输出，但只有 2 个包，`pip list` 等价。
- **N3**：Worker 记录了沙箱安装失败和批准重试，处理透明。

这三个问题都不影响 T39 的实质产出质量。

### 3.7 为什么不是 PASS_WITH_WARNINGS

T39 是一个 bounded reproducibility/bootstrap 任务。它的核心价值在于"在干净环境里验证最小依赖集"。在这个标准下：

- 干净环境确实独立于 DLEnv
- 三条验证命令确实通过（我独立复验了）
- 依赖规范确实干净、最小、诚实
- 没有越界操作
- 没有过度声明

N1-N3 都是 accepted 级别的小观察，不影响 verdict。
