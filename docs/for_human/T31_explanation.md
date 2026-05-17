# T31 任务说明与 Review 解释

## 1. 通俗解释：T31 在做什么

### 1.1 用日常比喻来理解

想象你有一间专用厨房（`DLEnv` 环境），里面什么工具都有，你在里面做饭（训练模型）一直很顺手。但现在你想告诉别人"你也可以做同样的菜"，问题就来了：

- 你的厨房里有台进口烤箱（`torch` nightly CUDA build），但菜谱上其实没说一定要用烤箱
- 你厨房里还堆了很多别的东西（无关包、Conda 本地构建），别人照抄你的厨房清单会很困惑
- 菜谱上写的默认做法其实用普通炉子（NumPy CPU）就行

T31 做的事情就是：**打开你的厨房门，清点一下里面有什么，然后写一份诚实的说明书**，说清楚：

- 菜谱（训练配置）实际上要求什么工具
- 哪些工具只是你厨房特有的，不是必须的
- 如果别人也想做这道菜，最小需要什么
- 但**不**假装你已经在别人的厨房里成功做了一遍

### 1.2 为什么需要这样做

在这之前，项目里有两个关于训练环境的文档：

- `requirements-recovery.txt`：只写了 benchmark recovery 需要的 `numpy + PyYAML`，不管训练
- `docs/training_chain_bootstrap.md`（T17 产出）：只写了"推荐用 DLEnv"，但没有分析训练链到底能不能脱离 torch 跑

这导致一个关键风险：如果有人（或未来的你自己）想在一台新机器上复现训练结果，会误以为必须安装和 DLEnv 一样复杂的环境，包括那个 nightly 版的 CUDA torch。

T31 就是来回答这个问题的。

---

## 2. 详细解释：任务的实现

### 2.1 任务目标

T31 的目标在任务包里写得很清楚：**产出一个可移植依赖锁策略计划**（produce a portable dependency-lock plan for the training chain），但不执行它。

关键限制：

- 不安装/升级/删除任何包
- 不运行训练、benchmark、TFLite、硬件或清理命令
- 不创建 `runs/` 或 `artifacts/`
- 不改源码、配置或 `requirements-recovery.txt`

### 2.2 任务流程

Worker 按以下步骤执行：

**第一步：解释器清点**

探测了三套 Python 解释器：

| 解释器 | 版本 | 关键发现 |
|--------|------|----------|
| `DLEnv` | Python 3.12.9 | 有 numpy、PyYAML、torch（nightly CUDA） |
| base Anaconda | Python 3.12.7 | 有 numpy、PyYAML，没有 torch |
| system Python | Python 3.13.7 | 缺 numpy，不可用 |

**第二步：训练入口依赖追踪**

Worker 分析了三条训练路径的配置和代码：

1. **Static theta 训练**（`experiment_static_theta_v2.yaml`）
2. **Residual-b 训练**（`experiment_runtime_b_residual.yaml`）
3. **Gated v5 / teacher 训练**（`experiment_runtime_b_residual_norm_gated_teacher_v5.yaml`）

核心发现：**这三份配置都没有显式写 `training.tiny_cnn.backend: torch`**。而 `tiny_cnn.py` 的默认值是 `backend="numpy"`，`device="auto"`。

这意味着：在配置语义层面，训练链并不强制要求 torch。torch 是可选加速路径，不是硬依赖。

**第三步：入口验证**

Worker 在没有 torch 的 base Anaconda 下运行了：

- `python -m cnn_fpga.model.train --help` — 成功
- `python -m cnn_fpga.data.dataset_builder --help` — 成功
- `python -m cnn_fpga.data.runtime_dataset_builder --help` — 成功

这进一步证明：代码入口在 import 层面不硬依赖 torch。

**第四步：策略制定**

基于以上发现，Worker 提出了"双 lane"策略：

- **CPU-portable lane**：`numpy + PyYAML`，Python 3.12.x，Windows first
- **GPU-local lane**：`DLEnv` 本机证据，dev torch + CUDA，不写入正式 lock

### 2.3 文档变化

T31 产出了 4 个文件：

1. **`docs/training_chain_portable_dependency_lock_plan.md`**（新建）
   - 核心产出，包含完整的 10 节内容：范围、解释器清单、包证据、入口依赖映射、锁策略、可提交 vs 本机证据、bootstrap 提案、非声明、下一步建议、探针命令记录

2. **`docs/review/T31_review.md`**（新建）
   - Worker 自检记录 + reviewer adversarial review

3. **`docs/for_human/T31_explanation.md`**（新建）
   - 面向人类的通俗说明

4. **`docs/tasks/Phase2/T31_training_chain_portable_dependency_lock_plan.md`**（修改）
   - 追加了 Worker Output 和 Verification Record 两节

### 2.4 对后续开发的意义

T31 的核心贡献是为 Milestone 2J（Reproducibility And Deployment Boundary）打下了训练链可移植性的基础：

1. **证据层面**：明确区分了"本机能跑训练"和"训练链可以脱离 torch 跑"这两个事实
2. **策略层面**：给出了 CPU-only lock 的具体执行路径，包括目标 Python 版本、依赖列表、验证步骤
3. **风险控制**：没有把 DLEnv 的复杂环境误写成可移植保证
4. **后续任务入口**：Section 9 的推荐下一步任务已经固定了明确的 scope、verification 和 forbidden scope

与项目其他文档的关系：

- `requirements-recovery.txt`：T31 确认了它的 scope 不变，只覆盖 recovery smoke
- `docs/training_chain_bootstrap.md`（T17）：T31 补充了 config 级别的依赖分析，两份文档互补但不冲突
- `docs/02_experiment_plan.md`：Section 6.2 的依赖矩阵现在可以引用 T31 的 CPU-lock 发现
- Milestone 2I 的 `Conditional Allow` 最弱项（clean-environment reproducibility）现在有了更具体的执行路径

---

## 3. 为什么我给出了 PASS 的 review 结果

### 3.1 任务完成度

T31 任务包要求的 8 类产出全部存在，且内容实质性地回答了每一项要求。不是空话或占位符。

### 3.2 代码级声明全部可验证

我对 Worker 的核心声明做了独立验证：

| 声明 | 验证方式 | 结果 |
|------|----------|------|
| 三份配置不强制 `backend=torch` | 直接读配置文件 | 确认：都没有写 `backend` 或 `device` |
| `tiny_cnn.py` 默认 `backend="numpy"` | 读源码 dataclass | 确认：第 29 行 |
| `train.py` 不在模块级导入 torch | 读源码 imports | 确认：只有 numpy + stdlib |
| `dataset_builder.py` 只依赖 numpy | 读源码 imports | 确认 |
| `runtime_dataset_builder.py` 只依赖 numpy | 读源码 imports | 确认 |
| `config.py` 有 YAML fallback parser | 读源码 | 确认：`_load_yaml_fallback()` |

所有代码级声明均准确。

### 3.3 没有越界

Git diff 和 status 显示恰好 4 个文件，全部在 Allowed Files 列表内。`runs/`、`artifacts/`、源码、配置、`requirements-recovery.txt` 均无变更。

### 3.4 诚实性

文档中没有任何把本机证据误写成可移植保证的地方。非声明（non-claims）部分列出了 7 项，每一项都对应真实的风险。Worker 也没有假装已经创建了 lockfile 或验证了干净环境。

### 3.5 发现的非阻塞问题

- **N1**：计划文档的子章节编号用了 `##` 而不是 `###`，是 Markdown 格式问题，不影响内容
- **N2**：T17 的 `training_chain_bootstrap.md` 可以在未来引用 T31 的双 lane 发现，但目前两份文档不矛盾
- **N3**：Worker 的自检文件结构合理

这些问题都不影响 T31 的实质产出质量。

### 3.6 为什么不是 PASS_WITH_WARNINGS

T31 是一个 planning/boundary 任务，不涉及代码实现或运行时验证。它的核心价值在于"把分析做对、把边界写清"。在这个标准下：

- 没有事实性错误
- 没有越界操作
- 没有把计划写成已验证事实
- 所有代码级声明经得起独立检查
- 非声明完整

唯一可以算作 warning 的是 N1 的格式问题，但它对内容没有影响，不足以降低 verdict 等级。

### 3.7 建议的后续动作

1. Captain 接受 T31 为 `PASS`
2. 更新 task board 标记 T31 完成
3. 按 T31 Section 9 创建下一个 bounded task（CPU-only clean-environment draft lock + dry-run bootstrap）
4. 更新 handoff 文档
