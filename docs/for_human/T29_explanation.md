# T29 人话版说明

## 一、这个任务是做什么的？——通俗解释

T29 修的是一个很小的报表格式 bug，小到只删了一行代码。

之前 T28 在 P4 benchmark 的 markdown 报告里加了一列 `Teacher Diag`（用于显示 teacher diagnostics 的状态），但加的时候忘了删掉旧的表头。结果是报告里出现了两行表头——一行 11 列（旧的），一行 12 列（新的）。markdown 表格要求所有行列数一致，所以这个报告渲染出来是歪的。

这就像 Excel 表格里第一行有 11 列标题，第二行又有 12 列标题，下面数据是 12 列——看起来就对不齐了。

T29 就是把那行多余的旧表头删掉，让表头、分隔线、数据行都是 12 列。

## 二、实现的详细解释

### 任务目标

删除 `_write_report()` 中重复的旧 markdown comparison-table header row，使报告表格结构正确。

### 任务流程

T28 → T28 review 发现重复表头 bug → Captain 将此 bug 独立为 T29 → Worker 执行单行修复

### 代码变化

**文件：** `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`

**删除的行（第 318 行）：**

```
"| Scenario | Mode | LER Mean | LER Std | Overflow Mean | Hist Sat Mean | Commit Mean | Slow Viol Mean | Fast Viol Mean | Dominant Source | Artifact |",
```

这是 T28 之前的旧 11 列表头，不含 `Teacher Diag`。

**保留的行（现在第 318 行）：**

```
"| Scenario | Mode | LER Mean | LER Std | Overflow Mean | Hist Sat Mean | Commit Mean | Slow Viol Mean | Fast Viol Mean | Dominant Source | Teacher Diag | Artifact |",
```

这是 T28 新增的 12 列表头，包含 `Teacher Diag`。

**验证方式：** Worker 用两种方式验证——`py_compile` 静态语法检查通过，然后直接调用 `_write_report()` 生成临时 markdown 并检查列数一致性（header=12, separator=12, data=12）。

### 配置文件变化

无。

### 对后续开发的意义

1. **报告格式干净了：** 后续任何需要阅读 P4 benchmark markdown 报告的人（包括论文审稿人）不会再看到歪掉的表格。
2. **T28 遗留的最后一个 code bug 清零：** T28 的语义修复（missing-vs-zero）是正确的，T28 的 CSV 输出是正确的，T28 review 指出的唯一代码 bug 就是这个重复表头。现在它也修好了。
3. **为 statcalib 和 seed 诊断扫清障碍：** T29 之后，Teacher diagnostics 的可观测性修复（T28）和报告格式修复（T29）都已完成。接下来做 `T26`（statcalib baseline feasibility gate）或 `T36`（seed=20260429 failure-mechanism diagnosis）时，不会继承任何格式或语义问题。

## 三、为什么 reviewer 给出了 PASS

### 这是极简修复，没有争议空间

改动只有一行删除。没有新增代码、没有新增逻辑、没有新增配置、没有运行 benchmark。唯一的变化就是删掉了一个多余的字符串常量。

### 验证是够用的

对于一行删除，做 `py_compile` 加上内存中的格式检查已经充分。不需要跑 benchmark，也不需要单元测试——这个改动不涉及任何计算逻辑。

### 没有任何越界

- 只改了 Allowed files 里的文件
- 没有碰 benchmark 口径、语义、场景、基线、历史数据
- 没有创建新的 run 目录
- 文档没有把计划写成事实

### 唯一的附注

`__pycache__` 里的 `.pyc` 文件又跟着变了（Worker 的格式验证触发了 recompile）。这跟 T28 的情况一样——不应作为有意义改动提交，但也不是 Worker 的过错。
