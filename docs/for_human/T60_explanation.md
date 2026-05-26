# T60 说明文档

## 1. 这个任务在做什么

可以把 `T59` 理解成“先把 `statcalib` 这条新比较器赛道接进系统，并证明它至少能跑通一次小型 smoke”。  
`T60` 做的不是再跑一次实验，而是补两个更基础的问题：

1. 这条新赛道会不会“串台”，偷偷影响别的已有模式。
2. 以后别人改代码时，这条赛道的关键语义会不会悄悄回退。

所以，`T60` 本质上是一次很小但很重要的“语义隔离 + 回归测试加固”任务。

---

## 2. 这次实现到底改了什么

### 2.1 任务目标

结合 `docs/04_task_board.md` 和 `docs/07_handoff.md`，当前项目仍处于 `Phase 2: Controlled Development`，而且 `T60` 是当时唯一允许推进的任务。  
它的目标不是扩实验，不是追 `FR8`，也不是补新 benchmark，而是解决 `T59` 留下来的两个前置问题：

1. `T59` 的 review 发现 `slow_loop.statcalib.teacher_mode` 可能通过通用 fallback 链影响非 `statcalib` 模式，这是语义隔离风险。
2. `T59` 虽然做了集成 smoke，但直接针对 estimator 分支和聚合逻辑的回归测试还不够。

换句话说，`T60` 是在“先把新 lane 接上”之后，把它从“能跑”提升到“边界更清楚、以后更不容易改坏”。

### 2.2 代码变化

真正的代码变化只发生在一个地方：`cnn_fpga/runtime/slow_loop_runtime.py`。

核心修正是 `SlowLoopRuntimeConfig.from_config()` 里的 `teacher_mode` 解析逻辑：

- 以前的问题：
  - 通用 fallback 链里会读到 `slow_loop.statcalib.teacher_mode`。
  - 这意味着即使当前模式不是 `statcalib`，理论上也可能被 `statcalib` 子树里的配置偷偷影响。
- 现在的做法：
  - 先单独算一个“共享默认值链” `shared_teacher_mode`。
  - 如果当前 `mode == "statcalib"`，才允许从 `statcalib_cfg` 读取 `teacher_mode`。
  - 如果当前不是 `statcalib`，就只看当前激活模式自己的配置，再回退到共享默认值链。

这个改动很小，但它的意义很直接：  
`statcalib` 不再能隐式改写 `hybrid_residual_b`、`ukf` 等别的模式的 teacher 选择。

### 2.3 测试变化

这次的主要增量其实在测试，而不是功能扩展。

#### A. runtime 语义隔离测试

`tests/test_statcalib_runtime_smoke.py` 新增了两类断言：

1. 当前模式是 `hybrid_residual_b` 时，即使 `slow_loop.statcalib.teacher_mode = ekf`，最终也不应该串到这个模式上。
2. 当前模式是 `statcalib` 时，`slow_loop.statcalib.teacher_mode` 应该正常生效。

这两条测试正好对应 `T59` review 里最重要的 deferred warning。

#### B. estimator 直接分支测试

新建 `tests/test_statcalib_estimator.py`，覆盖了：

1. 无效窗口
2. 直方图质量为 0
3. 信号低于阈值
4. `delta_b` 裁剪边界
5. 诊断错误分支

这里的意义是：  
以后就算没人跑 benchmark，只要这些单测还在，`statcalib` 最核心的分支语义就比较难被悄悄改坏。

#### C. aggregation / report 回归测试

新建 `tests/test_statcalib_aggregation.py`，覆盖了：

1. 非 `statcalib` 模式应保持 `not_applicable`
2. `generated` 状态的聚合统计
3. 混合状态 `mixed`
4. benchmark 侧状态字段聚合的确定性
5. markdown report 中 `Statcalib` 列是否真的写出来

这一步很重要，因为 `T59` 不只是加了一个运行模式，还让 `comparison.csv` / `summary.json` / report 多了一批新字段。  
如果只测运行、不测聚合，后续很容易出现“代码能跑，但结果表写错或写空”的回归。

### 2.4 没有改什么

同样重要的是，这次**没有**做下面这些事：

- 没有改 benchmark config
- 没有新建 run root
- 没有重跑 `T59` smoke
- 没有改 `ParamMapper`
- 没有改 `.tflite`、真板、训练链、cleanup、theory branch
- 没有把任何计划写成“已经完成的 FR8 事实”

这说明本次实现是收敛的，不是借题发挥式扩 scope。

### 2.5 对后续开发的意义

从项目主线看，`T60` 的意义不是“结果更强了”，而是“下一步能更稳地做真正该做的事”。

它的价值主要有三点：

1. 先把 `T59` 留下的语义漏洞补上，避免未来比较器 lane 研究建立在串台配置上。
2. 给 `statcalib` 新增一层直接单元测试保护，降低以后继续开发 comparator lane 时的回归风险。
3. 为后续真正可能发生的公平性/稳健性审查、甚至更后面的 `FR8` 前置任务，提供一个更干净的代码基础。

但它**没有**改变证据等级。  
`T60` 不是 formal comparator evidence，也没有回答“statcalib 为什么在 T59 smoke 里显得那么强”。

---

## 3. 为什么这次 review 给的是 PASS

我给 `PASS`，原因很直接：

1. 任务目标确实完成了。
   - `teacher_mode` 串台问题被实质性修掉了。
   - estimator 分支测试补上了。
   - aggregation / report 测试补上了。

2. 没有发现伪实现、mock 冒充完成态、或者硬编码扩散。
   - `statcalib` 本身仍然是一个很简化的 comparator，这件事在 `T59` 就已经是明说的事实。
   - `T60` 并没有把这个简化 comparator 包装成“正式最优方法”。

3. 没有缺失这次任务要求的验证。
   - 我重新跑了 `unittest`。
   - 我重新跑了 `py_compile`。
   - 我重新检查了 `runs/` 和 `cnn_fpga/config/`，确认没有新 run root、没有配置改动。

4. 没有明显过度工程。
   - 没有去重构 benchmark runner。
   - 没有新增复杂抽象层。
   - 只是对一个具体语义点做修复，并补最小必需测试。

5. 没有把计划写成事实。
   - closeout 文档仍然明确写着“没有新 smoke”“不是 FR8”“后面还要 fairness/robustness”。

当然，也有我保留在 review 里的提醒：

- `diagnostic_error` 的那条测试是通过故意传错参数类型触发的，它是合理的分支覆盖，但属于“合成错误路径”，不能误读成“真实运行中已经观察到该故障”。
- `T60` 关闭的是语义隔离和回归覆盖问题，不是 `T59` 的公平性/解释性问题。

这两个点都不构成 blocker，所以 verdict 仍然是 `PASS`。

---

## 4. Worker 已写文档时，我怎么看

### 4.1 对 worker 的 review 草稿

`docs/review/T60_review.md` 原草稿方向基本是对的：  
它也判断这次任务应当通过。

但原草稿偏简略，少了几件 reviewer 需要明确说清楚的事：

1. 没有把 `T60` 和 `T59` 的 deferred warning 逐一对应起来。
2. 没有强调“这次关闭的是语义隔离风险，不是 FR8 证据问题”。
3. 没有把我实际复核过的验证动作讲清楚。

所以我保留了结论方向，但把论证补完整了。

### 4.2 对 worker 的 explanation 草稿

worker 已经尝试写解释文档，但在我当前环境里原文件表现出明显的可读性/编码问题；同时它也缺少把这次任务放回 `task board + handoff + T59 deferred items` 背景里的解释。  
所以我直接重写成这版，重点补了：

1. 这次任务为什么存在
2. 它与 `T59` 的关系
3. 具体代码和测试到底改了什么
4. 为什么这次通过，但仍然不能把它说成 `FR8`

---

## 5. 一句话总结

`T60` 不是“让 statcalib 更强”的任务，而是“让 statcalib 这条新 lane 更不容易串台、更不容易被未来改坏”的任务。  
从 reviewer 角度看，这次实现完成度足够、范围控制也合格，因此应当通过。
