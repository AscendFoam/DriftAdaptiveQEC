# T45：paper-grade benchmark 扩展协议与缺口审计 —— 给人类的说明

## 1. T45 这轮到底做了什么

T45 没有跑任何 benchmark，也没有改任何代码或配置。  
它做的是一件更像“定规则”的工作：

- 把当前 `T24/T25` 冻结集 benchmark 到底能支持什么、不能支持什么写清楚；
- 决定论文是否还能只靠 frozen-set benchmark 继续往前写；
- 如果要做更强的 paper benchmark，先把“扩展到哪里、哪些该纳入、哪些先不要纳入”锁成协议。

产出文件是：

- `docs/protocols/benchmark/paper_benchmark_expansion_protocol.md`

这个文档的核心作用，是把“当前真实证据”和“未来可能扩展的 benchmark 车道”分开，防止后面把参考建议误写成已经完成的事实。

## 2. T45 的核心结论

### 2.1 frozen-set benchmark 还能不能用

可以用，但只能按比较保守的方式用。

当前 frozen set 已经能诚实支撑：

- `mock-backed software HIL` 范围内的 formal software revalidation
- 四个冻结场景、五个冻结 mode 的排名结论
- `hybrid_residual_b` 在这个冻结集里赢四个场景

但它还不能单独支撑：

- “paper-grade expanded benchmark” 这种更强说法
- 更广泛的 superiority claim
- 部署、`.tflite`、真板之类的结论

所以，**如果论文只想维持一个保守、证据有界的版本，frozen-set only 是可以的；但如果想把 benchmark 叙事写强，就不够。**

### 2.2 要不要开 benchmark 扩展车道

T45 的答案是：**如果目标是更强的论文 benchmark 说服力，应该开一个单独的、有边界的扩展车道。**

但这个扩展车道必须满足三条：

1. 不能改写 `T24` 现有 frozen-set 结果；
2. 只能作为“额外 expansion lane”单独标注；
3. 不能顺手把 `.tflite`、真板、部署恢复混进来。

## 3. 哪些扩展项被采纳了，哪些没有

### 3.1 被采纳为“后续可以做”的

- 保留 `T24` 作为 anchor table，不重写历史 frozen set
- 新增额外 drift family 的扩展车道
- 优先考虑：
  - `random_walk`
  - `burst_reset` / 突发后恢复型 drift
  - 一个未见 drift law 的 generalization holdout
- 把 `statcalib` 作为**单独标注**的 comparator lane 纳入候选
- 明确要求 learned modes 以后要区分 training seed 和 evaluation seed
- 以后如果真开扩展 benchmark，必须继续报告 commit / violation / saturation 这些系统约束指标

### 3.2 目前先 defer 的

- soft-information / correlation-aware comparator
- CI-driven stopping
- rollback / fallback 作为硬性 acceptance 字段

原因不是这些不重要，而是：

- 现在仓库里还没有准备好把它们作为当前主线 runnable lane；
- 如果现在强塞进去，会让任务失控、边界失真。

### 3.3 当前明确不建议放进主线 benchmark 扩展里的

- 把 `Gated v5`、FiLM-style、teacher-representation 这些分支直接并进当前 mainline benchmark
- 把 `.tflite` runtime 或 `real_board` 验证塞进同一个任务
- 把 `docs/reference/延伸改进思路.md` 当成当前主线 truth

这里的核心判断很简单：

**T45 要做的是“收窄协议”，不是“再打开一轮模型分支搜索”。**

## 4. 这对后续论文路线意味着什么

T45 之后，路线被分成了两种合法走法：

### 路线 A：保守 frozen-set paper

可以继续，但要非常老实地写成：

- frozen-set only
- software-HIL only
- evidence-bounded

这条路线的优点是现在就站得住；缺点是 benchmark 说服力偏弱，投稿定位会更保守。

### 路线 B：先开一个 bounded benchmark expansion lane

如果想把 benchmark 叙事写得更强，这条路线更合适。  
但它必须是一个新的 bounded task，而且只能扩已经在 T45 协议里预声明过的内容。

## 5. T45 没有改变什么

T45 没有：

- 运行 benchmark
- 新增 run dir
- 修改 benchmark code / config
- 改 baseline 语义
- 把 `statcalib` 写成已集成 comparator
- 把 `.tflite` 或真板写成 benchmark scope 内已恢复

所以它本质上是一个**协议锁定 + 缺口盘点**任务，不是结果任务。

## 6. 一句话总结

T45 把当前 benchmark 结论收得更稳了：

- `T24` 冻结集结果继续保留为主锚点；
- 不能把它直接写成更强的 paper-grade benchmark；
- 如果要更强，就必须新开一个边界清楚的 expansion lane，而不是偷偷改写当前结果。
