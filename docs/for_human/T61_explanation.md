# T61 说明文档

## 1. 这个任务在做什么

`T59` 已经证明了一件事：`statcalib` 这条新 comparator lane 能接进系统，并且在一个很小的 smoke 里跑出明显更好的结果。  
但当时有两个问题：

1. 那次 run 来自 dirty worktree，provenance 不够干净。
2. 结果强得有点反常，需要先做一次很小的 sanity rerun，看它是不是偶然。

`T60` 解决了代码语义和回归测试的问题。  
所以 `T61` 的任务就变成了一个很具体的问题：

> 用同样的小矩阵，在干净提交上再跑一次，看看 `statcalib` 的优势到底还在不在。

这不是正式 `FR8`，也不是扩 benchmark，只是一个“先确认这个现象值得继续认真看”的小关口。

---

## 2. 这次实现到底做了什么

### 2.1 任务目标

从 `docs/04_task_board.md` 和 `docs/07_handoff.md` 看，`T61` 是 `T60` 之后的唯一当前任务。  
它的目标很窄，只做两件事：

1. 修掉 `T59` 的 provenance 弱点，至少把 rerun 起点变成 clean committed worktree。
2. 在不扩 scope 的前提下，复查 `statcalib` 对 `ukf` / `hybrid_residual_b` 的异常强优势是否还存在。

所以它本质上是“bounded rerun + provenance audit”，不是代码开发任务。

### 2.2 实际做法

这次没有改 source、没有改 test、没有改 config。  
按任务包，worker 只允许：

- 新建一个 T61 专属 run root
- 写结果文档
- 写 review / explanation / worker summary

实际执行路径也是这样：

1. 先做 preflight：
   - `git status --short`
   - `git rev-parse --short HEAD`
2. 复用已有配置：
   - `cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml`
3. 只跑锁定矩阵：
   - scenarios: `static_bias_theta`, `linear_ramp`
   - modes: `ukf`, `hybrid_residual_b`, `statcalib`
   - `--paired-seeds`
   - `--repeats 2`
4. 把输出收敛到一个 run root：
   - `runs/p4_benchmark/T61_statcalib_fairness_sanity_20260527_015239`

从 diff 看，代码树本身没有被动过；`git diff --name-only` 对 `cnn_fpga/`、`tests/`、`cnn_fpga/config/`、治理文档、理论分支材料都没有新增改动。  
这说明 worker 在范围控制上是合格的，没有借机改功能。

### 2.3 结果层面发生了什么

从 `summary.json` 和 `comparison.csv` 看，这次 rerun 的结果很明确：

- `statcalib` 在两个 scenario 里都还是第一名
- `statcalib_status=generated`
- `statcalib_reason=statcalib_params_emitted`
- `statcalib_generated_windows_mean=600.0`

更具体一点：

- `static_bias_theta`：
  - `statcalib` 仍然远优于 `ukf` 和 `hybrid_residual_b`
  - T61 runner-up gap 约 `0.3793`
- `linear_ramp`：
  - `statcalib` 仍然远优于 `ukf` 和 `hybrid_residual_b`
  - T61 runner-up gap 约 `0.3581`

也就是说，单看数值结果，`T59` 的“statcalib 特别强”这个现象并没有塌掉。

### 2.4 真正的问题出在哪里

问题不在结果，而在 provenance。

任务开始前的 clean-start commit 是：

- `9174065`

但最终 `summary.json` 里记录的 `git_commit` 却是：

- `6058f42`

再结合 `git reflog`，可以看到在 benchmark 还没彻底结束时，仓库 `HEAD` 发生了 branch checkout。  
而当前 benchmark 产物的 `git_commit` 记录逻辑显然更接近“结束时的仓库状态”，而不是“启动时的仓库状态”。

这就导致一个关键问题：

> 虽然 run 是从 clean worktree 启动的，但最终产物没有保持住单一、可辩护的 commit 锚点。

这会直接影响 T61 的任务是否算完成，因为 T61 的任务名里就写着 `clean-provenance`。

### 2.5 对后续开发的意义

这次任务虽然没有把 provenance 问题彻底关掉，但它仍然有价值：

1. 它说明 `statcalib` 的强结果不是一次性的脏工作区偶然现象，至少在同一 bounded matrix 下又出现了一次。
2. 它把剩余 blocker 收敛得更清楚了：
   - 现在不是“结果是否还存在”不清楚
   - 而是“怎么把这个 rerun 变成 provenance 真正干净的证据”
3. 它为后续是否要做 `FR8` 提供了明确前置条件：
   - 先补 provenance-safe rerun
   - 再谈 formal comparator gate

所以，`T61` 没有白做，但它也没有达到“可以直接推进到 FR8”的程度。

---

## 3. 为什么这次 review 给的是 BLOCK

我给 `BLOCK`，不是因为 worker 乱改代码，也不是因为结果不好看。  
恰恰相反：

- 范围控制是合格的
- 没有伪实现、mock 冒充完成态、或硬编码新逻辑
- 没有破坏已有功能
- 结果文档也基本诚实

真正 blocking 的原因只有一个：

> 这个任务要求的是 clean-provenance fairness sanity rerun，但最终产物没有完成 clean-provenance 这个核心目标。

更直白地说：

1. fairness sanity 这半边做到了
   - `statcalib` 的优势确实 persisted
2. clean provenance 这半边没做完
   - final artifact 不是单一 clean-start commit 的稳定锚点

因此，从“任务是否真的完成”这个标准看，结论只能是 `BLOCK`。

这不是说 T61 没有产出价值，而是说：

- 它产出了有用的中间证据
- 但它没有完成它被指派去关闭的 blocker

所以 review 结果不能是 `PASS` 或 `PASS_WITH_WARNINGS`。

---

## 4. Worker 已写文档时，我怎么看

### 4.1 对 worker 的 review 草稿

worker 已经写了 `docs/review/T61_review.md` 草稿。  
方向上我认为它是对的：它也判断这次应该 `BLOCK`，而不是把结果 persisted 误写成“任务完成”。

我主要补充和强化了几件事：

1. 把 `BLOCK` 的根因写得更明确：
   - 不是结果失败
   - 是 provenance 目标失败
2. 把 blocking issue 和 task package 的目标直接对齐
3. 把“哪些地方其实是合格的”也写清楚，避免误以为 worker 改坏了主线
4. 补充了后续最合理动作：
   - provenance-safe rerun
   - 或者单开允许 source 改动的 launch-time commit capture 修复任务

### 4.2 对 worker 的 explanation 草稿

worker 原 explanation 太短，能说明“发生了什么”，但不够说明：

1. `T61` 为什么会出现在 `T59/T60` 之后
2. 这次为什么不是代码问题，而是证据治理问题
3. 为什么结果 persisted 了，review 还是 `BLOCK`
4. 这次对后续 `FR8` 判断到底推进了什么、又没推进什么

所以我重写了这份更完整的版本，把它放回整个任务链条里解释。

---

## 5. 一句话总结

`T61` 的真实结论不是“statcalib 不行”，而是：

> `statcalib` 在这个小矩阵里依然很强，但这次任务没有把 provenance 修干净，所以还不能拿它去开 `FR8`。

这也是为什么本次 review 的 verdict 是 `BLOCK`。
