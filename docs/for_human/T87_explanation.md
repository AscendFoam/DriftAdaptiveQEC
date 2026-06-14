# T87 说明

## 1. 这轮为什么不是新实验，而是更强的作者终检

`T74` 到 `T86` 已经把主线论文材料分成了几层相对稳定的事实：

- 主文主结果层：冻结四场景 benchmark 与保守机制解释；
- supporting boundary 层：training/material、isolated current-host `.tflite` runtime、boundary schematic；
- supplement / gate 层：`statcalib` extension lane 与 real-board gate/provenance；
- blocked / exclusion 层：default-env portability、full reproducibility、real-board execution、expanded benchmark、promoted comparator。

`T86` 已经把这些层怎么装配成 submission-facing package 说清楚了。到 `T87`，真正剩下的不是“再装一遍”，而是检查 note 里还有没有残余状态滞后、危险表述、或者会把作者后续人工润色误导成 claim promotion 的句子。

所以这轮任务的目标不是补实验，而是把作者终检纪律固定下来。

## 2. 这轮实际做了什么

本轮新增了 4 份作者终检文档：

1. `paper_author_final_qa_checklist.md`
2. `paper_presubmission_regression_gate.md`
3. `paper_submission_wording_redflag_register.md`
4. `paper_manual_finish_queue.md`

同时，主 note 只改了 3 个 section：

1. `Numerical Results`
2. `Discussion`
3. `Conclusion`

并且都补上了 `% T87-QA: ...` 注释。

这些修改只做一件事：把“下一步是什么”从 `T86` 的 submission-facing assembly，进一步收紧成 `author-final QA + bounded manual finish`。

## 3. 后续哪些事情还可以继续做

`T87` 之后仍允许继续做的，只能是人工终修，不是证据升级：

- 调整 `Numerical Results`、`Discussion`、`Conclusion` 的句法和段落流畅度；
- 在不改 route 的前提下，微调 appendix / supplement 的标题、顺序和 bridge 句；
- 在既有 manifest 允许范围内，决定 frozen result 更适合用 `T75-FIG-M01` 还是 `T74-TBL-01` 做主呈现；
- 继续精简 `T75-FIG-A01` / `T74-FIG-03` 这类 boundary schematic 的图注；
- 按目标 venue 做页数、断句、排版上的人工收束。

这些动作都已经被 `paper_manual_finish_queue.md` 明确限制住了。

## 4. 哪些内容仍然必须 blocked / excluded

即使 `T87` 成功，下面这些内容仍然不能被写强：

- `real-board execution / timing / resource`
- `default-env / cross-host .tflite portability`
- `full training reproducibility`
- `FR8/statcalib` mature comparator promotion / unique clean threshold
- expanded benchmark / stronger oracle baseline
- 任何 unified deployment closure / hardware-ready / submission-ready completed 叙事

原因很简单：这些都不是“再润色一句话”能解决的，而是需要新的 bounded evidence task。

## 5. 为什么这轮成功也不等于 submission-ready completed

`T87` 的 gate verdict 是：

`GO_FOR_BOUNDED_AUTHOR_MANUAL_FINISH_ONLY`

这个 verdict 的含义非常窄：

- 说明当前主线 note/material 已经足够支撑有边界的人工终修；
- 不说明论文已经进入正式投稿完成态；
- 不说明任何 blocked surface 已解锁；
- 也不说明当前主机之外的编译、部署或硬件条件已经补齐。

换句话说，`T87` 把“现在还能做什么、绝不能做什么”变成了可审计事实，但它没有把项目推到“已经可以直接对外宣称完成投稿包”的状态。
