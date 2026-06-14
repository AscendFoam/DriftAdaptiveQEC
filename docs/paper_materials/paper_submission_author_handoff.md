# T86 Submission Author Handoff

## 1. 当前 submission-facing package 已具备的内容

当前 mainline 已经具备一条可以继续人工终修的 submission-facing package 路线，内容分三层：

1. 主文：
   - `T75-FIG-M01` / `T74-TBL-01` 的 frozen benchmark 主结果
   - `T75-FIG-M02` 的 descriptive mechanism/intervention reading
2. 附录：
   - `T74-TBL-02`
   - `T74-TBL-03`
   - `T74-TBL-04`
   - `T74-TBL-05`
   - `T75-FIG-A01` / `T74-FIG-03`
3. 补充材料：
   - `T74-TBL-06`
   - `T74-TBL-07`
   - `T74-SUP-01` 到 `T74-SUP-04`

这些内容已经可以被装配成一套 simulation/material-first submission-facing package，但它们仍然必须保持分层，而不是被压扁成统一闭环叙事。

## 2. 当前仍未完成、必须继续排除的 surface

以下 surface 仍必须继续留在 exclusion / blocked 列，而不能被手工润色时写进 package 完成态：

1. real-board execution / timing / resource
2. default-env / cross-host `.tflite` portability
3. full training reproducibility
4. `statcalib` mature comparator promotion
5. broader expanded benchmark story
6. unified portability/deployment closure figure or prose

这些项目不是“还差一句文案”，而是“还差新的 bounded evidence task”。

## 3. 作者后续可以继续做的 bounded polishing / manual editorial action

作者后续可以继续做的事情，只应停留在装配和人工终修层：

- 调整主文 Results 与 Discussion 的句子流畅度，但不改变 evidence hierarchy。
- 在 appendix / supplement 中微调桥接句、排版顺序和标题风格，使主文到附录/补充材料的过渡更自然。
- 根据期刊/会议篇幅，选择 `T75-FIG-M01` 或 `T74-TBL-01` 作为 frozen result 的主要呈现方式。
- 对 `T75-FIG-A01` / `T74-FIG-03` 的标题或图注做更简洁的人工润色，但继续把它当成 boundary schematic，而不是 result figure。

## 4. 绝对不能写强的 claim / boundary

以下表述必须继续避免：

- `real-board execution succeeded`
- `hardware validated`
- `default-env recovered`
- `deployment closure is complete`
- `full reproducibility is closed`
- `statcalib is the promoted final comparator`
- `a unique clean threshold has been established`
- `the mechanism is causally proved`
- `the ranking is proven beyond the frozen set`

如果作者想表达这些方向，只能回退到已有的安全替代表述：

- `isolated current-host true runtime verified for selected preserved artifacts`
- `read-only real-board gate / regeneration / provenance boundary with current-host NO_GO`
- `canonical chain intact + one clean CPU-only bounded rerun`
- `separately labeled extension-lane closure with no-promotion and no unique clean threshold`
- `descriptive multi-seed mechanism evidence`

## 5. 本轮 note 中实际加强的装配性段落

为支持本轮 assembly / exclusion 收口，当前 note 中实际加强了以下 section：

- `Numerical Results`
- `Discussion`
- `Conclusion`

对应源码标记为：

- `% T86-ASSEMBLY: Numerical Results`
- `% T86-ASSEMBLY: Discussion`
- `% T86-ASSEMBLY: Conclusion`

这些修改只服务 route / exclusion 明确化，不改变主线结果等级，也不把当前 package 写成 submission-ready completion。
