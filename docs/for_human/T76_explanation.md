# T76 任务解释与 Review 说明

## 1. 先用通俗的话说，`T76` 在做什么

`T74` 和 `T75` 已经把论文材料整理到了“有表、有图、有安全写法”的阶段，但还差最后一层非常实际的检查：

- 这些图如果真的渲染成 PNG / PDF，会不会被裁切、挤压、看不清？
- 作者真正开始写 Results 段时，图、表、句子、附录桥接，是否已经有一套可以直接拿来装配的入口？

所以 `T76` 不是新实验，也不是恢复 full manuscript 扩写。它做的是“真实预览 + 人工可读性 QA + Results 段装配包”。

## 2. 这次实现具体做了什么

### 2.1 任务目标

结合 `docs/02_experiment_plan.md`、`docs/04_task_board.md` 和 `docs/07_handoff.md`，`T76` 的定位很明确：

- `T74` 负责 stable-ID 结果表、图表清单、caption 与 traceability；
- `T75` 负责 bounded Results authoring、三张 `T75-FIG-*` 资产、caption/placement lock 与 do-not-write guardrail；
- `T76` 负责把这些 `T75` 资产做一次真实 rendered preview，并补出作者可直接使用的 Results-section assembly 材料。

换句话说，主线已经从“有没有材料”推进到“这些材料能不能诚实、稳定地用于落稿”。

### 2.2 任务流程

这次交付实际分成四层：

1. 真实预览层
   - 生成了三张主预览 PNG：
     - `t75_fig_m01_preview.png`
     - `t75_fig_m02_preview.png`
     - `t75_fig_a01_preview.png`
   - 另外生成了：
     - `t75_preview_contact_sheet.png`
     - `t75_preview_bundle.pdf`

2. trace / QA 层
   - `render_manifest.json`
   - `preview_source_map.csv`
   - `visual_qa_checklist.md`

3. 面向作者的 paper materials 层
   - `paper_rendered_figure_qa.md`
   - `paper_results_section_assembly_pack.md`
   - `paper_results_callout_sheet.md`

4. 必要的 presentation-only 修正层
   - `T75-FIG-M01`：缩短图例说明，footer 改双行
   - `T75-FIG-M02`：footer 改双行
   - `T75-FIG-A01`：长句换行并收紧纵向间距

### 2.3 改了哪些文件，没改哪些文件

这次没有改任何源码、测试、benchmark、`runs/`、`artifacts/` 或治理文档。

改动集中在两类文件：

- 文档与索引：
  - `docs/paper_materials/README.md`
  - `docs/paper_materials/paper_rendered_figure_qa.md`
  - `docs/paper_materials/paper_results_section_assembly_pack.md`
  - `docs/paper_materials/paper_results_callout_sheet.md`
  - 若干 `T75` authoring 文档的同步说明

- 图资产与预览包：
  - `docs/figure_assets/T76_rendered_figure_qa_pack/*`
  - `docs/figure_assets/T75_maintext_results_authoring_pack/*`

这很重要，因为它说明 `T76` 维持了“docs-only / paper-facing / no evidence upgrade”的任务边界。

### 2.4 对后续开发的意义

`T76` 的意义不在于“把论文写完了”，而在于把主线又往前推进了一小步，但这一步是可靠的：

- 现在作者已经不只拥有 stable-ID 表和 bounded prose，还拥有真实可视预览；
- Results 段的推荐顺序、fallback 路线、callout 安全句式已经更集中；
- 如果下一步要判断“是否可以受控地 reopen prose drafting”，就不再是空谈，而是可以基于 `T74 -> T75 -> T76` 这一整套材料链做 gate 判断。

同时，`T76` 并没有改变这些红线：

- 不是 `.tflite` / real-board / `FR8` 的证据升级；
- 不是 full-manuscript reopen；
- 不是 `T74-FIG-04` blocked slot 的解除；
- 不是新的 benchmark 或 runtime 事实。

## 3. 为什么我的 review 结论是 `PASS_WITH_WARNINGS`

### 3.1 为什么不是 `BLOCK`

因为任务要求的核心交付已经真实存在，而且 reviewer 可以独立验证：

- 三张 preview PNG、contact sheet、PDF bundle 都真实存在且非空；
- `render_manifest.json` 可以解析；
- `preview_source_map.csv` 与 manifest 的 `preview_id` 集合一致；
- `paper_rendered_figure_qa.md`、`paper_results_section_assembly_pack.md`、`paper_results_callout_sheet.md` 都已经落地；
- 没有看到 mock/stub/placeholder 冒充完成态；
- 没有看到源码、测试、`runs/`、`artifacts/` 的越界改动。

更关键的是，真实预览图我已经实际看过，能确认这次不是“只写文档说自己渲染过了”，而是真的生成了可看的图。

### 3.2 为什么不是完全的 `PASS`

我保留 warning，主要因为还有三处收口质量问题：

1. `preview_source_map.csv` 的 schema 不够干净
   - 对 `T76-PREVIEW-CS01` 和 `T76-PREVIEW-PDF01`，`upstream_t74_ids` 列里写的是 `T76-PREVIEW-*`，不是 `T74-*` stable ID。
   - 这不影响人读，也不影响整包 traceability，因为 `render_manifest.json` 仍然足够清楚。
   - 但如果以后有人把 CSV 当成严格 source-map，这两行会产生语义混淆。

2. 工作区残留了 `.tmp_t76_*` 探针文件
   - 这说明 Worker 的真实渲染路径大概率是诚实的，因为确实留下了过程痕迹；
   - 但这些文件不属于正式交付，也在 Allowed files 之外，提交前应该清掉。

3. QA 文档里的上游 `T74-*` 绑定写得还不够“逐条内联”
   - 当前包整体可追溯；
   - 但 `paper_rendered_figure_qa.md` 更像“QA 结论 + 指向 manifest/source-map”，而不是“每张图的 QA 结论里直接把上游 stable IDs 写全”。

所以我的判断是：任务实质完成了，但交付边角还留了一点 schema/清洁度债务，更适合 `PASS_WITH_WARNINGS`，而不是无保留 `PASS`。

## 4. 如果 Worker 已经写了 review / explanation 文档，我怎么处理

我看过现有的 `docs/review/T76_review.md` 和 `docs/for_human/T76_explanation.md`。它们的大方向基本正确：

- 能正确把 `T76` 解释为 paper-facing rendered QA / assembly；
- 没有把这次工作写成新实验、新主结果或 deployment closure；
- 也能指出 `.tmp_t76_*` 是过程性噪声。

我这次补充和调整的地方主要有三点：

1. 把 verdict 从 `PASS` 调整为 `PASS_WITH_WARNINGS`
   - 原因不是任务没完成，而是我认为 source-map schema 语义不够干净，且过程文件残留应正式记为 warning。

2. 把 warning 说得更精确
   - 不是泛泛而谈“有些噪声”，而是具体指出 contact-sheet / PDF 行在 CSV 里复用了不完全匹配语义的列。

3. 把 `T74 -> T75 -> T76` 的链条讲得更完整
   - 这样后面无论是作者、Captain 还是下一位 Reviewer，都更容易理解：
     - `T74` 是 stable-ID 材料冻结；
     - `T75` 是 bounded authoring；
     - `T76` 是真实渲染 QA 和 Results 装配。

## 5. 我建议的下一步

如果只是收口当前任务，最合适的动作很小：

1. 清理 `.tmp_t76_*` 和 `.tmp_t76_fontcache/`。
2. 给 `preview_source_map.csv` 的 contact-sheet / PDF 聚合行补一个更明确的字段设计或说明。
3. 如果还想再提高一点文档自解释性，就在 `paper_rendered_figure_qa.md` 里把每张图对应的 `T74-*` stable IDs 再写得更显眼一些。

除此之外，不建议在 `T76` 内继续扩 scope。若未来需要真实期刊模板导出、版面适配或 manuscript reopen，都应该新开有界任务，而不是把这些事情继续堆到 `T76` 上。
