# T76 Review

Verdict: `PASS_WITH_WARNINGS`

## Blocking issues

- 无。

## Non-blocking issues

- `docs/figure_assets/T76_rendered_figure_qa_pack/preview_source_map.csv:5-6` 对 `T76-PREVIEW-CS01` 和 `T76-PREVIEW-PDF01` 复用了 `upstream_t74_ids` 列，但实际填入的是 `T76-PREVIEW-*`，而不是 `T74-*` stable ID。`render_manifest.json` 里仍然保留了足够的追溯信息，所以这不阻断任务完成；但当前 source-map schema 的语义不够干净，后续若要给作者或自动化脚本复用，建议显式拆成 `upstream_t74_ids` 与 `source_preview_ids`。
- 工作区仍残留 `.tmp_t76_render_a01.png`、`.tmp_t76_render_m02.png`、`.tmp_t76_render_probe.png` 与 `.tmp_t76_fontcache/`。这些看起来是本轮真实渲染过程中的探针/缓存产物，不影响对正式交付物的判断，但它们不属于 `T76` 正式 deliverable，也在 Allowed files 之外；提交前应单独清理或至少明确忽略策略。
- `docs/paper_materials/paper_rendered_figure_qa.md:21-48` 的逐图 QA 结论总体成立，但上游 `T74-*` 绑定主要通过 `render_manifest.json` / `preview_source_map.csv` 间接完成，而不是在每张图的 QA 结论里都内联列全。当前整包 traceability 仍然成立，只是文档粒度略弱于任务包对“逐条 QA 结论显式绑定上游 stable ID”的理想要求。

## Missing tests

- 无新增代码测试缺口。`T76` 是 docs-only / paper-facing 任务，核心验证点应是预览产物真实性、trace 一致性、文档边界诚实性与 git scope。
- 仍缺一个很轻量的 schema 级检查：当前只验证了 `preview_id` 集合与文件存在性，没有额外约束 `preview_source_map.csv` 中聚合产物行的字段语义。

## Suspicious implementation details

- 未发现 mock、stub、placeholder 冒充完成态的情况。三张 preview PNG、contact sheet 与 PDF bundle 都真实落地，且 reviewer 已对 PNG 预览做了实际目视检查。
- `T75` 相关 SVG 的修改看起来是 presentation-only，而不是 evidence-level 改写：
  - `T75-FIG-M01` 只缩短图例说明并把 footer 改成双行；
  - `T75-FIG-M02` 只把 footer 改成双行；
  - `T75-FIG-A01` 只把长句改成双行并收紧纵向间距。
- 当前渲染路径更像一次有界的本地主机 QA 产物生成，而不是可重复复用的长期 helper。对 `T76` 这类一次性 paper-facing 质量控制任务，这仍然是可接受的；但它解释了为什么工作区里会留下 `.tmp_t76_*` 噪声，以及为什么 trace CSV 出现了聚合行字段复用。

## Recommended next action

- 接受 `T76` 完成，但按 `PASS_WITH_WARNINGS` 收口。
- 提交前清理 `.tmp_t76_*` / `.tmp_t76_fontcache/`，避免把过程性噪声带入正式提交。
- 如需一个极小后续修补任务，优先做两件事：
  - 给 `preview_source_map.csv` 的 contact-sheet / PDF 行补一个明确的 `source_preview_ids` 字段或等价说明；
  - 在 `paper_rendered_figure_qa.md` 的逐图 QA 结论里补齐每张图对应的上游 `T74-*` stable IDs。

## Reviewer verification notes

- 已核对 `docs/figure_assets/T76_rendered_figure_qa_pack/render_manifest.json` 可解析，且包含 `T76-PREVIEW-M01`、`T76-PREVIEW-M02`、`T76-PREVIEW-A01`、`T76-PREVIEW-CS01`，并额外记录 `T76-PREVIEW-PDF01`。
- 已核对 `docs/figure_assets/T76_rendered_figure_qa_pack/preview_source_map.csv` 的 `preview_id` 集合与 manifest 一致。
- 已核对以下正式产物真实存在且非空：
  - `t75_fig_m01_preview.png`
  - `t75_fig_m02_preview.png`
  - `t75_fig_a01_preview.png`
  - `t75_preview_contact_sheet.png`
  - `t75_preview_bundle.pdf`
- 已实际查看三张 PNG preview 与 contact sheet，确认本轮修正后的主要裁切问题已消失，且没有把 `.tflite`、real-board、`FR8` 或 blocked slot 升级成更强叙事。
- 已核对：
  - `git diff --name-only -- runs` 为空；
  - `git diff --name-only -- artifacts` 为空；
  - `git diff --name-only -- cnn_fpga physics benchmark tests` 为空。
- 当前工作区确有既存治理文档 diff；结合用户补充说明，这些不应归因于 `T76`，因此本 review 不将其作为 blocker。
