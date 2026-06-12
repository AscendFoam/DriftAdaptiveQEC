# T76 Rendered Figure QA

## 1. 任务定位

`T76` 不是新实验，也不是重新打开 `T74/T75` 的 claim 层级。

本文件只记录一件事：`T75-FIG-*` 在真实栅格化预览下是否可读，以及为了消除裁切/排版问题所做的 presentation-only 修正。

## 2. 渲染路径

- 渲染后端：bundled Node runtime + `sharp 0.34.5`
- 栅格化库：`cairo 1.18.4`、`pango 1.57.0`
- 栅格化目标：`density=216`，主 preview 宽度 `1800 px`
- 联系页与 PDF：bundled Python + `Pillow` + `reportlab`
- 产物目录：`docs/figure_assets/T76_rendered_figure_qa_pack/`

这条路径只用于 paper-facing legibility QA，不改变任何上游 stable ID、数值来源或 evidence boundary。

## 3. 逐图 QA 结果

| `T75` 资产 | `T76` preview ID | 初始渲染问题 | `T76` 修正 | 当前状态 |
| --- | --- | --- | --- | --- |
| `T75-FIG-M01` | `T76-PREVIEW-M01` | 图例横条最右侧的 authoritative-source 说明在真实栅格化时被右边界裁切；底部 boundary note 单行过长 | 缩短横条说明文字；底部 boundary note 改为双行 | `PASS_AFTER_LAYOUT_FIX` |
| `T75-FIG-M02` | `T76-PREVIEW-M02` | 底部 boundary note 单行过长，真实栅格化时右侧裁切 | 底部 boundary note 改为双行 | `PASS_AFTER_LAYOUT_FIX` |
| `T75-FIG-A01` | `T76-PREVIEW-A01` | 三层说明中的长句在右侧裁切；第一次换行后第三层底部说明与 footer 轻微挤压 | 三层长句改为双行；第三层文字纵向收紧，避免与 footer 冲突 | `PASS_AFTER_LAYOUT_FIX` |

## 4. 当前人工视觉结论

### `T75-FIG-M01`

- 主标题、四个 scenario 标签、`Hybrid Residual-B` / `UKF runner-up` 图例与四个 `Δ` 注释均可读。
- `T74-TBL-01 remains the authoritative numeric source.` 已完整显示。
- 底部 boundary note 已完整显示。
- 该图仍然只是 `T74-TBL-01` 的 publication-facing visual compression，不能替代 authoritative numeric table 的角色。

### `T75-FIG-M02`

- 两个 panel 的标题、零线、seed 顺序与数值标注均可读。
- 颜色不是唯一编码手段；seed 标签、正负号与零线在灰度场景下仍能区分主要语义。
- 底部 boundary note 已完整显示。
- 该图仍然只支持 descriptive mechanism/intervention reading。

### `T75-FIG-A01`

- Layer 1 / 2 / 3 的长句裁切已消失。
- blocked slot、appendix authoring role、底部 boundary note 均可读。
- 图仍然表达的是 layered boundary，而不是 unified portability / deployment closure。
- `T74-FIG-04` 仍然保持 blocked slot，不得被补写成已完成图位。

## 5. 裁切修正对语义的影响

本轮所有修正都属于 presentation-only：

- 不改 `T75-FIG-*` 资产 ID；
- 不改 `T75` 到 `T74` 的 source chain；
- 不改任何数值、排序、caption 核心句或 claim boundary；
- 不把 `.tflite`、real-board、`FR8` 或 blocked slot 升级成更强叙事。

## 6. 与 `T76` 预览包的关系

本文件对应的实物预览产物为：

- `T76-PREVIEW-M01` -> `t75_fig_m01_preview.png`
- `T76-PREVIEW-M02` -> `t75_fig_m02_preview.png`
- `T76-PREVIEW-A01` -> `t75_fig_a01_preview.png`
- `T76-PREVIEW-CS01` -> `t75_preview_contact_sheet.png`
- `T76-PREVIEW-PDF01` -> `t75_preview_bundle.pdf`

追溯关系以 `docs/figure_assets/T76_rendered_figure_qa_pack/render_manifest.json` 与 `preview_source_map.csv` 为准。
