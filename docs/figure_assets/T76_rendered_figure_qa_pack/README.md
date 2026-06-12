# T76 Rendered Figure QA Pack

## 1. 目录用途

本目录保存 `T76` 生成的真实渲染预览包，用于回答两个问题：

1. `T75-FIG-*` 在真实栅格化后是否可读；
2. Results section 在引用这些图时，是否已经有一套稳定的 preview/source-map/QA trace。

本目录不产生新实验事实，也不提升任何 evidence level。

## 2. 目录内容

| 文件 | 作用 |
| --- | --- |
| `t75_fig_m01_preview.png` | `T75-FIG-M01` 的真实预览 PNG，对应 `T76-PREVIEW-M01` |
| `t75_fig_m02_preview.png` | `T75-FIG-M02` 的真实预览 PNG，对应 `T76-PREVIEW-M02` |
| `t75_fig_a01_preview.png` | `T75-FIG-A01` 的真实预览 PNG，对应 `T76-PREVIEW-A01` |
| `t75_preview_contact_sheet.png` | 三张预览的联系页，对应 `T76-PREVIEW-CS01` |
| `t75_preview_bundle.pdf` | 三张预览与联系页的 PDF bundle，对应 `T76-PREVIEW-PDF01` |
| `render_manifest.json` | 预览产物总清单，锁定 preview ID、渲染后端、源图与输出文件 |
| `preview_source_map.csv` | 把 `T76-PREVIEW-*` 追溯回 `T75-FIG-*` 与上游 `T74-*` |
| `visual_qa_checklist.md` | 本轮人工视觉 QA checklist |

## 3. 渲染边界

- 渲染后端：bundled Node runtime + `sharp 0.34.5`
- 预览目标：`density=216`，主 preview 宽度 `1800 px`
- 联系页与 PDF：bundled Python + `Pillow` + `reportlab`
- 若后续期刊模板需要不同尺寸或字体嵌入，本目录只能作为当前 host 的 rendered QA 证据，不代表最终出版社排版产物。

## 4. 与上游任务的关系

- `T75` 提供 authoring 资产与 `authoring_manifest.json`
- `T76` 只验证这些 authoring 资产在真实预览中可读，并补齐 section-assembly trace
- 如果要找 authoritative numerics、runtime boundary 或 real-board gate wording，仍需回到 `T74-*` 以及更上游 task/review/evidence path

## 5. 当前最重要的结论

- `T75-FIG-M01`、`T75-FIG-M02`、`T75-FIG-A01` 都已经过真实栅格化预览
- `T76` 发现并修复了三个 presentation-only 问题：`M01` 图例说明裁切、`M02` footer 裁切、`A01` 三层长句裁切
- 这些修正没有改变 stable ID、caption 核心句、上游 source chain 或 evidence boundary
