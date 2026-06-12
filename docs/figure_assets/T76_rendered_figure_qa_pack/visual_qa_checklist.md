# T76 Visual QA Checklist

## 1. 本轮检查项

| preview ID | 对应资产 | 无裁切 | 灰度可读性 | 边界文案保持 | 放置准备度 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| `T76-PREVIEW-M01` | `T75-FIG-M01` | `PASS` | `PASS` | `PASS` | `READY` | 修正后图例说明与双行 footer 均完整显示；`T74-TBL-01` 仍是 authoritative substitute |
| `T76-PREVIEW-M02` | `T75-FIG-M02` | `PASS` | `PASS` | `PASS` | `READY` | 零线、正负号、seed 标签与双行 footer 可读；仍是 descriptive only |
| `T76-PREVIEW-A01` | `T75-FIG-A01` | `PASS` | `PASS` | `PASS` | `READY_APPENDIX_ONLY` | 三层长句裁切已消失；blocked slot 与 appendix-only boundary 保持明确 |
| `T76-PREVIEW-CS01` | contact sheet | `PASS` | `PASS` | `PASS` | `READY_FOR_REVIEW` | 适合 reviewer / author 快速总览，不替代单图预览 |

## 2. 关键人工结论

1. `T75-FIG-M01`、`T75-FIG-M02`、`T75-FIG-A01` 已完成真实栅格化 QA，不再停留在“只看 SVG 文本结构”的层面。
2. 本轮修正全部属于版式可读性修正，没有新增图位、没有改数值、没有改 stable ID。
3. `T75-FIG-A01` 继续保持 appendix-only 边界说明定位，不能被主文压缩成 deployment closure。

## 3. 仍需保持的写作约束

- `T48` 只能写成 isolated current-host true runtime。
- `T49/T71/T72` 只能写成 read-only real-board gate/provenance with current-host `NO_GO`。
- `FR8` 仍然不能写成 promoted comparator。
- `T74-FIG-04` 仍然必须保持 blocked。
