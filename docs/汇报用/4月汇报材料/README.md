# 4月汇报材料

本目录保存 2026-04 前后形成的项目汇报材料。

## 文件说明

| 文件 | 内容 | 边界 |
| --- | --- | --- |
| `CNN_FPGA_GKP_项目汇报.pptx` | 原始项目汇报 PPT | 历史汇报材料 |
| `CNN_FPGA_GKP_项目汇报_增强版.pptx` | 增强版项目汇报 PPT | 历史汇报材料 |
| `artifact-build-manifest.json` | 增强版 PPT 的本机构建 manifest | 只记录历史本机生成信息；其中 Temp 路径不可复现 |
| `CNN_FPGA_GKP_PPT汇报内容方案.md` | PPT 内容方案 | 汇报草稿材料 |
| `P0-P4口头汇报.md` | P0-P4 口头汇报稿 | 汇报草稿材料 |
| `CNN_FPGA_GKP_工程化实验方案_讲解版.md` | 旧工程化实验方案讲解版 | 已不承担当前计划状态 |
| `Nvidia论文对比.md` | 对比材料 | 历史参考 |
| `回答.md` | 问答材料 | 历史参考 |
| `*.drawio` | 汇报图源文件 | 历史图源 |

## 使用规则

1. 本目录材料可用于回顾项目早期表达方式，不应直接作为当前实验事实来源。
2. 涉及 P4、`.tflite`、real-board、statcalib 或论文 claim 时，必须回到 `docs/evidence_packs/`、`docs/protocols/`、`docs/review/` 和治理文档核对。
3. `artifact-build-manifest.json` 中的 `previewDir`、`previewPaths`、`layoutDir`、`layoutResults` 和 `slides.modulePath` 是本机临时路径记录，不是可移植产物。
