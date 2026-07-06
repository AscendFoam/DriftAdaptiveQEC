# 投稿稿 BibTeX 覆盖审计

## 作用域

本文件服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex` 的投稿前引用整理。它基于本地文件：

- `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`
- `docs/paper_materials/zotero_export_for_literature_review.bib`
- `docs/paper_materials/zotero_literature_review_cards.md`
- `docs/paper_materials/投稿稿引用同步与核对清单.md`

本文件不联网检索，不新增外部引用，不把任何未核对硬件条目升级为强引用。

## 当前覆盖结论

| 项目 | 当前状态 |
| --- | --- |
| 投稿稿 active citation key 数 | 27 |
| 本地 Zotero BibTeX 条目数 | 38 |
| active TeX key 是否都有本地 Zotero / BibTeX 对应 | 是，已由 `投稿稿引用同步与核对清单.md` 逐项登记 |
| 当前 TeX 引用系统 | 已迁移到 `docs/paper_notes/CNN_FPGA_GKP_submission_refs.bib` active-key BibTeX；不是最终期刊引用格式 |
| 已移出强引用的硬件候选 | `AFS` / `Astrea`，本地稳定 DOI/arXiv/ACM/IEEE 元数据仍缺 |

## 已覆盖的主题组

| 主题组 | TeX key | 本地状态 |
| --- | --- | --- |
| GKP foundations / review | `gkp2001`, `grimsmo2021` | matched |
| GKP analog / soft information | `fukui2018`, `noh2020`, `noh2022`, `raveendran2022`, `berent2024`, `borah2025` | matched |
| GKP logical-channel / fidelity-oriented analysis | `hastrup2023`, `jafarzadeh2025`, `zheng2024` | matched；`zheng2024` preprint 仍需最终版本核验 |
| adaptive calibration / noise estimation | `spitz2018`, `wagner2021`, `dgr2023`, `chen2022`, `sivak2024` | matched |
| learned QEC modules | `bausch2024`, `chamberland2026`, `stein2026` | matched；2026 preprint 仍需最终版本核验 |
| real-time / FPGA decoders | `lilliput2022`, `helios2023`, `collision2025`, `qldpcfpga2025`, `caune2024`, `yang2026`, `ziad2024`, `maurer2025` | matched；部分条目需 publication-year / venue 规范化 |

## 投稿前仍需完成

1. 按目标期刊要求冻结 BibTeX style、导出 `.bbl` 或改用期刊模板；当前 active-key `.bib` 仍需最终格式处理。
2. 对 2025/2026 preprint 条目复核最终 arXiv / DOI / venue 状态。
3. 对 real-time / FPGA decoder 条目统一作者、题名、会议/期刊、年份和 DOI/arXiv 字段。
4. 继续保留 AFS/Astrea 为待核对候选；稳定元数据恢复前不得作为强 main-text citation。
5. 引用硬件 decoder 文献时，只能把其 latency/resource/closed-loop 指标写成外部标准，不能写成当前项目结果。
6. 引用 logical-channel / fidelity-oriented GKP 文献时，只能作为缺口和外部指标参照；当前 `final_ler` proxy 不能写成 fidelity、logical-channel tomography 或 process infidelity。

## 禁止外推

- 不把 active-key BibTeX 迁移写成“所有引用已达到最终投稿格式”；
- 不把文献卡图表核验写成对当前项目实验的验证；
- 不把硬件 decoder 文献的指标写成当前项目 hardware evidence；
- 不把 preprint 条目写成最终期刊版本，除非后续完成版本核验。
