# 投稿稿 BibTeX 编译入口同步记录

日期：2026-07-06

## 修改对象

- `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`
- `docs/paper_notes/README.md`
- `docs/paper_materials/README.md`

## 问题

投稿稿的主编译入口以仓库根目录为工作目录运行 `latexmk`，但 TeX 文件此前使用裸路径：

```tex
\bibliography{CNN_FPGA_GKP_submission_refs}
```

这会让 BibTeX 在根目录查找 `CNN_FPGA_GKP_submission_refs.bib`，而实际 active-key BibTeX 文件位于 `docs/paper_notes/CNN_FPGA_GKP_submission_refs.bib`。在旧 `.bbl` 存在时，PDF 仍可能编译成功，但该成功依赖缓存产物，不是干净可复现的引用编译入口。

## 本次同步

TeX bibliography database 已改为根目录可解析路径：

```tex
\bibliography{docs/paper_notes/CNN_FPGA_GKP_submission_refs}
```

## 边界

- 本次只修复 BibTeX 编译契约。
- 不新增 citation key。
- 不修改 BibTeX 条目内容。
- 不恢复 AFS/Astrea 等未核对硬件引用。
- 不完成目标期刊最终参考文献格式、`.bbl` 提交策略或 venue-specific style 核验。
- 不新增 benchmark、统计、hardware、finite-energy fidelity 或 deployment 证据。

## 需要验证

- 从仓库根目录运行投稿稿 `latexmk`。
- 确认 `.blg` 不再报告 `CNN_FPGA_GKP_submission_refs.bib` 查找失败。
- 确认没有 undefined citation / undefined reference。

## 本次验证结果

2026-07-06 已从仓库根目录运行：

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=D:\Codes\Quantum\DriftAdaptiveQEC\build\submission_draft_cd docs\paper_notes\CNN_FPGA_GKP_submission_draft.tex
```

结果：命令退出码为 `0`，输出 `build/submission_draft_cd/CNN_FPGA_GKP_submission_draft.pdf`。`build/submission_draft_cd/CNN_FPGA_GKP_submission_draft.blg` 记录：

```text
Database file #1: docs/paper_notes/CNN_FPGA_GKP_submission_refs.bib
```

日志扫描未发现 `I couldn't open database file`、`Citation ... undefined`、`There were undefined`、`LaTeX Warning`、`Package ... Warning` 或 BibTeX `Warning--`。`git diff --check` 仅报告既有 README 的 LF/CRLF 提示。

同日另以新输出目录 `build/submission_draft_bibtex_clean_20260706` 复跑 `latexmk`，退出码同为 `0`，生成的 PDF 长度为 `550383` bytes，`.bbl` 长度为 `9223` bytes，`.blg` 同样记录 `Database file #1: docs/paper_notes/CNN_FPGA_GKP_submission_refs.bib`。最终 `.log/.blg` 扫描未发现数据库打开失败、undefined citation/reference、LaTeX warning、package warning 或 BibTeX warning。
