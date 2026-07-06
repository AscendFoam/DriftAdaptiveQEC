# QA Report v2

- PPTX: `docs\paper_notes\ppt_output\CNN_FPGA_GKP_submission_draft_two_slide_cn_v2_no_font_garble.pptx`
- Slide images: `docs\paper_notes\ppt_output\slide1_v2_no_font_garble.png`, `docs\paper_notes\ppt_output\slide2_v2_no_font_garble.png`
- Slide count: 2
- Embedded media files: 2
- Main visible text is rasterized into full-slide PNGs to avoid PowerPoint CJK font substitution, mojibake, and question-mark glyph fallback.
- Special-symbol reduction: replaced subscript/Greek-heavy formula with plain `Delta = K*s + b`; avoided rare glyphs in visible text.
- Structural bounds check: PASS
- Evidence boundary: no real-board execution success, no measured FPGA latency/resource/power, no finite-energy logical-channel fidelity, no p-value/CI, no deployment closure.
- Rendered preview note: the delivered slide PNGs are the rendered preview source used inside the PPTX.
