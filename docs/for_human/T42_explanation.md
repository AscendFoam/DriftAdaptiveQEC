# T42: 论文 Background / Related Work 骨架与定位校准 —— 给人类的说明

## 1. T42 做了什么

T42 在已有论文骨架（`docs/paper_materials/paper_draft_skeleton.md`）基础上做了三件事：

1. **新增 Background / Related Work 章节**。这个章节规划了 6 个子节：
   - GKP 量子纠错的基本问题框架
   - 快回路/慢回路时间尺度分离的工程动机
   - 已有的 CNN/ML 辅助 QEC 解码工作
   - 经典自适应漂移跟踪方法（EKF、UKF、Window Variance、RLS）
   - "teacher-guided residual" 方法定位：为什么在经典 teacher 上学残差比让 CNN 学全部参数更合理
   - 量子系统论文中的 benchmark 和部署证据边界

   这个章节是骨架级的大纲（子标题 + 需要覆盖的内容范围 + 哪些 claim 可以引用、哪些不能），不是论文正文。

2. **校准了论文标题和定位**。产出了 `docs/paper_materials/paper_method_positioning_calibration.md`，对比了两种定位：
   - **保守定位**：把论文写成"bounded recovery / revalidation report"。最安全，但对目标会议/期刊（QCE、TQE、EPJ Quantum Technology）来说偏窄。
   - **方法向前定位**：把论文写成方法论文，中心贡献是"teacher-guided residual-b 双回路解码框架"，但在 abstract、introduction 和 limitations 里严格限制在当前证据边界内。
   
   **推荐方案**：方法向前的标题（"A Dual-Loop Teacher-Guided Residual Decoding Framework for Real-Time GKP Error Correction"），配合证据受限的正文。

3. **校准了 introduction 的 contribution bullets**。每条贡献明确绑定了 claim ID 和安全措辞，blocked claims（C6, C7, C8, C10, C11）不进入任何贡献条目。

## 2. 它没有改变什么

T42 没有改变任何事实：

- 所有 claim 状态（supported / partial / blocked）与 `docs/paper_materials/paper_claim_evidence_ledger.md` 完全一致。
- 没有写任何论文正文段落。
- 没有运行 benchmark、训练、`.tflite`、硬件或 cleanup。
- 没有修改源码、config、`runs/`、`artifacts/` 或阶段结论文档。

## 3. 下一步是什么

T42 完成后，论文骨架已经有了 Background / Related Work 的结构空间和明确的定位校准。下一步应该是：

1. **确认标题定位**（Captain / 人类决策）：方法向前标题是否被接受？如果是，后续正文写作就按这个定位推进。
2. **逐章节有界正文撰写**：从 Background / Related Work 的前几个子节开始，按 bounded task 逐段推进。每个 task 只允许写一个章节的 prose，不允许升级证据。
3. 在正文撰写过程中，持续对照 `docs/paper_materials/paper_claim_evidence_ledger.md` 和 `docs/paper_materials/paper_reviewer_risk_audit.md` 确保措辞不越界。
