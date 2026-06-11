# T43: 论文 Background / Related Work 有界正文草稿 —— 给人类的说明

## 1. T43 做了什么

T43 在 T42 产出的骨架基础上，为论文的 Background / Related Work 章节撰写了有界正文草稿。产出文件是 `docs/paper_materials/paper_background_related_work_draft.md`，包含 6 个子节的真实段落级正文（不是骨架或要点列表）：

1. **GKP 量子纠错与自适应解码问题**：介绍 GKP 编码的基本原理、syndrome 测量、线性解码规则 $\Delta = K s + b$，以及噪声参数漂移导致固定解码器失配的工程动机。

2. **双回路时间尺度分离**：解释快回路（~5μs 确定性线性解码）和慢回路（10–100ms 统计估计与参数更新）的架构设计，以及参数 bank 切换的原子性约束。

3. **机器学习辅助 QEC 解码**：综述 ML-based 解码在 surface code 等离散变量码上的进展，指出 GKP 连续变量场景的差异，以及本工作用 Tiny-CNN 作为慢回路估计器的定位。

4. **经典自适应漂移跟踪方法**：描述五个经典基线（EKF、UKF、Window Variance、RLS、Constant Residual-Mu），明确 UKF 是最强经典基线。引用 supported claims C2/C3。

5. **Teacher-Guided 残差修正定位**：核心方法定位子节。解释"为什么不让 CNN 直接回归绝对参数"→ 因为"离线训练改善 ≠ formal HIL 改善"（稳定结论 9.1 第 7 条）。阐述 teacher-guided residual-b 的两个机制：teacher 提供稳定锚点，residual 降维。引用 supported claim C3 和稳定结论。

6. **量子系统验证中的证据边界**：简短的文献讨论段落，说明软件模拟与硬件验证之间的证据落差。正确标注 C5 为 supported、C6/C7/C8 为 blocked。保持调查性语气而非自我辩护。

## 2. 它没有改变什么

T43 没有改变任何事实：

- 所有 claim 状态（supported / partial / blocked）与 `docs/paper_materials/paper_claim_evidence_ledger.md` 完全一致。
- 没有修改源码、config、`runs/`、`artifacts/` 或阶段结论文档。
- 没有运行 benchmark、训练、`.tflite`、硬件或 cleanup。
- 没有撰写 Abstract、Introduction、Method、Results、Conclusion 或任何超出 Background/Related Work 范围的正文。
- 草稿中保留了内部草稿标注（如 `[supported claim C3]`、`[stable conclusion 9.1 item 7]`），在后续论文组装时统一清理。

## 3. Review 结论

- Verdict: `PASS`
- Blocking issues: none
- Non-blocking issues:
  - N1: Subsection 6 可能被读作部分自我辩护；如果后续觉得不自然，可以并入 Limitations 章节。
  - N2: 引用标记 [1]--[7] 尚未对应具体文献列表；后续需建立共享文献文件。
  - N3/N4: 内部草稿标注格式不完全统一；论文组装时清理。

## 4. 下一步是什么

T43 完成后，论文的 Background / Related Work 章节已经有了第一版可用的正文草稿。下一步建议：

1. 确认 Background/Related Work 正文是否满足预期。
2. 选择下一个有界正文撰写任务：
   - **Introduction 正文**：基于 T42 校准过的 contribution bullets 撰写。
   - **Method/System 正文**：自然衔接 Background 子节 1--2 的架构描述。
3. 建立共享文献文件，统一管理引用标记。
4. 持续对照 claim ledger 和 risk audit 确保措辞不越界。
