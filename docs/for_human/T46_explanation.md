# T46: Multi-seed mechanism/intervention plan and trace pack —— 给人类的说明

## 1. T46 做了什么

T46 产了一份 docs-only 的机制证据计划文件 `docs/evidence_packs/mechanism_ablation/seed_mechanism_multi_seed_plan.md`。它回答了以下问题：

1. **当前能安全说什么**：`seed=20260429` 上存在 trace-supported 的 combined committed-`b` 不稳定性诊断证据（来自 T36+T38），但仅限单 seed，不是因果证明。

2. **什么还说不出来**：该不稳定性是否在其他 seed 上复现；降低残差幅度是否能稳定改善。

3. **最小 seed 选择逻辑**：现有 3 个 seed（20260427/20260428/20260429），建议新增 3 个（20260425/20260430/20260510），总上限 6 个。

4. **最小 trace 字段**：与 T38 一致（17 个字段），核心是 teacher_b、delta_b、committed_b 和 window_ler。

5. **干预/反事实矩阵**：识别了 3 个真正的机制测试（降低 residual clip、降低 residual scale、teacher-delta 衰减）和 3 个不属于机制测试的方向（新架构、新 loss、删特征）。推荐 I1（降低 residual_clip_b）为最高优先级。

6. **诊断证据 vs 因果证据的边界**：当前只有诊断证据；因果证据需要干预实验在多 seed 上一致改善。

7. **未来执行任务的 go/no-go 规则**：Go 条件是 ≤6 个 seed、冻结 4 场景、≤1 个干预变体；No-Go 触发是新场景、新基线、或把诊断写成因果。

## 2. 它没有改变什么

T46 没有改变任何事实：

- 没有运行任何 benchmark、训练、`.tflite`、硬件或 cleanup 命令。
- 没有修改源码、config、`runs/`、`artifacts/` 或治理文件。
- 没有把单 seed 诊断升级成多 seed 确认或因果证明。
- 没有重新打开 T45 冻结的 benchmark 边界。
- 所有 claim 状态（C4 仍为 partial）与 `docs/paper_materials/paper_claim_evidence_ledger.md` 完全一致。

## 3. Review 结论

- Verdict: `PASS`
- Blocking issues: none
- Non-blocking issues:
  - N1: 现有 3-seed 样本量较小，"outlier" 判定可能不稳健 = `accepted`
  - N2: 干预 I1 的 clip 降幅（0.12→0.06）可能过大 = `accepted`（具体值由执行任务锁定）
  - N3: 现有 seed 是否需要重跑 vs 只需重新导出 = `accepted`（执行层面决策）
  - N4: 干预 I3 可能需要修改代码而非仅改 config = `accepted`（已标为次要优先级）

## 4. 下一步是什么

T46 完成后，建议的下一步是：

1. **Phase A trace-only probe**：对 6 个 seed（含 3 个新 seed）执行 trace 导出，判断 committed-`b` 不稳定性是否在 20260429 以外也出现。
2. **Phase B（条件性）**：仅在 Phase A 产生正面信号后，对 6 个 seed 执行 1 个干预变体（I1）。
3. 如果 Phase A 显示模式不可复现，C4 保持 `partial`，论文用适当的 hedge 诊断措辞。
