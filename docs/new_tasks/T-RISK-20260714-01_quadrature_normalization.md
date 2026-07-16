# T-RISK-20260714-01 quadrature normalization

- **Task ID：** T-RISK-20260714-01
- **标题：** 冻结 quadrature normalization 并修复 Fourier-p audit
- **日期：** 2026-07-14
- **状态：** Done

## 输入材料

R-N041、T2.3.3 legacy Fourier-p audit、GKP/Campagne/Sivak 一手 convention、现有
finite-energy/Fock/noise-transfer/cross-fidelity 代码与 PC-N01。

## 实际完成内容

建立四 chart normalization contract；修复 damped-projector centers/width/envelope/Jacobian
完整 dilation、chart-qualified dB/variance、registered Fock bridge 与 canonical q/p folding；
保留 legacy ambiguous path 为负证据，并同步旧任务机器结果和 claim 边界。

## 产物路径

- `physics/quadrature_conventions.py`
- `tests/test_quadrature_conventions.py`
- `docs/quadrature_normalization_contract.md`
- `docs/t_risk_20260714_01_quadrature_validation.json`
- `docs/tasks/T-RISK-20260714-01_quadrature_normalization.md`

## 验证方式和结果

direct `32 passed`；quadrature/cross machine gates 分别 15/15、15/15；10/12 dB canonical
Fock 最大 q/p LER gap `1.51e-7`；legacy gap `>0.418`；full suite
`751 passed, 4 failed`，四项为既有 R-N012 缺失文档。

## 风险复核

R-N041 降为 Mitigated。PC-N01 仅关闭坐标/解析子门；device envelope/`nbar` 与 coherent
joint-axis fidelity 仍 fail closed。R-N037/R-N038/R-N040 不关闭。

## 是否需要插入新 task

否；剩余工作已由正常任务和 calibration/hardware gates 承接。

## 对任务板的同步说明

T-RISK-20260714-01 改为 Done，T2.3.4 改为 In Progress，当前推荐任务更新为 T2.3.4。

## 对论文 claim 的影响

允许写 chart-qualified axis-resolved q/p 对齐；禁止写 decoder axes 是 canonical pair、
coherent joint process fidelity、device-calibrated squeezing 或 infinite-cutoff theorem。
