# T5.0.1 文献趋势 reproduction table

## 1. 结论与使用规则

本表冻结 14 个趋势目标。当前 registry 为 `PASS`，含 17/17 machine gates 和 52 行 Source Data；这里的
`PASS` 只表示来源、目标、容差、calibration/holdout 用途和现有 artifact 绑定完整，**不表示 14 行都已
复现**。当前有 5 行 `REGISTERED_PENDING`、2 行 `REFERENCE_ONLY` 和 1 行
`REPORTING_TEMPLATE_ONLY`。

- `calibration_only` 可用于建立模型/有效域，不得再作为独立检验；
- `independent_holdout` 已使用与训练/选择分离的数据，且禁止回调选模；
- `future_holdout_preregistered` 只冻结未来门禁，当前不能写成结果；
- `reference_only_no_model_selection` 与 `reporting_template_only` 只规定对照或写法，不是项目验收数值；
- Knill/qunaught、P-Steane 和 trapped-ion 始终是 secondary，不进入 sBs 主排名。

机器镜像：`docs/t5_0_1_literature_trend_reproduction.json`；逐行证据：
`docs/t5_0_1_literature_trend_reproduction_source_data.csv`。

## 2. 一手来源核验

| Source ID | 来源 | 类型 | 本表用途 |
| --- | --- | --- | --- |
| `SRC-CAMPAGNE-2020` | [Campagne-Ibarcq et al., Nature 2020](https://www.nature.com/articles/s41586-020-2603-3), DOI `10.1038/s41586-020-2603-3` | 同行评议实验 | sharpen--trim 结构及 Pauli QEC-on/off 参考 |
| `SRC-SIVAK-2023` | [Sivak et al., Nature 2023](https://www.nature.com/articles/s41586-023-05782-6), DOI `10.1038/s41586-023-05782-6` | 同行评议实验 | sBs displacement、occupancy/correlation、gain/timing |
| `SRC-PUVIANI-PRL-2025` | [Puviani et al., PRL 2025](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.134.020601), DOI `10.1103/PhysRevLett.134.020601` | 同行评议理论仿真 | NMF 方向与 10→1000-cycle 外推目标 |
| `SRC-RALPH-2024` | [Ralph et al., Entropy 2024](https://www.mdpi.com/1099-4300/26/10/874), DOI `10.3390/e26100874` | 同行评议理论 | noise-transfer 高/低 squeezing 有效域 |
| `SRC-MARQVERSEN-2025` | [Marqversen et al., arXiv:2505.14775](https://arxiv.org/abs/2505.14775) | 一手预印本 | Knill/Steane 等价与 qunaught squeezing secondary |
| `SRC-CHEN-2026` | [Chen et al., arXiv:2604.08247](https://arxiv.org/abs/2604.08247) | 一手预印本 | P-Steane `(a,b)` noise shaping secondary |
| `SRC-FONTBOTE-2026` | [Fontboté-Schmidt et al., arXiv:2605.08009](https://arxiv.org/abs/2605.08009) | 一手预印本实验 | 双模 trapped-ion QEC-on/off 报告模板 |

前三篇的本地全文另有 8 个行级 fragment/hash 锚；其余近期论文元数据在 2026-07-16 从官方期刊或 arXiv
页面核验。预印本状态明确保留，不冒充同行评议定稿。

## 3. 逐趋势注册表

| ID | role | 数值/方向目标与容差 | calibration / holdout | 当前状态与证据 | 后续门禁 |
| --- | --- | --- | --- | --- | --- |
| `LT-2020-STRUCTURE` | cross-validation | 严格保持 `2 sharpen + 2 trim` 原生循环，不允许映射成 sBs cycle | calibration only | `STRUCTURE_IMPLEMENTED_NOT_NUMERIC_REPRODUCTION`；T2.2.2 fault overlay 与 non-executable secondary contract 通过 | T5.0.2 独立 protocol-trend holdout |
| `LT-2020-QEC-ON-OFF` | cross-validation | 外部方向为 X/Y/Z 均 QEC-on > off；外部 on 值 `275/160/275 us` 只作参考。本项目未来以逐 Pauli paired 95% CI 下界 `>0`，不要求匹配微秒 | future holdout；不可选模 | `REFERENCE_ONLY` | T5.3.1 logical channel |
| `LT-2023-DISPLACEMENT` | main | recovery depth 在 `lS/4` 附近最大；峰位置容差 `1/16`，左右 Spearman `|rho|>=0.95`，20-cycle recovery fraction `>=0.98` | calibration only | `QUALIFIED_DIRECTIONAL_PASS`；峰 `0.25`、rho `+1/-1`、recovery `1.0` | T5.2.1 独立注入 |
| `LT-2023-OCCUPANCY-CORRELATION` | main | occupancy 对 `0.825` 绝对差 `<=0.02`；tail shrink paired CI-low `>=3e-4`；post-removal tail `<=1.5e-3` | calibration only | `QUALIFIED_DIRECTIONAL_PASS`；误差 `0.006435`、CI-low `0.001684`、tail `0.000192` | T5.2.3 multilevel leakage/reset |
| `LT-2023-GAIN-TIMING` | main reference | 文献 `G=2.27±0.07`、constituent/full cycle `4.924/9.848 us` | reference only；不可选模 | `REFERENCE_ONLY`；尚无 project device timing | T5.1.5 与 T6 |
| `LT-2025-NMF-DIRECTION` | main | 只在已注册 PRL-like comparator lane 要求 paired NMF−MF logical-Z lifetime CI-low `>0` | independent holdout；不可回调选模 | `QUALIFIED_DIRECTIONAL_PASS`；差 `0.206114 [0.084161,0.328067]`。T4.4.5 exact-budget MF 跨 cutoff 排名反转仍保留 | T5.4.4/T5.4.5 |
| `LT-2025-NMF-HORIZON` | main | 文献 train 10/eval 1000；项目执行 2/5/10/32 training sweep 与 `1e3/1e5/1e6` cycles 全步 recurrence | future holdout；不可选模 | `QUALIFIED_RECURRENCE_PASS_PHYSICAL_GAIN_NOT_ESTABLISHED`；2-cycle 独立 32-cycle MSE `9.95e-5`、long worst `8.14e-4`；10/32-cycle long worst `<2.36e-5/<7.62e-6`；reset 最慢 20 half-cycles | T5.4.5 |
| `LT-2025-KNILL-EQUIVALENCE` | secondary | 独立参数网格上 special-case 最大绝对差 `<=1e-8` | future holdout；不可选模 | `REGISTERED_PENDING`；secondary executable=false | T5.0.2 |
| `LT-2025-QUNAUGHT-SQUEEZING` | secondary | 预注册 squeezing sweep 全点保持 qunaught-Knill 对其他 Knill variants 的对称 squeezing 方向 | future holdout；不可选模 | `REGISTERED_PENDING`；secondary executable=false | T5.0.2 |
| `LT-2026-PSTEANE-CONDITION` | secondary | 在小噪声且 data 比 ancilla 更噪的适用域验证 `2a=b` 对 q/p 输出噪声方差乘积的 stationarity/argmin；grid argmin error `<=0.01` | future holdout；不可选模 | `REGISTERED_PENDING`；secondary executable=false | T5.0.2 |
| `LT-2026-PSTEANE-NOISE-RATIO` | secondary | 在 disjoint data/ancilla noise-ratio grid 上验证选择 `(a,b)` 与相对 ME-Steane 方向；保留 `(1,1)` 与 `(1/sqrt2,sqrt2)` special cases | future holdout；不可选模 | `REGISTERED_PENDING`；secondary executable=false | T5.0.2 |
| `LT-2024-NOISE-TRANSFER-HIGH` | main | `>=10 dB`：noise-transfer/direct q-LER gap `<=5e-5`、effective z-score `<=2`、canonical q/p gap `<=1e-6` | calibration only | `QUALIFIED_DIRECTIONAL_PASS`；`3.92617e-5`、`1.70763`、`1.50991e-7` | T5.0.2 独立 cross-fidelity holdout |
| `LT-2024-NOISE-TRANSFER-LOW` | main negative | `3 dB` 必须保留 clipping failure：q-LER gap `>=0.01` 且 clipping ratio `<0.5` | calibration only | `NEGATIVE_BOUNDARY_VERIFIED`；gap `0.015408`、ratio `0.357665` | T5.0.2 negative holdout，禁止重调掩盖 |
| `LT-2026-TRAPPED-ION-REPORT` | secondary report | 必报 Pauli-resolved on/off、ratio uncertainty、wall-clock/round、reset recoil、parallel-control cost；外部数值不作验收阈值 | reporting template only；不可选模 | `REPORTING_TEMPLATE_ONLY` | T5.3.1/T5.3.4 |

## 4. 2026 trapped-ion 报告模板的精确边界

该论文在双模、单个 trapped ion 的 GKP Bell-state 实验中报告：Bell fidelity `0.69(1)`；XX/YY/ZZ 的
QEC-on lifetime 为 `5.0(7)/3.8(9)/5.3(8) ms`，off 为 `2.4(2)/2.3(4)/2.3(2) ms`，对应提升
`2.1(3)/1.7(5)/2.3(4)`，平均 `2.0(2)`；单轮 QEC 约 `500 us`。本项目只复用其报告结构：逐 Pauli、
on/off、带不确定度 ratio、物理时间、reset recoil 和并行控制代价。它不是 single-mode sBs、不是当前
superconducting cavity、不是 FPGA 实测，因此这些数值全部禁止迁移到 project pass/fail。

## 5. 非 demo 与 claim 审计

1. registry 绑定 6 个现有 artifacts 的 SHA-256，并重新检查各自 machine gates；不是手抄一张静态表。
2. 来源正文有 8 个 line fragment/hash 锚；官方 metadata、DOI/arXiv version 与核验日期另行保存。
3. 每个 target 都有目标、容差、用途、model-selection access、状态、证据、下一 gate 和 prohibited transfer。
4. `REGISTERED_PENDING` 明确不能计作通过；secondary 当前 `executable=false`，避免用解析口号冒充实现。
5. 同时保存 noise-transfer 正有效域和 3 dB 负边界；禁止只报告“看起来正确”的高 squeezing 点。
6. NMF 行强制携带 exact-budget MF cutoff reversal，禁止把局部 CI 升级成 universal NMF superiority。
7. external gain/lifetime/timing 与 reporting template 全部不可选模，避免跨平台数值泄漏到项目门槛。

因此 T5.0.1 完成的是可审计的预注册与证据路由；真正的新 independent holdout 从 T5.0.2 开始。

## 5. T5.0.2 后续验证快照（不改写本表历史状态）

T5.0.2 已在 disjoint 正式点执行。main cross-fidelity family 因 `10.25 dB` pooled effective/noise-transfer
最大 z-score `2.293338 > 2.0` 判为 `FAIL`；`11.75 dB` 和 `2.5 dB` 负边界通过各自冻结门禁。secondary
P-Steane 在 252 点全新解析网格上判为 `PASS`。T5.0.1 artifact 中 P-Steane 行仍保留当时的
`REGISTERED_PENDING`，以避免用后续结果覆写预注册快照；正式结果见
`docs/independent_cross_fidelity_holdout.md`。
