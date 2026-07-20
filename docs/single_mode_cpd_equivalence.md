# T6.17.1 single-mode Euclidean CPD 与 CI 等价边界

- verdict：`PASS_SINGLE_MODE_EUCLIDEAN_CPD_EQUALS_CI_WITH_MAP_BOUNDARIES`
- production q10×q10：1,048,576 points，mismatch=0，runtime=0.628 s
- boundary：1,000,000 points / 2,000,000 coordinates，exact-tie coordinates=332,386，mismatch=0，runtime=0.759 s
- gates/mutations：15/15、15/15；Source Data=58 rows

## 结论

对 `Λ=λZ²` 和 isotropic Euclidean metric，目标函数分解为两个独立平方项，因此 CPD 的 Voronoi cell 正是 CI 的半开区间笛卡尔积；正半边界按冻结规则归入较大整数。该结论只说明两个名称在这一行是重复 comparator，图表只能计一次。

完整 production syndrome code 位于 centered canonical cell，所以 hard action 全为 00；这本身是平凡事实。非平凡的 alias/parity 与 tie 证据由独立的一百万点 unwrapped boundary audit 提供，不能把 canonical-cell 全零误写成广义 MAP 优势。

## 不等价反例

| family | selection | CI | likelihood MAP | witness |
| --- | --- | ---: | ---: | --- |
| `biased` | canonical one-lattice mean shift; not outcome-tuned | 0 | 1 | `0.0` |
| `correlated` | first lexicographic mismatch on fixed 101x101 canonical midpoint grid | 0 | 2 | `[-0.4950495049504951, 0.00990099009900991]` |
| `finite_energy_likelihood` | first mismatch on fixed 2001-point canonical midpoint grid | 0 | 1 | `-0.49975012493753124` |

biased mean 会交换 periodic coset mass；correlated covariance 的 Mahalanobis cross term 破坏坐标独立性；finite-energy state likelihood 又引入非均匀峰权重/收缩。三者都说明 weighted/analog/coset MAP 不能改名成 CPD 或 CI。

## Claim 边界

- established：single-mode square/isotropic Euclidean CPD = CI（project-native mathematical/correctness evidence）。
- prohibited：把 CPD 与 CI 双计为两次胜场；声称 CPD=arbitrary MAP；从本实验推出 0.602 threshold、surface-GKP finite-size 或 multimode scaling。
- 本实验没有 LER/SOTA/FPGA measured claim；runtime/memory 仅是 Python correctness audit 的测量边界。

## 产物

- `cnn_fpga/benchmark/single_mode_cpd_equivalence.py`
- `docs/t6_17_1_single_mode_cpd_equivalence.json`
- `docs/t6_17_1_single_mode_cpd_equivalence_source_data.csv`（58 rows）
