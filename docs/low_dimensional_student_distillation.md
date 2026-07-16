# T4.4.3：低维指数递推 student 蒸馏

## 结论

对冻结的 T4.4.1 selected teacher 做严格 training/validation/evaluation 三分割蒸馏后，16/16 gates 通过。
validation-only 规则在 1、2、4 维候选中选择 4 维、restart 0；held-out evaluation physical-residual MSE 为
`6.083136e-6`。这证明一个 4-state、95-scalar 的可解释指数递推在当前 64-half-cycle、two-level、
matched-model 轨迹上高精度逼近 teacher 动作；不证明 student 保留物理 lifetime/fidelity gain。

## 冻结的拟合合同

每个候选都使用 outcome-specific affine recurrence：

`z[t+1] = a[m] z[t] + (1-a[m]) z_inf[m]`, `m in {g,e}`，

再通过 15-output affine head 与 `[2×14,1] * tanh(raw)` 生成 canonical physical residual。比较范围固定为
1/2/4 states，每个维度 3 个 fresh restarts、900 epochs；256 条 training、256 条 validation、256 条
evaluation trajectories 的 seeds 与内容哈希互异。restart 先按 validation MSE 选择，再选 validation MSE 位于
全局最佳 `5% + 1e-7` 容差内的最小维度；evaluation 在选择冻结后才报告。

训练器把逐步 affine recurrence 改写为数学等价的 cumulative-product 批量公式以消除小张量 GPU launch
开销。64 步随机序列与 literal step loop 的 float64 一致性在 `2e-13` 内；在线 artifact 仍执行原始逐步公式。

## 维度选择与 held-out 误差

| 状态维数 | 最佳 restart | validation MSE | evaluation MSE | scalars | healthy-step MAC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | `1.297436e-5` | `1.407074e-5` | 35 | 33 |
| 2 | 2 | `1.083297e-5` | `1.171606e-5` | 55 | 51 |
| 4 | 0 | `5.648504e-6` | `6.083136e-6` | 95 | 87 |

4 维是唯一进入 `6.030929e-6` validation eligibility threshold 的候选，因此没有为了缩小模型而接受明显
较差的 1/2 维，也没有查看 evaluation 后改选维度。

| evaluation comparator | MSE | selected student 相对降低 |
| --- | ---: | ---: |
| zero residual | `2.144444e-2` | `99.9716%` |
| latest-only affine | `1.373026e-3` | `99.5570%` |
| legacy T4.1.5 15-state student | `7.916932e-4` | `99.2316%` |
| selected 4-state student | `6.083136e-6` | — |

最大逐元素绝对误差为 `0.020634`。全部输出严格在 15 个 hard bounds 内；95 scalars 按 float32 仅为
380 bytes，但这里没有量化、RTL、综合、latency 或板测，所以不能把解析资源数写成 FPGA 结果。

## 在线安全与溯源

- `cnn_fpga/control/low_dimensional_recurrence.py` 只依赖 NumPy，不导入 torch、physics 或 teacher；
- student JSON 绑定 teacher checkpoint/state、T4.4.2 analysis、training/validation dataset SHA；任一字段篡改均拒绝；
- torch candidate 到 JSON artifact 最大误差 `2.22e-16`，逐步在线回放最大误差 `1.11e-16`；
- leakage、invalid、CRC fail、stale parameter 或 missed deadline 都 reset state 并返回 exact 15-zero residual；
- leakage 没有被伪造为 teacher 的第三个训练 token，也没有 `p(g)` 或 gain 语义。

## 非 demo 审计与限制

9 个候选保留全部 8,100 个 training-epoch rows、333 个 validation-checkpoint rows，以及三分割共 49,920
个 selected prediction rows；Source Data 总计 58,356 rows。所有参数张量在首 epoch 都获得 finite nonzero
gradient，fresh initializer 与最终 state hashes 全部不同。1 维三个候选在 175/175/225 epoch 达到 validation
最佳；2/4 维六个候选在 900-epoch budget cap 达峰，已经显式记录并禁止全局优化收敛 claim。当前误差远低于
所有强 comparator，继续延长训练不阻塞本 task，但 T4.4.4 必须用物理 trajectory/lifetime/fidelity/`p(g)`/
e-leakage burden 检验 gain retention，不能用 imitation MSE 代替。

产物：

- `cnn_fpga/benchmark/low_dimensional_student_distillation.py`
- `cnn_fpga/control/low_dimensional_recurrence.py`
- `tests/test_low_dimensional_student_distillation.py`
- `tests/test_low_dimensional_recurrence.py`
- `docs/t4_4_3_low_dimensional_student_validation.json`
- `docs/t4_4_3_low_dimensional_student_candidates.pt`
- `docs/t4_4_3_low_dimensional_student.json`
- `docs/t4_4_3_low_dimensional_student_source_data.csv`
