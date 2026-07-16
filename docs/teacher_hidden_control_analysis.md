# T4.4.2：teacher hidden state 与控制轨迹分析

## 结论

对 T4.4.1 validation-only 选出的 restart 601 做完全冻结的 post-hoc 分析后，17/17 gates 通过。当前固定/随机
trajectory 集上，10 维 GRU hidden 和 15 维 physical residual 都呈明显低维结构；单次相反 outcome 的影响在
约 10--12 个 half-cycles 内降到峰值的 1%。但是这只是 finite-cutoff、two-level、matched-model 轨迹中的
empirical response geometry，不是唯一物理 belief state、充分统计量或真实装置机制证明。

## 分析合同

- teacher checkpoint/state SHA 全程冻结，analysis optimizer step 为 0；
- 10 条原生 `g/e` fixed sequences 覆盖 all-run、alternation、16-block、双向 impulse、mid-run isolated event 和
  deterministic mixed；每条追踪 128 half-cycles；
- 前 20 half-cycles 另进入真实 differentiable simulator forced path，按 binary measurement 由 chosen outcome
  probability 还原 `p(g)`；
- 24/8 条随机 forced trajectories 以不同 seeds 分为 belief-probe training/evaluation；evaluation 不选 ridge、
  feature 或超参数；
- `leakage` 从未作为数值 2 输入 GRU。它只进入显式 OOD proxy：reset hidden，并令第一次 post-leak 动作
  exact zero residual/nominal control。

## 低维结构

| 对象 | 90% PC | 95% PC | 99% PC | 数值 rank |
| --- | ---: | ---: | ---: | ---: |
| 10-D hidden | 1 | 1 | 2 | 10 |
| 15-D control residual | 1 | 1 | 1 | 15 |

hidden 第一主成分解释 `96.90%` 方差；control 第一主成分解释 `99.89%`。这里的 rank 仍是满秩，因此结论是
“主要方差集中在低维线性子空间”，不是网络数学上退化成一维，也不是 T4.4.3 student 已经通过。

## `p(g)` 与 belief-like probe

forced-path `p(g)` 范围为 `[0.353157, 0.936084]`；最大 trace error `4.44e-16`，最小 final eigenvalue `0`。
trajectory-disjoint evaluation 结果为：

| probe | evaluation `R²` | MAE | RMSE |
| --- | ---: | ---: | ---: |
| 10-D hidden linear probe | 0.667797 | 0.049409 | 0.075305 |
| 5-D observed-history features | 0.487393 | 0.065644 | 0.093544 |

hidden 相对 last outcome/run/cumulative fraction/EWMA/time 这组简单 observed features 的 `R²` 高
`0.180404`。这支持 hidden 含额外的 history-conditioned、belief-like `p(g)` 信息；但 target 仍由同一
assumed simulator 产生，不能写成校准后的 cavity/transmon belief 或 device readout posterior。

## 指数饱和与明确反证

all-g/all-e 的 15 个 control residual 分别拟合 `c+d a^t`。30 个 fits 中 28 个 `R²≥0.95`，整体 median
`R²=0.999104`。两个未通过高拟合门的项被保留：

- all-g `virtual_rotation`：`R²=0.696368`；
- all-e `virtual_rotation`：`R²=0.934592`。

因此可以把“多数 gate parameters 呈单指数饱和”作为 T4.4.3 的候选压缩结构；不能声称全部 15 参数都由
单一指数精确描述。`virtual_rotation` 必须允许独立/非单指数 residual 或在 student scope gate 中显式处理。

## 有效记忆

| impulse / observable | `1/e` 以下 | 5% 以下 | 1% 以下 |
| --- | ---: | ---: | ---: |
| e 后持续 g：hidden | 3 | 7 | 10 |
| e 后持续 g：control | 4 | 8 | 10 |
| g 后持续 e：hidden | 3 | 8 | 11 |
| g 后持续 e：control | 5 | 9 | 12 |

阈值均在 128-half-cycle horizon 内 persistent crossing，没有右删失。固定点局部 Jacobian 的 spectral radius
为 `0.618031`（g）和 `0.596212`（e），对应 linearized time constant `2.0781/1.9336` half-cycles；fixed-point
residual 小于 `4e-17`。该局部线性时标与非线性 impulse 的 1% memory 长度不能混称。

## 非 demo 审计与产物

- native prefix replay、重复 trace 和 action hard-bound violation 均为 0；
- third token 动态负测确实被 analyzer 拒绝；
- 2,089-row Source Data 含 1,290 条 fixed native rows、129 条 leakage proxy rows、640 条严格 split belief
  rows 和 30 条逐参数 fit rows；
- 13 focused、87 adjacent tests 通过。

产物：

- `cnn_fpga/benchmark/bounded_residual_teacher_analysis.py`
- `tests/test_bounded_residual_teacher_analysis.py`
- `docs/t4_4_2_teacher_hidden_control_analysis.json`
- `docs/t4_4_2_teacher_hidden_control_source_data.csv`

T4.4.3 可以据此拟合低维 recurrence，但只能使用 training split 选择结构；T4.4.4 必须在独立物理同轨上验证
gain retention，不能用本任务的 PCA/linear-probe `R²` 替代。
