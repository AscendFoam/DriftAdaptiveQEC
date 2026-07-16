# T3.1.4 static sBs protocol-aware decoder

## 1. 结论与范围

本任务实现的是一个冻结的、只消费当前 deployable sBs classified observation 的静态
Bayes baseline：

- 在线输入仅为 `(observed_x, observed_z) in {g,e,leakage}^2`；
- 离线输入为 ideal Kraus-branch prior、preparation/readout/reset kernels 和显式
  leakage fallback cost；
- 输出为 `K_gg/K_ge/K_eg/K_ee/FALLBACK` 中的一个 branch action；
- hidden `g/e/f/higher` carry 用 stationary distribution 精确边缘化，不进入在线接口；
- 预测目标是 **ideal sBs Kraus branch**，不是 logical Pauli class、LER 或 full quantum
  recovery。

因此它满足 T3.1.4 的“finite-energy 或 sBs observation model”通过标准，但只允许写成
`static protocol branch-selection baseline`。它不是 T3.1.3 decoder oracle 的别名，也不是
finite-energy/channel-recovery optimum。

## 2. 为什么没有用一个改名的 finite-energy MAP 交差

执行前审计了 `physics/finite_energy_gkp.py`、`physics/logical_channel.py`、
`physics/finite_squeezing_noise.py` 与旧 shrinkage harness。对当前中心对称 finite-energy
logical-0/1 state，已有 folded-density posterior 在零偏置时的 hard MAP action 与 standard
全零 parity decision 完全相同；加入常数偏置后只是平移一条周期阈值。它可以作为 soft
response，但不足以证明存在新的 optimized hard decoder。

本任务因此选择任务板明确允许的 sBs observation/reset 路线：对 noisy `g/e/leakage`
分类、persistent hidden carry 和 conditional reset 建立完整静态 likelihood。finite-energy
wavefunction/Fock recovery 的更高保真度结论继续由 T5.3 验收，未被本任务冒充完成。

## 3. Exact likelihood

令 `c,c'` 为 constituent 前后的 hidden carry，`h` 为 pre-readout hidden state，`i` 为
ideal `g/e`，`o` 为 observed `g/e/leakage`。单 constituent 的 observation-conditioned
transition 为

```text
A_i(o)[c,c'] = sum_h P(h|c,i) P(o|h) P(c'|o,h).
```

无条件 transition 是 `A_i=sum_o A_i(o)`。对 ideal full-cycle branch `y`，按真实执行顺序
先 X、后 Z，冻结 branch prior `p(y)` 后的 carry transition 为

```text
T = sum_y p(y) A_{i_x(y)} A_{i_z(y)}.
```

实现从四个 pure carry state 分别迭代，只有全部收敛到同一 `pi=pi*T` 时才接受；可约的、
初态依赖的 stationary mixture 会 fail closed。随后精确计算

```text
L_y(o_x,o_z) = pi A_{i_x(y)}(o_x) A_{i_z(y)}(o_z) 1,
P(y|o_x,o_z) proportional to p(y) L_y(o_x,o_z).
```

部署端最终只保存 9×4 posterior LUT 和 action policy，不需要 hidden state 或在线更新。

## 4. Action 与 loss

四个 branch action 的 loss 为正确 `0`、错误 `1`。`FALLBACK` 使用每个场景显式冻结的
`0<cost<1`，且 **只有 observation 中至少一轴被分类为 leakage 才可选**：

```text
risk(K_j|o) = 1 - P(y=K_j|o)
risk(FALLBACK|o) = fallback_cost, only if observed leakage
```

非 leakage observation 必须选择 MAP branch，不能按低 confidence 任意拒绝样本。这一区别
把 safety fallback 与 T3.2.4 的 post-selection 诊断分开。报告同时保留 fallback rate、
observed-leakage rate、nonleak conditional branch error 和总 action cost。

## 5. 比较对象

| 行 | 在线输入 | 动作 | 定位 |
| --- | --- | --- | --- |
| `direct_observed_sbs_branch` | 当前 observed X/Z class | 非 leakage 直接按 class 选 branch；leakage fallback | simple protocol anchor |
| `static_sbs_branch_map` | 当前 observed X/Z class | exact posterior MAP，始终选 branch | 无 fallback 的静态 MAP |
| `static_sbs_observation_reset_bayes` | 当前 observed X/Z class | 非 leakage MAP；leakage 下按显式 cost 选 branch/fallback | 本任务 baseline |
| `ideal_sbs_branch_truth_reference` | hidden ideal branch | truth action | 不可部署零损失诊断下界 |

该比较已作为独立 `ideal-sBs-Kraus-branch target` 注册；`standard_binning` 不适用，禁止与
logical-coset decoder table 混表。

## 6. 生产验证

四个固定 synthetic assumption 场景，每个场景 8 个独立 seed、每 seed 20,000 cycles，
总计 640,000 个 Markov cycles。每条轨迹从解析 stationary carry 采样初态，ideal branch
序列与 simulator RNG 分离。参数不是设备标定值。

| 场景 | direct cost | forced MAP cost | protocol-aware cost | direct - protocol 95% seed-cluster CI | fallback rate | nonleak error: direct -> protocol |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced low-fault control | 0.151434 | 0.193794 | 0.151434 | `[0,0]` | 0.149213 | `0.134148 -> 0.134148` |
| biased branch prior | 0.367845 | 0.314556 | 0.293531 | `[0.072396,0.076231]` | 0.185625 | `0.390058 -> 0.296243` |
| asymmetric readout | 0.485669 | 0.541788 | 0.477850 | `[0.006884,0.008753]` | 0.314669 | `0.561736 -> 0.550327` |
| persistent leakage stress | 0.382403 | 0.452888 | 0.362047 | `[0.019164,0.021549]` | 0.542188 | `0.420778 -> 0.376314` |

场景等权、先在同一 seed 内平均后得到 aggregate direct-minus-protocol cost
`0.025622 [0.024873,0.026371]`。这种聚类方式避免把 4 个固定场景伪装成 32 个独立 seed。

Exact table 与独立 Markov Monte Carlo 的最大 likelihood 绝对误差为 `0.009812`；exact 与
empirical policy cost 的最大绝对误差为 `0.001771`。3 个非 control 场景的 paired CI 下界
为正，且同样 3 个场景存在 nonleak policy override，说明增益不是仅靠 leakage fallback。

## 7. 反简化审计

- 没有把对称 finite-energy soft response 的“posterior 存在”冒充 hard-action 改善；
- likelihood 不是 9 格手写 heuristic，而是完整 preparation/readout/reset kernel 的精确
  hidden-state marginalization；
- reset kernel sensitivity test 会改变 likelihood 和 parameter hash；
- reducible、初态依赖的 hidden carry model 会 fail closed；
- 640k 生产轨迹验证 exact likelihood/cost，Source Data 保留 32 个 scenario-seed 行；
- aggregate CI 以 8 个 seed 为 cluster，已修复首轮 scenario-seed pseudoreplication；
- fallback 只能由 observed leakage 触发，总成本和 survival/fallback cost 均保留；
- ideal truth 只在 evaluator/reference lane 使用，decoder signature 不接收 truth object；
- source hash 同时绑定 decoder、observation/reset model、sBs outcome 定义和 comparison registry。

## 8. 产物与复现

- 实现：`cnn_fpga/benchmark/static_protocol_decoder.py`
- 测试：`tests/test_static_protocol_decoder.py`
- JSON：`docs/t3_1_4_static_protocol_decoder_validation.json`
- Source Data：`docs/t3_1_4_static_protocol_decoder_source_data.csv`

```powershell
python -m cnn_fpga.benchmark.static_protocol_decoder
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests/test_static_protocol_decoder.py tests/test_sbs_observation_reset.py -q
```

## 9. Claim boundary

允许：在四个注册的 assumption-driven sBs 条件中，一个冻结的 observed-only static Bayes
policy 精确边缘化 preparation/readout/reset model，并相对 direct classified outcome 降低
显式 branch-action cost。

禁止：logical LER gain、full finite-energy recovery optimum、免费 post-selection、真实设备
readout/leakage 标定、实验闭环结果、或“CNN 超越 oracle”。
