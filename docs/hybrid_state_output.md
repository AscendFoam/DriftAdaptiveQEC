# T4.1.3 混合慢状态输出合同

## 1. 结论

T4.1.3 将 host estimator 输出冻结为一个 **future-only hybrid state**，包含连续
observed-noise/calibration 参数、四态 regime posterior、未来 leakage risk、estimated recovery
burden posterior、moving-block-bootstrap uncertainty 和 inactive parameter-bank recommendation。

它不是 fast controller：payload 不含逐周期 correction、frame update、pulse/gate action 或 logical
decision。cycle `t` 的输出最早只在 `t+1` 生效；真正的 bank stage/commit 仍由 `ParamBank` 负责。

## 2. 输出 schema

| Family | 字段 | 语义/单位 |
| --- | --- | --- |
| continuous noise/calibration | `mean_q/p`、`sigma_q/p`、`rho_qp`、tail/X-e/Z-e/leakage rates | q/p 为 lattice coordinate；其余 dimensionless/probability；来自 observed periodic moments |
| regime posterior | `normal/burst/leakage/calibration_shift` 四概率、entropy、most-likely class | 严格归一化；当前 producer 是 T4.1.1 注册 HMM |
| leakage/recovery | next-cycle risk、32-cycle horizon risk、depth 0--6 burden posterior | Beta posterior + observed run proxy；不是 simulator recovery-depth truth |
| uncertainty | 9×9 continuous covariance、标准误、三类 entropy、OOD score、recommendation confidence | moving-block bootstrap；当前只注册 synthetic-pilot scope |
| bank recommendation | `stage_candidate/hold_active`、profile、K/b、base version、validity、ID、CRC32 | future inactive-bank proposal；不是同周期 correction |

连续参数顺序固定为：
`mean_q, mean_p, sigma_q, sigma_p, rho_qp, tail_rate, x_e_rate, z_e_rate, leakage_rate`。
协方差必须对称 PSD；所有 posterior 严格归一化；概率/相关系数/单位/范围在 dataclass 构造时 fail closed。

## 3. observed-only estimator 路径

1. `ExperimentalHistorySample` 提供 T4.1.2 的 256×53 causal history；只使用 `mask=1` 且 `valid=1` 行。
2. q/p residual 通过 periodic characteristic moments 得到 centered mean 与完整 2×2 covariance；rate 参数直接来自 observed one-hot。
3. T4.1.1 checkpoint 按 manifest SHA-256 恢复 `RollingGaussianHMMAdapter`，每 32 cycles 消费 14 项 canonical observed summary，最多保留 8 windows。
4. leakage 使用 Jeffreys Beta prior 的 observed event posterior；horizon risk 为未来窗口 union probability。
5. recovery 输出是由当前 e/leakage runs 和 observed rates形成的 0--6 burden posterior，字段明确写 `truth_semantics=false`。
6. 9 个连续量用 moving-block bootstrap 估计 covariance；block 保留局部时间相关，不把 256 行冒充 256 个独立样本。

当前 HMM 仍走 T4.1.1 的 14-summary path，并未用 53-feature schema 重训。hybrid output 证明的是接口
兼容与 producer composition，不是 richer-input HMM 已优化；该缺口保留给 T4.1.4/T4.2/T5.4。

## 4. parameter-bank 语义

continuous covariance 被转换为 principal sigma/axis/ratio，复用现有 `ParamMapper` 的
`C(C+R)^-1` 路径产生 symmetric bounded K 和 bias；前一 active bank 仍参与 beta smoothing。
recommendation 同时绑定：

- `base_active_version`：防止 stale proposal 覆盖新 bank；
- `valid_from_cycle > as_of_cycle` 与 expiration：防止 hindsight/无限期重放；
- canonical JSON CRC32 与 deterministic recommendation ID：防止 payload 损坏/错配；
- `registered_synthetic_pilot` 或 `uncalibrated_contract` scope；后者必为 `hold_active`；
- OOD/regime-confidence gate：未通过时输出 fallback/hold，而不是偷偷 stage。

`stage_parameter_bank_recommendation()` 只写 inactive bank。已有 pending writer 会走真实
`ParameterUpdateConflictError`；active version 不匹配、CRC 损坏、错误 stage cycle 或 hold payload
都拒绝。只有下一 cycle 的 `commit_if_ready()` 才能原子切换 active bank。

## 5. 生产式验证

`cnn_fpga.benchmark.hybrid_state_output_validation` 使用 8 seeds×2,048 cycles；前 4 seeds 为
nominal scheduler lane，后 4 为 deadline/通信/CRC/stale/conflict stress lane。每条轨迹在 256-cycle
history ready 后每 32 cycles 输出一次，共 456 rows：

- nominal/stress：`228/228`；
- `stage_candidate/hold_active = 58/398`；
- 58 次 stage 全部在下一 cycle commit，staging K/b 与 recommendation 最大绝对误差 `0`；
- 五种 profiles 全覆盖：fallback `398`、normal `27`、X recovery `11`、Z recovery `16`、leakage hold `4`；
- OOD score 范围 `[0.46875,1.0]`；leakage horizon risk 范围 `[0.0604,0.9991]`；estimated recovery depth 范围 `[1.6007,5.1765]`；
- checkpoint hash/winner、future alignment、normalization、PSD、atomicity、conflict/stale/hold/CRC negative、deterministic replay、unique ID 和 source schema 共 17/17 gates 通过。

工件：

- `docs/t4_1_3_hybrid_state_output_validation.json`；
- `docs/t4_1_3_hybrid_state_output_source_data.csv`；
- `tests/test_hybrid_state_output.py`；
- `tests/test_hybrid_state_output_validation.py`。

## 6. 非 demo 审计与边界

首轮实现曾直接用 32-cycle horizon leakage probability 决定即时 profile。审计发现 union probability
会把小 per-cycle risk 放大，导致 recommendation 几乎全部进入 leakage hold；现已改为 current run +
next-cycle risk，horizon risk 只保留为慢状态输出。修复后 stage lane 覆盖 normal/X/Z/leakage 四类，
stress lane 保持 fallback/hold。

允许声称：observed-only hybrid software output、descriptive block-bootstrap uncertainty、version/CRC-bound
future bank proposal 与真实 ParamBank atomicity 已实现。禁止声称：recovery depth 真值被在线识别、risk/
uncertainty 已物理校准、53-feature HMM 已重训、parameter recommendation 有 logical/control gain、fixed-point/
RTL/board/device 已验证。

