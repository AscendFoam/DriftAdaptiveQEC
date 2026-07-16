# T3.1.3 full-state/regime/leakage oracle integration

## 1. 结论

当前 active T1.3.4 与 future T5 schemas 都显式选择并恰好接入一次
`full_state_model_oracle_map`。该行复用 T1.3.2 的 exact-`DriftState` periodic Gaussian-mixture
MAP，并新增 regime/burst provenance。它是不可部署的 assumed-model upper reference，不是实际
算法。

protocol leakage 另设 `hidden_leakage_flag_oracle`：读取 simulator hidden leakage kind 时只输出
`FLAG_LEAKAGE`，不凭空生成 Pauli correction。flagged cycles 的未知恢复成本用乐观/保守两端
包络报告，不能免费计为纠正成功。

## 2. Oracle taxonomy 与输入边界

| ID / alias | 真实角色 | 可部署 | 处理 |
| --- | --- | ---: | --- |
| `full_state_model_oracle_map` | exact hidden `DriftState` 下的 decoder model upper reference | 否 | 主要比较上界 |
| `hidden_leakage_flag_oracle` | exact hidden leakage kind 的 perfect erasure flag | 否 | 只报 cost envelope |
| legacy `oracle_delayed` | mock hidden target-param delayed reference | 否 | 不得称 decoder oracle |
| future trajectory lookup control oracle | control-policy/ansatz upper reference | 否 | 与本 decoder oracle 分列 |

普通 oracle decision 读取 centered syndrome，加上 exact hidden mean/covariance/outlier mixture、
regime、burst state。leakage flag 读取 `SyndromeTruthStep.leakage_kind`。`ObservedSyndromeStep` 的
deployable dict 不含 `DriftState`、hidden regime、leakage kind、true logical bits 或 target params；
`OracleHiddenContext.from_truth_step` 拒绝 observed dict。

T1.3.2 `OracleMAPResult/TrajectoryResult` 现保存 `state_regime` 与 `burst_active` provenance，
但 regime label 本身不被包装成额外性能：真正影响 likelihood 的仍是 exact state parameters。

## 3. Four-regime model-oracle matrix

四个 synthetic regimes 各用 4 个 evaluation seeds、每行 20,000 samples，共 320,000 paired
samples。static anchor 是四个 training states 等权 moment match；oracle 每个样本使用对应 exact
state。

| Regime | Static LER | Oracle LER | Static minus oracle |
| --- | ---: | ---: | ---: |
| quiet | 0.00814 | 0.00569 | 0.00245 |
| shifted | 0.06081 | 0.01961 | 0.04120 |
| correlated | 0.15079 | 0.00375 | 0.14704 |
| burst mixture | 0.08594 | 0.07136 | 0.01458 |
| aggregate | 0.076419 | 0.025103 | 0.051316 |

Aggregate paired 95% CI 为 `[0.050474,0.052157]`，static-only/oracle-only discordant failures
为 `18,075/1,654`。16/16 scenario-seed rows 的 CI 下界均为正；矩阵包含强相关与
`p_outlier=0.12, scale=2.5, burst_active=True`，不是单一 isotropic Gaussian demo。

该结果仍只是生成模型内 Bayes reference。quiet regime 中 oracle 与 standard 接近，burst
mixture 的 oracle LER 仍有 `~0.071`；“oracle”不意味着零错误或 channel-recovery optimum。

## 4. Protocol leakage flag 与成本包络

使用 T2.1.1 protocol-aligned stream 的 4 seeds、共 8,000 cycles：

| 指标 | 值 |
| --- | ---: |
| hidden leakage cycles | 1,616 |
| nonleakage cycles | 6,384 |
| flag sensitivity / specificity | 1.0 / 1.0 |
| nonleakage MAP errors | 1,272 |
| optimistic perfect-erasure lower bound | 0.159 |
| conservative leakage-as-failure rate | 0.361 |

两端差正好来自未知 leakage recovery cost。乐观端假定 flagged cycles 全部由外部免费恢复；保守端
把它们全计失败。当前任务不选择两端中的任何一个作为主 LER，也不把 perfect hidden flag 冒充
在线 leakage detector。T3.1.4/T5.2/T6 才能增加 protocol-aware action/cost/device evidence。

## 5. Machine artifact 与反简化验证

产物：

- `docs/t3_1_3_oracle_validation.json`；
- `docs/t3_1_3_oracle_source_data.csv`。

JSON/CSV 保存 16 个 regime rows、4 个 leakage rows、20 个 unique trace hashes、static training
hash、paired CI、flag/cost envelope、descriptor、schema gates、源码 hash 与 claim boundary。

反简化检查：

- full-state oracle 直接复用 T1.3.2 mixture likelihood，不用 scalar RMS adapter；
- `p_outlier=0/1`、相关 covariance、prior、loss proxy/separate 和 alias likelihood 的既有直接测试保留；
- 声明 `reference_anchor_method_id=full_state_model_oracle_map` 的 schemas 漏/重该 oracle 会失败；
- normal context bit-for-bit 对齐 `oracle_map_2d`；
- leakage context 的 `logical_class=None`、`logical_action=FLAG_LEAKAGE`、`map_result=None`；
- observed record 无 truth fields，mismatched regime/非法 kind/非法 syndrome 均负测；
- 10/10 machine gates，focused+adjacent `99 passed`。

T3.2.1 集成复核修正了旧 validator 的过宽假设：memory task 的 nondeployable reference 是
`full_episode_logical_truth_reference`，不是 full-state model oracle。本模块现只验证明确选择自身
reference ID 的 schemas；artifact schema 升级为 `t3.1.3-oracle-integration-v2`，gate 改名为
`full_state_oracle_present_in_declared_schemas`。

## 6. Claim 边界

允许：exact-state periodic-mixture oracle 是 nondeployable assumed-model upper reference；hidden
leakage 可以定义 perfect-erasure flag 的乐观/保守 cost envelope。

禁止：oracle 可部署；`oracle_delayed` 等同 decoder oracle；leakage 被免费完美纠正；该 oracle 是
finite-energy/protocol/channel-recovery/device optimum；CNN 可超过 oracle。
