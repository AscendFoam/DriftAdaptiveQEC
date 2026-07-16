# T2.0.2 sBs Kraus / error-space transition model

**日期：** 2026-07-14  
**实现：** `physics/sbs_error_space.py`  
**协议 ID：** `PROTO-SBS-MAIN`  
**证据等级：** protocol-aligned coarse-grained effective model；不是 Fock/pulse/device-calibrated sBs

## 1. 模型目标与边界

本模型实现 Sivak sBs flow diagram 的 error-space transition 层：支持 full-cycle `K_gg/K_ge/K_eg/K_ee`、`C_i` recovery hierarchy、逐级 trickle-down 和 deterministic Pauli-frame update。它是 T2.0.2 的通道层，不提前包含 T2.0.3 的 noisy readout/`|f>` reset/leakage，也不包含 T2.0.4 的实验 cycle timing。

模型保留每个 `C_i` 内的二维 logical density matrix，但主动去除不同 `C_i` 之间的相干。因此它不能替代文献中的 oscillator quadrature Kraus matrices、Fock-space 仿真、ECD pulse 或装置校准。

## 2. Grouped effective Kraus instrument

令 `b in {K_gg,K_ge,K_eg,K_ee}`，`T_b[j,i]` 是输入 `C_i`、输出 `C_j` 并产生 ideal branch `b` 的联合概率。矩阵约定为 `[target, source]`，并满足

\[
T_b[j,i]\ge 0,\qquad \sum_b\sum_j T_b[j,i]=1.
\]

每个非零 transition 构造一个 effective Kraus component：

\[
M_{b,j\leftarrow i}=\sqrt{T_b[j,i]}
|C_j\rangle\langle C_i|\otimes (X_L Z_L).
\]

同一 ideal outcome 下允许多个 component。这样，即使多个 error subspace 粗粒化后汇入同一目标，也不会用一个 many-to-one amplitude matrix 错误地产生非正交列。完整性为

\[
\sum_{b,j,i}M_{b,j\leftarrow i}^{\dagger}M_{b,j\leftarrow i}
=\sum_i\left(\sum_{b,j}T_b[j,i]\right)
|C_i\rangle\langle C_i|\otimes I_L=I.
\]

代码直接构造全部 component 并计算 `completeness_residual_norm()`；不是只检查 classical row/column sum。

## 3. Outcome 与 recovery-depth 约束

每个 `C_i` 有独立 `depth`；subspace index 不等于 depth，允许 `C1/C2` 同属 first error level。构造器强制：

| branch | 允许的 depth 变化 | 语义 |
| --- | ---: | --- |
| `K_gg` | 0，且保持同一 `C_i` | no recovery event；不等于“已在 code space” |
| `K_ge` | -1 | 一步 trickle-down |
| `K_eg` | -1 | 一步 trickle-down |
| `K_ee` | -2 | 两步 trickle-down |

任何向上跳、错误步数、`K_gg` 跨 subspace、负概率或固定 source 的总概率不为 1 都直接抛错。C0 的无误差极限被固定为 `P(K_gg|C0)=1`，其他三支为 0。

## 4. Pauli frame

文献 full X+Z cycle 的 deterministic logical effect 为 `X_L Z_L=-iY_L`。代码在每个 effective Kraus component 上保留 `X_L Z_L`，并将软件 frame 的 `x/z` 两个 GF(2) bit 同时翻转。结果同时返回：

- `unconditional_state`：包含物理 deterministic logical flip；
- `cycle_frame_corrected_state`：只撤销本 cycle 的 flip，不撤销 error-space population flow；
- `input_frame/output_frame`：两次 full cycle 后回到原 frame。

因此无误差 C0 中，未追踪 frame 的 logical state 会翻转，追踪后严格恢复原 density matrix；模型没有为了“identity test”偷偷删除文献中的 logical flip。

## 5. 显式 modeling assumptions

`make_trickle_down_chain` 提供一个可复现的 `C0...Cmax` recovery-depth chain。调用者必须显式给出：

- `one_step_probability`：`K_ge+K_eg` 总概率；
- `two_step_probability`：可行 depth 上 `K_ee` 概率；
- `ge_fraction`：一步事件在 `K_ge/K_eg` 间的分配。

这些值不是 Sivak 装置参数。builder 不提供“看似真实”的默认恢复率，也不裁剪非法概率。C0 强制 no-error；C1 不存在 two-step target，未使用的 two-step 概率不会被重分配给其他 branch。实际实验校准必须走 paper-parameter/calibration gate。

## 6. API 与产物

- `SBSErrorSpaceInstrument`：通用多 `C_i`/多同层 subspace CPTP instrument；
- `make_trickle_down_chain`：显式假设下的 chain builder；
- `apply_density_matrix`：所有 branch 的 subnormalized quantum state、概率、无条件态和 frame-corrected 态；
- `apply_population`：与 quantum path 同一 transition 的快速 population marginal；
- `sample_trajectory`：seeded ideal-Kraus trajectory，明确不是 noisy readout；
- `embed_logical_state`：把二维 logical density matrix 嵌入指定 `C_i`。

## 7. 验证与反 demo 审计

`tests/test_sbs_error_space.py` 覆盖：

1. 全 Hilbert space `sum M^dagger M=I`；
2. C0 no-error branch 与 Pauli-frame recovery；
3. 大误差不能一步投影到 C0；
4. `C1/C2` 同一 depth 的非链式 topology；
5. 随机复 density matrices 的 trace、PSD 与 population-path 一致性；
6. inter-subspace coherence 被显式去除；
7. seeded trajectory 的复现、单调 recovery depth 与 frame；
8. Monte Carlo branch frequency 与解析 transition 一致；
9. boundary probability 不裁剪/不重分配；
10. 非 TP、负概率、错误 depth jump、非物理 density matrix 全部 fail closed；
11. 代码 `protocol_id` 与 `docs/protocol_hierarchy.json` 主协议一致。

这比只写 `depth -= 1` 的 demo FSM 多验证了 quantum CP/TP、logical action、general topology 和负路径。但它仍是明确降级的 error-space instrument；Fock/coherent/device claims 必须等待 T2.3/T6 证据。
