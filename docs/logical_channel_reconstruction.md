# T5.3.1 六态 logical channel 重构

## 结论

T5.3.1 已在同一 finite-cutoff cavity model、同一正交化 GKP code basis、同一噪声参数和同一
`10 us` reporting interval 上，完成 nominal fixed-control sBs `qec_on` 与 matched idle `qec_off`
的六 Pauli eigenstate tomography。正式矩阵为：

- cutoff：`12/24/36/40`；
- noise profile：`high/medium/low`；
- intervention：`qec_off/qec_on`；
- 每条 lane：`30 cycles = 300 us`，逐周期保存 6 个 unnormalized `2x2` code-space outputs；
- 共 `24` 条 channel lanes、`17,266` 行 Source Data、`26/26` machine gates。

重构对象是线性的 CPTNI code subchannel

\[
\mathcal E_L(\rho_L)=V^\dagger\mathcal E(V\rho_LV^\dagger)V,
\]

而不是对 code-space survival 条件化后的非线性“post-selected channel”。因此 leakage 始终保留为
`1-Tr[E_L(rho)]`；conditional Bloch vector 只作诊断，不参与 PTM 或 lifetime 重构。

## 为什么不能复用旧结果改名

- `physics/logical_channel.py` 只由 parity confusion 产生 Pauli-twirled diagonal PTM，不能恢复 coherent/
  non-Pauli terms 或 code leakage。
- T4.4 teacher/student physical trajectories 使用不同的 learned-control horizon，未保存 matched 六态 channel
  output。
- T5.2 的 displacement、ancilla/readout、leakage/reset 结果是 component causal estimands，不能拼接成一个
  logical channel。

本任务只复用 T3.2.8 已验证的 finite-cutoff nominal sBs map 与 reference timing；三类异构结果在 artifact 中均为
`EXCLUDED`。

## PTM、non-Pauli 与 leakage 定义

六态按 `X+/X-/Y+/Y-/Z+/Z-` 输入。由正负态之和/差构造：

\[
\mathcal E(I)=\frac{1}{3}\sum_{a=X,Y,Z}
[\mathcal E(\rho_{a+})+\mathcal E(\rho_{a-})],\qquad
\mathcal E(\sigma_a)=\mathcal E(\rho_{a+})-\mathcal E(\rho_{a-}),
\]

\[
R_{\mu\nu}=\frac{1}{2}\mathrm{Tr}[\sigma_\mu\mathcal E(\sigma_\nu)].
\]

每个 cycle 同时报：

- 完整 `4x4` real PTM；
- Choi minimum eigenvalue 与 TNI survival-effect eigenvalues；
- 三个正负态 pair-sum linearity residual；
- Pauli block off-diagonal norm 与 antisymmetric coherent-rotation norm；
- non-unital code-flow、state-dependent survival；
- 六态 survival/leakage 与 conditional Bloch vector。

Pauli lifetime 不做单指数拟合。主数值是 raw code-weighted Pauli contrast 的 finite-horizon signed area，
分别以 cycle 和 `us` 报告；只有原始曲线真正越过 `e^-1` 时才线性插值给 crossing，否则保存
`right_censored`。所有 revival 和负值点均保留。

## 正式结果

### cutoff 40 终端参考

下表的 `A_X/A_Y/A_Z` 是 `0--30 cycles` 的 truncated signed-area lifetime；`L_30` 是 cycle 30 的
平均 code leakage。它们是本模型内数值，不是实验 lifetime 或 physical-memory LER。

| noise | lane | `A_X` | `A_Y` | `A_Z` | cycle-30 PTM diagonal `(I,X,Y,Z)` | `L_30` | off-diagonal norm |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| high | qec_off | 9.5241 | 8.1511 | 9.5241 | `(0.335307, 0.090726, 0.023977, 0.090727)` | 0.664693 | 0.022245 |
| high | qec_on | 18.0266 | 16.5393 | 18.0106 | `(0.651477, 0.517160, 0.413497, 0.516606)` | 0.348523 | 0.001706 |
| medium | qec_off | 14.3708 | 13.8319 | 14.3708 | `(0.318366, 0.241903, 0.183808, 0.241903)` | 0.681634 | 0.007156 |
| medium | qec_on | 21.7038 | 21.2078 | 21.6974 | `(0.739089, 0.694836, 0.656478, 0.694596)` | 0.260911 | 0.001903 |
| low | qec_off | 15.9258 | 15.6128 | 15.9258 | `(0.339004, 0.294448, 0.255765, 0.294448)` | 0.660996 | 0.004226 |
| low | qec_on | 22.4391 | 22.1105 | 22.4346 | `(0.758933, 0.728819, 0.703232, 0.728649)` | 0.241067 | 0.001905 |

`qec_off` 三轴均在 30-cycle horizon 内观察到 e-fold crossing；cutoff-40 `qec_on` 三轴均为
`right_censored`，不能填入虚构 crossing。formal high-cutoff area ratios 大于 1，但这仍不是
T5.3.3 的 simulated break-even 结论：本任务只重构 channel 与 lifetime curve，不冻结 operational boundary。

### cutoff 敏感度负结果

最初的 cutoff `12->16` 审计出现 active PTM 大幅变化，且 cutoff 12 的 QEC-on/off area ratios 仅约
`0.19--0.31`，会给出与高 cutoff 相反的性能方向。任务因此未用两点低 cutoff 扫描收口，而是扩展到
`12/24/36/40`。

正式 terminal `36->40` 结果为：

| noise | lane | final PTM Frobenius difference | final mean-leakage difference |
| --- | --- | ---: | ---: |
| high | qec_off | 0.000039 | 0.000029 |
| high | qec_on | 0.009690 | 0.000381 |
| medium | qec_off | 0.000049 | 0.000026 |
| medium | qec_on | 0.011078 | 0.000702 |
| low | qec_off | 0.000055 | 0.000027 |
| low | qec_on | 0.010497 | 0.000734 |

六条 terminal lanes 均通过预设数值 QA 容差 `PTM <=0.03`、`leakage <=0.02`。这只证明本次
36/40 finite-cutoff repeat 的稳定性，不是 infinite-cutoff convergence theorem；低 cutoff 反转继续保留在
Source Data 和 artifact 中。

## 物理性与防简化检查

- 最大六态 pair-sum linearity residual：`8.73e-14`；
- 最小 reconstructed Choi eigenvalue：`-1.71e-16`；
- full physical density 最大 trace error：`4.44e-16`；
- 最大 Hermiticity error：`0`；
- 最小 physical eigenvalue：`-7.07e-16`；
- QEC-on 每 cycle 计 `2` measurement、`2` reset、`18` active gates；QEC-off 三者均为 `0`；
- discarded/post-selected trajectories 均为 `0`；
- artifact validator 从 raw 六态 `2x2` outputs 全量重算 survival/leakage、conditional Bloch、PTM、Choi、
  lifetime、matched comparison 和 cutoff tables；PTM/lifetime/comparison mutation 均 fail closed。

通过：`33` direct/semantic-mutation tests；与 logical/Fock/SBS/wall-clock/cross-fidelity 相邻组合为
`196 passed`。

## 产物与边界

- 实现：`physics/fock_logical_channel.py`
- 编排：`cnn_fpga/benchmark/logical_channel_reconstruction.py`
- 测试：`tests/test_fock_logical_channel.py`、`tests/test_logical_channel_reconstruction.py`
- 机器产物：`docs/t5_3_1_logical_channel_reconstruction.json`
- Source Data：`docs/t5_3_1_logical_channel_source_data.csv`

允许表述：finite-cutoff matched-model CPTNI logical-channel reconstruction、non-Pauli/leakage diagnostics、
simulation-derived QEC-on/off Pauli-signal comparison。

禁止表述：实验 logical-channel tomography、multilevel leakage、physical-memory LER、beyond break-even、
device calibration、target-board/FPGA timing。`F_avg/F_e` 与不确定度留给 T5.3.2；operational boundary 留给
T5.3.3。
