# T3.2.9 有限时域 trajectory lookup control oracle

## 1. 角色与结论

本任务实现的是 `finite_horizon_control_oracle`：在固定 sBs gate ansatz、固定 finite-cutoff
assumed model 和两个 full cycles 内，为每个**已经观测到的 history prefix**分配独立的
15 维 bounded control residual，并用 exact branch enumeration 优化最终 state fidelity。
它不是知道真实 drift 的 decoder oracle，也不是允许任意 recovery channel 的 bound。

经过 3 restarts、每条 `300 + 250` epochs 的两阶段优化，cutoff 12 下：

| 策略 | 参数自由度 | expected fidelity | expected logical-Z | expected p(g) |
| --- | ---: | ---: | ---: | ---: |
| nominal standard | 固定 | 0.396787 | 0.380340 | 0.780881 |
| optimized time-indexed open-loop | 60 | 0.769403 | 0.750483 | 0.564769 |
| causal history lookup | 225 | 0.815799 | 0.802190 | 0.680229 |

lookup 相对 standard 的 fidelity 差为 `+0.419012`，相对已经优化的 open-loop 仍为
`+0.046396`。这证明当前模型/短 horizon 中 history-conditioned action 有额外数值空间；
不能证明 continuous non-convex ansatz 的全局最优，也不能外推为长时可部署策略。

## 2. 因果树，而非 hindsight table

每个 full cycle 有两个 g/e measurements。两 cycles 共 `H=4` 个 half-cycle decisions：

| 决策深度 j | 允许看到的输入 | 节点数 |
| ---: | --- | ---: |
| 0 | 空 history | 1 |
| 1 | `m0` | 2 |
| 2 | `m0,m1` | 4 |
| 3 | `m0,m1,m2` | 8 |

总 action nodes 为 `1+2+4+8=15`，每个节点 15 个参数，共 225 scalars；终端轨迹为
`2^4=16`。节点索引只由 `[0,j)` prefix 编码，同 prefix 的未来 suffix 无论如何变化，当前
action 都完全相同。若让节点读取当前或未来 outcome，那只能叫 hindsight trajectory bound，
不能进入本 control-oracle comparison。

## 3. 精确目标与嵌套对照

全部 `gggg` 到 `eeee` 分支同时进入 joint cavity--two-level-ancilla simulator，目标为

\[
J(\theta)=\sum_{\mathbf m}P_{\theta}(\mathbf m)
\mathcal F\!\left(\rho_{\theta}(T\mid\mathbf m),\rho_0\right).
\]

因此 autograd 同时包含 reward path 和早期 action 对后续 trajectory probability 的影响；
不是逐条 forced outcome 事后最大化再平均。16 个概率在 standard/open-loop/lookup、cutoff
12/16 六条 evaluation 中均归一到 1，所有 branch probability 为正，density trace/PSD gates
通过。

为排除“lookup gain 只是把 nominal controls 重新调了一遍”，另行优化 4×15 time-indexed
open-loop。其最佳表按每一深度复制到全部 prefix 后，与 lookup policy 的 objective 误差低于
`2e-10`；这个嵌入表作为 lookup 的一个正式 warm start，所以 lookup 至少保留嵌套子空间，
而不是靠随机 restart 碰巧超过它。

## 4. 优化、收敛与 checkpoint

- families：time-indexed open-loop、causal lookup；
- restarts：`101/211/307`，第一阶段 300 epochs，Adam learning rate `0.02`；
- refinement：每个 restart 都继续 250 epochs，learning rate `0.003`，再重新选模；
- selection：只按 cutoff-12 exact optimization-model fidelity；cutoff 16 不参与选择；
- precision/model：float64/complex128，cutoff 12，projector delta 0.34，high-noise timing；
- 总优化 wall time：open-loop phase/refinement `159.60/141.42 s`，lookup
  `166.32/137.79 s`。

选中 open-loop/lookup refinement 的最后 25 epoch 增益分别为 `1.58e-5` 与 `6.40e-5`，
均低于预设 `2e-4` gate。所有 3×2×2 个 phase/family runs 的每个 action node 都有非零
gradient coverage 并被实际改动。选中表没有靠撞 residual bound 获胜：open-loop/lookup 的
最大 `|tanh(raw)|` 分别约 0.783/0.880，均无 `>0.95` 项。

checkpoint 保存两阶段每个 restart 的 best table、seed、epoch、source/config hash；reload 后
cutoff 12/16 四个 selected fidelity 与 artifact 误差低于 `2e-10`。

## 5. cutoff transfer 与不可隐藏的代价

冻结 cutoff-12 checkpoint 后，不重训地在 cutoff 16 重放：

| 策略 | cutoff 12 fidelity | cutoff 16 fidelity |
| --- | ---: | ---: |
| standard | 0.396787 | 0.559221 |
| optimized open-loop | 0.769403 | 0.503415 |
| lookup | 0.815799 | 0.638688 |

lookup 在 cutoff 16 仍比 standard 高 `0.079467`，但 open-loop 排序反转，说明 continuous-control
结果具有明显 cutoff sensitivity。更重要的是 lookup 改写了 measurement distribution：cutoff
12 的最小/最大 terminal probability 为 `7.57e-5/0.6230`，standard 则为
`0.00474/0.4224`；lookup 的 p(g) 也比 standard 低 `0.10065`。因此 fidelity gain 伴随 branch
skew 和 ground-outcome burden，不能直接视为实验可取策略。

## 6. 指数资源增长

对 `C` 个 full cycles，`H=2C`，terminal branches 为 `2^H`，causal action nodes 为
`2^H-1`，action scalars 为 `15(2^H-1)`：

| full cycles | terminal branches | action nodes | action scalars | float64 table | exact terminal-state lower bound (cutoff 12) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 16 | 15 | 225 | 1.8 KB | 144 KiB |
| 5 | 1,024 | 1,023 | 15,345 | 120 KiB | 9 MiB |
| 8 | 65,536 | 65,535 | 983,025 | 7.5 MiB | 576 MiB |
| 10 | 1,048,576 | 1,048,575 | 15,728,625 | 120 MiB | 9 GiB |

9 GiB 只是 terminal density tensor 下界，不含每个 half-cycle 的 autograd intermediates、gates、
Kraus maps 和 optimizer state；实际训练成本更高。故两-cycle lookup 是离线短时 reference，
不是可扩展 controller architecture。

## 7. 证据与 claim 边界

- core：`physics/trajectory_lookup_control_oracle.py`；
- runner：`cnn_fpga/benchmark/trajectory_lookup_control_oracle.py`；
- machine artifact/checkpoint/Source Data：
  `docs/t3_2_9_trajectory_lookup_control_oracle.json/.pt/.csv`；
- tests：core 17 项，artifact 12 项；Source Data 共 3,418 rows；20/20 machine gates。

允许表述为“finite-horizon finite-cutoff assumed-model causal lookup control-policy empirical
reference”。禁止写成 globally certified ansatz optimum、decoder oracle、channel-recovery bound、
long-horizon deployable lookup controller、论文 Fig. S4 数值复现、multilevel/pulse/device 或
target-board 结果。

