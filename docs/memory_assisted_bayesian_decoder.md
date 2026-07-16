# T3.2.1 memory-assisted periodic Bayesian decoder

## 1. 结论

本任务实现了一个 **observed-only、有限 episode、二维相关、周期状态空间** 的 memory-assisted
Bayesian decoder。它消费 20 个连续 cycle 的 `residual_q/residual_p`，在每轮执行一次 prediction
和 likelihood update，并在 episode 末端输出一次四逻辑陪集 posterior/decision。在线 API 不接受
累计位移、lattice index、logical truth、hidden drift state 或 simulator regime。

生产验证覆盖 4 个冻结场景、8 个独立 evaluation seeds、每 seed 128 个 episode、每 episode
20 cycle，共 4,096 episodes / 81,920 cycles。与使用相同 H-step prior、同一末次观测、但丢弃前
19 个观测的 final-outcome static Bayesian comparator 相比，memory decoder 的 aggregate logical
error rate 降低 `0.303467`，seed-cluster Student-t 95% CI 为
`[0.291727, 0.315207]`；四个场景的 CI 下界均为正，且 NLL/Brier 均改善。

这是 syndrome-level synthetic evidence，不是 Wan 等论文有限能 Glancy–Knill 电路 fidelity
复现，不是在线参数辨识，不是 FPGA synthesis，也不是普适 history gain 结论。

## 2. 主来源与迁移边界

主来源为 Wan, Neville and Kolthammer, *Memory-assisted decoder for approximate
Gottesman-Kitaev-Preskill codes*, PRR 2, 043280 (2020)，[arXiv:1912.00829v3](https://arxiv.org/abs/1912.00829)。
本任务直接核对公开 v3 TeX/PDF 的 Eq. (7)、(9)、(12)--(15)、Fig. 2 与 tracking/truncation
appendices。task-scoped 双语证据包见
`docs/paper_readers/wan_memory_assisted_2020/`。

只迁移以下机制：

1. 多轮 syndrome history 联合形成 posterior；
2. 不在每轮执行中间纠正；
3. 在固定 episode 末端做一次 decision/correction；
4. 先验过程/观测噪声必须冻结并显式登记。

不迁移以下对象：

- finite-energy GKP comb wavefunction 和 Glancy–Knill q/p-SE circuit；
- 论文在 `sigma_0, Delta << sqrt(pi)` 下的 Gaussian/Laplace 闭式近似；
- 论文 density-matrix qubit fidelity 数值；
- offline squeezing 电路、装置 calibration 或硬件可实现性结论。

因此，当前实现称为 `periodic_memory_assisted_bayes`，而不是 “Wan decoder reproduction”。

## 3. 公平 history / observation contract

| 字段 | 冻结值 |
| --- | --- |
| history | 20 consecutive cycles |
| 每 cycle 消费 | `residual_q`, `residual_p` |
| episode 起点 | known zero logical-torus origin |
| 中间 action | 无 |
| action 时刻 | episode 末端一次 logical-coset decision |
| hidden truth inputs | 空 |
| production grid | `128 x 128` |
| reference grid | `256 x 256` |

`decode_observed_episode` 只接受 `ObservedSyndromeStep`，并检查 cycle 连续、`valid=True`、analog
syndrome 确实 wrap 到 residual syndrome；传入 full truth step、乱序、缺 cycle 或不一致 residual
均 fail closed。这个 budget 是后续 proposed controller 必须共享的合同；若 T4/T5 改变 history
或可见字段，必须重新运行本比较，不能沿用当前 gain。

T3.2.1 comparison registry 的四个角色为：

| 方法 | 角色 |
| --- | --- |
| `standard_binning` | 固定 no-memory anchor；当前 modular observation 下等价于中央陪集 |
| `final_outcome_static_periodic_bayes` | task-specific static anchor；同一 H-step prior，只看最后观测 |
| `periodic_memory_assisted_bayes` | deployable-algorithm candidate；消费全部 causal history |
| `full_episode_logical_truth_reference` | 只用于计分的 nondeployable reference |

注册表同时显式声明 task-specific static/reference anchors。T3.1.2 的
`static_training_average_map` 和 T3.1.3 的 `full_state_model_oracle_map` 只验证明确选择它们的
comparison，不再错误地强迫所有新 decoder task 采用同一语义的 static/oracle。

## 4. 数学实现

令 lattice spacing 为 `L=sqrt(2*pi)`，hidden cumulative displacement 的状态空间是
`[-L,L) x [-L,L)` 的 `2L x 2L` logical torus。四个半格 parity region 对应四个 logical cosets。

每轮 prediction 为二维 wrapped correlated-Gaussian circular convolution：

\[
p_t^-(x)=\sum_{x'} K_Q(x-x';\mu_Q,\Sigma_Q)p_{t-1}(x').
\]

实现用一次预计算 transition-kernel FFT，以及每 cycle 一次 forward/inverse 2D FFT；
`Sigma_Q` 的 off-diagonal correlation 被保留，没有拆成独立 q/p decoder。

观测 `y_t` 是 modulo-L residual syndrome，update 为：

\[
p_t(x)\propto p_t^-(x)K_R(y_t-x;0,\Sigma_R),
\]

其中 likelihood 按 period `L` wrap，所以相差一个 lattice cell 的 hidden states 具有相同 modular
observation likelihood，但属于不同 logical parity。状态网格使用 cell centre；known-zero origin 位于
四个 cells 的交点，因此初始 mass 对四邻 cell 各分 `1/4`，避免边界节点给某一 parity 人为加权。

运行时把观测量化到同一 grid step，随后用预计算 wrapped-Gaussian template 的 cyclic shift 做
`N^2` table lookup。最大输入量化误差为半个 grid step，并用 `128` 对 `256` reference grid 审计，
不是把连续 likelihood 重算成本隐藏在“每 cycle 二次型”口径里。

末端 logical posterior 是四个 parity region 的 posterior mass；decision 取最大 mass。static
comparator 从同一 zero prior 传播精确的离散 H-step transition spectrum，只在末端做一次相同
likelihood update，因而 history 是两者唯一的观测信息差异。

## 5. 生产结果

| 场景 | Standard LER | Final-static LER | Memory LER | static-memory gain (Student-t 95% CI) | Memory / static NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| quiet isotropic | 0.58496 | 0.58496 | 0.13379 | 0.45117 `[0.43776,0.46458]` | 0.3303 / 1.2913 |
| measurement limited | 0.68164 | 0.68164 | 0.59473 | 0.08691 `[0.05679,0.11704]` | 1.2694 / 1.3703 |
| correlated | 0.66602 | 0.67188 | 0.36328 | 0.30859 `[0.27421,0.34298]` | 0.8107 / 1.3617 |
| biased correlated | 0.79102 | 0.64453 | 0.27734 | 0.36719 `[0.33596,0.39841]` | 0.6647 / 1.3484 |

统计单位是 8 个 evaluation seed clusters，不把同 seed 内 128 episodes 当作独立重复。95% CI
使用 two-sided Student-t、`df=7`；首版的 normal critical value 已在深查中被替换。每个场景
Memory Brier 也小于 static Brier，防止只用 argmax error 掩盖 posterior 失准。

网格收敛的最坏结果：

- mean logical-posterior TV：`0.01805`；
- logical error-rate absolute delta：`0.015625`；
- pointwise decision disagreement：`0.046875`。

最后一项可由 near-tie posterior 放大，因此 gate 同时约束 TV 与 error-rate delta，decision
disagreement 保留报告但不单独冒充连续误差界。四场景 32 个 trace SHA-256 全部唯一。

## 6. 成本口径

production `128 x 128` grid 的 deterministic software/mapping proxy 为：

| 项目 | 值 |
| --- | ---: |
| logical-torus cells | 16,384 |
| 20-cycle raw residual history @24 bit | 960 bit |
| posterior storage @24 bit | 393,216 bit |
| transition kernel storage @24 bit | 393,216 bit |
| FFT grid points / cycle | 32,768 |
| complex-butterfly proxy / cycle | 458,752 |
| transition-kernel quadratic forms（一次性） | 147,456 |
| likelihood-template quadratic forms（一次性） | 147,456 |
| observation quantizations / cycle | 2 |
| likelihood table lookups / cycle | 16,384 |

LUT/FF/BRAM/DSP/Fmax 均为 `null`，`target_measured=False`。这一路径明显不是当前低成本 FPGA fast
path 候选；它首先是 history-aware Bayesian benchmark。后续 T5.5/T6 若没有具名 RTL、tool、part
与 synthesis/P&R 报告，不能把以上 operation/storage proxy 改写为硬件资源或 latency。

## 7. 反简化审计与修复

1. 首个小网格扫描有两个 gate 失败；没有降低阈值，而是查出网格边界节点使 parity argmax 不稳，
   改用 cell-centred grid 和 symmetric zero-prior split。
2. 初版每 cycle 重算所有 Gaussian exponentials，`64/96` 扫描超时；改为同网格观测量化 +
   cyclic template shift，并以 finer-grid convergence 审计误差。
3. 初版 CI 用 normal critical value；8 clusters 下改为 Student-t，并保留 df/method 字段。
4. 初版 cost 把一次性 template quadratic forms 误报成每 cycle；现拆为 one-time precompute、
   per-cycle quantization 与 table lookup。
5. registry 集成测试发现 static/oracle validators 把 task-specific comparator 错绑到 T3.1.2/
   T3.1.3；现增加显式 static/reference role anchors 和负向 mutation tests，并刷新 T3.1.1--T3.1.5
   source-bound artifacts。
6. 已覆盖 FFT-vs-explicit convolution、template-vs-direct wrapped Gaussian、one-cycle static/memory
   identity、early-history counterfactual、axis swap、correlation non-factorization、batch equivalence、
   observed-stream adapter、输入 fail-close、CSV 重算、artifact hash 与 claim boundary。

最终验证：focused `18 passed`；受 registry 影响的 8-file adjacent suite `167 passed`；显式
`tests/` 全量 `916 passed, 4 skipped, 4 failed`。4 个失败仅为已登记 R-N012 的两份缺失历史
FR8/P4 文档，没有本任务新增回归。`compileall`、reader asset/source-map audit、governance state
check 与 `git diff --check` 均通过；后者只有 Windows LF→CRLF 提示。

## 8. 产物与 claim boundary

- 实现：`cnn_fpga/benchmark/memory_assisted_bayesian_decoder.py`
- 测试：`tests/test_memory_assisted_bayesian_decoder.py`
- 机器证据：`docs/t3_2_1_memory_bayesian_validation.json`
- Source Data：`docs/t3_2_1_memory_bayesian_source_data.csv`
- 主来源阅读包：`docs/paper_readers/wan_memory_assisted_2020/`

允许写：在已注册的 bounded correlated-Gaussian modular-syndrome episodes 内，observed-only
periodic Bayesian filter 使用全部 causal history，并在同 trace/same prior 公平对照下优于
final-outcome static Bayesian decoder。

禁止写：精确复现 Wan finite-energy circuit fidelity、history 在任意模型中普遍有益、在线
device calibration、可部署 low-cost FPGA decoder、synthesis/实板 latency 或真实量子硬件增益。
