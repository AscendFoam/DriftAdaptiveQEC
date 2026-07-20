# Multimode GKP 强 baseline 与证据资格注册表

- **冻结日期：** 2026-07-21
- **对应任务：** T6.20.1
- **用途：** 为 Phase 6D 的 multimode 软件算法 lane 冻结最低可信 comparator、源码/公式来源、因果权限和排名资格
- **当前状态：** 文献与源码资格审计完成；除 T6.18.2 已完成的 CPD 复现外，本文不表示其余方法已经完成项目内复现

## 1. Phase 6D 主排名的 exact task signature

只有同时满足下列字段的方法才可进入同一 LER 主排名：

| 字段 | Phase 6D 主合同 |
| --- | --- |
| code family | surface-square multimode GKP，按距离 `d` 分层报告 |
| physical channel | Gaussian random displacement 的 stationary、smooth drift 与 abrupt/OOD 扩展；每个扩展保存完整生成参数 |
| observable | 当前轮 analog syndrome 与此前 syndrome history；不得读取 latent `theta`、scenario ID、future suffix 或 formal logical label |
| decision object | 当前轮 logical coset/action；不是最近物理位移、controller pulse 或 lifetime protocol |
| causal order | 用 `s_<t` 形成 prior，结合当前 `s_t` 解码，随后才以 `s_t` 更新，candidate bank 最早在 `t+1` 生效 |
| primary metric | per-round logical error rate `p_L`；按 `d`、`sigma`、scenario family 和 trajectory cluster 保留原始分母 |
| key secondary | worst-window/CVaR、adaptation lag、calibration、wall-clock、memory、deadline miss 与 fallback cost |
| comparison budget | 同 observation、decoder backend、update cadence、warm-up、precision、CPU/core、memory 与 online deadline；另设不参与 deployable 排名的 accuracy-ceiling 表 |

T6.18.3 的 `d=3` balanced heteroscedastic 结果已被访问，只能作为 development/headroom 依据。Phase 6D formal 必须使用新的 train/calibration/pilot/formal 四分割、不同 seeds 与未见 spatial pattern、variance law、transition rate/amplitude/duration，不能把旧数据改名为确认性 SOTA 证据。

## 2. 同任务 decoder backend：主排名最低集合

| 方法 | 一手来源 / 源码 | 与主合同关系 | Phase 6D 角色 | 资格标签 |
| --- | --- | --- | --- | --- |
| Closest integer / hard-decision MWPM | GKP 常规数字化；Fukui et al. 2017 中作为 analog 信息的对照 | 同 code/noise 的弱锚点，但丢弃 analog likelihood | 只作 sanity anchor，不得称 strongest baseline | `DIRECT_WEAK_ANCHOR` |
| Euclidean CPD / MWPM | [Lin, Chamberland & Noh, PRX Quantum 4, 040334 (2023)](https://arxiv.org/abs/2303.04702)；[官方仓库](https://github.com/amazon-science/LatticeAlgorithms.jl) | 与 surface-square GKP、Gaussian displacement 和当前 syndrome 直接匹配 | 主排名必选；消费 T6.18.2 已完成的官方复现 | `DIRECT_OFFICIAL_REPRODUCTION` |
| Nominal/estimated-metric weighted CPD | Lin 2023 的几何框架 + 项目异方差 adapter | 同 task，但 adapter 是项目实现，不是论文 exact reproduction | static 与 plug-in adaptive 几何基线 | `DIRECT_PROJECT_ADAPTER` |
| Periodic analog-likelihood MWPM | [Fukui et al., PRL 119, 180507 (2017)](https://arxiv.org/abs/1712.00294)；Lin 2023 的对照定义 | 必须使用 folded Gaussian even/odd coset likelihood；距离权重不能冒充 analog ML | 主排名必选，分别运行 static 与 adaptive frontend | `DIRECT_SOURCE_TRANSCRIBED` |
| Exact logical-coset MLD | [Lin & Noh, PRA 111, 052445 (2025)](https://arxiv.org/abs/2411.04277)；同一官方仓库 | surface-square GKP 直接适用；官方工作给出到 `d=39` 的 exact MLD | strongest decoder backend；`d=3` 必须与显式 coset sum 对拍 | `DIRECT_OFFICIAL_PENDING_REPRODUCTION` |
| `K` minimum-weight matchings (K-MWM) | [Lin, PRA 112, 042436 (2025)](https://arxiv.org/abs/2510.06531)；同一官方仓库 | surface-square GKP 直接适用；随 `K` 逼近 exact MLD | 主排名必选，报告 `K`—LER—时间—内存 Pareto | `DIRECT_OFFICIAL_PENDING_REPRODUCTION` |
| Static marginal/mixture exact MLD | exact MLD backend + 仅由 train/calibration syndrome 学得的冻结 prior | 同 task、可部署、比 Euclidean/static point estimate 强 | strongest frozen-static 候选，必须进入 deployable denominator | `DIRECT_PROJECT_NATIVE` |

本地官方源码固定为 `third_party/LatticeAlgorithms.jl@01f9bf1f6970b3e229b43aac9da3325c75518db8`，Apache-2.0。该快照同时包含 CPD、surface-square exact MLD 与 K-MWM 源码/示例。Phase 6D 不修改 upstream tree；任何 likelihood adapter 以独立 patch/module 保存并报告差异。

## 3. 因果 noise estimator / posterior provider：适配主排名

目前没有找到可不加修改就覆盖本项目完整 task signature 的“observed-only online drifting surface-square GKP decoder”。因此下表必须标成 `paper-inspired adapted baseline`，并统一接入相同 GKP likelihood 与相同 decoder backend；不得写成论文 exact reproduction。

| frontend | 一手来源 | 必须保留的原方法边界 | Phase 6D 最低实现 |
| --- | --- | --- | --- |
| frozen syndrome-estimated marginal/mixture | [Wagner et al. 2021](https://arxiv.org/abs/2010.02243) | syndrome statistics 对噪声参数的可识别性有条件，不能默认所有协方差均可识别 | train/calibration-only prior；formal 中完全冻结 |
| fixed sliding window / delayed overlapping window | [Bhardwaj et al. 2025](https://arxiv.org/abs/2511.09491) | 原 iterative/full-record 版本含非在线分析；目标时刻之后的数据不得倒灌 | 固定 `W`、单遍、prequential、最多用到 `s_<t` |
| EWMA / Kalman | 项目既有 baseline 与常规状态空间滤波 | 只能在统一 calibration 上冻结 `alpha/Q/R`，不能逐 family 调参 | 同 sufficient statistics、相同 update cadence 与 exact MLD backend |
| SMC-EAP / particle filter | [Kobori & Todo 2024/2025](https://arxiv.org/abs/2406.08981) | 原文是 surface-code syndrome + TN likelihood；换 GKP likelihood 后只能称 adapted | 固定 particle 数、ESS/重采样和 jitter；同时比较 EAP plug-in 与 posterior marginalization |
| causal Gaussian process | [Huo & Li 2017](https://arxiv.org/abs/1710.03636) | 适合 smooth/periodic；不得使用 full-sequence refit 或 correction-derived truth proxy | 只用 raw-syndrome statistic，冻结 kernel、inducing 数与 refit cadence |
| BOCPD | [Adams & MacKay 2007](https://arxiv.org/abs/0710.3742) | 是通用 online change-point 方法，不是 GKP decoder 论文 | 固定 hazard、run-length cap 和 posterior update；telegraph/step 的强 abrupt baseline |
| IMM/HMM | 多模型状态估计；QEC 中只作方法学先例 | state 数、transition matrix 和恢复阈值必须在 calibration 冻结 | 与 BOCPD 至少保留一个为主排名；另一个可作为消融 |

最低可信的 estimator × backend 主矩阵为：

1. static global-mixture exact MLD；
2. delayed-window plug-in exact MLD；
3. EWMA 与 Kalman plug-in exact MLD；
4. adapted SMC-EAP plug-in exact MLD；
5. BOCPD 或 IMM plug-in exact MLD；
6. proposed posterior-predictive exact MLD；
7. analog-MWPM 与 K-MWM 的 matched frontend 版本；
8. `true_theta + exact MLD` 只作不可部署上界。

## 4. 不同 task signature：只作分协议或边界参照

| 方法 | 一手来源 | 不进 direct 主排名的原因 | 正确实验位置 |
| --- | --- | --- | --- |
| MED / COR-MED | [Roy, Pousset & Royer 2025](https://arxiv.org/abs/2510.12677) | Steane-type QEC、noisy auxiliary、measurement back-action correlation；当前 Phase 6C 数据是直接位移 + ideal syndrome | 新建 noisy-auxiliary bridge；只有 task signature 对齐后在独立表排名。未发现作者官方代码，论文公式实现必须标 `SOURCE_TRANSCRIBED` |
| full-history dynamic analog 3D MWPM | [Noh, Chamberland & Brandão 2021/2022](https://arxiv.org/abs/2103.06994) | circuit-level repeated noisy measurement 与 space-time matching，不是单轮 ideal-syndrome surface-square benchmark | circuit-level/repeated-QEC 扩展，不和主表 raw LER 混排 |
| QLDPC-GKP soft decoding | [Borah et al. 2025](https://arxiv.org/abs/2505.06385) | outer code、check graph、circuit noise 与实时对象均不同 | code-family 边界/Related Work |
| Sivak-style decoder-prior optimization | [Sivak, Newman & Klimov 2024](https://arxiv.org/abs/2406.02700) | 使用实验逻辑结果离线调 prior，不是 formal-run syndrome-only 在线适应 | 同等 labeled pilot budget 的 frozen learned-prior 次级基线 |
| Puviani NMF / AQEC / FPGA QEC decoders | 既有 Phase 6A/6C registries | decision object、物理协议或 timing boundary 不同 | 独立 controller/lifetime/hardware lane，禁止填入 multimode LER denominator |

## 5. 因果合同与 posterior-predictive 定义

对第 `t` 轮，所有 deployable 方法遵守：

\[
q_t(\theta)=p(\theta_t\mid s_{<t}),\qquad
\hat L_t=\arg\max_L\int q_t(\theta)P(L,s_t\mid\theta)\,d\theta .
\]

输出 `\hat L_t` 后才允许用 `s_t` 更新 posterior，并最早在 `t+1` 通过 A/B bank 原子提交。若 baseline 只输出点估计 `\hat\theta_t`，则明确标成 plug-in：

\[
\hat L_t^{\mathrm{plugin}}=\arg\max_L P(L,s_t\mid\hat\theta_t).
\]

禁止把 `E[precision] -> weighted CPD/MWPM` 称为 posterior-predictive MLD。共享隐藏变量经边缘化后会诱导 mode 间相关性，正确实现必须在 logical-coset probability 层积分，而不是先分别边缘化坐标再相乘。

必须执行 future-suffix mutation、latent/scenario poisoning、formal-label denylist、round `t/t+1` bank-boundary 与 prefix-invariance 测试。RTS smoother、forward-backward/Viterbi full sequence、retrospective BOCPD、full-record FFT/GP 只能进 accuracy-ceiling 表。

## 6. Oracle 与参考项重新分层

| 名称 | 含义 | 排名 |
| --- | --- | --- |
| `true_metric_CPD_reference` | 知道真实当前 metric，但仍只找最可能物理位移/最近格点 | 不排名；不能再叫 decoder oracle |
| `static_mixture_exact_MLD` | 用冻结噪声分布对 logical coset probability 求和 | deployable 强 baseline |
| `causal_plugin_exact_MLD` | observed-only frontend 给点估计，exact MLD 作 backend | deployable 强 baseline |
| `causal_posterior_predictive_exact_MLD` | 对 observed-only posterior 在 coset probability 层积分 | proposed 主方法 |
| `true_theta_exact_MLD_oracle` | 当前真实 `theta_t` + exact logical-coset MLD | 不可部署上界，不参与 SOTA 排名 |

Exact MLD 优化的是逻辑陪集总概率，CPD 优化的是单个物理误差/最近格点。因此 `true_metric_CPD_reference` 仍可能被 exact MLD 击败，不能用它给 proposed 方法定义“已接近最优”。

## 7. 排名门与允许措辞

Phase 6D 只有同时满足以下条件，才允许写“在冻结 benchmark 上达到 SOTA”或更窄的等价措辞：

- 对每个 eligible strongest deployable baseline，aggregate relative-LER improvement 的 simultaneous paired 95% 下界均 `>10%`，且 absolute difference 下界 `>0`；
- calibration/telegraph 的预注册 tail endpoint（worst-window 或 CVaR）改善下界 `>0`；
- stationary degradation 的 95% 上界 `<=2%`，任一 OOD family degradation 上界 `<=5%`；
- `d=3` 与 `d=5`、多个 `sigma` 的方向一致，不能只选一个 crossing 或单一 family；
- exact MLD、K-MWM、analog-MWPM 与 strongest adapted estimator 已完成 source/implementation/budget 资格门；
- formal split 在任何结果访问前冻结，统计以 trajectory/transition block cluster 为单位，不把 round 当独立样本。

若缺少任一强 baseline，只能写“优于已实现 baselines”。若全部基线齐全但门未通过，则报告 frozen-benchmark negative/non-inferior 结果；不能通过删除 baseline、缩小 family 或转用旧 T6.18.3 数据恢复 SOTA 措辞。“世界范围/所有 multimode GKP decoder 的 universal SOTA”不在允许措辞内。

## 8. 对后续任务的映射

- T6.20.2—T6.20.4：冻结双 lane、全新 split 和 causal headroom；
- T6.21：复现 exact MLD/K-MWM/analog-MWPM，建立 static-mixture 与 noisy-auxiliary 边界；
- T6.22：实现 matched-budget adaptive baselines；
- T6.23：实现真正 posterior-predictive、risk-aware coset MLD；
- T6.24：运行新 benchmark 与 formal SOTA gate；
- T6.25：独立验证 single-mode RTL，不迁移 multimode LER 或 latency；
- T6.26：CNN/student 蒸馏与双 lane 论文证据汇合。

## 9. 非简化实现检查

本注册表没有把论文名称等同于可执行 comparator：

- exact MLD、K-MWM 有官方源码但仍需 stationary anchor、brute-force 对拍和 adapter 验证；
- analog-MWPM 与外部 adaptive 方法必须按公式适配，不能以距离权重、EWMA 别名或同名空壳代替；
- Roy noisy-auxiliary 与 Noh circuit-level 方法明确分离 task signature；
- 每个 frontend 都必须接同一 decoder backend，防止把“更强 estimator”与“更强 MLD head”混成一个贡献；
- 任何 reproduction 失败都保留 `BLOCKED/PARTIAL/NEGATIVE`，不允许寻找有利 substitute。
