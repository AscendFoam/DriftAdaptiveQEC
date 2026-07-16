# T2.3.6 可微 SBS teacher-training 资源可行性

**日期：** 2026-07-14  
**状态：** PASS；确认当前 host 的 2--10 full-cycle training-kernel envelope  
**实现：** `physics/differentiable_sbs_feasibility.py`  
**机器结果：** `docs/t2_3_6_differentiable_sbs_feasibility.json`  
**Source data：** `docs/t2_3_6_differentiable_sbs_feasibility.csv`

## 1. 结论与 claim 边界

在本机 NVIDIA GeForce RTX 4070 Laptop GPU、PyTorch
`2.8.0.dev20250405+cu128`、`float64/complex128` 下，cutoff 16、batch 16 的
2--10 full-cycle 点全部通过真实
`trajectory forward + reward/score backward + Adam update`。保守共同包络为：

> cutoff 16、batch 16、2--10 full cycles；每个 optimization step 的中位耗时
> `0.208--1.050 s`，CUDA peak allocation `83.2--303.8 MB`。

这只证明当前 finite-cutoff、two-level、literature-scenario simulator 的一次训练 kernel
可以执行。它**不证明** optimizer 收敛、seed 稳健性、NMF lifetime gain、物理 cutoff
收敛、pulse Hamiltonian、multilevel leakage/SPAM、device calibration 或 FPGA latency。

## 2. 不是 forward-only resource demo

每个计时点都实际执行以下链路：

1. GRU hidden size 10、两层 256-neuron MLP、15-output causal policy，共 72,913 参数；
2. policy 只读取此前 `g/e` history，输出当前 half-cycle 的 15 个 sBs residual controls；
3. joint cavity--two-level-ancilla density trajectory 随机采样完整 measurement history；
4. loss 同时包含 `E[dR/dtheta]` reward path 和
   `E[(R-b)d log P/dtheta]` score path，常数 baseline 为 `0.35`；
5. 执行 `Adam(lr=1e-4)` 的 `zero_grad -> backward -> step`；
6. 验证 gradient 与 parameter update 均有限、非零，并检查 trace、Hermiticity、PSD、
   trajectory probability；
7. 每点先 warm-up 1 次，再记录 3 次完整 optimization step 的 raw time、median 和 p90。

GPU 风险点各自在新 Python 子进程中执行。CUDA 使用
`max_memory_allocated/max_memory_reserved` 捕获包含 backward/Adam state 的峰值；CPU
用 2 ms sampler 记录进程 RSS 峰值。这样单点 OOM 不会污染后续 allocator，也不会把
初始化时间混入 steady-state step time。

## 3. 预注册网格与门

### 3.1 65 个实际点

- CUDA 56 点：
  - cutoff slice：`8/12/16/18/24/32/48`；
  - batch slice：`1/4/8/16/32/64/128/256/512`；
  - cutoff 16、batch `8/16` 覆盖每个 `2,3,...,10` cycles；
  - batch 32 覆盖 `2/4/6/8/10` cycles；
  - cutoff `18/24/32/48` 的 10-cycle high-cutoff anchors；
  - batch `512/576` 与 cutoff `48` 的隔离 frontier probes。
- CPU 9 点：cutoff、batch、horizon 三条代表性 RSS/runtime fallback slices。

### 3.2 通过标准

| 维度 | 门 |
| --- | --- |
| 数值 | objective/gradient/update finite；gradient norm `>1e-10` |
| density | trace/Hermiticity `<=2e-9`；最小本征值 `>=-2e-8` |
| runtime | median `<=10 s/step`；`<=2 s` 另标 preferred，不作为必过门 |
| memory | observed peak `<=75%` device total memory |
| envelope | cutoff 16 的每个 2--10 cycles 至少有 batch 8 可行，且有 cutoff `>=18` 的 batch `>=4`、10-cycle anchor |
| 反 demo | 多轴网格、三次 raw repeats、CPU RSS、CUDA allocated/reserved、真实资源拒绝点均存在 |

`10 s` 是本任务离线 teacher-kernel 的 bounded engineering gate，不是论文数字，也不是
`5 us` 物理 half-cycle 或 target-board deadline。

## 4. 结果

### 4.1 2--10 cycle 共同包络

cutoff 16、batch 16：

| full cycles | median step (s) | p90 (s) | CUDA peak allocated (MB) |
| ---: | ---: | ---: | ---: |
| 2 | 0.208 | 0.219 | 83.2 |
| 3 | 0.298 | 0.315 | 110.5 |
| 4 | 0.399 | 0.404 | 138.0 |
| 5 | 0.512 | 0.515 | 165.4 |
| 6 | 0.605 | 0.608 | 193.0 |
| 7 | 0.713 | 0.717 | 220.6 |
| 8 | 0.814 | 0.825 | 248.3 |
| 9 | 0.914 | 0.936 | 276.0 |
| 10 | 1.050 | 1.051 | 303.8 |

任务结论使用 batch 16，是因为只有 batch 8/16 对每个整数 horizon 都有实测；不能把
偶数 horizon 或 endpoint 的更大 batch 偷换成全区间上限。

### 4.2 实测资源 frontier

| 点 | 三次 step (s) | median/p90 (s) | peak allocated | 判定 |
| --- | --- | ---: | ---: | --- |
| cutoff 16, batch 512, 10 cycles | `8.624/8.624/8.617` | `8.624/8.624` | `6.233 GB`, `72.60%` | 最大已测可行 batch |
| cutoff 16, batch 576, 10 cycles | `9.551/9.514/9.501` | `9.514/9.544` | `7.015 GB`, `81.71%` | **memory exceeded** |
| cutoff 48, batch 8, 10 cycles | 三次均约 `7.5 s` | `7.480/7.706` | `3.129 GB`, `36.44%` | pass |
| cutoff 48, batch 16, 10 cycles | `13.954/13.929/13.985` | `13.954/13.979` | `4.040 GB`, `47.06%` | **runtime exceeded** |

这里的 “最大已测” 不是未扫描区域的数学上限。两个拒绝点表明 memory 与 runtime
frontier 均被实际触及，而不是从单点线性外推。

### 4.3 高 cutoff 与 CPU fallback

- cutoff 24、10 cycles 的 batch `4/8/16/32/64` 全通过，median 从 `1.156 s`
  增至 `6.788 s`，peak allocation 从 `0.422 GB` 增至 `2.051 GB`；
- cutoff 48、batch 8、10 cycles 仍在两条门内，说明结论不依赖 cutoff 8；
- CPU cutoff 8、batch 4 的 2/6/10-cycle median 为
  `0.232/0.453/0.764 s`，RSS increase 为 `121.8/141.7/165.3 MB`。

CPU/GPU 数字不可直接当作硬件优劣排名：它们是不同执行后端、kernel launch、线程库和
allocator 的当前 host 画像。

### 4.4 数值稳定性

65 点中 63 个资源门 pass、1 个 memory exceeded、1 个 runtime exceeded；资源拒绝点仍
数值稳定。全部点的：

- maximum trace error `6.66e-16`；
- maximum Hermiticity error `0`；
- minimum final eigenvalue `-9.37e-16`（float64 rounding scale）；
- minimum trajectory probability `2.52e-9`，没有 floor-clamped impossible branch；
- minimum gradient norm `0.3277`；
- minimum Adam parameter-update norm `0.0150`。

不存在 timeout、OOM、worker exception 或 numerical failure。

## 5. Figure contract 与图注

**核心结论：** 当前 host 支持 cutoff-16、batch-16 的 2--10-cycle Adam training
kernel，同时 memory/runtime 边界可被实测拒绝点定位。  
**archetype：** quantitative grid。  
**输出：** 183 mm 双栏，Python/matplotlib；SVG/PDF/TIFF/PNG，SVG/PDF 文字可编辑。  
**统计：** 每点 1 warm-up + 3 timed technical repeats；中心为 median，p90 和 raw repeats
保存在 source CSV；没有生物学重复或推断性显著性检验。

**Figure | Differentiable SBS teacher-kernel feasibility on the current host.**
**a,** cutoff 16 的已测 batch--horizon median step time；灰格为未扫描，不代表失败，红框
为资源拒绝点。**b,** 10-cycle runtime frontier；红虚线为 10 s engineering gate。
**c,** 含 backward/Adam state 的 CUDA peak allocation；红虚线为 75% VRAM gate。
**d,** cutoff 8、batch 4 的 CPU runtime/RSS fallback scaling。红叉只按对应 panel 的
runtime 或 memory gate 绘制。source data 是同名 CSV；图未做平滑、插值或缺失值填补。

## 6. 非 demo 审计与剩余风险

首版扫描的 43 点全部通过但只用到 5.75% VRAM，无法称为 envelope；因此扩大到 cutoff
48、batch 576，并实际找到两个 frontier。第二轮又发现仅 `backward()` 未执行 optimizer
update，遂加入真实 Adam state/update、非零 update norm gate，并废弃旧 scope 后完整重跑
65 点。最终实现没有以下简化：

- 不以 T2.3.4 的单点 resource counter 替代 scan；
- 不以 forward-only 或 `grad is not None` 替代 training step；
- 不用 float32 临时制造可行性；
- 不把未扫描灰格、OOM 解析估计或最大已测点写成绝对上限；
- 不把当前 host GPU timing 写成 5-us physical control 或 FPGA timing；
- 不把 resource feasibility 写成 RNN 已训练或 NMF 已优于 MF/standard。

R-N043 的 horizon/resource 子门已缓解，但 optimizer convergence、seed distribution 和
lifetime ranking 仍由 T2.3.7/T4.4 承接。无需插入新 task；下一项按任务板执行 T2.3.7。

## 7. 复现

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m physics.differentiable_sbs_feasibility --devices cuda cpu --warmup-steps 1 --repeats 3 --runtime-budget-seconds 10 --preferred-runtime-seconds 2 --output docs\t2_3_6_differentiable_sbs_feasibility.json
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m physics.plot_differentiable_sbs_feasibility
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m pytest tests\test_differentiable_sbs_feasibility.py -q --basetemp .pytest_tmp_t236
```
