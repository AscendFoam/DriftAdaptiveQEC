# T2.3.5 Feedback-GRAPE 随机轨迹梯度验证

**日期：** 2026-07-14  
**状态：** PASS  
**实现：** `physics/feedback_grape_gradient.py`  
**机器结果：** `docs/t2_3_5_feedback_grape_gradient_validation.json`

## 1. 验证对象

对 measurement trajectory `m`，一手论文给出的 Feedback-GRAPE 梯度为

\[
\frac{\partial \mathbb E[\mathcal R]}{\partial\theta}
=\mathbb E\!\left[\frac{\partial\mathcal R}{\partial\theta}\right]
+\mathbb E\!\left[\mathcal R
\frac{\partial\log P_\theta(\mathbf m)}{\partial\theta}\right].
\]

第一项是 conditional-state/reward path，第二项是 measurement trajectory probability
的 score path。只让 reward 反传、只检查 `grad is not None`，或把 sampling frequency
当常数而漏掉第二项，都会得到错误的 Feedback-GRAPE estimator。

## 2. 三层独立证据

### 2.1 完整 trajectory 穷举

在 cutoff 6、1 full cycle（2 次 measurement）的模型上穷举 `gg/ge/eg/ee`。使用共享
三参数 causal policy：static residual、latest-outcome response 与 history-mean response。

定义

\[
g_{exact}=\partial_\theta\sum_m P_mR_m,
\]

\[
g_R=\partial_\theta\sum_m\operatorname{stopgrad}(P_m)R_m,
\quad
g_S=\partial_\theta\sum_m\operatorname{stopgrad}(P_mR_m)\log P_m.
\]

结果：

- `sum_m P_m = 1.0`；
- `g_exact=[0.18949777,-0.07522291,0.00863952]`；
- `g_R=[0.11396006,-0.04754100,0.02017385]`；
- `g_S=[0.07553771,-0.02768191,-0.01153433]`；
- `max|g_exact-(g_R+g_S)|=5.55e-17`。

常数 baseline `b=E[R]` 不改变 exact score gradient，最大差 `1.11e-16`；同时
`E[d log P/dtheta]` 最大残差 `1.51e-16`，验证了 probability normalization identity。

### 2.2 分项 central finite difference

除总 expected-return 外，还分别冻结 base probability 或 base reward，独立差分：

\[
g_R^{FD}=\sum_mP_m(\theta)
\frac{R_m(\theta+h)-R_m(\theta-h)}{2h},
\]

\[
g_S^{FD}=\sum_mR_m(\theta)
\frac{P_m(\theta+h)-P_m(\theta-h)}{2h}.
\]

`h=1e-5` 时：

| 核对 | relative L2 error |
| --- | ---: |
| total autograd vs FD | `1.68e-10` |
| reward path vs independent FD | `2.44e-10` |
| score path vs independent FD | `3.22e-10` |

`h={3e-4,1e-4,3e-5,1e-5}` 的四点 sweep 全通过；最差 total/reward/score relative
error 分别约 `1.08e-7/2.19e-7/2.55e-7`，远低于预注册 `5e-5` 容差。这样排除了
总梯度中两项误差偶然相消的假通过。

### 2.3 真随机 trajectory estimator

生产验证使用 32 个独立 batch、每 batch 384 条随机 trajectory，共 12,288 条。每个
batch 独立计算 `g_R` 与带常数 baseline 的 `g_S`：

- sampled mean `=[0.19036077,-0.07595743,0.00940569]`；
- component z-score `=[1.120,-1.017,0.630]`，最大 `1.120 SE < 3.5`；
- ground-outcome fraction `0.6211`，并非单分支伪随机；
- score trace variance 从 `2.617e-4` 降到 `1.163e-5`，baseline ratio `0.04443`。

## 3. 非 demo 审计

- 两项梯度分别为非零并各有独立 finite difference，而非只核对总 loss；
- 1-cycle 四分支与 2-cycle 十六分支均检查 probability normalization 与 decomposition；
- baseline 的 unbiased identity 与 empirical variance reduction 分开检查；
- random estimator 用独立 seed/repeat 和 standard error，不用一次“看起来接近”的 batch；
- CPU exact/finite-difference 为 production 路线，CUDA exact parity 由 direct test 覆盖；
- 32 项 direct tests 通过。

## 4. 边界

本任务只证明 T2.3.4 finite-cutoff、two-level、literature-scenario simulator 中的梯度
计算正确。三参数 compact policy 是数值 audit probe，不是训练完成的 RNN teacher。
尚未证明：

- cutoff/batch/horizon 在 2--10 cycles 的可行资源 envelope（T2.3.6）；
- standard/MF/NMF 的方向性 ranking（T2.3.7）；
- optimizer convergence、RNN architecture、OOD generalization 或 teacher gain；
- pulse Hamiltonian、multilevel leakage/SPAM、device calibration 或硬件时序。

## 5. 复现

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m physics.feedback_grape_gradient --device cpu --batch-size 384 --repeats 32
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m pytest tests\test_feedback_grape_gradient.py -q
```
