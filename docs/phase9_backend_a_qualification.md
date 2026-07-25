# T9.2.2 Phase-9 高保真 physics backend A 资格报告

- verdict：`PASS_T9_2_2_BACKEND_A_QUALIFIED`
- backend：`PHASE9-BACKEND-A-JOINT-FOCK-QUTRIT-GKSL-V1`
- analysis：`95f11f0dc17a1799a97a69af6908765d1e00d75399ef3d07277d075ada74f624`
- parent T9.2.1：`78775658b4c9fa3a768252e2d529e39a0a90924c28158de2e4a527ba3052fe34`
- gates：20/20；mutations：20/20

## 实际模型

backend A 逐轮维护有限 Fock oscillator × `g/e/f` qutrit 的联合密度矩阵。恢复动作进入时间依赖 Hamiltonian；loss/dephasing/relaxation/excitation 进入 GKSL generator；Ramsey-like interaction 后由连续 IQ likelihood 构造对角 Kraus 并真实更新密度矩阵。reset 是 success/failure quantum instrument，失败支路保留 `e/f`，不是给标签加独立噪声。

动作还进入五维 latent drift 递推；共同 exogenous record 下，IDLE/X 的 joint-state trace distance=`0.125141`，drift L2=`0.00342637`。

## 物理性、极限与收敛

- Choi 最小特征值 `-2.51279e-19`，TP Frobenius `0`；
- full-round 最小特征值 `-5.91071e-17`，trace error `1.11022e-16`；
- ideal action distance `1.13596e-05`，zero-noise idle distance `0`；
- step 8→16 / 16→32 distance `0.000548302` / `0.000136498`，ratio `0.248946`；
- cutoff 8→12 distance `1.20991e-10`；
- large-reset `g` probability `1`，failed-reset `f` persistence `1`。

## 证据边界

本 PASS 只资格化 backend A 的实现。IQ 是 synthetic analog pre-frontend；T9.2.6 前不具备 ADC/Q-format/stream 资格。backend B、双后端对拍、codebook、LER、六态 lifetime、physical break-even、硬件/HIL、official Puviani、external SOTA 与 rank 全部保持 `null`。
