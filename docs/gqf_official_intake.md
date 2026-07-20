# T6.8.3：Puviani GQF 官方源码与运行环境 intake

## 结论

官方仓库 `https://github.com/Matteo-Puviani/GQF.git` 已固定到 commit
`c9ab1ef2b3ff6fa6d6d24cd95fbd06e2872e016d`，12 个 tracked files 的内容树
SHA-256 为 `64f849db145f3ae1653105ff7488e7c2b5315bfc81c3947a4f8861be4e7943fd`。
`third_party/GQF` 的 tracked checkout 保持未修改；MIT license、远端 URL、commit、文件表和
原始 requirements 均进入机器报告。

本 task 的结论只到 **official-source intake + CPU minimum real smoke**，不是论文精确复现，
也不支持“超过 Puviani NMF”。

## 上游缺陷与隔离补丁

官方 commit 的 `GQF/mesolve.py:13` 存在 `IndentationError`，不能直接导入；此外还存在字符串
identity 比较、官方 runner 的 `RNN/rNN` token 不一致，以及 TEST evaluation zero-buffer shape
错误。四项修复以独立 patch series 保存到：

`third_party/patches/GQF/c9ab1ef2b3ff6fa6d6d24cd95fbd06e2872e016d/`

补丁只应用到 `runs/t6_8_3_gqf_worktrees/...-local-git-clone-v2` 派生 clone。补丁后 tracked-tree
SHA-256 为 `3c04bb74d1f8c627b211fe760757f44543df8d23042fda960039bdae6323217a`，且变更文件精确为：

- `GQF/GKP_environment.py`
- `GQF/feedback_GRAPE.py`
- `GQF/mesolve.py`
- `GQF/runner.py`

这一区分避免把项目 adapter 误写成未修改的官方实现。

## 环境锁

隔离环境 `GQFEnv` 的核心版本为 Python 3.9.18、TensorFlow 2.10.1、NumPy 1.23.5、Gym 0.26.2、
Matplotlib 3.7.5、CUDA Toolkit 11.2.2 和 cuDNN 8.1.0.77。完整 conda explicit、pip freeze 和
history lock 位于 `configs/gqf_official/`，每个文件均有 SHA-256 绑定。

官方 `requirements.txt` 只列出四个未固定顶层依赖，不能单独视为可重建环境。

## 真实 smoke 与失败边界

CPU smoke 使用 cutoff 8 的有限 Fock GKP 态，真实构造 `FeedbackGRAPE`/`GKPEnv` 并执行一个 sBs
环境 step，而不是 import-only smoke：

- GKP state norm：`1.0`
- density trace：`0.9999999148 - 1.78e-15 i`
- Hermitian residual：`7.16e-8`
- minimum eigenvalue：`0.0`
- syndrome probability：`0.2625993`
- reward：`0.4522216`

归一性、迹、Hermitian、半正定容差、概率范围、binary syndrome 和非终止分支全部通过。

隔离 GPU probe 在 RTX 4070 / TensorFlow 2.10.1 / CUDA 11.2 路径上发生
`cuSolverDN` 初始化致命错误，进程返回非零且未生成成功 artifact。因此 GPU 状态固定为
`UNQUALIFIED_CUSOLVER_FATAL`。这不能被 CPU smoke 或基础 GPU matmul 替代。

## 反简化与可复核性

- 12/12 intake gates 通过；
- 12/12 target-specific semantic mutations 被拒绝，逐门覆盖 commit、pristine tree、文件树、
  CUDA lock、原始语法缺陷、patch series、工作树隔离、真实环境 step、GPU 伪成功、exact 伪升级、
  绑定哈希和突变计数；
- `tests/test_gqf_official_intake.py` 与任务板治理测试合计 `13 passed`；
- `verify_report` 会重新计算所有门，并核对 live upstream tree、tracked status、每个 artifact hash 和
  patch-series hash。

## 允许与禁止的表述

允许：已锁定并审计 Puviani 官方 GQF 源码；补丁隔离；CPU 最小真实环境步通过；当前 GPU 路径未合格。

禁止：paper-exact reproduction、NMF lifetime reproduction、Route-A 超过 NMF、官方 GPU reproduction。
这些结论必须由 T6.8.4 及其后续门单独建立。

