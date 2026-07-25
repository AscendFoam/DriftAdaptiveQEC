# T9.2.3 Phase-9 独立 physics backend B 资格报告

- verdict：`PASS_T9_2_3_BACKEND_B_QUALIFIED`
- backend：`PHASE9-BACKEND-B-DENSE-STRANG-ANALYTIC-KRAUS-V1`
- analysis：`4e46a0c7bf88356c38874112bf26c7ddc0d60d989c024c539185b52b3bef80aa`
- gates：22/22；mutations：22/22

## 独立实现证据

- solver：dense `scipy.linalg.expm` midpoint Strang splitting，噪声采用显式解析 Kraus/Schur channel。
- RNG：BLAKE2b 地址化 + Python `random.Random` + 手写 Box–Muller；未使用 backend A 的 NumPy RNG stream。
- IQ likelihood 与 squeezed-comb logical projector 均在 `physics/phase9_backend_b.py` 内独立实现。
- 静态隔离审计：forbidden import []，forbidden token []。

## 物理与数值资格

- pure-loss closed-form mean error：`4.441e-16`；qutrit relaxation population error：`0.000e+00`。
- Strang 8→16 / 16→32 distance：`5.483e-04` / `1.365e-04`，ratio `0.249`。
- action-induced f-population：`0.515`；IQ Kraus、reset failure、action-conditioned drift 与六态投影均由真实状态演化检查覆盖。

## Claim 边界

本报告只证明 backend B 的实现与资格门通过，不证明 A/B 分布一致、LER、六态寿命、physical break-even、official Puviani exact/surpass、硬件实测、external SOTA 或 rank。
上述十项字段全部保持 `null`；A/B 统计对拍由 T9.2.4 执行。
