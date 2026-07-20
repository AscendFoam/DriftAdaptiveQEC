# T7.1.3 主图 3--4 冻结合同

## Figure contract

- Backend：Python / matplotlib only；183 mm 双栏，白底、editable SVG/PDF、600-dpi LZW-TIFF。
- Fig.3 核心结论：Route-A 只在 pilot-locked EWMA aggregate 上有窄优势；Window 更低、static 对比和 oracle-gap closure 为负，abrupt/OOD 只建立安全/非劣且 fallback 代价高。
- Fig.4 核心结论：现有证据证明 fixed-point/CXXRTL 的百万周期确定性与三 seed open-source P&R estimate；不证明板上 latency/jitter/deadline/power 或速度优势。

## Panel map

- Fig.3a：七方法 smooth aggregate LER；oracle 明示 nondeployable。
- Fig.3b：EWMA 与 static 的 paired contrast，以及负的 static-to-oracle gap closure。
- Fig.3c：六 abrupt/OOD family 的 global worst-window counts，Route-A 与 locked EWMA 重合而非改善。
- Fig.3d：fallback rate 与 recovery lag 成本；Fig.3e 将 Phase 6C task-local positive 移至 Supplement。
- Fig.4a--d：million-cycle correctness、6-cycle/II=1 clock model、三 seed Fmax 与完整 profile resources。
- Fig.4e：42 个 board fields null；V5 quantized/formal/CXXRTL/P&R 为 not run/dropped。

## Reviewer-risk checks

1. `strongest deployable = Window` 与 `Route-A is not global best` 直接写在 Fig.3，不以 EWMA 预注册对比替代全方法排序。
2. tail plot 使用同轴 paired points；重合表示 safety/non-inferiority，不标 improvement。
3. host/software、CXXRTL、post-route estimate 与 board measured 分层；222.222 ns 只标 27-MHz clock conversion。
4. student profile 只作为 optional sidecar resource context；不驱动 fast action。

## Figure legends

**Fig. 3 | Restricted V4 performance and safety results.** Smooth results use 24 formal seed clusters and equal family weights. Route-A improves only over the pilot-locked EWMA aggregate; Window remains the strongest deployable method, the paired static contrast is negative, and static-to-oracle gap closure is below zero. Across six abrupt/OOD families, worst-window outcomes establish safety/non-inferiority rather than improvement and require substantial fallback. Phase 6C task-local results are excluded from the main ranking. Source data are provided.

**Fig. 4 | Pre-board deterministic execution evidence and physical-measurement boundary.** The integer/CXXRTL path completes 1,000,000 cycles without mismatch, undefined action or silent overflow. The integrated selected profile has a six-cycle, II=1 clock-model path and three-seed open-source P&R estimates; the student is an optional sidecar. All 42 physical-board fields remain null, and V5 hardware stages were not run. Source data are provided.
