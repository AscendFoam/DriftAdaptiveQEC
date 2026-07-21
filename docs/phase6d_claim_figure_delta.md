# T7.1.5 Phase 6D 双 lane claim/figure delta

## Figure contract

- Backend：Python / matplotlib only。
- 历史策略：T7.1.1--T7.1.4 只读保留；本任务只新增 Figure 5--6，不覆盖、改色或重命名旧图。
- Bundle 边界：两图无跨 lane 箭头、无共同性能轴、无 LER--latency 加权总分。
- Figure 5 核心结论：Phase-6D v1 does not establish frozen-benchmark multimode LER SOTA because causal headroom over static-mixture exact MLD is zero; opened task-local LER, tail and compute values remain context only.
- Figure 6 核心结论：The exact single-mode converged RTL provides a deterministic six-cycle, II=1, atomic and fail-closed pre-board implementation contract, while physical-board latency, jitter, deadline and power remain unmeasured.

## Panel map

### Figure 5 — multimode software only

- a：strongest-baseline LER and no-go gate。
- b：opened task-local LER, tail and host-compute context。
- c：pilot, formal and scaling evidence-state boundary。
- d：CNN/student dropped ablation inset。

### Figure 6 — exact single-mode RTL only

- a：six-cycle II=1 deterministic pipeline。
- b：atomic A/B bank, LKG and fail-closed transaction。
- c：formal and million-cycle CXXRTL qualification。
- d：three-seed whole-harness Fmax estimate。
- e：post-route resource utilization。
- f：board-null and speed-claim boundary。

## Evidence hierarchy and review risks

- Figure 5 hero 是 strongest-baseline 零 headroom；opened LER/tail/compute 只作 context，未执行的 scaling/pilot/formal 不填零。
- Figure 6 hero 是 cycle/transaction contract；formal/CXXRTL 是 pre-board validation，P&R 是 whole-harness estimate，board 字段保持 null。
- CNN/student 只在 Figure 5d 以 dropped/ablation 状态出现，不进入任一 primary verdict。
- current RTL 不执行 multimode MLD；single-mode timing 不能替 Figure 5 补门，Figure 5 的 LER 也不能证明 Figure 6 的硬件实现。

## Figure legends

Fig. 5 | Multimode software evidence remains below the frozen-benchmark promotion gate. a, Train-only causal headroom compares static-mixture exact MLD with the proposed risk action over 79,872 rounds; both have pL=0.111979 and the paired relative point estimate and 95% lower confidence bound are 0%, so the v1 branch is NO-GO. b, The opened d=3 task-local study (9.6 million cycles, 32 seed clusters and 20,000 bootstrap resamples) is retained only as context for LER, non-overlapping 512-cycle worst-window/CVaR95 tail and measured host runtime; it does not use the strongest denominator. c, Pilot, formal and frozen-benchmark scaling results were not accessed after the headroom stop. d, CNN/student is absent from the primary result and appears only as a dropped ablation status. No RTL timing or hardware claim is inferred from this figure. Source data are provided as a Source Data file.

Fig. 6 | Exact single-mode deterministic and fail-closed pre-board RTL evidence. a, The converged fast path has a six-cycle source-to-action contract and initiation interval one. b, CRC/version admission, inactive-bank staging, safe-boundary atomic commit and last-known-good recovery implement the stated fail-closed transaction. c, Seventeen of seventeen formal gates and 21 of 21 formal mutants pass; an independent one-million-cycle CXXRTL run compares the full 148-byte public vector with zero mismatch. d,e, Three GW2AR seeds pass 27 MHz (minimum 36.794 MHz), with resource utilization reported for the whole-harness observability top; all critical paths end in the observability fold, so Fmax is not a bare-core result. f, Board latency, jitter, deadline misses, power and physical transfer/commit latency remain null. These data establish neither measured hardware performance nor a speed ranking, and the current RTL does not execute the multimode decoder. Source data are provided as a Source Data file.

## Revocation rule

任一 raw/config/code/source hash 漂移、任一父门失败、删除 multimode NO-GO、填充 board-null、把 learning 提升为 primary、写入 global score、复用旧图输出路径或删掉 whole-harness caveat，均撤销本 delta。

Machine verdict: `PASS_PHASE6D_CLAIM_FIGURE_DELTA_RTL_ONLY`。
