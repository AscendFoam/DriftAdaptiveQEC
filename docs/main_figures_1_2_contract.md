# T7.1.2 主图 1--2 冻结合同

## Figure contract

- Backend：Python / matplotlib only。
- Archetype：两图均为 schematic-led composite；183 mm 双栏，白底、可编辑 SVG/PDF、600-dpi LZW-TIFF。
- Fig.1 核心结论：受支持的贡献是 observed-only host slow loop 通过 versioned trusted bank 接入 6-cycle/II=1 FPGA fast path 的预板合同；P&R 是 estimate，board measurement 仍为空。
- Fig.2 核心结论：安全由 typed event/action、freeze/switch/reset/LKG rollback 和 hysteresis 实现；IMM/BOCPD、posterior-mixture MAP 与 V5 risk compiler 明确标为 not run/dropped。

## Panel map

- Fig.1a：双回路 hero schematic；实线为逐轮 fast path，虚线为 host parameter update。
- Fig.1b：simulation → fixed-point → CXXRTL → P&R estimate → board-null 证据层。
- Fig.1c：6-cycle source-to-action、II=1、4000-cycle host cadence 与 board latency null。
- Fig.2a--c：observed-only 输入、四类证据分支与四类 typed action。
- Fig.2d：candidate → inactive bank → safe commit → LKG/hysteresis 事务链。
- Fig.2e：Dropped V5 与 blocked board 明示，不作为淡化脚注。

## Reviewer-risk checks

1. CNN/teacher/student 仅为 optional ablation sidecar，不驱动 fast action；HMM/posterior inference 位于软件慢回路。
2. `POST_ROUTE_ESTIMATE` 与 `BOARD_MEASURED` 颜色和标签不同；42 个 measured 字段不填零。
3. V5 early-stop 模块不使用已实现配色，也不连入 production arrows。
4. 所有节点/边均在 Source Data 中绑定 report/Source Data/code SHA-256 与 selector。

## Figure legends

**Fig. 1 | Evidence-bounded dual-loop Route-A contract.** a, Observed syndrome and integrity inputs feed a deterministic MAP-LUT/event fast path, while a software slow loop stages versioned parameter images through trusted A/B banks and last-known-good recovery. Solid and dashed arrows denote per-round and update-cadence paths, respectively. CNN/teacher/student modules remain optional ablations. b, Available evidence progresses from project-native simulation through fixed-point and CXXRTL qualification to a three-seed post-route estimate; physical-board measurement is blocked and all 42 measured fields remain null. c, The pre-board timing contract is six cycles with initiation interval one, whereas host updates occur every 4000 cycles. These values are not board-measured latency. Source data are provided as a Source Data file.

**Fig. 2 | Typed fail-closed adaptation and atomic parameter control.** a--c, Observed syndrome, health, integrity, version and age fields select typed normal/smooth, tail, leakage and integrity branches and their corresponding stage, trusted-bank, reset or last-known-good actions. d, Candidate images cross CRC/SHA/version checks before inactive-bank write and safe-boundary atomic commit; recovery requires hysteresis. e, IMM/BOCPD, posterior-mixture MAP, the V5 risk compiler and V5 implementation results were not run after the preregistered early stop, while board latency, jitter and power remain blocked. Source data are provided as a Source Data file.

Machine verdict: `PASS_MAIN_FIGURES_1_2_RESTRICTED_PREBOARD_CONTRACT`.
