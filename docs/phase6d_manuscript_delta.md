# T7.2.6 Phase 6D 双 lane 论文正文 delta

- Machine verdict：`PASS_PHASE6D_DUAL_LANE_MANUSCRIPT_DELTA_RTL_ONLY`。
- 正文：`docs/paper_notes/Phase6D_Dual_Lane_GKP_manuscript.tex`。
- 门：27/27；语义 mutation：27/27。
- Source Data：57 rows，逐行 canonical JSON + SHA-256 可逆。

## 中心论点

multimode 软件 lane 的 v1 strongest-baseline headroom 为 0%，因此不建立 frozen-benchmark SOTA；exact single-mode RTL lane 独立建立 six-cycle/II=1、atomic、fail-closed 的 pre-board 贡献。二者只共享 contract bridge，不共享性能分母。

## 正文消费边界

- Abstract、Introduction、Methods、Results、Discussion、Limitations、Conclusion 与 Supplement delta 均由逐节 token contract 验证。
- T7.1.5 的 10 条 final claims 原样绑定；multimode negative、board-null、speed prohibition 与 learning dropped 不可删除。
- 旧 51 页稿和 T7.1.1--T7.2.5 保持只读历史快照，不再充当 current manuscript verdict。
- Figure 5 只承载 multimode；Figure 6 只承载 exact single-mode RTL；无 global LER--latency score。

## 关键数值

- multimode：strongest baseline 与 causal risk 均 `p_L=0.1119791667`，relative point/LCB=`0%/0%`，pilot/formal/scaling 未访问。
- RTL：17/17 formal gates、21/21 formal mutants、1,000,000 cycles、998,435/998,435 II=1 pairs、0 mismatch，三 seed 最低 Fmax 36.794 MHz。
- 物理板：latency/jitter/deadline/power/transfer/commit 均 null；不声称 fastest 或 SOTA latency。

## Revocation

任一父 hash 漂移、删 strongest baseline/0% no-go、填 board-null、把 CNN/student 升为 primary、声称 current RTL 执行 multimode MLD、添加跨 lane 总分或把 post-route estimate 写成 measured，均撤销本合同。
