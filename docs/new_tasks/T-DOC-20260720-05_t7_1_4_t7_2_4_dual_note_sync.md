# T-DOC-20260720-05：T7.1.4 与 T7.2.1--T7.2.4 双 note 同步

- Task ID：`T-DOC-20260720-05`
- 标题：Supplement S1--S5、正式正文合同与旧 note 保守同步
- 日期：2026-07-20
- 状态：Done

## 输入材料

- T7.1.4 提交 `5cc0335`：792-row Supplement figure contract、S1--S5、17/17 gates/mutations、13/13 live parent verifiers 与 210 项分环境回归。
- T7.2.1--T7.2.4 提交 `5444c78`、`0f7a18d`、`5e0bdd4`、`2d6ad1f`：Introduction/Related Work、Methods、Results、Discussion/Conclusion 四份正文合同。
- 两份 note、既有 PPT 矢量图、正式 Fig.1--4、Supplement S1--S5 和 Phase 6C atlas。

## 实际完成内容

1. 复核 Route-A 新 note 已按 non-ranking 合同装配 S1--S5，并呈现噪声转移有效域、Petz/SDP 与 top-$K$ 非部署边界、all-seed/OOD、fixed-point retention、失败/board-null 和 Phase 6C 独立 lane。
2. 复核新 note 已形成完整论文主干：6 段 Introduction、5 个机制分组 Related Work、11 个 Methods 小节加 3 个统计小节、10 个有序 Results 小节和 8 个 Discussion 小节；四份机器合同保持 live。
3. 旧 note 的日期和总证据表更新到 T7.2.4；新增 formal-paper overlay 表与新旧稿权威关系；Discussion/Conclusion 增补 4,000-cycle host cadence、59--96% tail intervention occupancy、single-mode effective-model 外推边界和六级物理升级门。
4. 保留旧 note 的 CNN/affine/teacher--student/FPGA 主体、历史数值和两张 PPT 图，不把旧结果重写成当前主排名。
5. 视觉 QA 发现新增段落被四张浮动图切断；加入显式 `\clearpage` 后重新编译，使历史图组与正式论文覆盖层各自连续。

## 当前可声称优势

- **受限 LER 对照**：V4 相对预注册 locked EWMA 为 `+2.14%`；只能用于该锁定对照，不能写成相对 static/Window 或全局最优。
- **独立 task-local 算法证据**：multimode posterior-weighted CPD 的 absolute LER gain 为 `0.064668 [0.064413, 0.064926]`，32/32 seed clusters 同向；matched two-GKP-CNOT analog-ML reduction 为 `31.160%--67.982%`。
- **学习模块扩展证据**：四状态/95 参数 student 在 cutoff 12/16 的十周期有效保真寿命为固定参数的 `2.36x/1.58x`，至少保留 `98.15%` teacher gain，参数量相对 72,853 缩小约 `767x`；它不是当前 LER winner 或板测结果。
- **确定性预板执行证据**：V4 integrated path 为 six-cycle、II=1，并完成 1,000,000 CXXRTL cycles 的 0 mismatch/undefined/silent-overflow；P&R 只支持 target-device feasibility，不支持 measured latency、jitter、power 或 fastest/SOTA。
- **表示保真而非性能优势**：selected fixed-point profile 的 quantized-minus-float LER 为 `3.05e-5`，区间跨零；dense profile 在注册 replay 中为 0。该结果只支持 retention。

## 必须保留的负结果与缺口

- Window 和 static joint MAP 在主 smooth benchmark 中强于 Route-A；calibration tail 中 static MAP 显著更好。
- 六类 abrupt/OOD worst-window 只建立 locked-EWMA non-inferiority，多类 tail 的 fallback/unnecessary-fallback 占用约 59--96%。
- V5 在 observed-only causal/action headroom 入口早停，20 个后续 task Dropped、0 个 V5 downstream output；不得由 Phase 6C 或 legacy CNN 营救。
- 没有 calibrated cavity/transmon、真实 logical lifetime/beyond-break-even、board-resident training、闭环 QPU 或实板 speed/power；42 个 measured-board fields 全为 null。

## 产物路径

- `docs/paper_notes/Contract_Centric_Regime_Aware_GKP_note_draft.tex`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- `docs/new_tasks/T-DOC-20260720-05_t7_1_4_t7_2_4_dual_note_sync.md`
- `README.md`
- `docs/paper_notes/README.md`
- `docs/new_risks.md`
- `docs/new_task_board.md`

## 验证方式和结果

- 新 note：TeX Live/XeLaTeX/BibTeX 编译 45 页；日志无 overfull、undefined reference/citation、LaTeX/package warning、float-too-large 或 fatal error；45/45 页渲染检查通过。
- 旧 note：TeX Live/XeLaTeX 编译 33 页；同类日志扫描为 clean；33/33 页渲染检查通过，并以原始分辨率复核新增表格页和浮动图边界。
- 合计 78/78 页视觉 QA；`git diff --check` 通过。

## 风险复核

- R-N005、R-N110、R-N142、R-N145 继续由 T6.9.2、T7.2.5--T7.4 与 Phase 8 承接。
- T7.1.4 的补充图只增加有效域、失败域和证据完整性，不新增系统级优越性。
- 本轮不插入新 task；插入 V5 rescue、伪 lifetime、伪板测或跨任务总榜会违反冻结合同。

## 对任务板的同步

- 在进度日志登记 `T-DOC-20260720-05 User request -> Done`。
- 不改变 `T7.2.5 In Progress` 的当前推荐状态或执行顺序。
