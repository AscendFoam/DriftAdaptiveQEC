# 投稿稿 GKP 坐标与 FPGA fast path 理论补强记录

日期：2026-07-06

## 对象

- `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`

## 修改内容

- 在 `Approximate GKP syndrome model` 中补充 \(\lambda=\sqrt{2\pi}\) 的软件 residual coordinate 解释。
- 明确标准 GKP 叙事中的 logical separation / stabilizer period 与本稿归一化 two-quadrature residual coordinate 的关系，避免把软件常数误读为新的物理 code definition。
- 在 `FPGA-facing datapath contract` 中补充 register-level fast-path 描述：四个 \(K_t\) fixed-point registers、两个 \(b_t\) registers、clip-threshold registers、active-bank selector、commit acknowledgement 和 overflow/saturation status flags。
- 明确 per-shot correction cycle 只执行 active-bank read、syndrome clipping、two-row affine product、bias add、quantization 和 status-bit emission；histogram update、CNN inference、posterior branch search 和 parameter fitting 留在 slow loop。

## 证据边界

- 本次修改只补强理论坐标约定和可实现 datapath 叙事。
- 不新增实验、不改数值、不升级 statistical inference、finite-energy logical-channel fidelity、hardware、real-board、`.tflite`、benchmark 或 deployment 证据。
- \(\lambda=\sqrt{2\pi}\) 只作为当前软件 residual model 的统一 wrap / logical-boundary / source-data 坐标约定，不等于完成 physical finite-energy device calibration。
- FPGA fast path 仍是 future implementation contract；没有新增 synthesis、timing closure、resource、power、DMA/MMIO 或 source-vs-board 证据。

## 验证

- 预期验证：重新编译投稿稿，扫描 LaTeX warning、内部项目语体、审稿元语言残留、硬件/统计过强主张和行尾空白。
