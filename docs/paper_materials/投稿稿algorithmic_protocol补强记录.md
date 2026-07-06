# 投稿稿 algorithmic protocol 补强记录

日期：2026-07-03

对应稿件：`docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`

## 本轮修改

- 在 Method 的 `Overview` 与 `Affine fast path` 之间新增 `Algorithmic protocol` 小节。
- 将投稿稿方法描述从模块说明进一步收口为可复现的协议顺序：recent syndrome window -> candidate affine surface -> clipping / smoothing / fixed-point and saturation checks -> inactive-bank staging -> safe-boundary commit -> active affine fast path。
- 明确 teacher-anchored residual branch、statistical calibration rule 或 future estimator 的差异只体现在候选 \((K,b)\) 的生成方式；fast-loop contract 和 runtime logging fields 保持一致。
- 明确需要记录 active / staged \((K,b)\)、clipping / saturation status、commit acknowledgement、stale-parameter counters 和 residual-boundary events，以便未来与 board implementation 做 source-vs-board 对照。

## 边界

- 本轮不新增实验、不改结果数字、不运行 benchmark、不补硬件测量。
- 本轮不把当前软件协议写成真实 FPGA implementation、board commit latency 或 source-vs-board agreement。
- 本轮只提高 Method 的可复现叙述和审稿人可读性，不升级 `software-HIL`、`.tflite`、real-board、statistical 或 logical-channel evidence。

## 验证

- 需要重新扫描投稿稿中的内部治理词和过强部署表述。
- 需要重新运行 source-data / symbol-boundary 机械审计，确认新增方法文字没有破坏既有审计。
- 需要重新编译投稿稿，确认 LaTeX 仍可生成 PDF。
