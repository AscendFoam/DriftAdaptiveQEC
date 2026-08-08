# Route-A 真板前位流候选审计

- 结论：`PASS_PREBOARD_CANDIDATE_NOT_PHYSICAL_QUALIFICATION`
- 证据边界：这是综合、P&R、打包和 CXXRTL 证据，不是真板测量。
- 后路由 Fmax：83.97 MHz（约束 27.00 MHz）。
- 资源：LUT 6532/20736，DFF 2969/15552，BSRAM 8/46。
- UART：候选 27 MHz / 3 Mbaud（9 clocks/bit）；完整栈以 3 clocks/bit 加速回归，实际比率 PHY 另行通过独立 CXXRTL 测试。
- 重复帧：返回显式 duplicate 状态，不重新执行、不消耗序号。

## 尚未满足

- 未识别实际板卡、revision、串口/JTAG 路径。
- 未烧录候选位流，未采集真实 source-to-action、deadline miss、功耗或长序列数据。
- 逐帧 UART 模式包含链路间隙，不能替代 T6.4 的满速百万周期 HIL。

机器可读 manifest：`docs/t6_9_2_preboard_bitstream_candidate.json`。
