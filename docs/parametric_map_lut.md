# T4.2.1 version-bound parametric MAP-LUT

## 1. 任务结论

T4.2.1 已把逐周期 periodic-Gaussian axis MAP 冻结为“慢路径编译、快路径整数执行”的双层合同：

1. slow path 从真实 `ParamBank` 当前 active `K/b` 和 measurement covariance 反解有效 Gaussian
   mean/covariance，生成 X/Z 两张 hash/CRC/version-bound LLR ROM；
2. fast path 每 cycle 只接收 10-bit syndrome code、X/Z quadrature phase bit 和 latched active-bank
   version，执行地址拆分、两点 ROM read、整数线性插值和 LLR sign action；
3. 软件 pipeline contract 为固定 5 stages、worst-case 5 cycles、II=1，bank switch 发生时 in-flight
   request 继续使用提交时锁存的旧 image；
4. 当前结论仍是 software integer pipeline contract，不是 RTL synthesis、post-route timing、FPGA 或
   board/device measurement。

本任务只实现 X/Z phase-specific marginal MAP。虽然 slow path 保留并验证 full 2x2 covariance，但逐
phase ROM 只使用其 marginal sigma；不能宣称 full correlated 2D MAP optimality。

## 2. Active parameter bank 的可逆模型合同

`ParamMapper` 的有效模型为

\[
K=C(C+R)^{-1},\qquad b=\alpha(I-K)\mu,
\]

其中 `C` 是 decoder effective covariance，`R` 是 measurement covariance，`alpha` 是 bias contract。
编译器不直接相信 metadata 中可能过时的 prediction，而是从 active words 反解：

\[
\mu=(I-K)^{-1}b/\alpha,\qquad C=(I-K)^{-1}KR.
\]

随后重新计算 `C(C+R)^-1` 并与 active `K` 比较。缺少 `R/alpha`、`K` 非对称、`K` eigenvalue 不在
`(0,1)`、`R/C` 非正定或重构残差超限时一律拒绝编译。T4.1.3 以后由 `ParamMapper` 产生的 bank metadata
已补充 `alpha_bias`，从而不依赖隐式默认值。

## 3. LUT 与地址/插值合同

选定配置为 10-bit offset-binary ADC、8-bit ROM address、2-bit fraction、22-bit signed LLR
`Q9.12`。每个 phase 保存 256 intervals 加一个 guard node，共 257 entries。双 bank 的纯 table image
为 `2 banks x 2 phases x 257 x 22 = 22,616 bits`；这是精确表示大小，不是综合后的 BRAM/LUT 数。

ADC code `c` 表示 quantization bin center，而 ROM node 位于 coarse interval boundary。若
`c = address*2^F + f`，线性插值权重必须是

\[
w=\frac{2f+1}{2^{F+1}},
\]

不是 `f/2^F`。初版遗漏 `+1/2 bin` 会造成系统性 LLR code bias；已修复为 odd numerator 加
ties-to-even signed shift。在线 kernel 无浮点除法、`exp/log/sqrt/pow`；这些只允许出现在 slow compiler。

LLR 符号约定继续复用 T1.1.2：非负选择 even/`I`，负值选择 odd；phase 0 输出 `X` action label，
phase 1 输出 `Z` action label。这里只输出 logical-coset action，不提前实现 T4.2.2 的 event FSM、frame
accumulator、reset/fallback state。

## 4. 五级 pipeline

| stage | 固定功能 |
| --- | --- |
| S0 | latch input、active version 和 image identity |
| S1 | address/fraction 拆分与 X/Z phase select |
| S2 | synchronous-style `y0/y1` ROM read |
| S3 | integer linear interpolation、ties-to-even shift |
| S4 | LLR sign、`I/X/Z` action register |

每个 request 的 `valid_cycle=input_cycle+5`，连续 16 cycles 输入在 cycles 5--20 连续输出。测试在首个
request in-flight 时切换 image，旧请求仍返回旧 image/version，下一请求使用新 image/version。该行为是
可执行软件时序合同；目标器件 Fmax、BRAM inference、routing 和实测 deadline 仍为 `null`。

## 5. 穷举验证结果

8 个 registered banks 覆盖 zero/near-edge mean、q/p anisotropy、正负 correlation 和多档 measurement
covariance。每个 bank 在 1024 个 ADC codes 和 X/Z 两 phase 上穷举，共 16,384 行 Source Data：

| 指标 | 结果 |
| --- | ---: |
| hard action mismatch | `0 / 16,384` |
| mean absolute LLR code error | `0.387756` code |
| maximum absolute LLR code error | `20` codes（`0.004883` LLR） |
| nearest-ROM mean error | `89.353210` codes |
| interpolation / nearest mean-error ratio | `0.004340` |

address convergence 为：

| address bits | entries/phase（含 guard） | mean abs code error | max code error | action mismatch |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 33 | 15.910217 | 1,048 | 0 |
| 6 | 65 | 4.071472 | 316 | 0 |
| 7 | 129 | 1.114807 | 77 | 0 |
| 8 | 257 | 0.387756 | 20 | 0 |

20/20 production gates 还覆盖 K/b 反解、真实 ParamBank version sequence、phase table 非共享、image
CRC/SHA/source binding、stale version/bad code/bad phase/tamper fail-closed、5-cycle latency、II=1 和
non-measured resource boundary。

## 6. 反简化审计与边界

- 没有把旧 T2.4.3 单轴 LUT stress demo 直接改名；本实现连接真实 `DecoderRuntimeParams/ParamBank`，并
  逐 bank 反解/重构模型。
- 没有只测手选 syndrome；对全部 ADC codes、两个 phase、八个 bank 穷举，并扫描 5--8 address bits。
- 没有把 nearest lookup 冒充 interpolation；两者在相同 exact quantized reference 下单独计误差。
- 没有在 bank switch 时让 in-flight request 偷读新表；image/version 随 request 锁存。
- 没有把 analytic 5-cycle schedule 写成硬件 timing；LUT/FF/BRAM/DSP/Fmax/RTL/board 字段均为空。

允许表述：registered effective Gaussian parameter banks 上，存在 version-bound、整数、phase-specific
marginal periodic MAP-LUT software pipeline contract，并在完整 ADC 网格上保持 exact-quantized hard action。

禁止表述：full correlated 2D MAP optimality、event-FSM/frame/fallback 已集成、端到端 fixed-point LER、
RTL/synthesis/post-route timing、FPGA 或 board/device 结果。

## 7. 产物与复现

- `cnn_fpga/decoder/parametric_map_lut.py`
- `cnn_fpga/runtime/parametric_map_lut.py`
- `cnn_fpga/benchmark/parametric_map_lut_validation.py`
- `docs/t4_2_1_parametric_map_lut_validation.json`
- `docs/t4_2_1_parametric_map_lut_source_data.csv`
- `docs/t4_2_1_parametric_map_lut_bank_images.json`
- `tests/test_parametric_map_lut.py`
- `tests/test_parametric_map_lut_validation.py`

```powershell
python -m cnn_fpga.benchmark.parametric_map_lut_validation
```
