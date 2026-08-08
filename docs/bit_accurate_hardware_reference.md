# T5.5.1 Python bit-accurate hardware reference

## 结论

T5.5.1 已冻结可作为后续 RTL 对拍基准的 Python golden model，正式验证 `16/16` gates。它使用固定 packed
words、真实逐周期 pipeline、整数 MAP、health/event FSM、A/B parameter bank 和 hash-chained trace；不再用
`cycle-5` 构造好的 MAP decision 直接调用 FSM。

verdict 为：

`BIT_ACCURATE_PYTHON_RTL_GOLDEN_FROZEN_HARDWARE_UNMEASURED`

这不是 RTL、综合、时序收敛、布局布线、transport 或板测结果。

## 冻结的 word contract

所有字段从 bit 0 起连续排布；CRC 对 little-endian payload bytes 计算，最高不足一 byte 的 bits 补零。

| word | payload | CRC | total | 关键字段 |
| --- | ---: | ---: | ---: | --- |
| fast input | 42 bit | CRC16 | 58 bit | 10-bit syndrome、2+2-bit observed class、phase、8-bit OOD、16-bit age、reset/valid/deadline |
| registered output | 102 bit | CRC16 | 118 bit | valid/mode/action、Pauli/phase frame、6 counters、health/fault mask、version、22-bit LLR |
| post-cycle state trace | 216 bit | CRC16 | 232 bit | event state、frame、version、health state、14 个逐 fault 饱和 counters |

CRC 使用 CRC-16/CCITT-FALSE：poly `0x1021`、init `0xffff`、xorout `0`、无反射；标准向量
`123456789 -> 0x29b1`。正式审计对代表性 58-bit input 的每一位做 single-bit flip，58/58 均检测。
reserved observation code 即使 CRC 正确也会把 observation 标为 invalid。

在线 `step_word` 只接收 integer word 和内部 safe-boundary 信号；没有 float syndrome、truth、hidden state、
drift state、division、`exp/log/sqrt`。ADC/IQ 到 10-bit syndrome code 的设备转换仍在本任务边界之外。

## 时序与 FSM

每个 hardware cycle 的顺序固定为：

1. 发布上一个 cycle 注册的 output；
2. 在 safe boundary 尝试 atomic A/B image commit；
3. 在 MAP S0 锁存本 cycle input、active image、version 和 integrity；
4. 推进 5-stage MAP pipeline；
5. matured decision 与同 source metadata 一起进入 health/event FSM；
6. action 在下一 cycle 发布。

因此 source-to-output latency 恰为 6 cycles、II=1。正式 trace 输入 4,110 个连续 words，再 drain 6 cycles；
4,110 个 outputs 一一对应 source 0--4109，无重复/丢失，latency set 只有 `{6}`。

fault corpus 包含 bad input CRC、observation invalid、deadline miss、OOD、persistent leakage/reset ack 和连续
e-run。对应 fault mask 分别非零，leakage 触发 reset，ack 后清除；3-bit e-run 饱和在 7，不回绕。

## Binary parameter image 与 A/B bank

8 个 T4.2.1 frozen images 均编码为固定 binary image：

- 128-byte little-endian header；
- 两个 257-entry phase tables；
- logical LLR 为 signed 22 bit，当前 binary/BRAM container 为 signed 24 bit；
- CRC32 + SHA256 trailer；
- 每 image 1,706 bytes；8-image bundle 13,724 bytes。

8/8 image 与 bundle byte-exact roundtrip；对 v0 的全部 1,706 个 proper prefixes 和 1,706 个逐 byte
single-bit corruptions 均 fail closed。binary codec 验证后复用 T4.3.2 的 manifest/CRC/SHA/CAS/hysteresis/
residency/safe-boundary A/B transaction。

正式 trace 在 cycle 4000 因 unsafe boundary defer，在 4001 commit v1。source 4000 已在 S0 锁存 v0，
output 仍为 v0；source 4001 锁存 v1。in-flight request 不受 bank switch 污染，active version 单调不回退。

binary header 中 model mean/sigma 仅用于 image provenance/roundtrip，online runtime 不做 float 运算。24-bit
container 和 bundle bytes 是表示合同，不是 BRAM packing、LUT/FF/DSP 或利用率实测。

## 验证证据

- 8 images×2 phases×1,024 codes = 16,384 rows；runtime LLR/action 与独立整数 ties-to-even 重构 0 mismatch；
- 4,116-row golden trace，output/state CRC 全通过，final chain
  `2ef0aede273600351e461459821392f635780452101c39dbfef3f69fd75e68f1`；
- full reference 独立重建两次，staging、4,116 trace rows 和 final chain 全一致；
- 16,503-row Source Data；
- 25 focused tests；T4.2/T4.3 core adjacent `157 passed`；closed-loop adjacent `19 passed`；
- 10 类 semantic mutations 覆盖宽度、binary codec、MAP parity、latency、commit/in-flight、fault、repeatability、
  file binding 与 hardware claim，全部 fail closed。

## 产物与复现

- `cnn_fpga/runtime/bit_accurate_hardware_reference.py`
- `cnn_fpga/benchmark/bit_accurate_hardware_reference.py`
- `tests/test_bit_accurate_hardware_reference.py`
- `docs/t5_5_1_bit_accurate_hardware_reference.json`
- `docs/t5_5_1_bit_accurate_hardware_reference_source_data.csv`
- `docs/t5_5_1_bit_accurate_golden_trace.csv`
- `docs/t5_5_1_bit_accurate_parameter_bank.bin`

```powershell
$env:PYTHONPATH='.'
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' `
  -m cnn_fpga.benchmark.bit_accurate_hardware_reference
```

