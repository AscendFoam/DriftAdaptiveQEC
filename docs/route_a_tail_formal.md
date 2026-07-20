# Route-A abrupt/OOD tail formal（T6.7.2）

## 结论

T6.7.2 的预注册 tail safety 和 nominal non-inferiority 门全部通过，但必须按正确强度解释：

> Route-A 在六个 abrupt/OOD families 中相对 pilot 锁定 EWMA 未发生超过 margin 的 degradation；这不是 tail LER 优势。

六类里五类与 EWMA 在所有核心安全统计上完全相等；burst 只有两个 avoided、两个 induced，最大单窗口 excess 为 1。

## Formal 规模

- 24 formal seed clusters；
- step、telegraph、burst、readout/reset、leakage、compound 各 6 cells；
- nominal 1 cell；
- 888 trajectories；
- 每方法 43,646,976 scored decisions；
- 六方法逐窗口 raw Pauli + paired/action Source Data 686,104 rows。

## calibration-shift

| method | average LER | global worst |
| --- | ---: | ---: |
| static joint MAP | 0.01087 | 32/512 |
| Window MAP | 0.01243 | 182/512 |
| locked EWMA | 0.01227 | 181/512 |
| Route-A | 0.01227 | 181/512 |

Route-A=EWMA，因此严格 locked-baseline 门通过；但它明显没有超过 static。论文不能把“181=181”改写成“tail 改善”，也不能声称旧 static 反例已被性能上超越。

## 六类 safety contrast

- step：Route-A−EWMA 全部 0；
- telegraph：全部 0；
- burst：aggregate 差约 0，最大单窗口 +1；
- readout/reset：全部 0；
- leakage：全部 0；
- compound：全部 0。

所有预注册 catastrophic margins 通过。

## nominal

- average difference：0；
- fallback：0.119%；
- unnecessary fallback：0.118%；
- induced−avoided：0；
- 四个 non-inferiority 门全部通过。

## 动作代价

tail 场景 fallback signal 约 59%–96%，unnecessary fallback 约 59%–95%。每个 family 有 3456 个 scored commits，其中 2044–3365 个发生在 tail truth interval，按预注册定义属于 false update。

这不代表 fast decoder 一定换成 static；V4 实际仍使用持续更新的 trusted EWMA shadow。但它也意味着不能再写“tail 时冻结所有自适应更新”。正确表述是：

- posterior/event 激活 safety signal；
- Window promotion 被阻断；
- trusted EWMA shadow 仍持续更新并可提交；
- integrity failure 才走 LKG rollback。

## lag

- detection：0 censored；burst p95 96 decisions，其余多为 0；
- recovery-to-OPEN：0 censored；p95 256–320 decisions；
- step 为持续 shift，无 recovery event。

## Claim 边界

可以：

- 相对锁定 EWMA 的 abrupt/OOD non-inferiority；
- nominal fallback cost 在预注册 margin 内；
- calibration worst 相对锁定 EWMA 不再恶化。

不可以：

- tail LER 优势；
- 相对 static 的 calibration safety 优势；
- 低 fallback 或低 false-update；
- physical-device fault guarantee；
- measured FPGA deadline/latency。

## 复现

```powershell
python -m cnn_fpga.benchmark.route_a_tail_formal
python -m cnn_fpga.benchmark.route_a_tail_formal --verify-only
python -m pytest tests/test_route_a_tail_formal.py -q
```

