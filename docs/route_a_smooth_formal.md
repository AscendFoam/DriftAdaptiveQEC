# Route-A smooth formal matrix（T6.7.1）

## 结论

T6.7.1 已完成全部 untouched smooth formal matrix。按预注册规则，Route-A 相对 pilot 锁定的 EWMA baseline 通过 aggregate paired LER 门：

[
Delta p_L=p_L^{\mathrm{EWMA}}-p_L^{\mathrm{RouteA}}
=2.1687\times10^{-5},
quad 95\%\,\mathrm{CI}=[1.9003,2.4548]\times10^{-5}.
]

这个结果只支持“在冻结 simulator/protocol 和 EWMA primary contrast 下，Route-A 的 smooth aggregate LER 有小幅正改善”。它不支持“Route-A 是 formal smooth matrix 中最好的 decoder”。

## 设计

- 24 个独立 formal seed clusters；
- mean、variance、correlation、periodic 各 6 个 held-out cells；
- 每 trajectory 8 个不计分 preamble windows + 96 个计分 windows；
- 每个 window 512 decisions；
- 每方法 28,311,552 scored decisions；
- standard/static/Window/EWMA/Kalman/Route-A 共享同一 10-bit syndrome；
- hidden-state oracle 单独评价，不进入 online route；
- 20,000 次 paired seed-cluster bootstrap；
- 四 family 等权，禁止结果后 reweight。

## 主要结果

| method | average (p_L) | p95 | worst |
| --- | ---: | ---: | ---: |
| static joint MAP | 0.00096819 | 3/512 | 16/512 |
| Window MAP | 0.00089642 | 2/512 | 64/512 |
| locked EWMA | 0.00101443 | 2/512 | 59/512 |
| Route-A | 0.00099274 | 2/512 | 59/512 |
| hidden oracle | 0.00016234 | 1/512 | 7/512 |

Route-A 的改善集中在 periodic drift：

- mean：CI 跨 0；
- variance：point 0，CI 跨 0；
- correlation：完全相同；
- periodic：`8.618e-5 [7.545e-5,9.735e-5]`，唯一 Holm discovery。

static-to-oracle gap closure 为 `-0.03046 [-0.04966,-0.01119]`；这是明确负证据，说明 Route-A 比 static 更远离 hidden-state oracle。

## 安全/动作代价

- fallback signal rate：21.63%；
- unnecessary fallback decision rate：21.58%；
- avoided / induced errors：661 / 47；
- false updates：0；
- adaptation lag：1904 decisions，0 censored；
- Window/EWMA commits：2870/10954。

这里的 fallback 是 policy safety signal；它不一定代表 fast path 换成 static decoder，但仍表明 smooth 场景下 posterior 经常不给 OPEN。论文必须同时报告，不得只给 LER 改善。

## Pauli 分量

每个方法同时报告 (p_X,p_Y,p_Z) 点估计和 24-seed cluster bootstrap 描述性区间。逻辑类编码冻结为：

- 0：I
- 1：Z（p parity）
- 2：X（q parity）
- 3：Y

原始逐窗口计数见 Source Data。

## 可支持与不可支持的主张

可以：

- untouched formal smooth aggregate 中，Route-A 相对锁定 EWMA 有小幅 paired improvement；
- periodic drift 是当前唯一 family-level discovery；
- dual-bank route 产生更多 avoided than induced errors。

不可以：

- “Route-A 优于 static GKP decoder”；
- “Route-A 优于所有 adaptive baselines”；
- “四种 smooth drift 均有优势”；
- “oracle gap 已关闭”；
- abrupt/OOD、physical lifetime、break-even、Puviani NMF 或 measured FPGA speed claim。

## 复现

```powershell
python -m cnn_fpga.benchmark.route_a_smooth_formal
python -m cnn_fpga.benchmark.route_a_smooth_formal --verify-only
python -m pytest tests/test_route_a_smooth_formal.py -q
```

主产物：

- `docs/t6_7_1_smooth_formal_matrix.json`
- `docs/t6_7_1_smooth_formal_matrix_source_data.csv`
- `runs/t6_7_1_formal_access_ledger.json`

