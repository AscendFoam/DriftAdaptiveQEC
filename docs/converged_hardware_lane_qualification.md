# T6.25.4 converged hardware lane 三种子 P&R

## 结论

**`PASS_EXACT_CONVERGED_TOP_THREE_SEED_PREBOARD_HARDWARE_LANE`**。对 T6.25.3 exact qualified converged top 的 small-pin observability harness 完成 GW2AR-LV18QN88C8/I7 seeds 1/7/19 open-source synthesis/P&R；三次均通过 27 MHz 约束。

| Seed | Fmax (MHz) | LUT4 | DFF | BSRAM | DSP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37.869 | 6118 | 2495 | 8 | 2 |
| 7 | 37.736 | 6108 | 2495 | 8 | 2 |
| 19 | 36.794 | 6130 | 2495 | 8 | 2 |

- Fmax min/median/max=`36.794/37.736/37.869` MHz。
- 资源最大值：LUT4=6130，DFF=2495，BSRAM=8，MULT18X18=1，MULT9X9=1。
- 6-cycle clock-model latency：27 MHz 下 `222.222` ns；II=1。该数值不含 transport/CDC/pin/jitter。
- 动态功耗仅为解析敏感性 low/nominal/high=`2.676/16.058/64.231` mW；不是 vendor power 或板测。
- 19/19 semantic mutations 被 gate 重算拒绝。

## 证据边界

这是 two-state、open-source pre-board P&R estimate。bitstream、真实 transport/CDC、板测 latency/jitter/deadline/power 与跨工作 fastest/SOTA 均未建立；multimode decoder 未部署在该 RTL 中。
