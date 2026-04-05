# P3 Overflow Tuning Sweep

| Candidate | Phase | gain_clip | gain_scale | beta | alpha | LER | Overflow | dLER vs short baseline | Commits |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_current | short | [0.10, 1.20] | 1.00 | 0.20 | 0.90 | 1.064743 | 0.370296 | +0.000000 | 240 |
| gain_scale_097 | short | [0.10, 1.20] | 0.97 | 0.20 | 0.90 | 1.067802 | 0.369854 | +0.003059 | 240 |
| gain_scale_095 | short | [0.10, 1.20] | 0.95 | 0.20 | 0.90 | 1.068545 | 0.368538 | +0.003802 | 240 |
| gain_scale_093 | short | [0.10, 1.20] | 0.93 | 0.20 | 0.90 | 1.070314 | 0.367250 | +0.005571 | 240 |
| gain_scale_090 | short | [0.10, 1.20] | 0.90 | 0.20 | 0.90 | 1.075140 | 0.365824 | +0.010397 | 240 |
| gain_scale_095_alpha085 | short | [0.10, 1.20] | 0.95 | 0.20 | 0.85 | 1.068679 | 0.368424 | +0.003936 | 240 |
| gain_scale_093_alpha085 | short | [0.10, 1.20] | 0.93 | 0.20 | 0.85 | 1.070315 | 0.368260 | +0.005572 | 240 |
| gain_scale_090_alpha085 | short | [0.10, 1.20] | 0.90 | 0.20 | 0.85 | 1.073182 | 0.365947 | +0.008440 | 240 |
| gain_scale_090 | confirm | [0.10, 1.20] | 0.90 | 0.20 | 0.90 | 1.098839 | 0.382887 | +0.034097 | 900 |
| gain_scale_090_alpha085 | confirm | [0.10, 1.20] | 0.90 | 0.20 | 0.85 | 1.099284 | 0.382375 | +0.034541 | 900 |
| gain_scale_093 | confirm | [0.10, 1.20] | 0.93 | 0.20 | 0.90 | 1.091004 | 0.384720 | +0.026261 | 900 |
| gain_scale_093_alpha085 | confirm | [0.10, 1.20] | 0.93 | 0.20 | 0.85 | 1.090744 | 0.385207 | +0.026001 | 900 |

## Selection

- Shortlist: gain_scale_090, gain_scale_090_alpha085, gain_scale_093, gain_scale_093_alpha085, gain_scale_095_alpha085, gain_scale_095, gain_scale_097, baseline_current
- Long confirm: gain_scale_090, gain_scale_090_alpha085, gain_scale_093, gain_scale_093_alpha085
