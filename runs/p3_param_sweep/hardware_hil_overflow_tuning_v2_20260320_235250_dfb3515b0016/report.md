# P3 Overflow Tuning Sweep

| Candidate | Phase | gain_clip | beta | alpha | LER | Overflow | dLER vs short baseline | Commits |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_current | short | [0.20, 1.20] | 0.20 | 1.00 | 1.066796 | 0.369833 | +0.000000 | 240 |
| lowfloor_only | short | [0.10, 1.20] | 0.20 | 1.00 | 1.065486 | 0.368969 | -0.001309 | 240 |
| lower_alpha | short | [0.20, 1.20] | 0.20 | 0.90 | 1.065821 | 0.369590 | -0.000975 | 240 |
| lowfloor_lower_alpha | short | [0.10, 1.20] | 0.20 | 0.90 | 1.064774 | 0.369278 | -0.002022 | 240 |
| lower_beta | short | [0.20, 1.20] | 0.15 | 1.00 | 1.064053 | 0.369291 | -0.002743 | 240 |
| lowfloor_lower_beta | short | [0.10, 1.20] | 0.15 | 1.00 | 1.064153 | 0.369477 | -0.002643 | 240 |
| lower_alpha_beta | short | [0.20, 1.20] | 0.15 | 0.90 | 1.063229 | 0.369600 | -0.003567 | 240 |
| balanced_lowfloor_alpha_beta | short | [0.10, 1.20] | 0.15 | 0.90 | 1.063753 | 0.371252 | -0.003043 | 240 |
| lowfloor_only | confirm | [0.10, 1.20] | 0.20 | 1.00 | 1.069939 | 0.386968 | +0.003143 | 900 |
| lowfloor_lower_alpha | confirm | [0.10, 1.20] | 0.20 | 0.90 | 1.069234 | 0.386844 | +0.002438 | 900 |
| lower_beta | confirm | [0.20, 1.20] | 0.15 | 1.00 | 1.069300 | 0.386876 | +0.002504 | 900 |

## Selection

- Shortlist: lowfloor_only, lowfloor_lower_alpha, lower_beta, lowfloor_lower_beta, lower_alpha, lower_alpha_beta, baseline_current, balanced_lowfloor_alpha_beta
- Long confirm: lowfloor_only, lowfloor_lower_alpha, lower_beta
