# P3 Overflow Tuning Sweep

| Candidate | Phase | gain_clip | beta | alpha | LER | Overflow | dLER vs short baseline | Commits |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_current | short | [0.20, 1.20] | 0.20 | 1.00 | 1.065146 | 0.369573 | +0.000000 | 240 |
| mild_clip | short | [0.15, 0.80] | 0.15 | 0.90 | 1.062333 | 0.370747 | -0.002812 | 240 |
| mild_clip_lowfloor | short | [0.10, 0.80] | 0.15 | 0.90 | 1.061145 | 0.368967 | -0.004001 | 240 |
| mid_clip | short | [0.15, 0.75] | 0.10 | 0.90 | 1.059396 | 0.372780 | -0.005750 | 240 |
| mid_clip_lowfloor | short | [0.10, 0.75] | 0.10 | 0.90 | 1.058699 | 0.372085 | -0.006447 | 240 |
| strong_clip | short | [0.10, 0.70] | 0.10 | 0.80 | 1.059372 | 0.371414 | -0.005774 | 240 |
| strong_clip_slow | short | [0.05, 0.70] | 0.08 | 0.80 | 1.057828 | 0.371042 | -0.007318 | 240 |
| very_conservative | short | [0.05, 0.65] | 0.05 | 0.70 | 1.048549 | 0.370504 | -0.016597 | 240 |
| mild_clip_lowfloor | confirm | [0.10, 0.80] | 0.15 | 0.90 | 1.069704 | 0.388059 | +0.004558 | 900 |
| baseline_current | confirm | [0.20, 1.20] | 0.20 | 1.00 | 1.069886 | 0.387596 | +0.004740 | 900 |
| very_conservative | confirm | [0.05, 0.65] | 0.05 | 0.70 | 1.104497 | 0.386268 | +0.039351 | 900 |

## Selection

- Shortlist: mild_clip_lowfloor, baseline_current, very_conservative, mild_clip, strong_clip_slow, strong_clip, mid_clip_lowfloor, mid_clip
- Long confirm: mild_clip_lowfloor, baseline_current, very_conservative
