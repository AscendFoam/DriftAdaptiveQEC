# 投稿稿 affine local-validity 诊断记录

日期：2026-07-06

本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`，把已有 controlled sequence 和 holdout-stress source CSV 派生为一张局部仿射有效性诊断表。

该表只回答一个审稿问题：在受控局部高斯设置和未见漂移压力设置中，oracle affine 是否仍有 MSE headroom，naive wrapped posterior 是否自动支配 affine path，以及 stale commit 是否会吞掉这种 headroom。

## 生成文件

- `docs\paper_materials\submission_draft_affine_local_validity_diagnostic.csv`
- `docs\paper_materials\submission_draft_affine_local_validity_diagnostic.json`

## 结果摘要

| Surface | Layer | Oracle gain (%) | Branch risk delta | Lag risk delta | Readout |
| --- | --- | ---: | ---: | ---: | --- |
| `static_bias_theta` | `short_sequence_controlled` | 2.05 | 0.000000 | NA | `local_affine_not_dominated` |
| `linear_ramp` | `short_sequence_controlled` | 3.40 | 0.000142 | NA | `local_affine_headroom_visible` |
| `step_sigma_theta` | `short_sequence_controlled` | 10.00 | 0.000910 | NA | `local_affine_headroom_visible` |
| `periodic_drift` | `short_sequence_controlled` | 3.43 | 0.000122 | NA | `local_affine_headroom_visible` |
| `random_walk_drift` | `holdout_stress_controlled` | 7.84 | 0.006055 | 0.001182 | `local_affine_headroom_visible` |
| `burst_reset_drift` | `holdout_stress_controlled` | 7.06 | 0.003792 | 0.006478 | `stale_commit_can_erase_gain` |
| `faster_than_window_oscillation` | `holdout_stress_controlled` | 6.74 | 0.003740 | 0.005144 | `stale_commit_can_erase_gain` |

## 可写边界

- 可以写：该派生表把 oracle-affine MSE headroom、naive wrapped-posterior branch risk 和 stale-commit risk 放在同一个审稿可读框架中。
- 可以写：它支持“affine fast path 有局部有效域且 commit policy 是方法的一部分”的受限解释。
- 不能写：它补齐了正式 nearest-lattice / wrapped-decoder benchmark、CI/p-value、trained-branch holdout generalization、finite-energy logical-channel fidelity 或硬件证据。
