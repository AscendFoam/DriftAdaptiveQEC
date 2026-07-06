# 投稿稿 source-data 机械审计报告

## 作用边界

本文档由 `docs/figure_assets/submission_draft_python_figures/audit_submission_draft_source_data.py` 生成，服务 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它只检查当前投稿稿表格、figure source CSV、benchmark comparison/summary、controlled analysis CSV、literature metric crosswalk、benchmark expansion protocol、runner smoke pair、runner smoke matrix、row-level provenance、图件 manifest 与 source-data manifest 的机械一致性。

本文档不是新实验，不运行 benchmark，不报告 CI，不证明 full reproducibility，不证明 fallback-free runtime，也不改变 `.tflite`、real-board、statcalib 或硬件证据等级。

## 审计结论

- Status: `PASS_WITH_LIMITATIONS`
- Checks passed: `250`
- Checks failed: `0`
- 主结论：当前 TeX 表格、figure source data、生成分析 CSV、literature metric crosswalk、source-data coverage matrix、benchmark expansion protocol、runner smoke pair、runner smoke matrix、row-level provenance、图件 manifest 和 source-data manifest 在已检查字段上机械一致。
- 解释边界：这只能支持 manuscript table/source-data traceability；不等于强统计、expanded benchmark、硬件验证或完整复现包。

## 输入文件哈希

| 文件 | SHA256 |
| --- | --- |
| `tex` | `84fa953ec8fe1bacdb75862edc4877c74b06a688725790a933b01d2f9b684d95` |
| `t24_comparison_csv` | `c7f9cb551bff840cd3e64ef52c776ea1d07eff9aaaa4f67b54f33d003cdee8a4` |
| `t24_summary_json` | `f03e8721160c85855518ba6f3c44b9adb05deb197d554918dcd32a6f68feca6b` |
| `fig02_source_csv` | `0a3989a3f74d75aa9ef0aff76415f3f7e07b970d655dc070f8757af18fe33e02` |
| `fig02_paired_source_csv` | `44d75a22dcc8e5ad9ac1e8b27c758c95ee56a5f6671b906eecafd8f35f19c0e7` |
| `fig03_source_csv` | `76e87d766fe553d0e909d35e1ab6c5303c7eecb5673f17ea6f6fa66d6c6c33cc` |
| `fig04_source_csv` | `0a5e183206f60b97e86a9febed098abe33a094d089355b4570bd6dfc7bc973bc` |
| `fig05_source_csv` | `ee89f3c97ea8fc4b8133a82bde650db9ebe37eeb5ff735eaec96e78fa9a9ca87` |
| `controlled_oracle_affine_csv` | `a09789c9b33e0e7630f5cad7a61f03e8160061d9be0368b5f560584118bc0d6e` |
| `fast_path_cost_model_csv` | `83abaf5f0389156b23290e0960644d268dc35759e5a245bfd8304720c9c83c43` |
| `fixed_point_parity_csv` | `3d5c2602df37666ee0504b4ea3ee34e9011a46c54592b72013a9632049c41332` |
| `logical_channel_surrogate_csv` | `b1006ec768e201c6187720334425a446132e1bd0bc62b83f9b07ad26bba9967c` |
| `finite_energy_channel_sanity_csv` | `39fcc31503b2267a70904c7d48549bf2d1b2ed16b106bc22bc0dc5e01736b0ee` |
| `holdout_drift_stress_csv` | `8d542c3e76bf8ee8ec94d16089b60a8f40388711c31f11479617652db4860f71` |
| `affine_local_validity_diagnostic_csv` | `819d202b4f65a4bc7334b6d54b4c7f2c9d48cb25bd3fe2578396505dd6cedc8f` |
| `commit_lag_sweep_csv` | `7566a76715a57ac2ea6be8ce970e7c81e5aee3e0d6dc405873b078995fed61d1` |
| `commit_lag_sweep_json` | `27815c71646b57ad402ebcf287a8abf92126576feed5d49a94eb13cc61fc3df0` |
| `sequence_controlled_baseline_csv` | `763d53afad7613a929e1ce328bdf5a71b87f40fec9d9cd794c9944766a46a8d3` |
| `paired_uncertainty_csv` | `418863d8485c1a5ac31b8d388b06948488553b745a35b7261a35984619bb627e` |
| `ler_advantage_margin_csv` | `02f98536c13e3707773fe78a3088a68b979381f5bab9e1bc55daf9f27379b611` |
| `metric_readiness_csv` | `4afe5c218adc79228ca995a2b7860bc1f4c0f53815d6ffb4cc154111734811b6` |
| `literature_metric_crosswalk_csv` | `a9ca8b75ac921df1f44afe4544e435c73bc0cac6f4d7c1ee184490bede7a1913` |
| `closest_work_positioning_csv` | `b335165a5821e766b910cd01ae401be5def36dd9540360256612b3742bf7346d` |
| `source_data_coverage_matrix_csv` | `2f993274956f004c9cca2b72b494875ece033883c51030d036e54850f4f7edcd` |
| `source_data_coverage_matrix_json` | `9c4b60cf6f467b158df2b3d38c441c5a70032795ad988f3bec4e5c19626251d6` |
| `benchmark_expansion_protocol_csv` | `81aa8601f7e0cb8487724da688f7f41b7fe0be2d441a1701b9bdd84322c0aaef` |
| `phase_a_repeat_plan_csv` | `b7463f86e07ecadf7a229699b6a248abbb461c6a6621387b1b0eeb8fa15a4f33` |
| `phase_a_repeat_summary_csv` | `64a158292024868a0c2a7f75de780070fa22be1146f609a9aff09d60d27c745d` |
| `phase_a_paired_interval_csv` | `84214160c3997dc046e2252e0417213dca94e61ab2a83214d574af3b4e2e38b0` |
| `phase_a_upgrade_gate_csv` | `7225e5f4ece6f52d589c4fa41960cab6d8e2b25bf14d0c4dae868f808b1bec5d` |
| `runner_smoke_pair_csv` | `41d925a9f1d8c6c00dd941eadb91d5a5388aca244397b9e595e5088a1603cb9c` |
| `runner_smoke_matrix_csv` | `af154d63ac34e382c00851d5d5cb705fc88ce7cd795caf783a9e298f318f4464` |
| `row_provenance_manifest_csv` | `aa2d43e580f203653177d0bcb485038a9248c50c7acadb4ca829b6c7b1190753` |
| `row_provenance_manifest_json` | `7f9d7fdb51ead52a014d703a82a55faaf96e3456a2a0ddd6c2aaca059f93e88a` |
| `runtime_discipline_csv` | `a182da068536e20bef946b3db39eafc07a1dbb49da1228417168adb382cb2469` |
| `gkp_boundary_sensitivity_csv` | `aa983727d3b5f48a2bb50c4f60b02e9705efefa61f97fbfa001d0bc7f533be0a` |
| `submission_source_data_manifest_csv` | `1640012f69e4545ebfb8460f5fa0b79798457ff740cb29855dce7ef5cb5e57e0` |
| `submission_source_data_manifest_json` | `889a5d75731a41c9cbba9909aeef7578fdd46c5d2fee0800541663c26abdad57` |
| `figure_manifest_json` | `a542fbd9c4bc8c72f20ba9d4ef09847651072523099afae7e6d3a3fd66b609bb` |

## 模型 artifact 哈希

| Artifact | SHA256 |
| --- | --- |
| `artifacts\models\runtime_b_residual_v1\tiny_cnn_20260401_083648_2fc740424c0d.npz` | `585ee844b0bd9b9046f01eb97835a2da788050d4197846265545f44ec8d5aabb` |

## 检查明细

| Check | Status | Detail |
| --- | --- | --- |
| `T24 protocol shape` | `PASS` | summary.json records four scenarios, five predeclared modes, paired seeds and repeats=2. |
| `T24 row coverage` | `PASS` | comparison.csv contains 20 rows with completed_repeats=expected_repeats=2 and coverage=1.0. |
| `TeX main table static_bias_theta/ekf` | `PASS` | TeX 0.838110 vs T24 rounded 0.838110. |
| `TeX main table static_bias_theta/ukf` | `PASS` | TeX 0.825370 vs T24 rounded 0.825370. |
| `TeX main table static_bias_theta/constant_residual_mu` | `PASS` | TeX 0.836658 vs T24 rounded 0.836658. |
| `TeX main table static_bias_theta/rls_residual_b` | `PASS` | TeX 0.837577 vs T24 rounded 0.837577. |
| `TeX main table static_bias_theta/hybrid_residual_b` | `PASS` | TeX 0.810902 vs T24 rounded 0.810902. |
| `TeX main table linear_ramp/ekf` | `PASS` | TeX 0.819200 vs T24 rounded 0.819200. |
| `TeX main table linear_ramp/ukf` | `PASS` | TeX 0.811201 vs T24 rounded 0.811201. |
| `TeX main table linear_ramp/constant_residual_mu` | `PASS` | TeX 0.816911 vs T24 rounded 0.816911. |
| `TeX main table linear_ramp/rls_residual_b` | `PASS` | TeX 0.819373 vs T24 rounded 0.819373. |
| `TeX main table linear_ramp/hybrid_residual_b` | `PASS` | TeX 0.787755 vs T24 rounded 0.787755. |
| `TeX main table step_sigma_theta/ekf` | `PASS` | TeX 0.822365 vs T24 rounded 0.822365. |
| `TeX main table step_sigma_theta/ukf` | `PASS` | TeX 0.811548 vs T24 rounded 0.811548. |
| `TeX main table step_sigma_theta/constant_residual_mu` | `PASS` | TeX 0.819784 vs T24 rounded 0.819784. |
| `TeX main table step_sigma_theta/rls_residual_b` | `PASS` | TeX 0.821493 vs T24 rounded 0.821493. |
| `TeX main table step_sigma_theta/hybrid_residual_b` | `PASS` | TeX 0.788800 vs T24 rounded 0.788800. |
| `TeX main table periodic_drift/ekf` | `PASS` | TeX 0.832192 vs T24 rounded 0.832192. |
| `TeX main table periodic_drift/ukf` | `PASS` | TeX 0.821558 vs T24 rounded 0.821558. |
| `TeX main table periodic_drift/constant_residual_mu` | `PASS` | TeX 0.829670 vs T24 rounded 0.829670. |
| `TeX main table periodic_drift/rls_residual_b` | `PASS` | TeX 0.832334 vs T24 rounded 0.832334. |
| `TeX main table periodic_drift/hybrid_residual_b` | `PASS` | TeX 0.806392 vs T24 rounded 0.806392. |
| `Fig2 source mean static_bias_theta/ekf` | `PASS` | source_data mean 0.838109861111 vs T24 0.8381098611111111. |
| `Fig2 source sd static_bias_theta/ekf` | `PASS` | source_data SD 0.000832638889 vs T24 0.0008326388888889036. |
| `Fig2 repeats static_bias_theta/ekf` | `PASS` | source_data n_repeats matches T24 completed_repeats. |
| `Fig2 source mean static_bias_theta/ukf` | `PASS` | source_data mean 0.825370416667 vs T24 0.8253704166666667. |
| `Fig2 source sd static_bias_theta/ukf` | `PASS` | source_data SD 0.000672916667 vs T24 0.0006729166666666897. |
| `Fig2 repeats static_bias_theta/ukf` | `PASS` | source_data n_repeats matches T24 completed_repeats. |
| `Fig2 source mean static_bias_theta/constant_residual_mu` | `PASS` | source_data mean 0.836658333333 vs T24 0.8366583333333333. |
| `Fig2 source sd static_bias_theta/constant_residual_mu` | `PASS` | source_data SD 0.000128611111 vs T24 0.000128611111111121. |
| `Fig2 repeats static_bias_theta/constant_residual_mu` | `PASS` | source_data n_repeats matches T24 completed_repeats. |
| `Fig2 source mean static_bias_theta/rls_residual_b` | `PASS` | source_data mean 0.837576805556 vs T24 0.8375768055555556. |
| `Fig2 source sd static_bias_theta/rls_residual_b` | `PASS` | source_data SD 0.000734027778 vs T24 0.0007340277777777571. |
| `Fig2 repeats static_bias_theta/rls_residual_b` | `PASS` | source_data n_repeats matches T24 completed_repeats. |
| `Fig2 source mean static_bias_theta/hybrid_residual_b` | `PASS` | source_data mean 0.810901527778 vs T24 0.8109015277777778. |
| `Fig2 source sd static_bias_theta/hybrid_residual_b` | `PASS` | source_data SD 0.001188472222 vs T24 0.0011884722222222366. |
| `Fig2 repeats static_bias_theta/hybrid_residual_b` | `PASS` | source_data n_repeats matches T24 completed_repeats. |
| `Fig2 source mean linear_ramp/ekf` | `PASS` | source_data mean 0.819199583333 vs T24 0.8191995833333334. |
| `Fig2 source sd linear_ramp/ekf` | `PASS` | source_data SD 0.000170138889 vs T24 0.00017013888888889328. |
| `Fig2 repeats linear_ramp/ekf` | `PASS` | source_data n_repeats matches T24 completed_repeats. |
| `Fig2 source mean linear_ramp/ukf` | `PASS` | source_data mean 0.811200833333 vs T24 0.8112008333333334. |
| `Fig2 source sd linear_ramp/ukf` | `PASS` | source_data SD 0.000868333333 vs T24 0.0008683333333333043. |
| `Fig2 repeats linear_ramp/ukf` | `PASS` | source_data n_repeats matches T24 completed_repeats. |
| `Fig2 source mean linear_ramp/constant_residual_mu` | `PASS` | source_data mean 0.816911111111 vs T24 0.8169111111111111. |
| `Fig2 source sd linear_ramp/constant_residual_mu` | `PASS` | source_data SD 0.000124444444 vs T24 0.00012444444444442704. |
| `Fig2 repeats linear_ramp/constant_residual_mu` | `PASS` | source_data n_repeats matches T24 completed_repeats. |
| `Fig2 source mean linear_ramp/rls_residual_b` | `PASS` | source_data mean 0.819373472222 vs T24 0.8193734722222222. |
| `Fig2 source sd linear_ramp/rls_residual_b` | `PASS` | source_data SD 0.000092083333 vs T24 9.208333333332597e-05. |
| `Fig2 repeats linear_ramp/rls_residual_b` | `PASS` | source_data n_repeats matches T24 completed_repeats. |
| `Fig2 source mean linear_ramp/hybrid_residual_b` | `PASS` | source_data mean 0.787755138889 vs T24 0.7877551388888888. |
| `Fig2 source sd linear_ramp/hybrid_residual_b` | `PASS` | source_data SD 0.000439305556 vs T24 0.0004393055555555469. |
| `Fig2 repeats linear_ramp/hybrid_residual_b` | `PASS` | source_data n_repeats matches T24 completed_repeats. |
| `Fig2 source mean step_sigma_theta/ekf` | `PASS` | source_data mean 0.822365416667 vs T24 0.8223654166666667. |
| `Fig2 source sd step_sigma_theta/ekf` | `PASS` | source_data SD 0.000170138889 vs T24 0.00017013888888889328. |
| `Fig2 repeats step_sigma_theta/ekf` | `PASS` | source_data n_repeats matches T24 completed_repeats. |
| `Fig2 source mean step_sigma_theta/ukf` | `PASS` | source_data mean 0.811547500000 vs T24 0.8115475. |
| `Fig2 source sd step_sigma_theta/ukf` | `PASS` | source_data SD 0.000761111111 vs T24 0.0007611111111111013. |
| `Fig2 repeats step_sigma_theta/ukf` | `PASS` | source_data n_repeats matches T24 completed_repeats. |
| `Fig2 source mean step_sigma_theta/constant_residual_mu` | `PASS` | source_data mean 0.819784166667 vs T24 0.8197841666666666. |
| `Fig2 source sd step_sigma_theta/constant_residual_mu` | `PASS` | source_data SD 0.000275833333 vs T24 0.000275833333333364. |
| `Fig2 repeats step_sigma_theta/constant_residual_mu` | `PASS` | source_data n_repeats matches T24 completed_repeats. |
| `Fig2 source mean step_sigma_theta/rls_residual_b` | `PASS` | source_data mean 0.821492500000 vs T24 0.8214925. |
| `Fig2 source sd step_sigma_theta/rls_residual_b` | `PASS` | source_data SD 0.000420000000 vs T24 0.00042000000000003146. |
| `Fig2 repeats step_sigma_theta/rls_residual_b` | `PASS` | source_data n_repeats matches T24 completed_repeats. |
| `Fig2 source mean step_sigma_theta/hybrid_residual_b` | `PASS` | source_data mean 0.788799722222 vs T24 0.7887997222222223. |
| `Fig2 source sd step_sigma_theta/hybrid_residual_b` | `PASS` | source_data SD 0.001069166667 vs T24 0.001069166666666621. |
| `Fig2 repeats step_sigma_theta/hybrid_residual_b` | `PASS` | source_data n_repeats matches T24 completed_repeats. |
| `Fig2 source mean periodic_drift/ekf` | `PASS` | source_data mean 0.832191666667 vs T24 0.8321916666666667. |
| `Fig2 source sd periodic_drift/ekf` | `PASS` | source_data SD 0.000557777778 vs T24 0.0005577777777777682. |
| `Fig2 repeats periodic_drift/ekf` | `PASS` | source_data n_repeats matches T24 completed_repeats. |
| `Fig2 source mean periodic_drift/ukf` | `PASS` | source_data mean 0.821558055556 vs T24 0.8215580555555555. |
| `Fig2 source sd periodic_drift/ukf` | `PASS` | source_data SD 0.001884722222 vs T24 0.0018847222222221904. |
| `Fig2 repeats periodic_drift/ukf` | `PASS` | source_data n_repeats matches T24 completed_repeats. |
| `Fig2 source mean periodic_drift/constant_residual_mu` | `PASS` | source_data mean 0.829670138889 vs T24 0.8296701388888889. |
| `Fig2 source sd periodic_drift/constant_residual_mu` | `PASS` | source_data SD 0.000345416667 vs T24 0.0003454166666666536. |
| `Fig2 repeats periodic_drift/constant_residual_mu` | `PASS` | source_data n_repeats matches T24 completed_repeats. |
| `Fig2 source mean periodic_drift/rls_residual_b` | `PASS` | source_data mean 0.832334166667 vs T24 0.8323341666666666. |
| `Fig2 source sd periodic_drift/rls_residual_b` | `PASS` | source_data SD 0.000040277778 vs T24 4.027777777776409e-05. |
| `Fig2 repeats periodic_drift/rls_residual_b` | `PASS` | source_data n_repeats matches T24 completed_repeats. |
| `Fig2 source mean periodic_drift/hybrid_residual_b` | `PASS` | source_data mean 0.806391944444 vs T24 0.8063919444444444. |
| `Fig2 source sd periodic_drift/hybrid_residual_b` | `PASS` | source_data SD 0.000289166667 vs T24 0.0002891666666666737. |
| `Fig2 repeats periodic_drift/hybrid_residual_b` | `PASS` | source_data n_repeats matches T24 completed_repeats. |
| `Fig2 paired source static_bias_theta/repeat_0` | `PASS` | source paired delta 0.013953333333 and relative 1.689177 vs raw_rows repeat 0. |
| `Fig2 paired source static_bias_theta/repeat_1` | `PASS` | source paired delta 0.014984444444 and relative 1.816963 vs raw_rows repeat 1. |
| `Fig2 paired source linear_ramp/repeat_0` | `PASS` | source paired delta 0.023874722222 and relative 2.939986 vs raw_rows repeat 0. |
| `Fig2 paired source linear_ramp/repeat_1` | `PASS` | source paired delta 0.023016666667 and relative 2.840398 vs raw_rows repeat 1. |
| `Fig2 paired source step_sigma_theta/repeat_0` | `PASS` | source paired delta 0.023055833333 and relative 2.843638 vs raw_rows repeat 0. |
| `Fig2 paired source step_sigma_theta/repeat_1` | `PASS` | source paired delta 0.022439722222 and relative 2.762463 vs raw_rows repeat 1. |
| `Fig2 paired source periodic_drift/repeat_0` | `PASS` | source paired delta 0.016761666667 and relative 2.035559 vs raw_rows repeat 0. |
| `Fig2 paired source periodic_drift/repeat_1` | `PASS` | source paired delta 0.013570555556 and relative 1.655605 vs raw_rows repeat 1. |
| `TeX paired-delta table static_bias_theta` | `PASS` | TeX mean/min 0.014469/0.013953 and relative 1.75 vs source paired deltas. |
| `TeX paired-delta table linear_ramp` | `PASS` | TeX mean/min 0.023446/0.023017 and relative 2.89 vs source paired deltas. |
| `TeX paired-delta table step_sigma_theta` | `PASS` | TeX mean/min 0.022748/0.022440 and relative 2.80 vs source paired deltas. |
| `TeX paired-delta table periodic_drift` | `PASS` | TeX mean/min 0.015166/0.013571 and relative 1.85 vs source paired deltas. |
| `TeX paired-uncertainty table static_bias_theta` | `PASS` | TeX n/mean/low/high/direction 2/0.014469/0.013953/0.014984/2/2 vs paired uncertainty CSV. |
| `TeX paired-uncertainty table linear_ramp` | `PASS` | TeX n/mean/low/high/direction 2/0.023446/0.023017/0.023875/2/2 vs paired uncertainty CSV. |
| `TeX paired-uncertainty table step_sigma_theta` | `PASS` | TeX n/mean/low/high/direction 2/0.022748/0.022440/0.023056/2/2 vs paired uncertainty CSV. |
| `TeX paired-uncertainty table periodic_drift` | `PASS` | TeX n/mean/low/high/direction 2/0.015166/0.013571/0.016762/2/2 vs paired uncertainty CSV. |
| `TeX paired-uncertainty table all_scenarios` | `PASS` | TeX n/mean/low/high/direction 8/0.018957/0.016044/0.021892/8/8 vs paired uncertainty CSV. |
| `TeX LER advantage margin table static_bias_theta` | `PASS` | TeX UKF/Hybrid/delta/relative/direction/scale 0.825370/0.810902/0.014469/1.75/2/2/12.17 vs LER advantage margin CSV. |
| `TeX LER advantage margin table linear_ramp` | `PASS` | TeX UKF/Hybrid/delta/relative/direction/scale 0.811201/0.787755/0.023446/2.89/2/2/27.00 vs LER advantage margin CSV. |
| `TeX LER advantage margin table step_sigma_theta` | `PASS` | TeX UKF/Hybrid/delta/relative/direction/scale 0.811548/0.788800/0.022748/2.80/2/2/21.28 vs LER advantage margin CSV. |
| `TeX LER advantage margin table periodic_drift` | `PASS` | TeX UKF/Hybrid/delta/relative/direction/scale 0.821558/0.806392/0.015166/1.85/2/2/8.05 vs LER advantage margin CSV. |
| `GKP boundary sensitivity table sigma=0.15` | `PASS` | TeX squeezing/p_cross/p_any/infidelity 16.48/0.000000/0.000000/0.000000 vs GKP boundary-sensitivity CSV. |
| `GKP boundary sensitivity table sigma=0.20` | `PASS` | TeX squeezing/p_cross/p_any/infidelity 13.98/0.000000/0.000000/0.000000 vs GKP boundary-sensitivity CSV. |
| `GKP boundary sensitivity table sigma=0.25` | `PASS` | TeX squeezing/p_cross/p_any/infidelity 12.04/0.000001/0.000001/0.000001 vs GKP boundary-sensitivity CSV. |
| `GKP boundary sensitivity table sigma=0.30` | `PASS` | TeX squeezing/p_cross/p_any/infidelity 10.46/0.000029/0.000059/0.000039 vs GKP boundary-sensitivity CSV. |
| `GKP boundary sensitivity table sigma=0.35` | `PASS` | TeX squeezing/p_cross/p_any/infidelity 9.12/0.000342/0.000685/0.000456 vs GKP boundary-sensitivity CSV. |
| `GKP boundary sensitivity table sigma=0.40` | `PASS` | TeX squeezing/p_cross/p_any/infidelity 7.96/0.001729/0.003454/0.002303 vs GKP boundary-sensitivity CSV. |
| `Ablation source ukf` | `PASS` | TeX avg/delta 0.817382/0.000000 vs source 0.817382/0.000000. |
| `Ablation source hybrid_full` | `PASS` | TeX avg/delta 0.798545/-0.018837 vs source 0.798545/-0.018837. |
| `Ablation source hybrid_no_hist_deltas` | `PASS` | TeX avg/delta 0.826723/0.009341 vs source 0.826723/0.009341. |
| `Ablation source hybrid_no_teacher_prediction` | `PASS` | TeX avg/delta 0.807251/-0.010131 vs source 0.807251/-0.010131. |
| `Ablation source hybrid_no_teacher_params` | `PASS` | TeX avg/delta 0.749621/-0.067761 vs source 0.749621/-0.067761. |
| `Ablation source hybrid_no_teacher_deltas` | `PASS` | TeX avg/delta 0.800329/-0.017053 vs source 0.800329/-0.017053. |
| `Mechanism source 20260425` | `PASS` | TeX deltas 0.000907/0.163289 vs source 0.000907/0.163289. |
| `Mechanism source 20260427` | `PASS` | TeX deltas -0.145352/0.287166 vs source -0.145352/0.287166. |
| `Mechanism source 20260428` | `PASS` | TeX deltas -0.078998/0.057395 vs source -0.078998/0.057395. |
| `Mechanism source 20260429` | `PASS` | TeX deltas -0.127948/0.322245 vs source -0.127948/0.322245. |
| `Mechanism source 20260430` | `PASS` | TeX deltas -0.170777/-0.024372 vs source -0.170777/-0.024372. |
| `Mechanism source 20260510` | `PASS` | TeX deltas -0.003953/-0.035533 vs source -0.003953/-0.035533. |
| `StatCalib source static_bias_theta/UKF` | `PASS` | TeX 0.825370 vs source 0.825370. |
| `StatCalib source static_bias_theta/Hybrid-b` | `PASS` | TeX 0.810902 vs source 0.810902. |
| `StatCalib source static_bias_theta/StatCalib supp.` | `PASS` | TeX 0.431708 vs source 0.431708. |
| `StatCalib T24 anchor static_bias_theta` | `PASS` | UKF and Hybrid-b extension-lane anchors match T24 rounded values. |
| `StatCalib source linear_ramp/UKF` | `PASS` | TeX 0.811201 vs source 0.811201. |
| `StatCalib source linear_ramp/Hybrid-b` | `PASS` | TeX 0.787755 vs source 0.787755. |
| `StatCalib source linear_ramp/StatCalib supp.` | `PASS` | TeX 0.467083 vs source 0.467083. |
| `StatCalib T24 anchor linear_ramp` | `PASS` | UKF and Hybrid-b extension-lane anchors match T24 rounded values. |
| `StatCalib source step_sigma_theta/UKF` | `PASS` | TeX 0.811548 vs source 0.811548. |
| `StatCalib source step_sigma_theta/Hybrid-b` | `PASS` | TeX 0.788800 vs source 0.788800. |
| `StatCalib source step_sigma_theta/StatCalib supp.` | `PASS` | TeX 0.460016 vs source 0.460016. |
| `StatCalib T24 anchor step_sigma_theta` | `PASS` | UKF and Hybrid-b extension-lane anchors match T24 rounded values. |
| `StatCalib source periodic_drift/UKF` | `PASS` | TeX 0.821558 vs source 0.821558. |
| `StatCalib source periodic_drift/Hybrid-b` | `PASS` | TeX 0.806392 vs source 0.806392. |
| `StatCalib source periodic_drift/StatCalib supp.` | `PASS` | TeX 0.438751 vs source 0.438751. |
| `StatCalib T24 anchor periodic_drift` | `PASS` | UKF and Hybrid-b extension-lane anchors match T24 rounded values. |
| `Controlled oracle-affine table static_bias_theta` | `PASS` | TeX nearest/fixed/oracle/wrapped-mean/wrapped-map MSE 0.232823/0.049199/0.048095/0.047955/0.048203 vs controlled oracle-affine CSV. |
| `Controlled oracle-affine table linear_ramp_midpoint` | `PASS` | TeX nearest/fixed/oracle/wrapped-mean/wrapped-map MSE 0.230689/0.061417/0.060251/0.062341/0.063105 vs controlled oracle-affine CSV. |
| `Controlled oracle-affine table step_after_jump` | `PASS` | TeX nearest/fixed/oracle/wrapped-mean/wrapped-map MSE 0.233626/0.108216/0.092541/0.105815/0.110339 vs controlled oracle-affine CSV. |
| `Controlled oracle-affine table periodic_high_phase` | `PASS` | TeX nearest/fixed/oracle/wrapped-mean/wrapped-map MSE 0.231307/0.081757/0.076028/0.082349/0.084002 vs controlled oracle-affine CSV. |
| `Fast-path cost model Affine fast path` | `PASS` | TeX branches/mult/add/nonlinear/state 1/4/4/0/6 vs fast-path cost CSV. |
| `Fast-path cost model Wrapped MAP, 3x3 branches` | `PASS` | TeX branches/mult/add/nonlinear/state 9/49/40/0/33 vs fast-path cost CSV. |
| `Fast-path cost model Wrapped posterior mean, 3x3 branches` | `PASS` | TeX branches/mult/add/nonlinear/state 9/99/98/18/33 vs fast-path cost CSV. |
| `Fixed-point parity table static_bias_theta` | `PASS` | TeX max/p99/MSE-delta/crossing-delta/quant-sat 0.000001/0.000001/0.000000/0.000000/0.000000 vs fixed-point parity CSV. |
| `Fixed-point parity table linear_ramp_midpoint` | `PASS` | TeX max/p99/MSE-delta/crossing-delta/quant-sat 0.000001/0.000001/0.000000/0.000000/0.000000 vs fixed-point parity CSV. |
| `Fixed-point parity table step_after_jump` | `PASS` | TeX max/p99/MSE-delta/crossing-delta/quant-sat 0.000002/0.000001/0.000000/0.000000/0.000000 vs fixed-point parity CSV. |
| `Fixed-point parity table periodic_high_phase` | `PASS` | TeX max/p99/MSE-delta/crossing-delta/quant-sat 0.000001/0.000001/0.000000/0.000000/0.000000 vs fixed-point parity CSV. |
| `Runtime discipline table Constant Residual-Mu` | `PASS` | TeX commits/slow-viol/fast-viol/overflow/correction-sat 899.8/0/0.0000158/0.002582/0 vs runtime-discipline CSV. |
| `Runtime discipline table EKF` | `PASS` | TeX commits/slow-viol/fast-viol/overflow/correction-sat 899.8/0/0.0000158/0.002559/0 vs runtime-discipline CSV. |
| `Runtime discipline table Hybrid Residual-B` | `PASS` | TeX commits/slow-viol/fast-viol/overflow/correction-sat 899.9/0/0.0000158/0.002536/0 vs runtime-discipline CSV. |
| `Runtime discipline table RLS Residual-B` | `PASS` | TeX commits/slow-viol/fast-viol/overflow/correction-sat 899.8/0/0.0000158/0.002538/0 vs runtime-discipline CSV. |
| `Runtime discipline table UKF` | `PASS` | TeX commits/slow-viol/fast-viol/overflow/correction-sat 899.8/0/0.0000158/0.002571/0 vs runtime-discipline CSV. |
| `Logical-channel surrogate table static_bias_theta` | `PASS` | TeX fixed/oracle/wrapped-mean/wrapped-map p_any 0.000000/0.000000/0.000000/0.000017 vs logical-channel surrogate CSV. |
| `Logical-channel surrogate table linear_ramp_midpoint` | `PASS` | TeX fixed/oracle/wrapped-mean/wrapped-map p_any 0.000000/0.000000/0.000042/0.000183 vs logical-channel surrogate CSV. |
| `Logical-channel surrogate table step_after_jump` | `PASS` | TeX fixed/oracle/wrapped-mean/wrapped-map p_any 0.000467/0.000467/0.002417/0.006450 vs logical-channel surrogate CSV. |
| `Logical-channel surrogate table periodic_high_phase` | `PASS` | TeX fixed/oracle/wrapped-mean/wrapped-map p_any 0.000042/0.000042/0.000617/0.001867 vs logical-channel surrogate CSV. |
| `Logical-channel surrogate fidelity table static_bias_theta` | `PASS` | TeX fixed/oracle/wrapped-mean/wrapped-map F_avg_surr 1.000000/1.000000/1.000000/0.999989 vs logical-channel surrogate CSV. |
| `Logical-channel surrogate fidelity table linear_ramp_midpoint` | `PASS` | TeX fixed/oracle/wrapped-mean/wrapped-map F_avg_surr 1.000000/1.000000/0.999972/0.999878 vs logical-channel surrogate CSV. |
| `Logical-channel surrogate fidelity table step_after_jump` | `PASS` | TeX fixed/oracle/wrapped-mean/wrapped-map F_avg_surr 0.999689/0.999689/0.998389/0.995700 vs logical-channel surrogate CSV. |
| `Logical-channel surrogate fidelity table periodic_high_phase` | `PASS` | TeX fixed/oracle/wrapped-mean/wrapped-map F_avg_surr 0.999972/0.999972/0.999589/0.998756 vs logical-channel surrogate CSV. |
| `Lattice logical-channel sanity row coverage` | `PASS` | Lattice logical-channel sanity CSV carries the four expected controlled methods. |
| `Lattice logical-channel sanity boundaries` | `PASS` | Every lattice logical-channel sanity row states the finite-energy channel-fidelity non-claim boundary. |
| `Lattice logical-channel sanity table fixed_affine` | `PASS` | TeX mean/worst p_any and mean/worst F_avg_surr 0.000127/step_after_jump/0.000467/0.999915/0.999689 vs lattice logical-channel sanity CSV. |
| `Lattice logical-channel sanity table oracle_affine` | `PASS` | TeX mean/worst p_any and mean/worst F_avg_surr 0.000127/step_after_jump/0.000467/0.999915/0.999689 vs lattice logical-channel sanity CSV. |
| `Lattice logical-channel sanity table wrapped_gaussian_posterior_mean` | `PASS` | TeX mean/worst p_any and mean/worst F_avg_surr 0.000769/step_after_jump/0.002417/0.999488/0.998389 vs lattice logical-channel sanity CSV. |
| `Lattice logical-channel sanity table wrapped_gaussian_map` | `PASS` | TeX mean/worst p_any and mean/worst F_avg_surr 0.002129/step_after_jump/0.006450/0.998581/0.995700 vs lattice logical-channel sanity CSV. |
| `Finite-energy toy-channel sanity row coverage` | `PASS` | Finite-energy toy-channel sanity CSV carries three deltas across hard-nearest, fixed-affine and oracle-affine methods. |
| `Finite-energy toy-channel sanity boundaries` | `PASS` | Every finite-energy toy-channel row states the calibrated finite-energy channel-fidelity non-claim boundary. |
| `Finite-energy toy-channel table delta=0.18/hard_nearest_syndrome` | `PASS` | TeX mean/worst p_any and mean F_avg_surr 0.000148/step_after_jump/0.000517/0.999901 vs finite-energy toy-channel CSV. |
| `Finite-energy toy-channel table delta=0.18/fixed_affine` | `PASS` | TeX mean/worst p_any and mean F_avg_surr 0.000148/step_after_jump/0.000517/0.999901 vs finite-energy toy-channel CSV. |
| `Finite-energy toy-channel table delta=0.18/oracle_affine` | `PASS` | TeX mean/worst p_any and mean F_avg_surr 0.000148/step_after_jump/0.000517/0.999901 vs finite-energy toy-channel CSV. |
| `Finite-energy toy-channel table delta=0.26/hard_nearest_syndrome` | `PASS` | TeX mean/worst p_any and mean F_avg_surr 0.000160/step_after_jump/0.000475/0.999893 vs finite-energy toy-channel CSV. |
| `Finite-energy toy-channel table delta=0.26/fixed_affine` | `PASS` | TeX mean/worst p_any and mean F_avg_surr 0.000129/step_after_jump/0.000458/0.999914 vs finite-energy toy-channel CSV. |
| `Finite-energy toy-channel table delta=0.26/oracle_affine` | `PASS` | TeX mean/worst p_any and mean F_avg_surr 0.000129/step_after_jump/0.000458/0.999914 vs finite-energy toy-channel CSV. |
| `Finite-energy toy-channel table delta=0.34/hard_nearest_syndrome` | `PASS` | TeX mean/worst p_any and mean F_avg_surr 0.001002/step_after_jump/0.001500/0.999332 vs finite-energy toy-channel CSV. |
| `Finite-energy toy-channel table delta=0.34/fixed_affine` | `PASS` | TeX mean/worst p_any and mean F_avg_surr 0.000163/step_after_jump/0.000617/0.999892 vs finite-energy toy-channel CSV. |
| `Finite-energy toy-channel table delta=0.34/oracle_affine` | `PASS` | TeX mean/worst p_any and mean F_avg_surr 0.000165/step_after_jump/0.000625/0.999890 vs finite-energy toy-channel CSV. |
| `Sequence controlled baseline static_bias_theta` | `PASS` | TeX nearest/fixed/oracle/wrapped-mean/wrapped-map sequence proxy 0.000214/0.000000/0.000000/0.000000/0.000000 vs sequence controlled baseline CSV. |
| `Sequence controlled baseline linear_ramp` | `PASS` | TeX nearest/fixed/oracle/wrapped-mean/wrapped-map sequence proxy 0.000264/0.000015/0.000015/0.000158/0.000575 vs sequence controlled baseline CSV. |
| `Sequence controlled baseline step_sigma_theta` | `PASS` | TeX nearest/fixed/oracle/wrapped-mean/wrapped-map sequence proxy 0.000580/0.000310/0.000310/0.001221/0.003220 vs sequence controlled baseline CSV. |
| `Sequence controlled baseline periodic_drift` | `PASS` | TeX nearest/fixed/oracle/wrapped-mean/wrapped-map sequence proxy 0.000229/0.000005/0.000005/0.000127/0.000519 vs sequence controlled baseline CSV. |
| `Holdout drift stress table random_walk_drift` | `PASS` | TeX fixed/lagged/oracle/wrapped-mean/wrapped-map residual MSE and oracle F_avg_surr 0.078869/0.073867/0.072685/0.078740/0.080857/0.999915 vs holdout drift stress CSV. |
| `Holdout drift stress table burst_reset_drift` | `PASS` | TeX fixed/lagged/oracle/wrapped-mean/wrapped-map residual MSE and oracle F_avg_surr 0.066865/0.068622/0.062144/0.065936/0.067789/0.999780 vs holdout drift stress CSV. |
| `Holdout drift stress table faster_than_window_oscillation` | `PASS` | TeX fixed/lagged/oracle/wrapped-mean/wrapped-map residual MSE and oracle F_avg_surr 0.068354/0.068890/0.063745/0.067485/0.068764/0.999969 vs holdout drift stress CSV. |
| `Affine local-validity diagnostic row coverage` | `PASS` | Affine local-validity CSV carries four short-sequence rows and three holdout-stress rows with explicit non-claim boundaries. |
| `Affine local-validity table static_bias_theta` | `PASS` | TeX layer/gain/branch/lag/readout Short sequence/2.05/0.000000/--/local affine not dominated vs affine local-validity CSV. |
| `Affine local-validity table linear_ramp` | `PASS` | TeX layer/gain/branch/lag/readout Short sequence/3.40/0.000142/--/local affine headroom visible vs affine local-validity CSV. |
| `Affine local-validity table step_sigma_theta` | `PASS` | TeX layer/gain/branch/lag/readout Short sequence/10.00/0.000910/--/local affine headroom visible vs affine local-validity CSV. |
| `Affine local-validity table periodic_drift` | `PASS` | TeX layer/gain/branch/lag/readout Short sequence/3.43/0.000122/--/local affine headroom visible vs affine local-validity CSV. |
| `Affine local-validity table random_walk_drift` | `PASS` | TeX layer/gain/branch/lag/readout Holdout stress/7.84/0.006055/0.001182/local affine headroom visible vs affine local-validity CSV. |
| `Affine local-validity table burst_reset_drift` | `PASS` | TeX layer/gain/branch/lag/readout Holdout stress/7.06/0.003792/0.006478/stale commit can erase gain vs affine local-validity CSV. |
| `Affine local-validity table faster_than_window_oscillation` | `PASS` | TeX layer/gain/branch/lag/readout Holdout stress/6.74/0.003740/0.005144/stale commit can erase gain vs affine local-validity CSV. |
| `Commit-lag sweep row coverage` | `PASS` | Commit-lag sweep CSV carries three holdout scenarios across six simulation-step lag settings with a fixed 64-step commit interval. |
| `Commit-lag sweep non-claim boundary` | `PASS` | Every commit-lag sweep row states that it is simulation-step diagnostic data, not measured hardware latency or trained-branch holdout proof. |
| `Commit-lag sweep table random_walk_drift` | `PASS` | TeX lag-0/8/16/32/64/128 residual MSE values 0.073831/0.073976/0.074130/0.074410/0.075173/0.076093 vs commit-lag sweep CSV. |
| `Commit-lag sweep table burst_reset_drift` | `PASS` | TeX lag-0/8/16/32/64/128 residual MSE values 0.067454/0.068193/0.068657/0.068694/0.068583/0.066913 vs commit-lag sweep CSV. |
| `Commit-lag sweep table faster_than_window_oscillation` | `PASS` | TeX lag-0/8/16/32/64/128 residual MSE values 0.069251/0.069406/0.068989/0.069039/0.068842/0.068249 vs commit-lag sweep CSV. |
| `Metric readiness row coverage` | `PASS` | TeX metric-readiness table and CSV carry the same five metric axes. |
| `Metric readiness table Logical-error proxy` | `PASS` | Metric-readiness TeX row is backed by the CSV axis and has non-empty current metric, supported statement and missing-evidence cells. |
| `Metric readiness table Logical-channel fidelity` | `PASS` | Metric-readiness TeX row is backed by the CSV axis and has non-empty current metric, supported statement and missing-evidence cells. |
| `Metric readiness table Drift robustness` | `PASS` | Metric-readiness TeX row is backed by the CSV axis and has non-empty current metric, supported statement and missing-evidence cells. |
| `Metric readiness table Fast-path cost and latency` | `PASS` | Metric-readiness TeX row is backed by the CSV axis and has non-empty current metric, supported statement and missing-evidence cells. |
| `Metric readiness table Hardware-facing validation` | `PASS` | Metric-readiness TeX row is backed by the CSV axis and has non-empty current metric, supported statement and missing-evidence cells. |
| `Literature metric crosswalk row coverage` | `PASS` | Literature metric crosswalk carries 23 rows across the six external-comparison axes. |
| `Literature metric crosswalk active citations` | `PASS` | Every citation_key in the literature crosswalk appears in the active submission-draft citation surface. |
| `Literature metric crosswalk boundaries` | `PASS` | Every literature crosswalk row has a reported metric, source anchor, manuscript use and explicit non-claim boundary. |
| `Literature metric crosswalk anchor policies` | `PASS` | Every literature crosswalk row records anchor strength, manuscript number policy and final pinning follow-up. |
| `Literature metric crosswalk source anchors` | `PASS` | Every literature crosswalk row points back to a local literature-card anchor and either a figure/table/equation anchor or an explicit card-level/public-text limitation. |
| `Literature metric crosswalk hardware pinning` | `PASS` | Hardware comparison rows include a source anchor and either figure/table checked status or page/table/figure pinning before strong per-value claims. |
| `Closest-work positioning row coverage` | `PASS` | Closest-work positioning CSV and TeX table carry the same five adjacent-work families. |
| `Closest-work positioning active citations` | `PASS` | Every citation key used by the closest-work positioning CSV appears in the active manuscript citation surface. |
| `Closest-work positioning boundaries` | `PASS` | Every closest-work row has an explicit non-claim boundary and a local literature-card source anchor. |
| `Closest-work positioning table Analog and surface-GKP decoding` | `PASS` | TeX closest-work row aligns with the generated CSV family, metric standard, distinction and boundary. |
| `Closest-work positioning table Calibration-aware and learned QEC decoders` | `PASS` | TeX closest-work row aligns with the generated CSV family, metric standard, distinction and boundary. |
| `Closest-work positioning table Finite-energy logical-channel analyses` | `PASS` | TeX closest-work row aligns with the generated CSV family, metric standard, distinction and boundary. |
| `Closest-work positioning table Runtime pre-decoders and calibration-conditioned neural modules` | `PASS` | TeX closest-work row aligns with the generated CSV family, metric standard, distinction and boundary. |
| `Closest-work positioning table Real-time FPGA and hardware-tailored decoders` | `PASS` | TeX closest-work row aligns with the generated CSV family, metric standard, distinction and boundary. |
| `Source-data coverage matrix row coverage` | `PASS` | Source-data coverage matrix carries the expected manuscript coverage groups in CSV and JSON. |
| `Source-data coverage matrix boundaries` | `PASS` | Every source-data coverage matrix row has a status, source-file surface and explicit non-claim boundary. |
| `Source-data coverage matrix TeX labels` | `PASS` | TeX source-data coverage table carries the same expected coverage-group labels as the CSV. |
| `Benchmark expansion protocol row coverage` | `PASS` | Benchmark expansion protocol carries four Phase A scenario rows plus Phase A reporting, Phase B holdout and Phase C provenance rows. |
| `Benchmark expansion repeat budget` | `PASS` | Every Phase A benchmark-expansion scenario has a minimum repeat budget, target repeat budget and non-claim boundary. |
| `Benchmark expansion holdout families` | `PASS` | Phase B predeclares random-walk, burst/reset and faster-than-window holdout drift families without treating them as current results. |
| `Phase A repeat plan coverage` | `PASS` | Phase A repeat plan carries three formal chunks and one smoke feasibility row for each predeclared scenario. |
| `Phase A repeat plan non-claim boundary` | `PASS` | Every Phase A plan row carries an explicit non-claim or not-yet-upgraded boundary. |
| `Phase A repeat summary boundary` | `PASS` | Any completed Phase A summary rows retain a non-claim boundary; an empty summary is allowed before Phase A runs exist. |
| `Phase A paired interval source boundary` | `PASS` | Phase A paired interval source data covers completed formal scenario(s): linear_ramp, static_bias_theta; the broader gate remains blocked. |
| `Phase A paired interval TeX values` | `PASS` | TeX formal Phase A interval table matches the generated paired-interval CSV for all completed scenarios. |
| `Phase A upgrade gate coverage` | `PASS` | Phase A upgrade gate separates descriptive, short-run, formal repeat, holdout and hardware evidence classes. |
| `Phase A upgrade gate non-claim boundary` | `PASS` | Every Phase A upgrade-gate row states a forbidden inference. |
| `Phase A upgrade gate TeX labels` | `PASS` | TeX Phase A upgrade-gate table carries all evidence-class labels from the CSV. |
| `Runner smoke pair row coverage` | `PASS` | Runner smoke-pair source data carries exactly one UKF row and one Hybrid row for one static-bias scenario. |
| `Runner smoke pair non-claim boundary` | `PASS` | Runner smoke-pair rows are explicitly bounded as one-repeat feasibility data, not expanded benchmark evidence. |
| `Runner smoke matrix row coverage` | `PASS` | Runner smoke matrix covers all four predeclared scenarios in CSV and TeX. |
| `Runner smoke matrix table static_bias_theta` | `PASS` | TeX UKF/Hybrid/delta/relative/positive-pair values match the runner smoke matrix CSV for static_bias_theta. |
| `Runner smoke matrix table linear_ramp` | `PASS` | TeX UKF/Hybrid/delta/relative/positive-pair values match the runner smoke matrix CSV for linear_ramp. |
| `Runner smoke matrix table step_sigma_theta` | `PASS` | TeX UKF/Hybrid/delta/relative/positive-pair values match the runner smoke matrix CSV for step_sigma_theta. |
| `Runner smoke matrix table periodic_drift` | `PASS` | TeX UKF/Hybrid/delta/relative/positive-pair values match the runner smoke matrix CSV for periodic_drift. |
| `Runner smoke matrix non-claim boundary` | `PASS` | Runner smoke matrix rows are explicitly bounded as all-scenario smoke feasibility data, not expanded benchmark evidence. |
| `Figure manifest source-data list` | `PASS` | figure_manifest.json lists the source CSV files used by Fig. 2-5 and paired Fig. 2 deltas. |
| `Figure manifest outputs exist` | `PASS` | All figure_manifest.json PDF outputs are present. |
| `Source-data manifest row coverage` | `PASS` | CSV and JSON source-data manifests carry the same 97 manuscript-facing source/script rows. |
| `Source-data manifest file paths` | `PASS` | Every source_path listed in the source-data manifest exists in the current checkout. |
| `Source-data manifest hashes` | `PASS` | Every source-data manifest SHA-256 matches the current file content. |
| `T24 artifact path stratification` | `PASS` | Hybrid rows share one model artifact path; non-hybrid baseline rows have no model artifact path. |
| `Row-level provenance coverage` | `PASS` | Row-level provenance manifest covers four scenarios, five modes and two repeats. |
| `Row-level provenance hashes` | `PASS` | Every row-level provenance entry has source summary, launch, comparison, config, runner and repeat summary/status hashes. |
| `Row-level provenance non-claim boundary` | `PASS` | Every row-level provenance entry is explicitly bounded as source trace rather than new evidence. |
| `Hybrid model artifact hash` | `PASS` | The shared hybrid .npz artifact exists and has a SHA256 hash recorded in this audit. |

## 已知限制

- The helper checks selected manuscript tables and the Fig. 5 validation-contract source summary, not every table in the draft.
- Supplementary StatCalib values are checked against the current source CSV and TeX table; this helper does not re-open FR8 artifacts.
- The controlled oracle-affine and wrapped-Gaussian table is checked against its generated CSV; the helper does not make that CSV a formal benchmark, CI analysis or hardware result.
- The sequence-controlled baseline table is checked against its generated CSV; the helper does not make that CSV a formal benchmark, CI analysis, holdout drift run or hardware result.
- The GKP boundary-sensitivity table is checked against an analytical Gaussian residual-boundary CSV; the helper does not turn it into finite-energy GKP logical-channel simulation, process tomography, hardware logical error rate or benchmark evidence.
- The fast-path cost table is checked against an analytical count CSV; the helper does not turn it into FPGA synthesis, timing closure, power/resource measurement or hardware validation.
- The fixed-point parity table is checked against a software-emulation CSV; the helper does not turn it into FPGA synthesis, timing closure, power/resource measurement, source-vs-board agreement or hardware validation.
- The runtime-discipline table is checked against software-in-the-loop counters; the helper does not turn it into board commit latency, hardware reliability, rollback proof, source-vs-board agreement or FPGA timing/resource evidence.
- The logical-channel surrogate, lattice sanity and finite-energy toy-channel tables are checked against residual-boundary and toy measurement-channel CSVs; the helper does not turn them into calibrated finite-energy GKP logical-channel tomography, process fidelity or hardware fidelity.
- The holdout drift stress table is checked against a controlled stress-test CSV; the helper does not turn it into a formal expanded benchmark, confidence-interval analysis, trained-branch generalization proof or hardware validation.
- The affine local-validity diagnostic table is checked against a derived CSV; the helper does not turn it into a formal nearest-lattice or wrapped-decoder benchmark, inferential analysis, trained-branch holdout proof or hardware validation.
- The commit-lag sweep table is checked against a controlled simulation-step CSV; the helper does not turn it into measured FPGA latency, source-vs-board agreement, trained-branch holdout proof or hardware validation.
- The paired-uncertainty table is checked against a descriptive resampling CSV; the helper does not turn it into a confidence interval, standard error, p-value, statistical significance claim or robustness proof.
- The LER advantage-margin table is checked against a descriptive source-data CSV; the helper does not turn delta/max SD into a confidence interval, standard error, p-value, statistical significance claim, expanded benchmark or hardware evidence.
- The metric-readiness table is checked against a manuscript-positioning CSV; the helper does not estimate channel fidelity, hardware latency or statistical significance.
- The literature metric crosswalk checks manuscript-positioning and anchor-policy coverage only; it does not normalize prior results or convert them into baselines for this study.
- The closest-work positioning table checks adjacent-work family coverage and non-claim boundaries only; it does not normalize external metrics or convert them into this manuscript's results.
- The source-data coverage matrix checks coverage classification only; it does not make unchecked or planning surfaces into result evidence.
- The benchmark-expansion protocol checks planning coverage only; it does not run the repeat-expanded benchmark, establish holdout robustness or provide CI/p-values.
- The Phase A repeat plan checks command-shape and planning coverage only; it does not run benchmarks, establish holdout robustness or provide CI/p-values.
- The Phase A repeat summary checks completed run summaries only when present; short-run rows remain feasibility-only and are not manuscript performance evidence.
- The Phase A upgrade gate checks wording boundaries only; it does not run benchmarks, compute intervals or provide hardware validation.
- The runner smoke-pair rows check feasibility/source traceability only; they are one-scenario, one-repeat smoke data and are not used as main-text performance evidence.
- The runner smoke-matrix rows check all-scenario short-run feasibility/source traceability only; they are not the main benchmark, an expanded benchmark, confidence-interval evidence, holdout robustness or hardware evidence.
- The row-level provenance manifest checks scenario/mode/repeat source traceability for the existing software-HIL rows; it is not recursive historical run-directory hash closure and excludes hil_events.json hashes.
- The source-data manifest checks file-level hashes for manuscript-facing source files and scripts, not recursive historical run-directory hash closure.
- The audit does not provide confidence intervals, p-values, holdout drift families or repeated-run closure.
- The audit does not validate real-board, default-environment .tflite or deployment behavior.
