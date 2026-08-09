# Phase 5 · Milestone 5.5

本页索引 `10` 个冻结机器证据。文件仍保留原路径，以维持自哈希和 release-pin；优先阅读“人类文档”列。

| Task | 机器证据 | 内容概览 | 人类文档 |
| --- | --- | --- | --- |
| `T5.5.1` | [t5_5_1_bit_accurate_hardware_reference.json](../../../t5_5_1_bit_accurate_hardware_reference.json) · 38.2 KiB | task_id=T5.5.1；protocol_id=PACKED-WORD-TRUE-PIPELINE-ATOMIC-BANK-RTL-GOLDEN-V1；status=PASS；verdict=BIT_ACCURATE_PYTHON_RTL_GOLDEN_FROZEN_HARDWARE_UNMEASURED | [bit_accurate_hardware_reference.md](../../../bit_accurate_hardware_reference.md) |
| `T5.5.2` | [t5_5_2_target_device_synthesis.json](../../../t5_5_2_target_device_synthesis.json) · 25.9 KiB | task_id=T5.5.2；status=PASS；verdict=TARGET_DEVICE_POST_ROUTE_ESTIMATE_PASSES_27MHZ_NOT_BOARD_MEASURED | [target_device_synthesis.md](../../../target_device_synthesis.md) |
| `T5.5.2` | [t5_5_2_target_device_synthesis_source_data.csv](../../../t5_5_2_target_device_synthesis_source_data.csv) · 11.0 KiB | 195 rows；7 columns（section, seed, name, value, …） | [target_device_synthesis.md](../../../target_device_synthesis.md) |
| `T5.5.3` | [t5_5_3_precision_resource_pareto.json](../../../t5_5_3_precision_resource_pareto.json) · 123.8 KiB | task_id=T5.5.3；status=PASS；verdict=SELECT_P10_A8_Q9_12_K4_REFERENCE_STATE4_SERIAL_DSP_POST_ROUTE_PASS_NOT_BOARD_MEASURED | [precision_resource_performance_pareto.md](../../../precision_resource_performance_pareto.md) |
| `T5.5.3` | [t5_5_3_precision_resource_pareto_source_data.csv](../../../t5_5_3_precision_resource_pareto_source_data.csv) · 18.4 KiB | 108 rows；25 columns（candidate_id, precision_id, topk_k, student_dimension, …） | [target_device_synthesis.md](../../../target_device_synthesis.md) |
| `T5.5.3` | [t5_5_3_student_rtl_equivalence.json](../../../t5_5_3_student_rtl_equivalence.json) · 1.9 KiB | task_id=T5.5.3；status=PASS | [precision_resource_performance_pareto.md](../../../precision_resource_performance_pareto.md) |
| `T5.5.4` | [t5_5_4_gru_student_hardware_feasibility.json](../../../t5_5_4_gru_student_hardware_feasibility.json) · 40.6 KiB | task_id=T5.5.4；status=PASS；verdict=DISTILLED_STUDENT_ONLY_QUANTIZED_GRU_DROPPED_FULL_GRU_OFFLINE_TEACHER | [gru_student_hardware_feasibility.md](../../../gru_student_hardware_feasibility.md) |
| `T5.5.4` | [t5_5_4_gru_student_hardware_feasibility_source_data.csv](../../../t5_5_4_gru_student_hardware_feasibility_source_data.csv) · 965 B | 4 rows；20 columns（candidate_id, role, functional_model, stored_parameters, …） | [target_device_synthesis.md](../../../target_device_synthesis.md) |
| `T5.5.4` | [t5_5_4_quantized_gru_workload_trace.csv](../../../t5_5_4_quantized_gru_workload_trace.csv) · 109 B | 1 rows；6 columns（cycles_after_start, weight_macs_completed, biases_consumed, done, …） | [T5.5.4_gru_student_hardware_feasibility.md](../../../new_tasks/T5.5.4_gru_student_hardware_feasibility.md) |
| `T5.5.4` | [t5_5_4_toolchain_provenance.json](../../../t5_5_4_toolchain_provenance.json) · 5.8 KiB | JSON 机器证据 | [T5.5.4_gru_student_hardware_feasibility.md](../../../new_tasks/T5.5.4_gru_student_hardware_feasibility.md) |
