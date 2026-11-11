# Source index

Imports are listed for navigation; optional and external modules are included.

| Script | Imports |
|---|---|
| `pipeline/battery_degradation_model.py` | __future__, dataclasses, math, pandas, typing |
| `pipeline/battery_degradation_model_pinn.py` | __future__, dataclasses, numpy, pandas, pathlib, random, torch, typing |
| `pipeline/build_ev_dataset.py` | __future__, json, math, pandas, pathlib, typing |
| `pipeline/build_grid_dataset.py` | __future__, dataclasses, numpy, pandas, pathlib, re, typing |
| `pipeline/build_joint_scenarios.py` | __future__, dataclasses, numpy, pandas, pathlib, train_ev_generator, train_grid_generator, typing |
| `pipeline/build_llm_timing_scenario_pilot.py` | argparse, hashlib, itertools, json, numpy, pandas, pathlib, scipy, shutil |
| `pipeline/build_optimization_inputs.py` | __future__, dataclasses, json, numpy, pandas, pathlib, typing |
| `pipeline/build_optimization_inputs_flexible.py` | __future__, dataclasses, json, numpy, pandas, pathlib, typing |
| `pipeline/build_plot_data_from_existing_results.py` | json, numpy, pandas, pathlib, typing |
| `pipeline/build_stress_scenarios.py` | __future__, dataclasses, json, numpy, pandas, pathlib, typing |
| `pipeline/check_aging_derating_boundaries.py` | argparse, itertools, json, lock_recovery_holdout, numpy, pandas, pathlib, run_directional_pinn_validation, run_llm_timing_pilot, screen_aging_derating_overlap, screen_recovery_departure, torch |
| `pipeline/check_low_confidence_risk_boundary.py` | argparse, dataclasses, itertools, json, lock_recovery_holdout, numpy, pandas, pathlib, run_joint_context_warm_start, run_llm_timing_pilot, screen_adaptive_recovery_reserve, screen_aging_derating_overlap, screen_forecast_adaptive_reserve, screen_low_confidence_recovery, torch |
| `pipeline/compare_debt_recovery.py` | argparse, check_low_confidence_risk_boundary, diagnose_low_confidence_reversals, json, lock_recovery_holdout, pandas, pathlib, run_joint_context_warm_start, run_llm_timing_pilot, screen_adaptive_recovery_reserve, screen_aging_derating_overlap, screen_forecast_adaptive_reserve, torch |
| `pipeline/compare_one_shot_vs_receding_horizon.py` | dataclasses, matplotlib, numpy, pandas, pathlib |
| `pipeline/compare_stress_modes.py` | dataclasses, json, matplotlib, numpy, pandas, pathlib, typing |
| `pipeline/compare_stress_modes_pretty.py` | dataclasses, json, matplotlib, numpy, pandas, pathlib, typing |
| `pipeline/compare_with_without_degradation.py` | matplotlib, numpy, pandas, pathlib |
| `pipeline/debt_aware_recovery.py` | nonanticipative_scheduler, numpy, run_directional_pinn_validation, scipy |
| `pipeline/diagnose_low_confidence_reversals.py` | argparse, check_low_confidence_risk_boundary, itertools, json, lock_recovery_holdout, numpy, pandas, pathlib, run_directional_pinn_validation, run_llm_timing_pilot, screen_adaptive_recovery_reserve, screen_forecast_adaptive_reserve, torch |
| `pipeline/ev_generator_learned_vae.py` | __future__, dataclasses, json, matplotlib, numpy, pandas, pathlib, pickle, sklearn, torch, typing |
| `pipeline/evaluate_mean_cvar_objective.py` | dataclasses, json, matplotlib, numpy, pandas, pathlib, typing |
| `pipeline/lock_peak_physical_development.py` | argparse, build_optimization_inputs, build_stress_scenarios, lock_recovery_holdout, numpy, pandas, pathlib, run_directional_pinn_validation, run_llm_timing_pilot, scipy, torch |
| `pipeline/lock_recovery_holdout.py` | argparse, build_joint_scenarios_f, build_optimization_inputs, hashlib, json, nonanticipative_scheduler, numpy, pandas, pathlib, run_directional_pinn_validation, run_llm_timing_pilot, screen_recovery_departure, sys, torch, train_ev_generator_f, train_grid_generator_f, types |
| `pipeline/nonanticipative_scheduler.py` | battery_degradation_model, dataclasses, numpy, scipy |
| `pipeline/plot_four_scenarios.py` | matplotlib, numpy, pandas, pathlib |
| `pipeline/plot_stress_figures.py` | matplotlib, numpy, os, pandas, pathlib, typing |
| `pipeline/replay_archived_budget8.py` | argparse, build_optimization_inputs, build_stress_scenarios, concurrent, json, lock_recovery_holdout, numpy, pandas, pathlib, rerun_archived_gpt6_benchmark, run_directional_pinn_validation, torch |
| `pipeline/replay_peak_physical_holdout.py` | argparse, json, numpy, pandas, pathlib, run_gemini_feedback_pair, torch |
| `pipeline/replay_peak_recovery_curve.py` | argparse, json, numpy, pandas, pathlib, replay_peak_physical_holdout, run_directional_pinn_validation, run_gemini_feedback_pair, torch |
| `pipeline/request_joint_context_seeds.py` | json, os, pathlib, sys, urllib |
| `pipeline/request_llm_timing_pilot_seeds.py` | json, os, pathlib, sys, time, urllib |
| `pipeline/request_real_llm_warmstarts.py` | json, os, pathlib, sys, time, urllib |
| `pipeline/rerun_archived_gpt6_benchmark.py` | argparse, concurrent, contextlib, importlib, json, llm_top_config_gemini, numpy, os, pandas, pathlib, re, subprocess, sys, tempfile, torch |
| `pipeline/run_aging_capacity_diagnostics.py` | argparse, dataclasses, nonanticipative_scheduler, numpy, pandas, pathlib, run_section_iv_replay |
| `pipeline/run_binding_archive_recovery.py` | argparse, json, nonanticipative_scheduler, numpy, pandas, pathlib, run_paper_archive_recovery, run_section_iv_replay, scipy |
| `pipeline/run_calibrated_recovery_floor.py` | argparse, numpy, pandas, pathlib, run_fine_grid_fallback_diagnostics, run_full_service_recovery_pilot, run_minimal_flex_pilot, run_section_iv_replay, run_service_fallback_pilot, unittest |
| `pipeline/run_dense_binding_archive.py` | argparse, json, nonanticipative_scheduler, numpy, pandas, pathlib, run_binding_archive_recovery, run_paper_archive_recovery, run_section_iv_replay, scipy |
| `pipeline/run_directional_pinn_ablation.py` | argparse, hashlib, nonanticipative_scheduler, numpy, pandas, pathlib, run_directional_pinn_validation, torch |
| `pipeline/run_directional_pinn_robustness.py` | argparse, itertools, nonanticipative_scheduler, numpy, pandas, pathlib, run_directional_pinn_validation, torch |
| `pipeline/run_directional_pinn_validation.py` | argparse, hashlib, nonanticipative_scheduler, numpy, pandas, pathlib, scipy, sys, torch, train_battery_degradation_pinn_temporal_dual_head |
| `pipeline/run_fine_grid_fallback_diagnostics.py` | argparse, nonanticipative_scheduler, numpy, pandas, pathlib, run_full_service_recovery_pilot, run_minimal_flex_pilot, run_section_iv_replay, run_service_fallback_pilot, unittest |
| `pipeline/run_fixed_service_ablation.py` | argparse, nonanticipative_scheduler, numpy, pandas, pathlib, run_section_iv_replay |
| `pipeline/run_frozen_recovery_validation.py` | argparse, itertools, numpy, pandas, pathlib, run_calibrated_recovery_floor, run_fine_grid_fallback_diagnostics, run_full_service_recovery_pilot, run_minimal_flex_pilot, run_section_iv_replay, run_service_fallback_pilot, unittest |
| `pipeline/run_full_service_recovery_pilot.py` | argparse, concurrent, itertools, nonanticipative_scheduler, numpy, pandas, pathlib, run_minimal_flex_pilot, run_section_iv_replay, scipy, unittest |
| `pipeline/run_gemini_feedback_pair.py` | argparse, json, lock_recovery_holdout, numpy, os, pandas, pathlib, re, run_directional_pinn_validation, run_llm_timing_pilot, scipy, sklearn, subprocess, tempfile, time, torch, urllib, warnings |
| `pipeline/run_joint_context_warm_start.py` | argparse, json, lock_recovery_holdout, numpy, pandas, pathlib, run_llm_timing_pilot, screen_adaptive_recovery_reserve, screen_aging_derating_overlap, screen_forecast_adaptive_reserve, torch |
| `pipeline/run_lambda_sweep_stochastic_optimization.py` | dataclasses, json, matplotlib, numpy, pandas, pathlib, typing |
| `pipeline/run_lambda_sweep_stochastic_optimization2.py` | dataclasses, json, matplotlib, numpy, pandas, pathlib, typing |
| `pipeline/run_legacy_tail_sensitivity.py` | argparse, build_optimization_inputs_flexible, build_stress_scenarios, concurrent, nonanticipative_scheduler, numpy, pandas, pathlib, run_section_iv_replay, train_ev_generator, train_grid_generator |
| `pipeline/run_llm_timing_pilot.py` | argparse, debt_aware_recovery, importlib, json, nonanticipative_scheduler, numpy, pandas, pathlib, run_directional_pinn_validation, sys, torch |
| `pipeline/run_locked_holdout_warm_starts.py` | argparse, importlib, json, lock_recovery_holdout, numpy, pandas, pathlib, run_directional_pinn_validation, run_llm_timing_pilot, scipy, screen_recovery_departure, torch |
| `pipeline/run_minimal_flex_pilot.py` | argparse, concurrent, itertools, nonanticipative_scheduler, numpy, pandas, pathlib, run_aging_capacity_diagnostics, run_section_iv_replay |
| `pipeline/run_minimum_archive_recovery_additions.py` | argparse, json, numpy, pandas, pathlib, run_binding_archive_recovery, run_paper_archive_recovery, run_section_iv_replay, run_visible_deadline_archive |
| `pipeline/run_paper_archive_recovery.py` | argparse, concurrent, itertools, json, nonanticipative_scheduler, numpy, pandas, pathlib, run_full_service_recovery_pilot, run_section_iv_replay, unittest |
| `pipeline/run_peak_exploration_pair.py` | argparse, json, numpy, pandas, pathlib, run_gemini_feedback_pair |
| `pipeline/run_real_llm_warmstart_pair.py` | argparse, contextlib, hashlib, importlib, json, numpy, os, pandas, pathlib, sys, unittest |
| `pipeline/run_receding_horizon_optimization.py` | dataclasses, json, numpy, pandas, pathlib, typing |
| `pipeline/run_scenario_optimization.py` | __future__, dataclasses, json, matplotlib, numpy, pandas, pathlib, typing |
| `pipeline/run_scenario_optimization2.py` | dataclasses, json, matplotlib, numpy, pandas, pathlib, typing |
| `pipeline/run_section_iv_replay.py` | argparse, battery_degradation_model, concurrent, itertools, matplotlib, nonanticipative_scheduler, numpy, pandas, pathlib |
| `pipeline/run_service_fallback_pilot.py` | argparse, concurrent, nonanticipative_scheduler, numpy, pandas, pathlib, run_full_service_recovery_pilot, run_minimal_flex_pilot, run_section_iv_replay, unittest |
| `pipeline/run_stochastic_scenario_optimization_flexible.py` | dataclasses, json, matplotlib, numpy, pandas, pathlib, typing |
| `pipeline/run_stochastic_scenario_optimization_with_degradation.py` | battery_degradation_model, dataclasses, json, matplotlib, numpy, pandas, pathlib, typing |
| `pipeline/run_stochastic_scenario_optimization_with_degradation_pinn.py` | __future__, battery_degradation_model, dataclasses, json, numpy, pandas, pathlib, typing |
| `pipeline/run_visible_deadline_archive.py` | argparse, json, nonanticipative_scheduler, numpy, pandas, pathlib, run_binding_archive_recovery, run_paper_archive_recovery, run_section_iv_replay, scipy |
| `pipeline/screen_adaptive_recovery_reserve.py` | argparse, dataclasses, itertools, json, lock_recovery_holdout, numpy, pandas, pathlib, run_directional_pinn_validation, run_llm_timing_pilot, screen_aging_derating_overlap, screen_recovery_departure, torch |
| `pipeline/screen_aging_derating_overlap.py` | argparse, dataclasses, itertools, json, lock_recovery_holdout, numpy, pandas, pathlib, run_directional_pinn_validation, run_llm_timing_pilot, screen_recovery_departure, torch |
| `pipeline/screen_departure_price.py` | argparse, itertools, json, lock_recovery_holdout, numpy, pandas, pathlib, run_directional_pinn_validation, run_llm_timing_pilot, screen_recovery_departure, torch |
| `pipeline/screen_forecast_adaptive_reserve.py` | argparse, itertools, json, lock_recovery_holdout, numpy, pandas, pathlib, run_directional_pinn_validation, run_llm_timing_pilot, screen_adaptive_recovery_reserve, screen_aging_derating_overlap, torch |
| `pipeline/screen_low_confidence_recovery.py` | argparse, itertools, json, lock_recovery_holdout, numpy, pandas, pathlib, run_joint_context_warm_start, run_llm_timing_pilot, screen_adaptive_recovery_reserve, screen_aging_derating_overlap, screen_forecast_adaptive_reserve, screen_recovery_departure, torch |
| `pipeline/screen_recovery_departure.py` | argparse, hashlib, itertools, numpy, pandas, pathlib, run_directional_pinn_validation, run_llm_timing_pilot |
| `pipeline/simulate_repeated_operation_soh.py` | battery_degradation_model, dataclasses, matplotlib, numpy, pandas, pathlib, typing |
| `pipeline/train_conditional_ev_vae.py` | __future__, dataclasses, json, matplotlib, numpy, pandas, pathlib, pickle, sklearn, torch, typing |
| `pipeline/train_ev_generator.py` | __future__, dataclasses, matplotlib, numpy, pandas, pathlib, pickle, sklearn, typing |
| `pipeline/train_ev_generator2.py` | __future__, dataclasses, matplotlib, numpy, pandas, pathlib, pickle, sklearn, typing |
| `pipeline/train_grid_generator.py` | __future__, dataclasses, matplotlib, numpy, pandas, pathlib, pickle, sklearn, typing |
| `pipeline/validate_fresh_evening_peak_feedback.py` | argparse, build_optimization_inputs, build_stress_scenarios, json, lock_recovery_holdout, numpy, pandas, pathlib, run_directional_pinn_validation, run_gemini_feedback_pair, run_llm_timing_pilot, torch |
| `pipeline/ev_grid_world_model/ev_generator_world_model.py` | __future__, numpy, pandas, pathlib, torch, typing, world_model_common |
| `pipeline/ev_grid_world_model/example_usage.py` | ev_generator_world_model, grid_generator_world_model, joint_scenarios_world_model, numpy, pandas, pathlib, world_model_common |
| `pipeline/ev_grid_world_model/grid_generator_world_model.py` | __future__, numpy, pandas, pathlib, torch, typing, world_model_common |
| `pipeline/ev_grid_world_model/joint_scenarios_world_model.py` | __future__, dataclasses, ev_generator_world_model, grid_generator_world_model, numpy, pandas, typing |
| `pipeline/ev_grid_world_model/world_model_common.py` | __future__, dataclasses, json, math, numpy, pandas, pathlib, random, torch, typing |
| `pipeline/final_world_model_exact_matched/build_joint_scenarios_f.py` | __future__, dataclasses, numpy, pandas, pathlib, train_ev_generator, train_grid_generator, typing |
| `pipeline/final_world_model_exact_matched/train_ev_generator_f.py` | __future__, dataclasses, matplotlib, numpy, pandas, pathlib, pickle, torch, typing, world_model_common |
| `pipeline/final_world_model_exact_matched/train_grid_generator_f.py` | __future__, dataclasses, matplotlib, numpy, pandas, pathlib, pickle, sklearn, torch, typing, world_model_common |
| `pipeline/final_world_model_exact_matched/validate_available_components.py` | argparse, json, numpy, pandas, pathlib, scipy, torch, train_ev_generator_f, train_grid_generator_f, world_model_common |
| `pipeline/final_world_model_exact_matched/world_model_common.py` | __future__, dataclasses, json, numpy, pathlib, random, torch, typing |
| `pipeline/final_aligned_world_model_code/build_joint_scenarios.py` | __future__, dataclasses, numpy, pandas, pathlib, train_ev_generator, train_grid_generator, typing |
| `pipeline/final_aligned_world_model_code/train_ev_generator.py` | __future__, dataclasses, matplotlib, numpy, pandas, pathlib, pickle, torch, typing, world_model_common |
| `pipeline/final_aligned_world_model_code/train_grid_generator.py` | __future__, dataclasses, matplotlib, numpy, pandas, pathlib, pickle, sklearn, torch, typing, world_model_common |
| `pipeline/final_aligned_world_model_code/world_model_common.py` | __future__, dataclasses, json, numpy, pandas, pathlib, random, torch, typing |
| `legacy/data_processing.py` |  |
| `legacy/EV-battery-large system1.py` | dataclasses, matplotlib, numpy, pandas, pathlib |
| `legacy/EV-battery-large system2.py` | dataclasses, matplotlib, numpy, pandas, pathlib |
| `legacy/EV-battery-large system3.py` | dataclasses, matplotlib, numpy, pandas, pathlib |
| `legacy/EV-battery-large system4.py` | dataclasses, matplotlib, numpy, pandas, pathlib |
| `legacy/EV-battery-large system5.py` | dataclasses, matplotlib, numpy, pandas, pathlib |
| `legacy/EV-battery-small system.py` | dataclasses, matplotlib, numpy, pandas, pathlib |
| `legacy/EV-battery-small system2.py` | dataclasses, matplotlib, numpy, pandas, pathlib |
| `legacy/EV-battery-small system3.py` | dataclasses, matplotlib, numpy, pandas, pathlib |
| `legacy/plot_us_maps.py` | numpy, pandas, pathlib, plotly |
| `legacy/plot_us_maps_CDIVMSAR.py` | numpy, pandas, pathlib, plotly |
| `legacy/prepare_eia930_region_profiles.py` | numpy, os, pandas, requests |
| `legacy/prepare_nhts_us.py` | argparse, matplotlib, numpy, pandas, pathlib |
| `legacy/prepare_nhts_us_CDIVMSAR.py` | argparse, matplotlib, numpy, pandas, pathlib |
| `legacy/prepare_real_sessions.py` | __future__, argparse, json, matplotlib, numpy, pandas, pathlib, typing |
| `legacy/region.py` | numpy, pandas |
| `tools/download_nyiso_ev_data.py` | __future__, argparse, bs4, dataclasses, datetime, json, os, pathlib, re, requests, sys, time, typing, urllib |
| `figures/plot_fig6_directional_ablation.py` | matplotlib, numpy, pandas, sys |
