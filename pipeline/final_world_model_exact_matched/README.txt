
PINN context enhancement for EV generator, Grid generator, and build_joint_scenarios
===================================================================================

Files
-----
- train_ev_generator.py
- train_grid_generator.py
- build_joint_scenarios.py

What was added
--------------
1) EV session-level PINN context
   - source_code, is_weekend_num, hour_bin, hour_sin, hour_cos, month_sin, month_cos
   - risk_lambda, risk_peak_hour, risk_month_end
   - battery_capacity_kwh, max_charge_power_kw
   - estimated_init_soc, estimated_target_soc
   - requested_avg_charge_power_kw, requested_c_rate
   - delta_soc_requested, required_energy_remaining_kwh
   - feasible_charge_hours, charging_slack_hours
   - temperature_c
   - placeholders:
       charge_power_kw, soc, soh, cumulative_throughput_kwh, delta_soh_true, soh_next

2) Grid hourly PINN context
   - temperature_c proxy if weather is unavailable
   - temperature_is_proxy
   - is_peak_hour
   - high_price_flag, high_congestion_flag, high_load_flag
   - grid_risk_score
   - price_emission_score
   - placeholders:
       charge_power_kw, soc, soh, cumulative_throughput_kwh, delta_soh_true, soh_next

3) Joint-scenario exports for PINN training
   For each scenario:
   - pinn_session_context.csv
   - pinn_hourly_context.csv

   For all scenarios:
   - pinn_session_context_all.csv
   - pinn_hourly_context_all.csv

Backward compatibility
----------------------
The public interfaces used by the existing pipeline are preserved:
- EVGenerator.load(...)
- EVGenerator.sample_sessions_for_horizon(...)
- GridGenerator.load(...)
- GridGenerator.load_data(...)
- GridGenerator.sample_trajectory(...)
- build_joint_scenarios(...)

Important fix
-------------
The uploaded grid generator reinitialized neural models inside load_data(),
which could wipe loaded weights when build_joint_scenarios() called:
    grid_gen = GridGenerator.load(...)
    grid_gen.load_data()

This enhanced version keeps load_data() from resetting already loaded models.

How to use
----------
1) Replace your existing three files with these versions.
2) Keep world_model_common.py unchanged.
3) Build scenarios as before:
       python build_joint_scenarios.py
4) Use the exported PINN context files as the scaffold to join:
   - stochastic optimization outputs
   - teacher-model degradation labels
