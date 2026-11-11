#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from battery_degradation_model import DegradationConfig, BatteryState, BatteryDegradationModel


@dataclass
class StochasticOptimizationConfig:
    input_dir: Path = Path("optimization_inputs_joint")
    out_dir: Path | None = None
    scenario_prefix: str = "scenario_"
    time_step_hours: float = 1.0

    w_price: float = 1.0
    w_emission: float = 30.0
    w_congestion: float = 0.2
    w_health: float = 5.0

    lambda_std: float = 0.5
    lambda_cvar: float = 1.0
    cvar_alpha: float = 0.9

    capacity_floor_kw: float = 1500.0
    capacity_std_factor: float = 0.2
    capacity_blend_actual: float = 0.3

    allow_unmet: bool = True
    unmet_penalty_per_kwh: float = 2.0

    deg_p_ref_kw: float = 7.0
    deg_soc_threshold: float = 0.70
    deg_min_soh: float = 0.50
    deg_k_throughput: float = 1.0e-4
    deg_k_power: float = 8.0e-5
    deg_k_soc_swing: float = 6.0e-5
    deg_k_high_soc: float = 1.2e-4
    deg_k_calendar: float = 2.0e-5
    deg_k_high_soc_dwell: float = 5.0e-5
    deg_power_exponent: float = 2.0
    deg_high_soc_exponent: float = 2.0

    build_pinn_training_dataset: bool = True
    augmentation_multipliers: tuple[float, ...] = (0.70, 0.85, 1.15, 1.30)
    augmentation_noise_std_kw: float = 0.50
    temperature_jitter_c: float = 1.0


# ---------- helpers ----------
def resolve_output_dir(config: StochasticOptimizationConfig) -> Path:
    if config.out_dir is not None:
        return Path(config.out_dir)
    input_name = Path(config.input_dir).name.lower()
    if "stress" in input_name:
        return Path("stochastic_results_stress_with_deg")
    return Path("stochastic_results_joint_with_deg")


def build_degradation_model(config: StochasticOptimizationConfig) -> BatteryDegradationModel:
    deg_cfg = DegradationConfig(
        dt_hours=config.time_step_hours,
        p_ref_kw=config.deg_p_ref_kw,
        soc_threshold=config.deg_soc_threshold,
        min_soh=config.deg_min_soh,
        k_throughput=config.deg_k_throughput,
        k_power=config.deg_k_power,
        k_soc_swing=config.deg_k_soc_swing,
        k_high_soc=config.deg_k_high_soc,
        k_calendar=config.deg_k_calendar,
        k_high_soc_dwell=config.deg_k_high_soc_dwell,
        power_exponent=config.deg_power_exponent,
        high_soc_exponent=config.deg_high_soc_exponent,
    )
    return BatteryDegradationModel(deg_cfg)


def initialize_battery_states(ev_df: pd.DataFrame) -> Dict[str, BatteryState]:
    states = {}
    for _, ev in ev_df.iterrows():
        states[ev["ev_id"]] = BatteryState(
            soc=float(ev["init_soc"]),
            soh=1.0,
            cumulative_throughput_kwh=0.0,
        )
    return states


def list_scenario_dirs(input_dir: Path, prefix: str):
    return sorted([p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)])


def normalize_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    mn = np.nanmin(x.values)
    mx = np.nanmax(x.values)
    if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mn) / (mx - mn)


def cvar_of_array(x: np.ndarray, alpha: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    q = np.quantile(x, alpha)
    tail = x[x >= q]
    if len(tail) == 0:
        return float(q)
    return float(np.mean(tail))


def load_all_grid_scenarios(scenario_dirs):
    out = []
    for sdir in scenario_dirs:
        df = pd.read_csv(sdir / "optimizer_grid.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["scenario"] = sdir.name
        out.append(df)
    return out


def build_robust_hourly_signals(grid_scenarios, config: StochasticOptimizationConfig) -> pd.DataFrame:
    all_grid = pd.concat(grid_scenarios, ignore_index=True)
    for c in ["price_kwh", "emission_ton_per_kwh", "congestion_intensity", "ev_capacity_limit_kw"]:
        all_grid[c] = pd.to_numeric(all_grid[c], errors="coerce")

    rows = []
    for t, g in all_grid.groupby("t", as_index=False):
        cap_mean = float(np.nanmean(g["ev_capacity_limit_kw"].values))
        cap_std = float(np.nanstd(g["ev_capacity_limit_kw"].values))
        robust_capacity = max(cap_mean - config.capacity_std_factor * cap_std, config.capacity_floor_kw)
        rows.append({
            "t": int(t),
            "timestamp": g["timestamp"].iloc[0],
            "price_mean": float(np.nanmean(g["price_kwh"].values)),
            "price_std": float(np.nanstd(g["price_kwh"].values)),
            "price_cvar": cvar_of_array(g["price_kwh"].values, config.cvar_alpha),
            "emission_mean": float(np.nanmean(g["emission_ton_per_kwh"].values)),
            "emission_std": float(np.nanstd(g["emission_ton_per_kwh"].values)),
            "emission_cvar": cvar_of_array(g["emission_ton_per_kwh"].values, config.cvar_alpha),
            "congestion_mean": float(np.nanmean(g["congestion_intensity"].values)),
            "congestion_std": float(np.nanstd(g["congestion_intensity"].values)),
            "congestion_cvar": cvar_of_array(g["congestion_intensity"].values, config.cvar_alpha),
            "capacity_mean_kw": cap_mean,
            "capacity_std_kw": cap_std,
            "robust_capacity_kw": robust_capacity,
        })
    robust_df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    robust_df["price_risk"] = robust_df["price_mean"] + config.lambda_std * robust_df["price_std"] + config.lambda_cvar * robust_df["price_cvar"]
    robust_df["emission_risk"] = robust_df["emission_mean"] + config.lambda_std * robust_df["emission_std"] + config.lambda_cvar * robust_df["emission_cvar"]
    robust_df["congestion_risk"] = robust_df["congestion_mean"] + config.lambda_std * robust_df["congestion_std"] + config.lambda_cvar * robust_df["congestion_cvar"]
    robust_df["price_norm"] = normalize_series(robust_df["price_risk"])
    robust_df["emission_norm"] = normalize_series(robust_df["emission_risk"])
    robust_df["congestion_norm"] = normalize_series(robust_df["congestion_risk"])
    robust_df["base_score"] = config.w_price * robust_df["price_norm"] + config.w_emission * robust_df["emission_norm"] + config.w_congestion * robust_df["congestion_norm"]
    return robust_df


def schedule_one_scenario_with_robust_signals(scenario_dir: Path, robust_df: pd.DataFrame, out_dir: Path, config: StochasticOptimizationConfig):
    grid_df = pd.read_csv(scenario_dir / "optimizer_grid.csv")
    ev_df = pd.read_csv(scenario_dir / "optimizer_ev_sessions.csv")
    grid_df["timestamp"] = pd.to_datetime(grid_df["timestamp"], errors="coerce")
    for c in ["price_kwh", "emission_ton_per_kwh", "congestion_intensity", "ev_capacity_limit_kw"]:
        if c in grid_df.columns:
            grid_df[c] = pd.to_numeric(grid_df[c], errors="coerce")

    T = len(robust_df)
    dt = config.time_step_hours
    deg_model = build_degradation_model(config)
    battery_states = initialize_battery_states(ev_df)

    actual_capacity = pd.to_numeric(grid_df.get("ev_capacity_limit_kw", pd.Series(np.full(T, np.nan))), errors="coerce").fillna(config.capacity_floor_kw).values.astype(float)
    robust_capacity = pd.to_numeric(robust_df["robust_capacity_kw"], errors="coerce").fillna(config.capacity_floor_kw).values.astype(float)
    remaining_capacity = ((1.0 - config.capacity_blend_actual) * robust_capacity + config.capacity_blend_actual * np.maximum(actual_capacity, config.capacity_floor_kw))
    remaining_capacity = np.maximum(remaining_capacity, config.capacity_floor_kw)

    schedule_rows = []
    session_rows = []

    for _, ev in ev_df.iterrows():
        ev_id = ev["ev_id"]
        arr = max(int(ev["arrival_step"]), 0)
        dep = min(int(ev["departure_step"]), T)
        required_energy = max(float(ev["required_energy_kwh"]), 0.0)
        max_power_kw = float(ev["max_charging_power_kw"])
        capacity_kwh = float(ev["battery_capacity_kwh"])

        feasible_hours = list(range(arr, dep))
        ranked = []
        for t in feasible_hours:
            nominal_charge = min(max_power_kw, max(0.0, remaining_capacity[t]))
            state = battery_states[ev_id]
            deg_info = deg_model.compute_increment(soc=state.soc, charge_power_kw=max(nominal_charge, 0.0), battery_capacity_kwh=capacity_kwh)
            ranked.append((t, float(robust_df.loc[t, "base_score"]) + config.w_health * deg_info["delta_soh_total"]))
        ranked_hours = [t for t, _ in sorted(ranked, key=lambda x: x[1])]

        served_energy = total_cost = total_emission = total_health = total_delta = 0.0
        for t in ranked_hours:
            if served_energy >= required_energy - 1e-9:
                break
            available_kw = max(0.0, remaining_capacity[t])
            if available_kw <= 1e-9:
                continue
            charge_kw = min(max_power_kw, available_kw, (required_energy - served_energy) / dt)
            if charge_kw <= 1e-9:
                continue
            current_state = battery_states[ev_id]
            deg_info = deg_model.compute_increment(soc=current_state.soc, charge_power_kw=charge_kw, battery_capacity_kwh=capacity_kwh)
            new_state = deg_model.update_state(state=current_state, charge_power_kw=charge_kw, battery_capacity_kwh=capacity_kwh)
            battery_states[ev_id] = new_state
            energy_kwh = charge_kw * dt
            served_energy += energy_kwh
            remaining_capacity[t] -= charge_kw
            price_kwh = float(grid_df.loc[t, "price_kwh"])
            emission_kwh = float(grid_df.loc[t, "emission_ton_per_kwh"])
            congestion = float(grid_df.loc[t, "congestion_intensity"])
            temperature_c = float(grid_df.loc[t, "temperature_c"]) if "temperature_c" in grid_df.columns else 25.0
            cost = price_kwh * energy_kwh
            emission = emission_kwh * energy_kwh
            total_cost += cost; total_emission += emission; total_health += deg_info["delta_soh_total"]; total_delta += deg_info["delta_soh_total"]
            schedule_rows.append({
                "ev_id": ev_id, "t": t, "timestamp": grid_df.loc[t, "timestamp"],
                "charge_kw": charge_kw, "energy_kwh": energy_kwh,
                "actual_price_kwh": price_kwh, "actual_emission_ton_per_kwh": emission_kwh,
                "actual_congestion_intensity": congestion, "temperature_c": temperature_c,
                "robust_score": float(robust_df.loc[t, "base_score"]),
                "health_penalty": deg_info["delta_soh_total"],
                "delta_soh_total": deg_info["delta_soh_total"],
                "delta_soh_cycling": deg_info["delta_soh_cycling"],
                "delta_soh_calendar": deg_info["delta_soh_calendar"],
                "soc_before": deg_info["soc_before"], "soc_after": deg_info["soc_after"],
                "soh_before": current_state.soh, "soh_after_update": new_state.soh,
                "cumulative_throughput_before_kwh": current_state.cumulative_throughput_kwh,
                "cumulative_throughput_after_kwh": new_state.cumulative_throughput_kwh,
                "cost": cost, "emission_ton": emission,
            })
        unmet = max(required_energy - served_energy, 0.0)
        final_state = battery_states[ev_id]
        session_rows.append({
            "ev_id": ev_id, "arrival_step": arr, "departure_step": dep,
            "required_energy_kwh": required_energy, "served_energy_kwh": served_energy,
            "unmet_energy_kwh": unmet, "service_ratio": 0.0 if required_energy <= 1e-9 else served_energy / required_energy,
            "energy_cost": total_cost, "emission_ton": total_emission,
            "health_penalty_total": total_health, "delta_soh_total": total_delta,
            "final_soc": final_state.soc, "final_soh": final_state.soh,
            "cumulative_throughput_kwh": final_state.cumulative_throughput_kwh,
        })

    schedule_df = pd.DataFrame(schedule_rows)
    session_df = pd.DataFrame(session_rows)
    scenario_out = out_dir / scenario_dir.name
    scenario_out.mkdir(parents=True, exist_ok=True)
    schedule_df.to_csv(scenario_out / "charging_schedule.csv", index=False)
    session_df.to_csv(scenario_out / "session_results.csv", index=False)

    summary = {
        "scenario": scenario_dir.name,
        "n_ev_sessions": int(len(session_df)),
        "total_required_energy_kwh": float(session_df["required_energy_kwh"].sum()) if len(session_df) else 0.0,
        "total_served_energy_kwh": float(session_df["served_energy_kwh"].sum()) if len(session_df) else 0.0,
        "total_unmet_energy_kwh": float(session_df["unmet_energy_kwh"].sum()) if len(session_df) else 0.0,
        "service_ratio": float(session_df["served_energy_kwh"].sum() / max(session_df["required_energy_kwh"].sum(), 1e-9)) if len(session_df) else 0.0,
        "total_energy_cost": float(schedule_df["cost"].sum()) if len(schedule_df) else 0.0,
        "total_emission_ton": float(schedule_df["emission_ton"].sum()) if len(schedule_df) else 0.0,
        "total_health_penalty": float(schedule_df["health_penalty"].sum()) if len(schedule_df) else 0.0,
        "total_delta_soh": float(schedule_df["delta_soh_total"].sum()) if len(schedule_df) else 0.0,
        "mean_final_soh": float(session_df["final_soh"].mean()) if len(session_df) else np.nan,
        "peak_ev_power_kw": float(schedule_df["charge_kw"].max()) if len(schedule_df) else 0.0,
        "weighted_objective_proxy": float((schedule_df["cost"].sum() if len(schedule_df) else 0.0) + config.w_emission * (schedule_df["emission_ton"].sum() if len(schedule_df) else 0.0) + config.w_health * (schedule_df["health_penalty"].sum() if len(schedule_df) else 0.0)),
    }
    return summary


def build_risk_summary(summary_df: pd.DataFrame, config: StochasticOptimizationConfig):
    obj = pd.to_numeric(summary_df["weighted_objective_proxy"], errors="coerce").values
    srv = pd.to_numeric(summary_df["service_ratio"], errors="coerce").values
    unmet = pd.to_numeric(summary_df["total_unmet_energy_kwh"], errors="coerce").values
    alpha = config.cvar_alpha
    return {
        "source": Path(config.input_dir).name,
        "n_scenarios": int(len(summary_df)),
        "expected_objective": float(np.nanmean(obj)),
        "worst_case_objective": float(np.nanmax(obj)),
        "cvar_objective": cvar_of_array(obj, alpha),
        "expected_service_ratio": float(np.nanmean(srv)),
        "worst_case_service_ratio": float(np.nanmin(srv)),
        "expected_unmet_energy_kwh": float(np.nanmean(unmet)),
    }


def build_pinn_training_dataset(input_dir: Path, out_dir: Path, config: StochasticOptimizationConfig) -> pd.DataFrame:
    session_context = pd.read_csv(input_dir / "pinn_session_context_all.csv")
    hourly_context = pd.read_csv(input_dir / "pinn_hourly_context_all.csv")
    hourly_context["timestamp"] = pd.to_datetime(hourly_context["timestamp"], errors="coerce")
    deg_model = build_degradation_model(config)
    rng = np.random.default_rng(123)
    all_rows = []
    scenario_dirs = sorted([p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith(config.scenario_prefix)])
    for sdir in scenario_dirs:
        schedule_path = sdir / "charging_schedule.csv"
        if not schedule_path.exists():
            continue
        schedule_df = pd.read_csv(schedule_path)
        if schedule_df.empty:
            continue
        schedule_df["timestamp"] = pd.to_datetime(schedule_df["timestamp"], errors="coerce")
        scenario_id = int(sdir.name.split("_")[-1])
        sess = session_context[session_context["scenario_id"] == scenario_id].copy()
        hr = hourly_context[hourly_context["scenario_id"] == scenario_id].copy()
        df = schedule_df.merge(sess, on="ev_id", how="left", suffixes=("", "_sess")).merge(hr, on=["scenario_id", "t", "timestamp"], how="left", suffixes=("", "_hour"))
        df["sample_type"] = "optimal"
        df["scenario_name"] = sdir.name
        df["charge_power_kw"] = pd.to_numeric(df["charge_kw"], errors="coerce")
        df["soc"] = pd.to_numeric(df["soc_before"], errors="coerce")
        df["soh"] = pd.to_numeric(df["soh_before"], errors="coerce")
        df["delta_soh_true"] = pd.to_numeric(df["delta_soh_total"], errors="coerce")
        df["soh_next"] = pd.to_numeric(df["soh_after_update"], errors="coerce")
        df["soc_next"] = pd.to_numeric(df["soc_after"], errors="coerce")
        df["battery_capacity_kwh"] = pd.to_numeric(df["battery_capacity_kwh"], errors="coerce")
        df["cumulative_throughput_kwh"] = pd.to_numeric(df["cumulative_throughput_before_kwh"], errors="coerce")
        optimal_cols = [c for c in ["sample_type","scenario_name","scenario_id","ev_id","session_id","timestamp","t","soc","soh","charge_power_kw","battery_capacity_kwh","temperature_c","price_kwh","emission_ton_per_kwh","congestion_intensity","grid_risk_score","is_peak_hour","risk_lambda","risk_peak_hour","risk_month_end","arrival_step","departure_step","required_energy_kwh","max_charging_power_kw","init_soc","target_soc","delta_soc_requested","feasible_charge_hours","charging_slack_hours","requested_avg_charge_power_kw","requested_c_rate","cumulative_throughput_kwh","delta_soh_true","delta_soh_cycling","delta_soh_calendar","soc_next","soh_next"] if c in df.columns]
        all_rows.append(df[optimal_cols].copy())
        # augmented teacher labels
        for _, row in df.iterrows():
            soc = float(row["soc_before"])
            soh = float(row["soh_before"])
            cap = float(row["battery_capacity_kwh"])
            temp = float(row["temperature_c"]) if pd.notna(row.get("temperature_c", np.nan)) else 25.0
            tp = float(row["cumulative_throughput_before_kwh"])
            for mult in config.augmentation_multipliers:
                p_aug = max(float(row["charge_kw"]) * mult + rng.normal(0.0, config.augmentation_noise_std_kw), 0.0)
                info = deg_model.compute_increment(soc=soc, charge_power_kw=p_aug, battery_capacity_kwh=cap)
                next_state = deg_model.update_state(BatteryState(soc=soc, soh=soh, cumulative_throughput_kwh=tp), charge_power_kw=p_aug, battery_capacity_kwh=cap)
                new = {k: row[k] for k in optimal_cols if k in row.index}
                new["sample_type"] = f"perturbed_x{mult:.2f}"
                new["charge_power_kw"] = p_aug
                new["temperature_c"] = temp + rng.normal(0.0, config.temperature_jitter_c)
                new["delta_soh_true"] = info["delta_soh_total"]
                new["delta_soh_cycling"] = info["delta_soh_cycling"]
                new["delta_soh_calendar"] = info["delta_soh_calendar"]
                new["soc_next"] = info["soc_after"]
                new["soh_next"] = next_state.soh
                all_rows.append(pd.DataFrame([new]))
    dataset = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    dataset.to_csv(out_dir / "pinn_training_dataset.csv", index=False)
    with open(out_dir / "pinn_training_dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_rows": int(len(dataset)),
            "n_optimal_rows": int((dataset["sample_type"] == "optimal").sum()) if len(dataset) else 0,
            "n_augmented_rows": int((dataset["sample_type"] != "optimal").sum()) if len(dataset) else 0,
            "n_scenarios": int(dataset["scenario_id"].nunique()) if len(dataset) else 0,
        }, f, indent=2, ensure_ascii=False)
    return dataset


def run_all(config: StochasticOptimizationConfig) -> pd.DataFrame:
    input_dir = Path(config.input_dir)
    out_dir = resolve_output_dir(config)
    scenario_dirs = list_scenario_dirs(input_dir, config.scenario_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)
    robust_df = build_robust_hourly_signals(load_all_grid_scenarios(scenario_dirs), config)
    robust_df.to_csv(out_dir / "robust_hourly_signals.csv", index=False)
    rows = [schedule_one_scenario_with_robust_signals(sdir, robust_df, out_dir, config) for sdir in scenario_dirs]
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "stochastic_results_summary.csv", index=False)
    risk_summary = build_risk_summary(summary_df, config)
    with open(out_dir / "stochastic_risk_summary.json", "w", encoding="utf-8") as f:
        json.dump({"input_dir": str(config.input_dir), "config": asdict(config), "risk_summary": risk_summary}, f, indent=2, ensure_ascii=False, default=str)
    if config.build_pinn_training_dataset:
        build_pinn_training_dataset(input_dir, out_dir, config)
    return summary_df


def main():
    config = StochasticOptimizationConfig()
    summary_df = run_all(config)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
