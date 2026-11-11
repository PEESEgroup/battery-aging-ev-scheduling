#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_receding_horizon_optimization.py

Receding-horizon stochastic EV scheduling.

At each time step:
1) observe current system state
2) build a short forward robust scenario window
3) optimize charging over that window
4) execute only the first-step decision
5) update EV states and roll forward

Expected input:
    optimization_inputs_joint/
        scenario_00/
            optimizer_grid.csv
            optimizer_ev_sessions.csv
        ...
or
    optimization_inputs_stress/
        ...

Outputs:
    receding_horizon_results/
        scenario_00/
            executed_schedule.csv
            hourly_execution_results.csv
            session_results.csv
            summary.json
        ...
        receding_horizon_summary.csv
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class RHConfig:
    input_dir: Path = Path("optimization_inputs_stress")
    out_dir: Path = Path("receding_horizon_results")
    scenario_prefix: str = "scenario_"

    horizon_hours: int = 6
    control_step_hours: int = 1
    time_step_hours: float = 1.0

    w_price: float = 1.0
    w_emission: float = 30.0
    w_congestion: float = 0.2
    w_health: float = 0.8

    lambda_risk: float = 0.3
    lambda_std: float = 0.5
    cvar_alpha: float = 0.9

    capacity_floor_kw: float = 1500.0
    capacity_std_factor: float = 0.2
    capacity_blend_actual: float = 0.3

    soc_threshold: float = 0.65
    alpha_throughput: float = 1.0
    beta_high_soc: float = 4.0
    gamma_high_power: float = 2.5

    unmet_penalty_per_kwh: float = 2.0
    verbose_debug: bool = True


def list_scenario_dirs(input_dir: Path, prefix: str) -> List[Path]:
    return sorted([p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)])


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


def normalize_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    if len(x) == 0:
        return pd.Series(dtype=float)
    mn = np.nanmin(x.values)
    mx = np.nanmax(x.values)
    if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mn) / (mx - mn)


def health_penalty_proxy(
    init_soc: float,
    max_power_kw: float,
    battery_capacity_kwh: float,
    charge_kw: float,
    cumulative_energy_before_kwh: float,
    dt_hours: float,
    config: RHConfig,
) -> float:
    soc_before = init_soc + cumulative_energy_before_kwh / max(battery_capacity_kwh, 1e-6)
    soc_after = soc_before + charge_kw * dt_hours / max(battery_capacity_kwh, 1e-6)
    avg_soc = 0.5 * (soc_before + soc_after)

    throughput_term = config.alpha_throughput * (charge_kw * dt_hours)
    high_soc_term = 0.0
    if avg_soc > config.soc_threshold:
        high_soc_term = config.beta_high_soc * (avg_soc - config.soc_threshold) ** 2

    power_ratio = charge_kw / max(max_power_kw, 1e-6)
    high_power_term = config.gamma_high_power * (power_ratio ** 2)
    return throughput_term + high_soc_term + high_power_term


def load_all_grid_scenarios(scenario_dirs: List[Path]) -> Dict[str, pd.DataFrame]:
    out = {}
    for sdir in scenario_dirs:
        df = pd.read_csv(sdir / "optimizer_grid.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        out[sdir.name] = df
    return out


def build_window_robust_signals(
    all_grid: Dict[str, pd.DataFrame],
    t0: int,
    horizon: int,
    config: RHConfig,
) -> pd.DataFrame:
    rows = []
    keys = list(all_grid.keys())

    for tau in range(t0, t0 + horizon):
        price_vals, emission_vals, congestion_vals, capacity_vals, ts_vals = [], [], [], [], []

        for k in keys:
            df = all_grid[k]
            if tau >= len(df):
                continue
            row = df.iloc[tau]
            ts_vals.append(row["timestamp"])
            price_vals.append(pd.to_numeric(row.get("price_kwh"), errors="coerce"))
            emission_vals.append(pd.to_numeric(row.get("emission_ton_per_kwh"), errors="coerce"))
            congestion_vals.append(pd.to_numeric(row.get("congestion_intensity"), errors="coerce"))
            capacity_vals.append(pd.to_numeric(row.get("ev_capacity_limit_kw"), errors="coerce"))

        if len(price_vals) == 0:
            continue

        cap_vals = np.asarray(capacity_vals, dtype=float)
        cap_mean = float(np.nanmean(cap_vals))
        cap_std = float(np.nanstd(cap_vals))
        robust_capacity = max(cap_mean - config.capacity_std_factor * cap_std, config.capacity_floor_kw)

        rows.append({
            "t_global": tau,
            "t_local": tau - t0,
            "timestamp": ts_vals[0],
            "price_mean": float(np.nanmean(price_vals)),
            "price_std": float(np.nanstd(price_vals)),
            "price_cvar": cvar_of_array(np.asarray(price_vals), config.cvar_alpha),
            "emission_mean": float(np.nanmean(emission_vals)),
            "emission_std": float(np.nanstd(emission_vals)),
            "emission_cvar": cvar_of_array(np.asarray(emission_vals), config.cvar_alpha),
            "congestion_mean": float(np.nanmean(congestion_vals)),
            "congestion_std": float(np.nanstd(congestion_vals)),
            "congestion_cvar": cvar_of_array(np.asarray(congestion_vals), config.cvar_alpha),
            "capacity_mean_kw": cap_mean,
            "capacity_std_kw": cap_std,
            "robust_capacity_kw": robust_capacity,
        })

    robust_df = pd.DataFrame(rows)
    if robust_df.empty:
        return robust_df

    lam = config.lambda_risk
    robust_df["price_risk"] = (
        (1.0 - lam) * robust_df["price_mean"]
        + lam * robust_df["price_cvar"]
        + config.lambda_std * robust_df["price_std"]
    )
    robust_df["emission_risk"] = (
        (1.0 - lam) * robust_df["emission_mean"]
        + lam * robust_df["emission_cvar"]
        + config.lambda_std * robust_df["emission_std"]
    )
    robust_df["congestion_risk"] = (
        (1.0 - lam) * robust_df["congestion_mean"]
        + lam * robust_df["congestion_cvar"]
        + config.lambda_std * robust_df["congestion_std"]
    )

    robust_df["price_norm"] = normalize_series(robust_df["price_risk"])
    robust_df["emission_norm"] = normalize_series(robust_df["emission_risk"])
    robust_df["congestion_norm"] = normalize_series(robust_df["congestion_risk"])

    robust_df["base_score"] = (
        config.w_price * robust_df["price_norm"]
        + config.w_emission * robust_df["emission_norm"]
        + config.w_congestion * robust_df["congestion_norm"]
    )

    return robust_df


def initialize_ev_state(ev_df: pd.DataFrame) -> pd.DataFrame:
    if ev_df.empty:
        return ev_df.copy()

    df = ev_df.copy()
    numeric_cols = [
        "arrival_step", "departure_step", "required_energy_kwh",
        "max_charging_power_kw", "battery_capacity_kwh", "init_soc"
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["remaining_energy_kwh"] = df["required_energy_kwh"].copy()
    df["served_energy_kwh"] = 0.0
    df["current_soc"] = df["init_soc"].copy()
    df["is_finished"] = False
    return df


def select_active_evs(ev_state: pd.DataFrame, t0: int) -> pd.DataFrame:
    if ev_state.empty:
        return ev_state.copy()

    active = ev_state[
        (~ev_state["is_finished"]) &
        (ev_state["arrival_step"] <= t0) &
        (ev_state["departure_step"] > t0) &
        (ev_state["remaining_energy_kwh"] > 1e-9)
    ].copy()
    return active.reset_index(drop=True)


def one_step_schedule(
    active_evs: pd.DataFrame,
    actual_grid_row: pd.Series,
    robust_df: pd.DataFrame,
    ev_state: pd.DataFrame,
    config: RHConfig,
    t0: int,
) -> List[Dict[str, Any]]:
    if active_evs.empty or robust_df.empty:
        return []

    actual_capacity = pd.to_numeric(actual_grid_row.get("ev_capacity_limit_kw"), errors="coerce")
    if not np.isfinite(actual_capacity):
        actual_capacity = config.capacity_floor_kw

    robust_capacity = float(robust_df.iloc[0]["robust_capacity_kw"])
    remaining_capacity = max(
        (1.0 - config.capacity_blend_actual) * robust_capacity
        + config.capacity_blend_actual * max(actual_capacity, config.capacity_floor_kw),
        config.capacity_floor_kw
    )

    ranked = []
    for _, ev in active_evs.iterrows():
        hp = health_penalty_proxy(
            init_soc=float(ev["init_soc"]),
            max_power_kw=float(ev["max_charging_power_kw"]),
            battery_capacity_kwh=float(ev["battery_capacity_kwh"]),
            charge_kw=min(float(ev["max_charging_power_kw"]), remaining_capacity),
            cumulative_energy_before_kwh=float(ev["served_energy_kwh"]),
            dt_hours=config.time_step_hours,
            config=config,
        )
        score = float(robust_df.iloc[0]["base_score"]) + config.w_health * hp
        slack = float(ev["departure_step"] - t0)
        ranked.append((ev["ev_id"], score - 0.02 * (1.0 / max(slack, 1.0))))

    ranked_ids = [eid for eid, _ in sorted(ranked, key=lambda x: x[1])]

    executed = []
    for ev_id in ranked_ids:
        if remaining_capacity <= 1e-9:
            break

        idxs = ev_state.index[ev_state["ev_id"] == ev_id]
        if len(idxs) == 0:
            continue
        idx = idxs[0]
        ev = ev_state.loc[idx]

        max_power = float(ev["max_charging_power_kw"])
        remaining_energy = float(ev["remaining_energy_kwh"])
        charge_kw = min(max_power, remaining_capacity, remaining_energy / config.time_step_hours)
        if charge_kw <= 1e-9:
            continue

        energy_kwh = charge_kw * config.time_step_hours
        hp = health_penalty_proxy(
            init_soc=float(ev["init_soc"]),
            max_power_kw=max_power,
            battery_capacity_kwh=float(ev["battery_capacity_kwh"]),
            charge_kw=charge_kw,
            cumulative_energy_before_kwh=float(ev["served_energy_kwh"]),
            dt_hours=config.time_step_hours,
            config=config,
        )

        price_kwh = float(pd.to_numeric(actual_grid_row.get("price_kwh"), errors="coerce"))
        emission_kwh = float(pd.to_numeric(actual_grid_row.get("emission_ton_per_kwh"), errors="coerce"))
        congestion = float(pd.to_numeric(actual_grid_row.get("congestion_intensity"), errors="coerce"))

        executed.append({
            "ev_id": ev_id,
            "t": t0,
            "timestamp": actual_grid_row["timestamp"],
            "charge_kw": charge_kw,
            "energy_kwh": energy_kwh,
            "price_kwh": price_kwh,
            "emission_ton_per_kwh": emission_kwh,
            "congestion_intensity": congestion,
            "health_penalty": hp,
            "cost": price_kwh * energy_kwh,
            "emission_ton": emission_kwh * energy_kwh,
            "robust_score": float(robust_df.iloc[0]["base_score"]),
        })

        ev_state.loc[idx, "served_energy_kwh"] += energy_kwh
        ev_state.loc[idx, "remaining_energy_kwh"] = max(float(ev_state.loc[idx, "remaining_energy_kwh"]) - energy_kwh, 0.0)
        ev_state.loc[idx, "current_soc"] = min(
            float(ev_state.loc[idx, "current_soc"]) + energy_kwh / max(float(ev["battery_capacity_kwh"]), 1e-6),
            1.0
        )
        if ev_state.loc[idx, "remaining_energy_kwh"] <= 1e-9:
            ev_state.loc[idx, "is_finished"] = True

        remaining_capacity -= charge_kw

    return executed


def finalize_session_results(ev_state: pd.DataFrame) -> pd.DataFrame:
    if ev_state.empty:
        return pd.DataFrame(columns=[
            "ev_id", "required_energy_kwh", "served_energy_kwh", "unmet_energy_kwh", "service_ratio"
        ])

    df = ev_state.copy()
    df["unmet_energy_kwh"] = np.maximum(df["remaining_energy_kwh"], 0.0)
    df["service_ratio"] = np.where(
        df["required_energy_kwh"] > 1e-9,
        df["served_energy_kwh"] / df["required_energy_kwh"],
        0.0
    )
    return df[[
        "ev_id", "arrival_step", "departure_step",
        "required_energy_kwh", "served_energy_kwh",
        "unmet_energy_kwh", "service_ratio"
    ]].copy()


def run_one_scenario(target_dir: Path, all_grid: Dict[str, pd.DataFrame], config: RHConfig) -> Dict[str, Any]:
    target_grid = pd.read_csv(target_dir / "optimizer_grid.csv")
    target_ev = pd.read_csv(target_dir / "optimizer_ev_sessions.csv")

    target_grid["timestamp"] = pd.to_datetime(target_grid["timestamp"], errors="coerce")
    for c in ["price_kwh", "emission_ton_per_kwh", "congestion_intensity", "ev_capacity_limit_kw", "load_mw", "price_mwh", "emission_ton_per_mwh"]:
        if c in target_grid.columns:
            target_grid[c] = pd.to_numeric(target_grid[c], errors="coerce")

    T = len(target_grid)
    ev_state = initialize_ev_state(target_ev)

    executed_rows = []
    hourly_rows = []

    for t0 in range(T):
        robust_df = build_window_robust_signals(
            all_grid=all_grid,
            t0=t0,
            horizon=min(config.horizon_hours, T - t0),
            config=config,
        )
        active_evs = select_active_evs(ev_state, t0)
        exec_now = one_step_schedule(
            active_evs=active_evs,
            actual_grid_row=target_grid.iloc[t0],
            robust_df=robust_df,
            ev_state=ev_state,
            config=config,
            t0=t0,
        )
        executed_rows.extend(exec_now)

        tmp = pd.DataFrame(exec_now)
        hourly_rows.append({
            "t": t0,
            "timestamp": target_grid.iloc[t0]["timestamp"],
            "load_mw": float(target_grid.iloc[t0].get("load_mw", np.nan)),
            "price_mwh": float(target_grid.iloc[t0].get("price_mwh", np.nan)),
            "emission_ton_per_mwh": float(target_grid.iloc[t0].get("emission_ton_per_mwh", np.nan)),
            "congestion_intensity": float(target_grid.iloc[t0].get("congestion_intensity", np.nan)),
            "scheduled_ev_power_kw": float(tmp["charge_kw"].sum()) if len(tmp) else 0.0,
            "scheduled_ev_energy_kwh": float(tmp["energy_kwh"].sum()) if len(tmp) else 0.0,
            "scheduled_ev_cost": float(tmp["cost"].sum()) if len(tmp) else 0.0,
            "scheduled_ev_emission_ton": float(tmp["emission_ton"].sum()) if len(tmp) else 0.0,
            "scheduled_ev_health_penalty": float(tmp["health_penalty"].sum()) if len(tmp) else 0.0,
        })

    executed_df = pd.DataFrame(executed_rows)
    hourly_df = pd.DataFrame(hourly_rows)
    session_df = finalize_session_results(ev_state)

    total_required = float(session_df["required_energy_kwh"].sum()) if len(session_df) else 0.0
    total_served = float(session_df["served_energy_kwh"].sum()) if len(session_df) else 0.0
    total_unmet = float(session_df["unmet_energy_kwh"].sum()) if len(session_df) else 0.0
    service_ratio = 0.0 if total_required <= 1e-9 else total_served / total_required

    summary = {
        "scenario": target_dir.name,
        "horizon_hours": int(T),
        "rolling_horizon_hours": int(config.horizon_hours),
        "n_ev_sessions": int(len(session_df)),
        "total_required_energy_kwh": total_required,
        "total_served_energy_kwh": total_served,
        "total_unmet_energy_kwh": total_unmet,
        "service_ratio": service_ratio,
        "total_energy_cost": float(hourly_df["scheduled_ev_cost"].sum()) if len(hourly_df) else 0.0,
        "total_emission_ton": float(hourly_df["scheduled_ev_emission_ton"].sum()) if len(hourly_df) else 0.0,
        "total_health_penalty": float(hourly_df["scheduled_ev_health_penalty"].sum()) if len(hourly_df) else 0.0,
        "peak_ev_power_kw": float(hourly_df["scheduled_ev_power_kw"].max()) if len(hourly_df) else 0.0,
        "congestion_exposure": float(
            (pd.to_numeric(hourly_df["scheduled_ev_energy_kwh"], errors="coerce").fillna(0.0)
             * pd.to_numeric(hourly_df["congestion_intensity"], errors="coerce").fillna(0.0)).sum()
        ) if len(hourly_df) else 0.0,
    }
    summary["weighted_objective_proxy"] = float(
        summary["total_energy_cost"]
        + config.w_emission * summary["total_emission_ton"]
        + config.w_health * summary["total_health_penalty"]
        + config.unmet_penalty_per_kwh * summary["total_unmet_energy_kwh"]
    )

    scenario_out = config.out_dir / target_dir.name
    scenario_out.mkdir(parents=True, exist_ok=True)

    executed_df.to_csv(scenario_out / "executed_schedule.csv", index=False)
    hourly_df.to_csv(scenario_out / "hourly_execution_results.csv", index=False)
    session_df.to_csv(scenario_out / "session_results.csv", index=False)

    with open(scenario_out / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "input_dir": str(config.input_dir),
            "config": asdict(config),
            "summary": summary,
        }, f, indent=2, ensure_ascii=False, default=str)

    return summary


def main() -> None:
    config = RHConfig()
    config.out_dir.mkdir(parents=True, exist_ok=True)

    scenario_dirs = list_scenario_dirs(config.input_dir, config.scenario_prefix)
    if not scenario_dirs:
        raise FileNotFoundError(f"No scenario folders found under {config.input_dir}")

    all_grid = load_all_grid_scenarios(scenario_dirs)

    rows = []
    for sdir in scenario_dirs:
        if config.verbose_debug:
            print(f"[run] receding-horizon scenario -> {sdir.name}")
        rows.append(run_one_scenario(sdir, all_grid, config))

    summary_df = pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)
    summary_df.to_csv(config.out_dir / "receding_horizon_summary.csv", index=False)

    print(f"[ok] receding-horizon results built -> {config.out_dir}")
    print(f"[ok] scenarios processed            -> {len(summary_df)}")
    print(f"[ok] summary saved                  -> {config.out_dir / 'receding_horizon_summary.csv'}")
    print("\\nReceding-horizon summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
