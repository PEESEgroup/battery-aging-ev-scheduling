
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from train_ev_generator import EVGenerator
from train_grid_generator import GridGenerator


@dataclass
class JointScenarioConfig:
    ev_model_path: Path = Path("processed_ev/ev_generator.pkl")
    grid_model_path: Path = Path("processed_grid/grid_generator.pkl")
    out_dir: Path = Path("joint_scenarios")
    start_time: str = "2025-10-01 00:00:00"
    horizon_hours: int = 24
    n_scenarios: int = 10
    ev_source: str = "acn"
    grid_stochastic: bool = True
    ev_tail_strength: float = 0.6
    grid_tail_strength: float = 0.6
    ev_use_tail_enhancement: bool = True

    # new PINN-context export options
    export_pinn_context: bool = True
    pinn_default_battery_capacity_kwh: float = 60.0
    pinn_default_max_charge_power_kw: float = 11.0
    pinn_default_initial_soh: float = 1.0


def build_ev_hourly_features(ev_sessions: pd.DataFrame, start_ts: pd.Timestamp, horizon_hours: int) -> pd.DataFrame:
    """
    Build EV hourly aggregates. If extra PINN session context columns exist,
    their hourly means are also exported.
    """
    if ev_sessions.empty:
        hours = pd.date_range(start=start_ts, periods=horizon_hours, freq="h")
        out = pd.DataFrame({
            "timestamp": hours,
            "ev_n_sessions": 0,
            "ev_total_energy_kwh": 0.0,
            "ev_mean_energy_kwh": 0.0,
            "ev_mean_parking_h": 0.0,
            "ev_mean_init_soc": 0.0,
            "ev_mean_target_soc": 0.0,
            "ev_mean_requested_power_kw": 0.0,
            "ev_mean_requested_c_rate": 0.0,
            "ev_mean_temperature_c": 25.0,
            "ev_mean_battery_capacity_kwh": 0.0,
            "ev_mean_max_charge_power_kw": 0.0,
        })
        return out

    df = ev_sessions.copy()
    df["timestamp_hour"] = pd.to_datetime(df["timestamp_hour"], errors="coerce").dt.floor("h")

    hourly = (
        df.groupby("timestamp_hour", as_index=False)
        .agg(
            ev_n_sessions=("session_id", "count"),
            ev_total_energy_kwh=("energy_delivered_kwh", "sum"),
            ev_mean_energy_kwh=("energy_delivered_kwh", "mean"),
            ev_mean_parking_h=("parking_duration_hours", "mean"),
        )
        .rename(columns={"timestamp_hour": "timestamp"})
    )

    # optional hourly means from PINN session context
    optional_map = {
        "estimated_init_soc": "ev_mean_init_soc",
        "estimated_target_soc": "ev_mean_target_soc",
        "requested_avg_charge_power_kw": "ev_mean_requested_power_kw",
        "requested_c_rate": "ev_mean_requested_c_rate",
        "temperature_c": "ev_mean_temperature_c",
        "battery_capacity_kwh": "ev_mean_battery_capacity_kwh",
        "max_charge_power_kw": "ev_mean_max_charge_power_kw",
    }
    for src, dst in optional_map.items():
        if src in df.columns:
            agg = (
                df.groupby("timestamp_hour", as_index=False)[src]
                .mean()
                .rename(columns={"timestamp_hour": "timestamp", src: dst})
            )
            hourly = hourly.merge(agg, on="timestamp", how="left")

    hours = pd.DataFrame({
        "timestamp": pd.date_range(start=start_ts, periods=horizon_hours, freq="h")
    })
    hourly = hours.merge(hourly, on="timestamp", how="left")

    fill_zero_cols = [
        "ev_n_sessions", "ev_total_energy_kwh", "ev_mean_energy_kwh", "ev_mean_parking_h",
        "ev_mean_init_soc", "ev_mean_target_soc", "ev_mean_requested_power_kw",
        "ev_mean_requested_c_rate", "ev_mean_battery_capacity_kwh", "ev_mean_max_charge_power_kw"
    ]
    for c in fill_zero_cols:
        if c in hourly.columns:
            hourly[c] = hourly[c].fillna(0.0)

    if "ev_mean_temperature_c" in hourly.columns:
        hourly["ev_mean_temperature_c"] = hourly["ev_mean_temperature_c"].fillna(25.0)

    hourly["ev_n_sessions"] = hourly["ev_n_sessions"].astype(int)
    return hourly


def build_joint_hourly(grid_hourly: pd.DataFrame, ev_hourly: pd.DataFrame) -> pd.DataFrame:
    out = grid_hourly.merge(ev_hourly, on="timestamp", how="left")

    out["ev_avg_power_kw_proxy"] = out["ev_total_energy_kwh"]
    out["ev_load_to_grid_ratio"] = np.where(
        out["system_load_mw"] > 0,
        out["ev_avg_power_kw_proxy"] / (out["system_load_mw"] * 1000.0),
        0.0,
    )
    out["price_emission_score"] = out["weighted_lbmp_mwh"] * out["weighted_imer_co2_ton_per_mwh"]

    # extra context for aggregate-level PINN training
    if "grid_risk_score" in out.columns:
        out["joint_risk_score"] = 0.5 * out["grid_risk_score"] + 0.5 * np.clip(out["ev_load_to_grid_ratio"] * 10.0, 0.0, 1.0)
    else:
        out["joint_risk_score"] = np.clip(out["ev_load_to_grid_ratio"] * 10.0, 0.0, 1.0)

    if "temperature_c" in out.columns and "ev_mean_temperature_c" in out.columns:
        out["temperature_c_combined"] = 0.5 * out["temperature_c"] + 0.5 * out["ev_mean_temperature_c"]
    elif "temperature_c" in out.columns:
        out["temperature_c_combined"] = out["temperature_c"]
    elif "ev_mean_temperature_c" in out.columns:
        out["temperature_c_combined"] = out["ev_mean_temperature_c"]
    else:
        out["temperature_c_combined"] = 25.0

    return out


def build_pinn_session_context(
    ev_gen: EVGenerator,
    ev_sessions: pd.DataFrame,
    grid_hourly: pd.DataFrame,
    scenario_id: int,
    config: JointScenarioConfig,
) -> pd.DataFrame:
    """
    Build session-level context rows that can be joined with stochastic
    optimization outputs and teacher-model degradation labels.
    """
    temp_df = None
    if "temperature_c" in grid_hourly.columns:
        temp_df = grid_hourly[["timestamp", "temperature_c"]].rename(columns={"timestamp": "timestamp_hour"})

    pinn_sessions = ev_gen.build_pinn_session_context(
        ev_sessions=ev_sessions,
        temperature_by_hour=temp_df,
        battery_capacity_kwh=config.pinn_default_battery_capacity_kwh,
        max_charge_power_kw=config.pinn_default_max_charge_power_kw,
    )

    if pinn_sessions.empty:
        pinn_sessions["scenario_id"] = pd.Series(dtype="int64")
        return pinn_sessions

    grid_ctx_cols = [
        "timestamp", "system_load_mw", "weighted_lbmp_mwh",
        "weighted_imer_co2_ton_per_mwh", "congestion_intensity",
        "temperature_c", "grid_risk_score", "is_peak_hour",
        "high_price_flag", "high_congestion_flag", "high_load_flag",
        "price_emission_score",
    ]
    grid_ctx_cols = [c for c in grid_ctx_cols if c in grid_hourly.columns]
    grid_ctx = grid_hourly[grid_ctx_cols].copy().rename(columns={"timestamp": "timestamp_hour"})

    pinn_sessions = pinn_sessions.merge(grid_ctx, on="timestamp_hour", how="left")

    rename_map = {
        "weighted_lbmp_mwh": "grid_price_mwh",
        "weighted_imer_co2_ton_per_mwh": "grid_carbon_ton_per_mwh",
        "congestion_intensity": "grid_congestion_intensity",
        "system_load_mw": "grid_system_load_mw",
    }
    pinn_sessions = pinn_sessions.rename(columns=rename_map)

    pinn_sessions["scenario_id"] = int(scenario_id)
    pinn_sessions["hour_index"] = (
        (pd.to_datetime(pinn_sessions["timestamp_hour"]) - pd.to_datetime(pinn_sessions["timestamp_hour"]).min())
        .dt.total_seconds() / 3600.0
    ).astype(int)

    # placeholders for optimizer outputs and teacher labels
    pinn_sessions["initial_soh"] = float(config.pinn_default_initial_soh)
    pinn_sessions["optimizer_charge_power_kw"] = np.nan
    pinn_sessions["optimizer_soc_start"] = pinn_sessions["estimated_init_soc"]
    pinn_sessions["optimizer_soc_end"] = np.nan
    pinn_sessions["optimizer_soh_start"] = pinn_sessions["initial_soh"]
    pinn_sessions["optimizer_soh_end"] = np.nan
    pinn_sessions["optimizer_cumulative_throughput_kwh"] = np.nan
    pinn_sessions["delta_soh_true"] = np.nan
    pinn_sessions["soh_next"] = np.nan
    pinn_sessions["sample_source"] = "joint_session_context"
    return pinn_sessions


def build_pinn_hourly_context(
    joint_hourly: pd.DataFrame,
    scenario_id: int,
    config: JointScenarioConfig,
) -> pd.DataFrame:
    """
    Build aggregate hourly rows for settings where the stochastic
    optimization is formulated at the fleet or station level.
    """
    out = joint_hourly.copy()
    out["scenario_id"] = int(scenario_id)
    out["hour_index"] = (
        (pd.to_datetime(out["timestamp"]) - pd.to_datetime(out["timestamp"]).min())
        .dt.total_seconds() / 3600.0
    ).astype(int)

    out["battery_capacity_kwh"] = float(config.pinn_default_battery_capacity_kwh)
    out["max_charge_power_kw"] = float(config.pinn_default_max_charge_power_kw)
    out["initial_soh"] = float(config.pinn_default_initial_soh)

    # expected optimization / labeling columns
    out["charge_power_kw"] = np.nan
    out["soc"] = out.get("ev_mean_init_soc", 0.0)
    out["soh"] = out["initial_soh"]
    out["cumulative_throughput_kwh"] = np.nan
    out["delta_soh_true"] = np.nan
    out["soh_next"] = np.nan
    out["sample_source"] = "joint_hourly_context"

    # align one temperature column name with the PINN interface
    if "temperature_c_combined" in out.columns:
        out["temperature_c"] = out["temperature_c_combined"]
    elif "temperature_c" not in out.columns:
        out["temperature_c"] = 25.0

    # aggregate proxies helpful for later labeling and optimization joins
    if "ev_total_energy_kwh" in out.columns:
        out["required_energy_remaining_kwh"] = out["ev_total_energy_kwh"]
    if "ev_mean_parking_h" in out.columns:
        out["time_to_departure_h"] = out["ev_mean_parking_h"]
    return out


def summarize_scenario(joint_hourly: pd.DataFrame, scenario_id: int, n_pinn_sessions: int = 0) -> Dict[str, Any]:
    out = {
        "scenario_id": scenario_id,
        "start_time": str(joint_hourly["timestamp"].min()),
        "end_time": str(joint_hourly["timestamp"].max()),
        "hours": len(joint_hourly),
        "total_ev_sessions": int(joint_hourly["ev_n_sessions"].sum()),
        "total_ev_energy_kwh": float(joint_hourly["ev_total_energy_kwh"].sum()),
        "mean_system_load_mw": float(joint_hourly["system_load_mw"].mean()),
        "peak_system_load_mw": float(joint_hourly["system_load_mw"].max()),
        "mean_price_mwh": float(joint_hourly["weighted_lbmp_mwh"].mean()),
        "mean_emission_ton_per_mwh": float(joint_hourly["weighted_imer_co2_ton_per_mwh"].mean()),
        "mean_congestion": float(joint_hourly["congestion_intensity"].mean()),
        "pinn_session_rows": int(n_pinn_sessions),
    }
    if "temperature_c" in joint_hourly.columns:
        out["mean_temperature_c"] = float(joint_hourly["temperature_c"].mean())
    if "joint_risk_score" in joint_hourly.columns:
        out["mean_joint_risk_score"] = float(joint_hourly["joint_risk_score"].mean())
    return out


def build_joint_scenarios(config: JointScenarioConfig) -> pd.DataFrame:
    config.out_dir.mkdir(parents=True, exist_ok=True)

    ev_gen = EVGenerator.load(config.ev_model_path)
    grid_gen = GridGenerator.load(config.grid_model_path)

    grid_data = grid_gen.load_data()
    grid_seed_row = grid_data.iloc[-1].copy()
    start_ts = pd.Timestamp(grid_seed_row["timestamp"])

    summary_rows: List[Dict[str, Any]] = []
    all_pinn_sessions: List[pd.DataFrame] = []
    all_pinn_hourly: List[pd.DataFrame] = []

    for s in range(config.n_scenarios):
        scenario_dir = config.out_dir / f"scenario_{s:02d}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        ev_sessions = ev_gen.sample_sessions_for_horizon(
            str(start_ts),
            hours=config.horizon_hours,
            source=config.ev_source,
            tail_strength=config.ev_tail_strength,
            use_tail_enhancement=config.ev_use_tail_enhancement,
            include_pinn_context=False,
        )
        ev_sessions["scenario_id"] = s

        grid_hourly = grid_gen.sample_trajectory(
            grid_seed_row,
            horizon=config.horizon_hours,
            stochastic=config.grid_stochastic,
            tail_strength=config.grid_tail_strength,
        )
        grid_hourly["scenario_id"] = s

        pinn_sessions = build_pinn_session_context(
            ev_gen=ev_gen,
            ev_sessions=ev_sessions,
            grid_hourly=grid_hourly,
            scenario_id=s,
            config=config,
        )
        ev_hourly = build_ev_hourly_features(
            pinn_sessions if not pinn_sessions.empty else ev_sessions,
            start_ts + pd.Timedelta(hours=1),
            config.horizon_hours,
        )
        ev_hourly["scenario_id"] = s

        joint_hourly = build_joint_hourly(grid_hourly=grid_hourly, ev_hourly=ev_hourly)
        joint_hourly["scenario_id"] = s

        pinn_hourly = build_pinn_hourly_context(
            joint_hourly=joint_hourly,
            scenario_id=s,
            config=config,
        )

        ev_sessions.to_csv(scenario_dir / "ev_sessions.csv", index=False)
        ev_hourly.to_csv(scenario_dir / "ev_hourly.csv", index=False)
        grid_hourly.to_csv(scenario_dir / "grid_hourly.csv", index=False)
        joint_hourly.to_csv(scenario_dir / "joint_hourly.csv", index=False)

        if config.export_pinn_context:
            pinn_sessions.to_csv(scenario_dir / "pinn_session_context.csv", index=False)
            pinn_hourly.to_csv(scenario_dir / "pinn_hourly_context.csv", index=False)
            all_pinn_sessions.append(pinn_sessions)
            all_pinn_hourly.append(pinn_hourly)

        summary_rows.append(
            summarize_scenario(joint_hourly, s, n_pinn_sessions=len(pinn_sessions))
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(config.out_dir / "joint_scenarios_summary.csv", index=False)

    if config.export_pinn_context:
        if all_pinn_sessions:
            pd.concat(all_pinn_sessions, ignore_index=True).to_csv(
                config.out_dir / "pinn_session_context_all.csv", index=False
            )
        else:
            pd.DataFrame().to_csv(config.out_dir / "pinn_session_context_all.csv", index=False)

        if all_pinn_hourly:
            pd.concat(all_pinn_hourly, ignore_index=True).to_csv(
                config.out_dir / "pinn_hourly_context_all.csv", index=False
            )
        else:
            pd.DataFrame().to_csv(config.out_dir / "pinn_hourly_context_all.csv", index=False)

    return summary_df


def main() -> None:
    config = JointScenarioConfig()
    summary_df = build_joint_scenarios(config)
    print(f"[ok] joint scenarios built -> {config.out_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
