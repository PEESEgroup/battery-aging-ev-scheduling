#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_joint_scenarios.py

Build aligned joint EV-grid scenarios by combining:
1) EV generator  -> synthetic EV sessions
2) Grid generator -> synthetic grid trajectories

Expected files:
    processed/ev_generator.pkl
    processed_grid/grid_generator.pkl

Outputs:
    joint_scenarios/
        scenario_00/
            ev_sessions.csv
            ev_hourly.csv
            grid_hourly.csv
            joint_hourly.csv
        ...
        joint_scenarios_summary.csv
"""

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


def build_ev_hourly_features(ev_sessions: pd.DataFrame,
                             start_ts: pd.Timestamp,
                             horizon_hours: int) -> pd.DataFrame:
    if ev_sessions.empty:
        hours = pd.date_range(start=start_ts, periods=horizon_hours, freq="H")
        return pd.DataFrame({
            "timestamp": hours,
            "ev_n_sessions": 0,
            "ev_total_energy_kwh": 0.0,
            "ev_mean_energy_kwh": 0.0,
            "ev_mean_parking_h": 0.0,
        })

    df = ev_sessions.copy()
    df["timestamp_hour"] = pd.to_datetime(df["timestamp_hour"], errors="coerce").dt.floor("H")

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

    hours = pd.DataFrame({
        "timestamp": pd.date_range(start=start_ts, periods=horizon_hours, freq="H")
    })

    hourly = hours.merge(hourly, on="timestamp", how="left")
    for c in ["ev_n_sessions", "ev_total_energy_kwh", "ev_mean_energy_kwh", "ev_mean_parking_h"]:
        hourly[c] = hourly[c].fillna(0.0)

    hourly["ev_n_sessions"] = hourly["ev_n_sessions"].astype(int)
    return hourly


def build_joint_hourly(grid_hourly: pd.DataFrame, ev_hourly: pd.DataFrame) -> pd.DataFrame:
    out = grid_hourly.merge(ev_hourly, on="timestamp", how="left")
    out["ev_avg_power_kw_proxy"] = out["ev_total_energy_kwh"]
    out["ev_load_to_grid_ratio"] = np.where(
        out["system_load_mw"] > 0,
        out["ev_avg_power_kw_proxy"] / (out["system_load_mw"] * 1000.0),
        0.0
    )
    out["price_emission_score"] = out["weighted_lbmp_mwh"] * out["weighted_imer_co2_ton_per_mwh"]
    return out


def summarize_scenario(joint_hourly: pd.DataFrame, scenario_id: int) -> Dict[str, Any]:
    return {
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
    }


def build_joint_scenarios(config: JointScenarioConfig) -> pd.DataFrame:
    config.out_dir.mkdir(parents=True, exist_ok=True)

    ev_gen = EVGenerator.load(config.ev_model_path)
    grid_gen = GridGenerator.load(config.grid_model_path)
    print(type(grid_gen))

    grid_data = grid_gen.load_data()
    grid_seed_row = grid_data.iloc[-1].copy()

    #start_ts = pd.Timestamp(config.start_time)
    start_ts = pd.Timestamp(grid_seed_row["timestamp"])
    summary_rows: List[Dict[str, Any]] = []

    for s in range(config.n_scenarios):
        scenario_dir = config.out_dir / f"scenario_{s:02d}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        ev_sessions = ev_gen.sample_sessions_for_horizon(
            start_time=str(start_ts),
            hours=config.horizon_hours,
            source=config.ev_source,
        )
        ev_sessions["scenario_id"] = s

        grid_hourly = grid_gen.sample_trajectory(
            seed_row=grid_seed_row,
            horizon=config.horizon_hours,
            stochastic=config.grid_stochastic,
        )
        grid_hourly["scenario_id"] = s

        ev_hourly = build_ev_hourly_features(
            ev_sessions=ev_sessions,
            start_ts=start_ts + pd.Timedelta(hours=1),
            horizon_hours=config.horizon_hours,
        )
        ev_hourly["scenario_id"] = s

        joint_hourly = build_joint_hourly(grid_hourly=grid_hourly, ev_hourly=ev_hourly)
        joint_hourly["scenario_id"] = s

        ev_sessions.to_csv(scenario_dir / "ev_sessions.csv", index=False)
        ev_hourly.to_csv(scenario_dir / "ev_hourly.csv", index=False)
        grid_hourly.to_csv(scenario_dir / "grid_hourly.csv", index=False)
        joint_hourly.to_csv(scenario_dir / "joint_hourly.csv", index=False)

        summary_rows.append(summarize_scenario(joint_hourly, s))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(config.out_dir / "joint_scenarios_summary.csv", index=False)
    return summary_df


def main() -> None:
    config = JointScenarioConfig()
    summary_df = build_joint_scenarios(config)

    print(f"[ok] joint scenarios built -> {config.out_dir}")
    print(f"[ok] number of scenarios   -> {len(summary_df)}")
    print(f"[ok] summary saved         -> {config.out_dir / 'joint_scenarios_summary.csv'}")
    print("\\nScenario summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
