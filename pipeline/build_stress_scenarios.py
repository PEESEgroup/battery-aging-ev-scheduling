#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_stress_scenarios.py

Create stress-test scenarios from existing joint EV-grid scenarios.

Input:
    joint_scenarios/
        scenario_00/
            ev_sessions.csv
            ev_hourly.csv
            grid_hourly.csv
            joint_hourly.csv
        ...
        joint_scenarios_summary.csv

Output:
    stress_scenarios/
        scenario_00/
            ev_sessions.csv
            ev_hourly.csv
            grid_hourly.csv
            joint_hourly.csv
            stress_metadata.json
        ...
        stress_scenarios_summary.csv
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class StressScenarioConfig:
    input_dir: Path = Path("joint_scenarios")
    out_dir: Path = Path("stress_scenarios")
    scenario_prefix: str = "scenario_"

    stress_mode: str = "combined_extreme"
    random_seed: int = 42

    ev_shift_fraction: float = 0.35
    ev_shift_window_start_hour: int = 17
    ev_shift_window_end_hour: int = 21
    ev_energy_multiplier: float = 1.15
    ev_duration_multiplier: float = 1.05

    peak_hours_start: int = 17
    peak_hours_end: int = 21
    price_multiplier_peak: float = 1.35
    emission_multiplier_peak: float = 1.15
    congestion_multiplier_peak: float = 1.40

    sustained_start_hour: int = 14
    sustained_end_hour: int = 22
    price_multiplier_sustained: float = 1.25
    emission_multiplier_sustained: float = 1.08
    congestion_multiplier_sustained: float = 1.15


def list_scenario_dirs(input_dir: Path, prefix: str) -> List[Path]:
    return sorted([p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)])


def _hour_mask(df: pd.DataFrame, start_hour: int, end_hour: int, ts_col: str = "timestamp") -> pd.Series:
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    h = ts.dt.hour
    return (h >= start_hour) & (h <= end_hour)


def _clip_nonnegative(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c] = np.maximum(out[c], 0.0)
    return out


def stress_ev_sessions(ev_df: pd.DataFrame, config: StressScenarioConfig) -> pd.DataFrame:
    if ev_df.empty:
        return ev_df.copy()

    rng = np.random.default_rng(config.random_seed)
    df = ev_df.copy()

    for c in ["arrival_time", "departure_time", "timestamp_hour"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    for c in ["arrival_hour", "parking_duration_hours", "energy_delivered_kwh"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if config.stress_mode not in {"evening_peak_cluster", "combined_extreme"}:
        return df

    n = len(df)
    n_shift = max(1, int(config.ev_shift_fraction * n))
    shift_idx = rng.choice(df.index.to_numpy(), size=min(n_shift, n), replace=False)
    peak_hours = np.arange(config.ev_shift_window_start_hour, config.ev_shift_window_end_hour + 1)
    new_hours = rng.choice(peak_hours, size=len(shift_idx), replace=True)

    for idx, new_hour in zip(shift_idx, new_hours):
        arr_time = df.at[idx, "arrival_time"] if "arrival_time" in df.columns else pd.NaT
        dep_time = df.at[idx, "departure_time"] if "departure_time" in df.columns else pd.NaT
        if pd.isna(arr_time):
            continue

        delta_h = int(new_hour - arr_time.hour)
        df.at[idx, "arrival_time"] = arr_time + pd.Timedelta(hours=delta_h)

        if "departure_time" in df.columns and pd.notna(dep_time):
            df.at[idx, "departure_time"] = dep_time + pd.Timedelta(hours=delta_h)

        if "timestamp_hour" in df.columns and pd.notna(df.at[idx, "timestamp_hour"]):
            df.at[idx, "timestamp_hour"] = df.at[idx, "timestamp_hour"] + pd.Timedelta(hours=delta_h)

        if "arrival_hour" in df.columns and pd.notna(df.at[idx, "arrival_hour"]):
            frac = float(df.at[idx, "arrival_hour"]) - np.floor(float(df.at[idx, "arrival_hour"]))
            df.at[idx, "arrival_hour"] = float(new_hour) + frac

    if "energy_delivered_kwh" in df.columns:
        df.loc[shift_idx, "energy_delivered_kwh"] = (
            pd.to_numeric(df.loc[shift_idx, "energy_delivered_kwh"], errors="coerce") * config.ev_energy_multiplier
        )

    if "parking_duration_hours" in df.columns:
        df.loc[shift_idx, "parking_duration_hours"] = (
            pd.to_numeric(df.loc[shift_idx, "parking_duration_hours"], errors="coerce") * config.ev_duration_multiplier
        )
        if "departure_time" in df.columns and "arrival_time" in df.columns:
            valid = df["arrival_time"].notna() & df["parking_duration_hours"].notna()
            df.loc[valid, "departure_time"] = df.loc[valid, "arrival_time"] + pd.to_timedelta(
                df.loc[valid, "parking_duration_hours"], unit="h"
            )

    return df


def build_ev_hourly_from_sessions(ev_sessions: pd.DataFrame, grid_hourly: pd.DataFrame) -> pd.DataFrame:
    if ev_sessions.empty:
        out = grid_hourly[["timestamp"]].copy()
        out["ev_n_sessions"] = 0
        out["ev_total_energy_kwh"] = 0.0
        out["ev_mean_energy_kwh"] = 0.0
        out["ev_mean_parking_h"] = 0.0
        return out

    df = ev_sessions.copy()
    if "timestamp_hour" not in df.columns:
        if "arrival_time" in df.columns:
            df["timestamp_hour"] = pd.to_datetime(df["arrival_time"], errors="coerce").dt.floor("H")
        else:
            raise ValueError("EV sessions missing both timestamp_hour and arrival_time.")

    size_col = "session_id" if "session_id" in df.columns else None
    if size_col is not None:
        hourly = (
            df.groupby("timestamp_hour", as_index=False)
            .agg(
                ev_n_sessions=(size_col, "count"),
                ev_total_energy_kwh=("energy_delivered_kwh", "sum"),
                ev_mean_energy_kwh=("energy_delivered_kwh", "mean"),
                ev_mean_parking_h=("parking_duration_hours", "mean"),
            )
            .rename(columns={"timestamp_hour": "timestamp"})
        )
    else:
        hourly = (
            df.groupby("timestamp_hour", as_index=False)
            .agg(
                ev_total_energy_kwh=("energy_delivered_kwh", "sum"),
                ev_mean_energy_kwh=("energy_delivered_kwh", "mean"),
                ev_mean_parking_h=("parking_duration_hours", "mean"),
            )
            .rename(columns={"timestamp_hour": "timestamp"})
        )
        hourly["ev_n_sessions"] = hourly.index.to_series().map(lambda _: 1)
        hourly["ev_n_sessions"] = df.groupby("timestamp_hour").size().values

    out = grid_hourly[["timestamp"]].merge(hourly, on="timestamp", how="left")
    for c in ["ev_n_sessions", "ev_total_energy_kwh", "ev_mean_energy_kwh", "ev_mean_parking_h"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out["ev_n_sessions"] = out["ev_n_sessions"].astype(int)
    return out


def stress_grid_hourly(grid_df: pd.DataFrame, config: StressScenarioConfig) -> pd.DataFrame:
    df = grid_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    num_cols = [
        "system_load_mw",
        "weighted_lbmp_mwh",
        "weighted_imer_co2_ton_per_mwh",
        "congestion_intensity",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    peak_mask = _hour_mask(df, config.peak_hours_start, config.peak_hours_end)
    sustained_mask = _hour_mask(df, config.sustained_start_hour, config.sustained_end_hour)

    if config.stress_mode in {"evening_peak_cluster", "combined_extreme"}:
        if "weighted_lbmp_mwh" in df.columns:
            df.loc[peak_mask, "weighted_lbmp_mwh"] *= config.price_multiplier_peak
        if "weighted_imer_co2_ton_per_mwh" in df.columns:
            df.loc[peak_mask, "weighted_imer_co2_ton_per_mwh"] *= config.emission_multiplier_peak
        if "congestion_intensity" in df.columns:
            df.loc[peak_mask, "congestion_intensity"] *= config.congestion_multiplier_peak
        if "system_load_mw" in df.columns:
            df.loc[peak_mask, "system_load_mw"] *= 1.05

    if config.stress_mode in {"sustained_high_price", "combined_extreme"}:
        if "weighted_lbmp_mwh" in df.columns:
            df.loc[sustained_mask, "weighted_lbmp_mwh"] *= config.price_multiplier_sustained
        if "weighted_imer_co2_ton_per_mwh" in df.columns:
            df.loc[sustained_mask, "weighted_imer_co2_ton_per_mwh"] *= config.emission_multiplier_sustained
        if "congestion_intensity" in df.columns:
            df.loc[sustained_mask, "congestion_intensity"] *= config.congestion_multiplier_sustained

    return _clip_nonnegative(df, num_cols)


def build_joint_hourly(grid_hourly: pd.DataFrame, ev_hourly: pd.DataFrame) -> pd.DataFrame:
    out = grid_hourly.merge(ev_hourly, on="timestamp", how="left")

    for c in ["ev_n_sessions", "ev_total_energy_kwh", "ev_mean_energy_kwh", "ev_mean_parking_h"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    out["ev_n_sessions"] = out["ev_n_sessions"].astype(int)
    out["ev_avg_power_kw_proxy"] = out["ev_total_energy_kwh"]

    if "system_load_mw" in out.columns:
        out["ev_load_to_grid_ratio"] = np.where(
            pd.to_numeric(out["system_load_mw"], errors="coerce") > 0,
            out["ev_avg_power_kw_proxy"] / (pd.to_numeric(out["system_load_mw"], errors="coerce") * 1000.0),
            0.0
        )
    else:
        out["ev_load_to_grid_ratio"] = 0.0

    if "weighted_lbmp_mwh" in out.columns and "weighted_imer_co2_ton_per_mwh" in out.columns:
        out["price_emission_score"] = (
            pd.to_numeric(out["weighted_lbmp_mwh"], errors="coerce")
            * pd.to_numeric(out["weighted_imer_co2_ton_per_mwh"], errors="coerce")
        )
    else:
        out["price_emission_score"] = np.nan

    return out


def summarize_stress_scenario(joint_hourly: pd.DataFrame, scenario_name: str, config: StressScenarioConfig) -> Dict[str, Any]:
    return {
        "scenario": scenario_name,
        "stress_mode": config.stress_mode,
        "hours": int(len(joint_hourly)),
        "total_ev_sessions": int(joint_hourly["ev_n_sessions"].sum()) if "ev_n_sessions" in joint_hourly.columns else 0,
        "total_ev_energy_kwh": float(joint_hourly["ev_total_energy_kwh"].sum()) if "ev_total_energy_kwh" in joint_hourly.columns else 0.0,
        "mean_system_load_mw": float(pd.to_numeric(joint_hourly.get("system_load_mw"), errors="coerce").mean()),
        "peak_system_load_mw": float(pd.to_numeric(joint_hourly.get("system_load_mw"), errors="coerce").max()),
        "mean_price_mwh": float(pd.to_numeric(joint_hourly.get("weighted_lbmp_mwh"), errors="coerce").mean()),
        "peak_price_mwh": float(pd.to_numeric(joint_hourly.get("weighted_lbmp_mwh"), errors="coerce").max()),
        "mean_emission_ton_per_mwh": float(pd.to_numeric(joint_hourly.get("weighted_imer_co2_ton_per_mwh"), errors="coerce").mean()),
        "mean_congestion": float(pd.to_numeric(joint_hourly.get("congestion_intensity"), errors="coerce").mean()),
        "peak_congestion": float(pd.to_numeric(joint_hourly.get("congestion_intensity"), errors="coerce").max()),
    }


def process_one_scenario(scenario_dir: Path, out_dir: Path, config: StressScenarioConfig) -> Dict[str, Any]:
    ev_sessions = pd.read_csv(scenario_dir / "ev_sessions.csv")
    grid_hourly = pd.read_csv(scenario_dir / "grid_hourly.csv")

    grid_hourly["timestamp"] = pd.to_datetime(grid_hourly["timestamp"], errors="coerce")

    stressed_ev_sessions = stress_ev_sessions(ev_sessions, config)
    stressed_grid_hourly = stress_grid_hourly(grid_hourly, config)
    stressed_ev_hourly = build_ev_hourly_from_sessions(stressed_ev_sessions, stressed_grid_hourly)
    stressed_joint_hourly = build_joint_hourly(stressed_grid_hourly, stressed_ev_hourly)

    scenario_out = out_dir / scenario_dir.name
    scenario_out.mkdir(parents=True, exist_ok=True)

    stressed_ev_sessions.to_csv(scenario_out / "ev_sessions.csv", index=False)
    stressed_ev_hourly.to_csv(scenario_out / "ev_hourly.csv", index=False)
    stressed_grid_hourly.to_csv(scenario_out / "grid_hourly.csv", index=False)
    stressed_joint_hourly.to_csv(scenario_out / "joint_hourly.csv", index=False)

    summary = summarize_stress_scenario(stressed_joint_hourly, scenario_dir.name, config)

    metadata = {
        "scenario": scenario_dir.name,
        "stress_mode": config.stress_mode,
        "config": asdict(config),
        "summary": summary,
    }
    with open(scenario_out / "stress_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    return summary


def build_all(config: StressScenarioConfig) -> pd.DataFrame:
    scenario_dirs = list_scenario_dirs(config.input_dir, config.scenario_prefix)
    if not scenario_dirs:
        raise FileNotFoundError(f"No scenario folders found under {config.input_dir}")

    config.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario_dir in scenario_dirs:
        rows.append(process_one_scenario(scenario_dir, config.out_dir, config))

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(config.out_dir / "stress_scenarios_summary.csv", index=False)
    return summary_df


def main() -> None:
    #config = StressScenarioConfig()
    config = StressScenarioConfig(
        input_dir=Path("joint_scenarios"),
        out_dir=Path("stress_scenarios_combined_extreme"),
        stress_mode="combined_extreme",
    )
    summary_df = build_all(config)

    print(f"[ok] stress scenarios built -> {config.out_dir}")
    print(f"[ok] scenarios processed    -> {len(summary_df)}")
    print(f"[ok] summary saved          -> {config.out_dir / 'stress_scenarios_summary.csv'}")
    print("\\nStress scenario summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
