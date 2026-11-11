#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_optimization_inputs.py

Convert joint EV-grid scenarios into optimization-ready inputs for
battery-aware EV scheduling.

Expected structure:
    joint_scenarios/
        scenario_00/
            ev_sessions.csv
            ev_hourly.csv
            grid_hourly.csv
            joint_hourly.csv
        ...
        joint_scenarios_summary.csv

Outputs for each scenario:
    optimization_inputs/
        scenario_00/
            optimizer_grid.csv
            optimizer_ev_sessions.csv
            optimizer_ev_hourly.csv
            optimizer_metadata.json
        ...
        optimization_inputs_summary.csv
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class OptimizationInputConfig:
    joint_dir: Path = Path("joint_scenarios")
    out_dir: Path = Path("optimization_inputs")

    scenario_prefix: str = "scenario_"
    time_step_hours: float = 1.0

    default_battery_capacity_kwh: float = 60.0
    default_max_charging_power_kw: float = 7.0
    min_max_charging_power_kw: float = 3.3
    max_max_charging_power_kw: float = 11.0

    default_init_soc: float = 0.30
    min_init_soc: float = 0.10
    max_init_soc: float = 0.80
    target_soc_margin: float = 0.05
    min_target_soc: float = 0.50
    max_target_soc: float = 0.95

    use_capacity_proxy: bool = True
    capacity_quantile: float = 0.90
    capacity_margin: float = 0.95


def list_scenario_dirs(joint_dir: Path, prefix: str) -> List[Path]:
    return sorted([p for p in joint_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)])


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_grid_input(joint_hourly: pd.DataFrame, config: OptimizationInputConfig) -> pd.DataFrame:
    df = joint_hourly.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["t"] = np.arange(len(df), dtype=int)

    df["load_mw"] = safe_numeric(df["system_load_mw"])
    df["price_mwh"] = safe_numeric(df["weighted_lbmp_mwh"])
    df["emission_ton_per_mwh"] = safe_numeric(df["weighted_imer_co2_ton_per_mwh"])
    df["congestion_intensity"] = safe_numeric(df["congestion_intensity"])

    df["price_kwh"] = df["price_mwh"] / 1000.0
    df["emission_ton_per_kwh"] = df["emission_ton_per_mwh"] / 1000.0

    if config.use_capacity_proxy:
        peak_ref = float(df["load_mw"].quantile(config.capacity_quantile))
        capacity_limit_mw = peak_ref * config.capacity_margin
        df["ev_capacity_limit_kw"] = np.maximum((capacity_limit_mw - df["load_mw"]) * 1000.0, 0.0)
    else:
        df["ev_capacity_limit_kw"] = np.nan

    out = df[[
        "t",
        "timestamp",
        "load_mw",
        "price_mwh",
        "price_kwh",
        "emission_ton_per_mwh",
        "emission_ton_per_kwh",
        "congestion_intensity",
        "ev_capacity_limit_kw",
        "ev_n_sessions",
        "ev_total_energy_kwh",
        "ev_mean_energy_kwh",
        "ev_mean_parking_h",
        "price_emission_score",
        "ev_load_to_grid_ratio",
    ]].copy()

    return out


def infer_max_power(energy_kwh: float, parking_h: float, config: OptimizationInputConfig) -> float:
    if pd.isna(energy_kwh) or pd.isna(parking_h) or parking_h <= 0:
        return config.default_max_charging_power_kw
    proxy = energy_kwh / max(parking_h, 1e-6)
    proxy = max(proxy, config.min_max_charging_power_kw)
    proxy = min(proxy, config.max_max_charging_power_kw)
    return float(proxy)


def infer_soc_fields(energy_kwh: float, capacity_kwh: float, config: OptimizationInputConfig) -> tuple[float, float]:
    if pd.isna(energy_kwh) or pd.isna(capacity_kwh) or capacity_kwh <= 0:
        init_soc = config.default_init_soc
        target_soc = min(max(init_soc + 0.30, config.min_target_soc), config.max_target_soc)
        return float(init_soc), float(target_soc)

    energy_ratio = energy_kwh / capacity_kwh

    target_soc = min(
        max(config.min_target_soc, config.default_init_soc + energy_ratio + config.target_soc_margin),
        config.max_target_soc
    )
    init_soc = max(config.min_init_soc, target_soc - energy_ratio)

    init_soc = min(init_soc, config.max_init_soc)
    target_soc = max(target_soc, init_soc + 0.05)
    target_soc = min(target_soc, config.max_target_soc)

    return float(init_soc), float(target_soc)


def build_ev_session_input(ev_sessions: pd.DataFrame,
                           grid_input: pd.DataFrame,
                           config: OptimizationInputConfig) -> pd.DataFrame:
    df = ev_sessions.copy()
    df["arrival_time"] = pd.to_datetime(df["arrival_time"], errors="coerce")
    df["departure_time"] = pd.to_datetime(df["departure_time"], errors="coerce")

    horizon_start = pd.to_datetime(grid_input["timestamp"].min())
    horizon_end = pd.to_datetime(grid_input["timestamp"].max()) + pd.Timedelta(hours=config.time_step_hours)

    df = df[
        (df["arrival_time"] < horizon_end) &
        (df["departure_time"] > horizon_start)
    ].copy()

    if df.empty:
        return pd.DataFrame(columns=[
            "ev_id", "arrival_time", "departure_time", "arrival_step", "departure_step",
            "parking_duration_hours", "energy_delivered_kwh", "battery_capacity_kwh",
            "max_charging_power_kw", "init_soc", "target_soc", "required_energy_kwh"
        ])

    df["arrival_time_clipped"] = df["arrival_time"].clip(lower=horizon_start)
    df["departure_time_clipped"] = df["departure_time"].clip(upper=horizon_end)

    dt_hours = config.time_step_hours
    df["arrival_step"] = np.floor((df["arrival_time_clipped"] - horizon_start).dt.total_seconds() / 3600.0 / dt_hours).astype(int)
    df["departure_step"] = np.ceil((df["departure_time_clipped"] - horizon_start).dt.total_seconds() / 3600.0 / dt_hours).astype(int)

    df = df[df["departure_step"] > df["arrival_step"]].copy()

    df["parking_duration_hours"] = safe_numeric(df["parking_duration_hours"])
    df["energy_delivered_kwh"] = safe_numeric(df["energy_delivered_kwh"])

    df["battery_capacity_kwh"] = config.default_battery_capacity_kwh
    df["max_charging_power_kw"] = [
        infer_max_power(e, p, config)
        for e, p in zip(df["energy_delivered_kwh"], df["parking_duration_hours"])
    ]

    soc_pairs = [
        infer_soc_fields(e, config.default_battery_capacity_kwh, config)
        for e in df["energy_delivered_kwh"]
    ]
    df["init_soc"] = [x[0] for x in soc_pairs]
    df["target_soc"] = [x[1] for x in soc_pairs]
    df["required_energy_kwh"] = (df["target_soc"] - df["init_soc"]) * df["battery_capacity_kwh"]

    df["ev_id"] = [f"ev_{i:04d}" for i in range(len(df))]

    out = df[[
        "ev_id",
        "arrival_time",
        "departure_time",
        "arrival_step",
        "departure_step",
        "parking_duration_hours",
        "energy_delivered_kwh",
        "battery_capacity_kwh",
        "max_charging_power_kw",
        "init_soc",
        "target_soc",
        "required_energy_kwh",
    ]].copy()

    return out.reset_index(drop=True)


def build_ev_hourly_input(ev_hourly: pd.DataFrame,
                          grid_input: pd.DataFrame) -> pd.DataFrame:
    df = ev_hourly.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    base = grid_input[["t", "timestamp"]].merge(df, on="timestamp", how="left")

    for c in ["ev_n_sessions", "ev_total_energy_kwh", "ev_mean_energy_kwh", "ev_mean_parking_h"]:
        if c in base.columns:
            base[c] = safe_numeric(base[c]).fillna(0.0)

    if "ev_n_sessions" in base.columns:
        base["ev_n_sessions"] = base["ev_n_sessions"].astype(int)

    return base


def summarize_optimizer_inputs(grid_input: pd.DataFrame,
                               ev_sessions_input: pd.DataFrame,
                               scenario_name: str) -> Dict[str, Any]:
    summary = {
        "scenario": scenario_name,
        "horizon_hours": int(len(grid_input)),
        "n_ev_sessions": int(len(ev_sessions_input)),
        "total_required_energy_kwh": float(ev_sessions_input["required_energy_kwh"].sum()) if len(ev_sessions_input) else 0.0,
        "mean_price_mwh": float(grid_input["price_mwh"].mean()),
        "mean_emission_ton_per_mwh": float(grid_input["emission_ton_per_mwh"].mean()),
        "peak_load_mw": float(grid_input["load_mw"].max()),
        "mean_congestion_intensity": float(grid_input["congestion_intensity"].mean()),
    }
    if "ev_capacity_limit_kw" in grid_input.columns:
        summary["mean_ev_capacity_limit_kw"] = float(grid_input["ev_capacity_limit_kw"].mean())
        summary["min_ev_capacity_limit_kw"] = float(grid_input["ev_capacity_limit_kw"].min())
    return summary


def process_one_scenario(scenario_dir: Path, out_dir: Path, config: OptimizationInputConfig) -> Dict[str, Any]:
    joint_hourly_path = scenario_dir / "joint_hourly.csv"
    ev_sessions_path = scenario_dir / "ev_sessions.csv"
    ev_hourly_path = scenario_dir / "ev_hourly.csv"

    joint_hourly = pd.read_csv(joint_hourly_path)
    ev_sessions = pd.read_csv(ev_sessions_path)
    ev_hourly = pd.read_csv(ev_hourly_path)

    grid_input = build_grid_input(joint_hourly, config)
    ev_sessions_input = build_ev_session_input(ev_sessions, grid_input, config)
    ev_hourly_input = build_ev_hourly_input(ev_hourly, grid_input)

    scenario_out = out_dir / scenario_dir.name
    scenario_out.mkdir(parents=True, exist_ok=True)

    grid_input.to_csv(scenario_out / "optimizer_grid.csv", index=False)
    ev_sessions_input.to_csv(scenario_out / "optimizer_ev_sessions.csv", index=False)
    ev_hourly_input.to_csv(scenario_out / "optimizer_ev_hourly.csv", index=False)

    summary = summarize_optimizer_inputs(grid_input, ev_sessions_input, scenario_dir.name)

    metadata = {
        "scenario": scenario_dir.name,
        "config": asdict(config),
        "summary": summary,
    }
    with open(scenario_out / "optimizer_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    return summary


def build_all(config: OptimizationInputConfig) -> pd.DataFrame:
    scenario_dirs = list_scenario_dirs(config.joint_dir, config.scenario_prefix)
    if not scenario_dirs:
        raise FileNotFoundError(f"No scenario folders found under {config.joint_dir}")

    config.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario_dir in scenario_dirs:
        rows.append(process_one_scenario(scenario_dir, config.out_dir, config))

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(config.out_dir / "optimization_inputs_summary.csv", index=False)
    return summary_df


def main() -> None:
    config = OptimizationInputConfig()
    summary_df = build_all(config)

    print(f"[ok] optimization inputs built -> {config.out_dir}")
    print(f"[ok] scenarios processed       -> {len(summary_df)}")
    print(f"[ok] summary saved             -> {config.out_dir / 'optimization_inputs_summary.csv'}")
    print("\\nOptimization input summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
