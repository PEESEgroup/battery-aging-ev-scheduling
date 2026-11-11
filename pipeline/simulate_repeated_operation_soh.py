#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
simulate_repeated_operation_soh.py

Repeat one daily optimized charging schedule over multiple days and
track cumulative SOH change using BatteryDegradationModel.

Inputs
------
Example:
    stochastic_results_joint_with_deg/
        scenario_00/
            charging_schedule.csv
            hourly_results.csv
            session_results.csv

Outputs
-------
    repeated_operation_soh/
        repeated_hourly_schedule.csv
        repeated_soh_summary.csv
        repeated_schedule_and_soh.png
        representative_ev_soh.png
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from battery_degradation_model import (
    DegradationConfig,
    BatteryState,
    BatteryDegradationModel,
)


@dataclass
class RepeatSOHConfig:
    input_scenario_dir: Path = Path("stochastic_results_joint_with_deg/scenario_00")
    out_dir: Path = Path("repeated_operation_soh")

    n_days: int = 60
    time_step_hours: float = 1.0
    top_k_evs: int = 6

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


def build_deg_model(config: RepeatSOHConfig) -> BatteryDegradationModel:
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


def load_daily_profiles(config: RepeatSOHConfig):
    scenario_dir = config.input_scenario_dir
    charging = pd.read_csv(scenario_dir / "charging_schedule.csv")
    hourly = pd.read_csv(scenario_dir / "hourly_results.csv")
    sessions = pd.read_csv(scenario_dir / "session_results.csv")

    if "t" not in charging.columns:
        raise ValueError("charging_schedule.csv must contain column 't'.")

    for c in ["charge_kw", "energy_kwh", "soc_before", "soc_after", "soh_after_update"]:
        if c in charging.columns:
            charging[c] = pd.to_numeric(charging[c], errors="coerce")

    for c in ["served_energy_kwh", "final_soc", "final_soh", "cumulative_throughput_kwh"]:
        if c in sessions.columns:
            sessions[c] = pd.to_numeric(sessions[c], errors="coerce")

    T = int(hourly["t"].max()) + 1 if "t" in hourly.columns else int(charging["t"].max()) + 1

    daily_profiles = (
        charging.pivot_table(index="ev_id", columns="t", values="charge_kw", aggfunc="sum", fill_value=0.0)
        .reindex(columns=range(T), fill_value=0.0)
    )

    cap_map: Dict[str, float] = {}
    init_soc_map: Dict[str, float] = {}

    for ev_id, g in charging.groupby("ev_id"):
        g = g.sort_values("t")
        soc0 = g["soc_before"].dropna().iloc[0] if "soc_before" in g.columns and g["soc_before"].dropna().shape[0] else 0.30
        init_soc_map[ev_id] = float(soc0)

        cap_candidates = []
        if "energy_kwh" in g.columns and "soc_before" in g.columns and "soc_after" in g.columns:
            for _, row in g.iterrows():
                ds = row.get("soc_after", np.nan) - row.get("soc_before", np.nan)
                ek = row.get("energy_kwh", np.nan)
                if pd.notna(ds) and pd.notna(ek) and abs(ds) > 1e-6:
                    cap_candidates.append(float(ek / ds))
        cap_map[ev_id] = float(np.median(cap_candidates)) if len(cap_candidates) > 0 else 60.0

    served_map = {}
    if "ev_id" in sessions.columns and "served_energy_kwh" in sessions.columns:
        served_map = sessions.set_index("ev_id")["served_energy_kwh"].to_dict()

    system_daily_power = (
        pd.to_numeric(hourly["scheduled_ev_power_kw"], errors="coerce").fillna(0.0).values
        if "scheduled_ev_power_kw" in hourly.columns else np.zeros(T)
    )

    return daily_profiles, cap_map, init_soc_map, served_map, system_daily_power, T


def apply_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_schedule_and_mean_soh(repeated_hourly: pd.DataFrame, summary_daily: pd.DataFrame, config: RepeatSOHConfig):
    apply_plot_style()

    fig = plt.figure(figsize=(13, 7.2), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(repeated_hourly["t_global"], repeated_hourly["scheduled_ev_power_kw"], linewidth=1.8)
    ax1.set_title("Repeated Multi-day Charging Schedule")
    ax1.set_ylabel("Scheduled EV Power (kW)")
    ax1.set_xlabel("Global Hour")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(summary_daily["day"], summary_daily["mean_soh"], linewidth=2.4, label="Mean SOH")
    ax2.plot(summary_daily["day"], summary_daily["min_soh"], linewidth=1.8, linestyle="--", label="Min SOH")
    ax2.fill_between(summary_daily["day"], summary_daily["min_soh"], summary_daily["max_soh"], alpha=0.15, label="SOH Range")
    ax2.set_title("SOH Evolution over Repeated Daily Operation")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("SOH")
    ax2.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    plt.savefig(config.out_dir / "repeated_schedule_and_soh.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_representative_ev_soh(rep_histories: Dict[str, pd.DataFrame], summary_daily: pd.DataFrame, config: RepeatSOHConfig):
    apply_plot_style()

    fig, ax = plt.subplots(figsize=(12.5, 6.0), facecolor="white")
    for ev_id, hist in rep_histories.items():
        daily_end = hist.groupby("day", as_index=False).agg(soh_after_update=("soh_after_update", "last"))
        ax.plot(daily_end["day"], daily_end["soh_after_update"], linewidth=2.0, label=str(ev_id))

    ax.plot(summary_daily["day"], summary_daily["mean_soh"], linewidth=3.0, linestyle="--", label="Fleet Mean SOH")
    ax.set_title("Representative EV SOH Trajectories")
    ax.set_xlabel("Day")
    ax.set_ylabel("SOH")
    ax.legend(frameon=False, ncol=2, loc="best")

    plt.tight_layout()
    plt.savefig(config.out_dir / "representative_ev_soh.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def simulate_repeated_soh(config: RepeatSOHConfig):
    daily_profiles, cap_map, init_soc_map, served_map, system_daily_power, T = load_daily_profiles(config)
    deg_model = build_deg_model(config)
    config.out_dir.mkdir(parents=True, exist_ok=True)

    repeated_rows = []
    for d in range(config.n_days):
        for t in range(T):
            repeated_rows.append({
                "day": d + 1,
                "t_day": t,
                "t_global": d * T + t,
                "scheduled_ev_power_kw": float(system_daily_power[t]),
            })
    repeated_hourly = pd.DataFrame(repeated_rows)

    all_states_daily = []
    representative_ev_histories = {}

    ev_order = list(daily_profiles.index)
    ev_order.sort(key=lambda x: served_map.get(x, 0.0), reverse=True)
    rep_evs = ev_order[: min(config.top_k_evs, len(ev_order))]

    for ev_id in daily_profiles.index:
        daily_power = daily_profiles.loc[ev_id].values.astype(float)
        repeated_power = np.tile(daily_power, config.n_days)

        init_state = BatteryState(
            soc=float(init_soc_map.get(ev_id, 0.30)),
            soh=1.0,
            cumulative_throughput_kwh=0.0,
        )
        battery_capacity_kwh = float(cap_map.get(ev_id, 60.0))

        hist = deg_model.simulate_profile(
            initial_state=init_state,
            charge_power_profile_kw=repeated_power.tolist(),
            battery_capacity_kwh=battery_capacity_kwh,
        )
        hist["day"] = hist["step"] // T + 1
        hist["t_day"] = hist["step"] % T
        hist["ev_id"] = ev_id

        daily_end = (
            hist.groupby("day", as_index=False)
            .agg(
                soh_after_update=("soh_after_update", "last"),
                soc_after_update=("soc_after_update", "last"),
                cumulative_throughput_after_kwh=("cumulative_throughput_after_kwh", "last"),
            )
        )
        daily_end["ev_id"] = ev_id
        all_states_daily.append(daily_end)

        if ev_id in rep_evs:
            representative_ev_histories[ev_id] = hist.copy()

    all_daily = pd.concat(all_states_daily, ignore_index=True)
    summary_daily = (
        all_daily.groupby("day", as_index=False)
        .agg(
            mean_soh=("soh_after_update", "mean"),
            min_soh=("soh_after_update", "min"),
            max_soh=("soh_after_update", "max"),
            mean_soc=("soc_after_update", "mean"),
            mean_cumulative_throughput_kwh=("cumulative_throughput_after_kwh", "mean"),
        )
    )

    repeated_hourly.to_csv(config.out_dir / "repeated_hourly_schedule.csv", index=False)
    summary_daily.to_csv(config.out_dir / "repeated_soh_summary.csv", index=False)

    plot_schedule_and_mean_soh(repeated_hourly, summary_daily, config)
    plot_representative_ev_soh(representative_ev_histories, summary_daily, config)
    return repeated_hourly, summary_daily, representative_ev_histories


def main():
    config = RepeatSOHConfig()
    repeated_hourly, summary_daily, rep_histories = simulate_repeated_soh(config)

    print(f"[ok] outputs saved -> {config.out_dir}")
    print(f"[ok] repeated schedule -> {config.out_dir / 'repeated_hourly_schedule.csv'}")
    print(f"[ok] SOH summary -> {config.out_dir / 'repeated_soh_summary.csv'}")
    print(f"[ok] figure -> {config.out_dir / 'repeated_schedule_and_soh.png'}")
    print(f"[ok] figure -> {config.out_dir / 'representative_ev_soh.png'}")
    print("\\nSOH summary preview:")
    print(summary_daily.head().to_string(index=False))


if __name__ == "__main__":
    main()
