#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对每个 scenario 单独优化
每辆 EV 按小时 score 排序
逐小时贪心分配能量
"""

from __future__ import annotations
"""
run_scenario_optimization.py

Run a simple battery-aware EV scheduling optimization for each scenario in:
    optimization_inputs/
        scenario_00/
            optimizer_grid.csv
            optimizer_ev_sessions.csv
            optimizer_ev_hourly.csv
            optimizer_metadata.json
        ...

Outputs:
    optimization_results/
        scenario_00/
            charging_schedule.csv
            hourly_results.csv
            session_results.csv
            objective_summary.json
        ...
        optimization_results_summary.csv

Method
------
Greedy battery-aware scheduler under hourly grid capacity:
1) Rank hours for each EV by a weighted score:
      score_t = w_price * price_kwh
              + w_emission * emission_ton_per_kwh
              + w_congestion * congestion_intensity_norm
              + w_health * health_penalty_proxy(ev, t)
2) Allocate charging energy to the lowest-score feasible hours
   subject to:
      - arrival/departure window
      - per-EV max charging power
      - system EV capacity limit
3) Build schedule and summarize cost / carbon / degradation metrics.

This is a practical first solver for scenario comparison.
"""


import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

@dataclass
class OptimizationRunConfig:
    input_dir: Path = Path("optimization_inputs")
    out_dir: Path = Path("optimization_results")
    scenario_prefix: str = "scenario_"
    time_step_hours: float = 1.0

    # objective weights
    w_price: float = 1.0
    w_emission: float = 30.0
    w_congestion: float = 0.2
    w_health: float = 0.8

    # health proxy parameters
    soc_threshold: float = 0.65
    alpha_throughput: float = 1.0
    beta_high_soc: float = 4.0
    gamma_high_power: float = 2.5

    # feasibility / service
    allow_unmet: bool = True
    unmet_penalty_per_kwh: float = 2.0


def list_scenario_dirs(input_dir: Path, prefix: str) -> List[Path]:
    return sorted([p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)])


def normalize_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    mn = np.nanmin(x.values) if len(x) else 0.0
    mx = np.nanmax(x.values) if len(x) else 1.0
    if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mn) / (mx - mn)


def health_penalty_proxy(
    init_soc: float,
    target_soc: float,
    max_power_kw: float,
    battery_capacity_kwh: float,
    charge_kw: float,
    cumulative_energy_before_kwh: float,
    dt_hours: float,
    config: OptimizationRunConfig,
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


def build_hour_score_table(grid_df: pd.DataFrame, config: OptimizationRunConfig) -> pd.DataFrame:
    df = grid_df.copy()
    df["price_norm"] = normalize_series(df["price_kwh"])
    df["emission_norm"] = normalize_series(df["emission_ton_per_kwh"])
    df["congestion_norm"] = normalize_series(df["congestion_intensity"])

    df["base_score"] = (
        config.w_price * df["price_norm"]
        + config.w_emission * df["emission_norm"]
        + config.w_congestion * df["congestion_norm"]
    )
    return df


def run_one_scenario(scenario_dir: Path, out_dir: Path, config: OptimizationRunConfig) -> Dict[str, Any]:
    grid_df = pd.read_csv(scenario_dir / "optimizer_grid.csv")
    ev_df = pd.read_csv(scenario_dir / "optimizer_ev_sessions.csv")

    grid_df["timestamp"] = pd.to_datetime(grid_df["timestamp"], errors="coerce")
    if ev_df.empty:
        ev_df = pd.DataFrame(columns=[
            "ev_id", "arrival_time", "departure_time", "arrival_step", "departure_step",
            "parking_duration_hours", "energy_delivered_kwh", "battery_capacity_kwh",
            "max_charging_power_kw", "init_soc", "target_soc", "required_energy_kwh"
        ])
    else:
        if "arrival_time" in ev_df.columns:
            ev_df["arrival_time"] = pd.to_datetime(ev_df["arrival_time"], errors="coerce")
        if "departure_time" in ev_df.columns:
            ev_df["departure_time"] = pd.to_datetime(ev_df["departure_time"], errors="coerce")

    grid_df = build_hour_score_table(grid_df, config)
    T = len(grid_df)
    dt = config.time_step_hours

    # available EV charging capacity each hour
    if "ev_capacity_limit_kw" in grid_df.columns:
        remaining_capacity = pd.to_numeric(grid_df["ev_capacity_limit_kw"], errors="coerce").fillna(np.inf).values.astype(float)
    else:
        remaining_capacity = np.full(T, np.inf, dtype=float)

    schedule_rows: List[Dict[str, Any]] = []
    session_rows: List[Dict[str, Any]] = []

    for _, ev in ev_df.iterrows():
        ev_id = ev["ev_id"]
        arr = int(ev["arrival_step"])
        dep = int(ev["departure_step"])
        dep = min(dep, T)
        arr = max(arr, 0)

        if dep <= arr:
            session_rows.append({
                "ev_id": ev_id,
                "required_energy_kwh": float(ev["required_energy_kwh"]),
                "served_energy_kwh": 0.0,
                "unmet_energy_kwh": float(ev["required_energy_kwh"]),
                "service_ratio": 0.0,
                "avg_charging_power_kw": 0.0,
            })
            continue

        max_power_kw = float(ev["max_charging_power_kw"])
        capacity_kwh = float(ev["battery_capacity_kwh"])
        init_soc = float(ev["init_soc"])
        target_soc = float(ev["target_soc"])
        required_energy = max(float(ev["required_energy_kwh"]), 0.0)

        feasible_hours = list(range(arr, dep))

        # dynamic score per hour for this EV
        ev_hour_rows = []
        cumulative_energy_before = 0.0
        for t in feasible_hours:
            nominal_charge = min(max_power_kw, remaining_capacity[t])  # proxy for ranking
            hp = health_penalty_proxy(
                init_soc=init_soc,
                target_soc=target_soc,
                max_power_kw=max_power_kw,
                battery_capacity_kwh=capacity_kwh,
                charge_kw=max(nominal_charge, 0.0),
                cumulative_energy_before_kwh=cumulative_energy_before,
                dt_hours=dt,
                config=config,
            )
            score = float(grid_df.loc[t, "base_score"]) + config.w_health * hp
            ev_hour_rows.append((t, score))

        # rank hours from lowest score to highest
        ranked_hours = [t for t, _ in sorted(ev_hour_rows, key=lambda x: x[1])]

        served_energy = 0.0
        total_cost = 0.0
        total_emission = 0.0
        total_health = 0.0
        total_congestion_cost = 0.0

        hourly_ev_alloc = []

        for t in ranked_hours:
            if served_energy >= required_energy - 1e-9:
                break

            available_kw = max(0.0, remaining_capacity[t])
            if available_kw <= 1e-9:
                continue

            charge_kw = min(max_power_kw, available_kw, (required_energy - served_energy) / dt)
            if charge_kw <= 1e-9:
                continue

            hp = health_penalty_proxy(
                init_soc=init_soc,
                target_soc=target_soc,
                max_power_kw=max_power_kw,
                battery_capacity_kwh=capacity_kwh,
                charge_kw=charge_kw,
                cumulative_energy_before_kwh=served_energy,
                dt_hours=dt,
                config=config,
            )

            energy_kwh = charge_kw * dt
            price_kwh = float(grid_df.loc[t, "price_kwh"])
            emission_kwh = float(grid_df.loc[t, "emission_ton_per_kwh"])
            congestion = float(grid_df.loc[t, "congestion_intensity"])

            served_energy += energy_kwh
            remaining_capacity[t] -= charge_kw

            cost = price_kwh * energy_kwh
            emission = emission_kwh * energy_kwh
            congestion_cost = congestion * energy_kwh

            total_cost += cost
            total_emission += emission
            total_health += hp
            total_congestion_cost += congestion_cost

            hourly_ev_alloc.append({
                "ev_id": ev_id,
                "t": t,
                "timestamp": grid_df.loc[t, "timestamp"],
                "charge_kw": charge_kw,
                "energy_kwh": energy_kwh,
                "price_kwh": price_kwh,
                "emission_ton_per_kwh": emission_kwh,
                "congestion_intensity": congestion,
                "health_penalty": hp,
                "cost": cost,
                "emission_ton": emission,
            })

        schedule_rows.extend(hourly_ev_alloc)

        unmet = max(required_energy - served_energy, 0.0)
        session_rows.append({
            "ev_id": ev_id,
            "arrival_step": arr,
            "departure_step": dep,
            "required_energy_kwh": required_energy,
            "served_energy_kwh": served_energy,
            "unmet_energy_kwh": unmet,
            "service_ratio": 0.0 if required_energy <= 1e-9 else served_energy / required_energy,
            "avg_charging_power_kw": 0.0 if served_energy <= 1e-9 else served_energy / max((dep - arr) * dt, 1e-6),
            "energy_cost": total_cost,
            "emission_ton": total_emission,
            "health_penalty_total": total_health,
            "congestion_cost_proxy": total_congestion_cost,
        })

    schedule_df = pd.DataFrame(schedule_rows)
    session_df = pd.DataFrame(session_rows)

    # hourly aggregation
    if schedule_df.empty:
        hourly_results = grid_df[["t", "timestamp", "load_mw", "price_mwh", "emission_ton_per_mwh", "congestion_intensity"]].copy()
        hourly_results["scheduled_ev_power_kw"] = 0.0
        hourly_results["scheduled_ev_energy_kwh"] = 0.0
        hourly_results["scheduled_ev_cost"] = 0.0
        hourly_results["scheduled_ev_emission_ton"] = 0.0
        hourly_results["scheduled_ev_health_penalty"] = 0.0
    else:
        hourly_sched = (
            schedule_df.groupby(["t", "timestamp"], as_index=False)
            .agg(
                scheduled_ev_power_kw=("charge_kw", "sum"),
                scheduled_ev_energy_kwh=("energy_kwh", "sum"),
                scheduled_ev_cost=("cost", "sum"),
                scheduled_ev_emission_ton=("emission_ton", "sum"),
                scheduled_ev_health_penalty=("health_penalty", "sum"),
            )
        )
        hourly_results = grid_df[["t", "timestamp", "load_mw", "price_mwh", "emission_ton_per_mwh", "congestion_intensity", "ev_capacity_limit_kw"]].merge(
            hourly_sched, on=["t", "timestamp"], how="left"
        )
        for c in ["scheduled_ev_power_kw", "scheduled_ev_energy_kwh", "scheduled_ev_cost", "scheduled_ev_emission_ton", "scheduled_ev_health_penalty"]:
            hourly_results[c] = pd.to_numeric(hourly_results[c], errors="coerce").fillna(0.0)

    total_required = float(session_df["required_energy_kwh"].sum()) if len(session_df) else 0.0
    total_served = float(session_df["served_energy_kwh"].sum()) if len(session_df) else 0.0
    total_unmet = float(session_df["unmet_energy_kwh"].sum()) if len(session_df) else 0.0
    unmet_penalty = config.unmet_penalty_per_kwh * total_unmet if config.allow_unmet else 0.0

    objective_summary = {
        "scenario": scenario_dir.name,
        "n_ev_sessions": int(len(session_df)),
        "horizon_hours": int(T),
        "total_required_energy_kwh": total_required,
        "total_served_energy_kwh": total_served,
        "total_unmet_energy_kwh": total_unmet,
        "service_ratio": 0.0 if total_required <= 1e-9 else total_served / total_required,
        "total_energy_cost": float(hourly_results["scheduled_ev_cost"].sum()),
        "total_emission_ton": float(hourly_results["scheduled_ev_emission_ton"].sum()),
        "total_health_penalty": float(hourly_results["scheduled_ev_health_penalty"].sum()),
        "peak_ev_power_kw": float(hourly_results["scheduled_ev_power_kw"].max()),
        "mean_ev_power_kw": float(hourly_results["scheduled_ev_power_kw"].mean()),
        "unmet_penalty_cost": unmet_penalty,
        "weighted_objective_proxy": float(
            hourly_results["scheduled_ev_cost"].sum()
            + config.w_emission * hourly_results["scheduled_ev_emission_ton"].sum()
            + config.w_health * hourly_results["scheduled_ev_health_penalty"].sum()
            + unmet_penalty
        ),
    }

    scenario_out = out_dir / scenario_dir.name
    scenario_out.mkdir(parents=True, exist_ok=True)

    schedule_df.to_csv(scenario_out / "charging_schedule.csv", index=False)
    hourly_results.to_csv(scenario_out / "hourly_results.csv", index=False)
    session_df.to_csv(scenario_out / "session_results.csv", index=False)

    with open(scenario_out / "objective_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "scenario": scenario_dir.name,
            "config": asdict(config),
            "objective_summary": objective_summary,
        }, f, indent=2, ensure_ascii=False, default=str)

    fig_path = scenario_out / "ev_scheduling_plot.png"
    plot_ev_scheduling_results(
        hourly_results=hourly_results,
        session_results=session_df,
        save_path=fig_path,
        scenario_name=scenario_dir.name,
    )

    heatmap_path = scenario_out / "ev_charging_heatmap.png"
    plot_ev_charging_heatmap(
        schedule_df=schedule_df,
        save_path=heatmap_path,
        scenario_name=scenario_dir.name,
        max_evs=25,
    )

    return objective_summary

def plot_ev_scheduling_results(
    hourly_results: pd.DataFrame,
    session_results: pd.DataFrame,
    save_path: Path,
    scenario_name: str,
) -> None:
    """
    Create a cleaner, presentation-ready EV scheduling figure.
    """
    hr = hourly_results.copy()
    sr = session_results.copy()

    hr["timestamp"] = pd.to_datetime(hr["timestamp"], errors="coerce")

    numeric_cols_hr = [
        "scheduled_ev_power_kw",
        "scheduled_ev_energy_kwh",
        "scheduled_ev_cost",
        "scheduled_ev_emission_ton",
        "scheduled_ev_health_penalty",
        "price_mwh",
        "emission_ton_per_mwh",
        "ev_capacity_limit_kw",
        "congestion_intensity",
    ]
    for c in numeric_cols_hr:
        if c in hr.columns:
            hr[c] = pd.to_numeric(hr[c], errors="coerce")

    numeric_cols_sr = ["served_energy_kwh", "unmet_energy_kwh", "required_energy_kwh", "service_ratio"]
    for c in numeric_cols_sr:
        if c in sr.columns:
            sr[c] = pd.to_numeric(sr[c], errors="coerce")

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig = plt.figure(figsize=(13, 9), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.0, 1.15], hspace=0.35)

    # -------------------------
    # Panel 1: Aggregate charging schedule
    # -------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("white")

    ax1.plot(
        hr["timestamp"],
        hr["scheduled_ev_power_kw"],
        linewidth=2.5,
        label="Scheduled EV Power",
        zorder=3,
    )

    if "ev_capacity_limit_kw" in hr.columns:
        ax1.plot(
            hr["timestamp"],
            hr["ev_capacity_limit_kw"],
            linewidth=2.0,
            linestyle="--",
            label="Capacity Limit",
            zorder=2,
        )
        ax1.fill_between(
            hr["timestamp"],
            0,
            hr["ev_capacity_limit_kw"],
            alpha=0.12,
            zorder=1,
        )

    ax1.set_title(f"{scenario_name}: Battery-Aware EV Charging Schedule", pad=10)
    ax1.set_ylabel("Power (kW)")
    ax1.grid(True, alpha=0.25, linewidth=0.8)
    ax1.legend(frameon=False, loc="upper right")

    # -------------------------
    # Panel 2: Price and emission signals
    # -------------------------
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.set_facecolor("white")

    l1 = ax2.plot(
        hr["timestamp"],
        hr["price_mwh"],
        linewidth=2.2,
        label="Price",
    )[0]
    ax2.set_ylabel("Price ($/MWh)")
    ax2.grid(True, alpha=0.25, linewidth=0.8)

    ax2b = ax2.twinx()
    l2 = ax2b.plot(
        hr["timestamp"],
        hr["emission_ton_per_mwh"],
        linewidth=2.0,
        linestyle="--",
        label="Emission",
    )[0]
    ax2b.set_ylabel("Emission (ton/MWh)")

    ax2.legend([l1, l2], ["Price", "Emission"], frameon=False, loc="upper right")
    ax2.set_title("Grid Signals")

    # -------------------------
    # Panel 3: Session-level service
    # -------------------------
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor("white")

    if len(sr) > 0:
        sr_plot = sr.sort_values("required_energy_kwh", ascending=False).head(15).copy()
        x = np.arange(len(sr_plot))

        ax3.bar(
            x,
            sr_plot["served_energy_kwh"],
            width=0.75,
            label="Served",
            zorder=3,
        )
        ax3.bar(
            x,
            sr_plot["unmet_energy_kwh"],
            width=0.75,
            bottom=sr_plot["served_energy_kwh"],
            label="Unmet",
            zorder=3,
        )

        # service ratio text
        for i, (_, row) in enumerate(sr_plot.iterrows()):
            ratio = row["service_ratio"] if pd.notna(row["service_ratio"]) else 0.0
            ax3.text(
                i,
                row["served_energy_kwh"] + row["unmet_energy_kwh"] + 0.5,
                f"{ratio:.0%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax3.set_xticks(x)
        ax3.set_xticklabels(sr_plot["ev_id"], rotation=35, ha="right")
        ax3.set_ylabel("Energy (kWh)")
        ax3.set_title("Session-Level Charging Service (Top 15 EVs)")
        ax3.legend(frameon=False, loc="upper right")
        ax3.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    else:
        ax3.text(0.5, 0.5, "No EV sessions scheduled", ha="center", va="center", fontsize=12)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title("Session-Level Charging Service")

    # -------------------------
    # Beautify axes
    # -------------------------
    for ax in [ax1, ax2, ax3]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_alpha(0.4)
        ax.spines["bottom"].set_alpha(0.4)

    ax2b.spines["top"].set_visible(False)
    ax2b.spines["left"].set_visible(False)
    ax2b.spines["right"].set_alpha(0.4)

    # nicer datetime ticks
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle("EV Scheduling Results Under Joint EV-Grid Scenarios", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

def plot_ev_charging_heatmap(
    schedule_df: pd.DataFrame,
    save_path: Path,
    scenario_name: str,
    max_evs: int = 25,
) -> None:
    """
    Plot EV charging heatmap.
    Rows: EVs
    Columns: time steps
    Values: charging power (kW)
    """
    import matplotlib.pyplot as plt

    df = schedule_df.copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No EV charging schedule available", ha="center", va="center", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{scenario_name}: EV Charging Heatmap")
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["charge_kw"] = pd.to_numeric(df["charge_kw"], errors="coerce").fillna(0.0)

    # keep top EVs by total charged energy for readability
    ev_rank = (
        df.groupby("ev_id", as_index=False)["energy_kwh"]
        .sum()
        .sort_values("energy_kwh", ascending=False)
    )
    keep_evs = ev_rank["ev_id"].head(max_evs).tolist()
    df = df[df["ev_id"].isin(keep_evs)].copy()

    pivot = df.pivot_table(
        index="ev_id",
        columns="timestamp",
        values="charge_kw",
        aggfunc="sum",
        fill_value=0.0
    )

    # reorder EVs by total charging energy
    pivot["__total__"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("__total__", ascending=False).drop(columns="__total__")

    fig, ax = plt.subplots(figsize=(13, 7), facecolor="white")
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        interpolation="nearest",
        origin="upper"
    )

    ax.set_title(f"{scenario_name}: EV Charging Heatmap", pad=10)
    ax.set_ylabel("EV Sessions")
    ax.set_xlabel("Time")

    # x-axis ticks
    n_cols = pivot.shape[1]
    step = max(1, n_cols // 8)
    xticks = np.arange(0, n_cols, step)
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [pd.to_datetime(pivot.columns[i]).strftime("%m-%d %H:%M") for i in xticks],
        rotation=30,
        ha="right"
    )

    # y-axis ticks
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Charging Power (kW)")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

def run_all(config: OptimizationRunConfig) -> pd.DataFrame:
    scenario_dirs = list_scenario_dirs(config.input_dir, config.scenario_prefix)
    if not scenario_dirs:
        raise FileNotFoundError(f"No scenario folders found under {config.input_dir}")

    config.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario_dir in scenario_dirs:
        rows.append(run_one_scenario(scenario_dir, config.out_dir, config))

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(config.out_dir / "optimization_results_summary.csv", index=False)
    return summary_df


def main() -> None:
    config = OptimizationRunConfig()
    summary_df = run_all(config)

    print(f"[ok] optimization results built -> {config.out_dir}")
    print(f"[ok] scenarios processed        -> {len(summary_df)}")
    print(f"[ok] summary saved              -> {config.out_dir / 'optimization_results_summary.csv'}")
    print("\nOptimization results summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
