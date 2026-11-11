#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_stochastic_scenario_optimization.py

Stochastic scenario-based battery-aware EV scheduling.

Input:
    optimization_inputs/
        scenario_00/
            optimizer_grid.csv
            optimizer_ev_sessions.csv
            optimizer_ev_hourly.csv
            optimizer_metadata.json
        ...

Output:
    stochastic_optimization_results/
        robust_hourly_signals.csv
        scenario_00/
            charging_schedule.csv
            hourly_results.csv
            session_results.csv
            objective_summary.json
            stochastic_ev_scheduling_plot.png
        ...
        stochastic_results_summary.csv
        stochastic_risk_summary.json
        robust_hourly_signals_plot.png
        stochastic_summary_plot.png
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
class StochasticOptimizationConfig:
    input_dir: Path = Path("optimization_inputs")
    out_dir: Path = Path("stochastic_optimization_results")
    scenario_prefix: str = "scenario_"
    time_step_hours: float = 1.0

    # objective weights
    w_price: float = 1.0
    w_emission: float = 30.0
    w_congestion: float = 0.2
    w_health: float = 0.8

    # risk weights
    lambda_std: float = 0.5
    lambda_cvar: float = 1.0
    cvar_alpha: float = 0.9

    # robust capacity settings
    capacity_floor_kw: float = 1500.0
    capacity_std_factor: float = 0.2
    capacity_blend_actual: float = 0.3   # blend actual scenario capacity into robust capacity

    # health proxy
    soc_threshold: float = 0.65
    alpha_throughput: float = 1.0
    beta_high_soc: float = 4.0
    gamma_high_power: float = 2.5

    # unmet service
    allow_unmet: bool = True
    unmet_penalty_per_kwh: float = 2.0

    # debug
    verbose_debug: bool = True


def list_scenario_dirs(input_dir: Path, prefix: str) -> List[Path]:
    return sorted([p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)])


def normalize_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    if len(x) == 0:
        return pd.Series(dtype=float)
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


def health_penalty_proxy(
    init_soc: float,
    max_power_kw: float,
    battery_capacity_kwh: float,
    charge_kw: float,
    cumulative_energy_before_kwh: float,
    dt_hours: float,
    config: StochasticOptimizationConfig,
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


def load_all_grid_scenarios(scenario_dirs: List[Path]) -> List[pd.DataFrame]:
    grid_list = []
    for sdir in scenario_dirs:
        df = pd.read_csv(sdir / "optimizer_grid.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["scenario"] = sdir.name
        grid_list.append(df)
    return grid_list


def build_robust_hourly_signals(
    grid_scenarios: List[pd.DataFrame],
    config: StochasticOptimizationConfig,
) -> pd.DataFrame:
    all_grid = pd.concat(grid_scenarios, ignore_index=True)

    for c in ["price_kwh", "emission_ton_per_kwh", "congestion_intensity", "ev_capacity_limit_kw"]:
        all_grid[c] = pd.to_numeric(all_grid[c], errors="coerce")

    grouped = all_grid.groupby("t", as_index=False)

    rows = []
    for t, g in grouped:
        price_vals = g["price_kwh"].values
        emission_vals = g["emission_ton_per_kwh"].values
        congestion_vals = g["congestion_intensity"].values
        capacity_vals = g["ev_capacity_limit_kw"].values

        cap_mean = float(np.nanmean(capacity_vals))
        cap_std = float(np.nanstd(capacity_vals))
        robust_capacity = max(
            cap_mean - config.capacity_std_factor * cap_std,
            config.capacity_floor_kw
        )

        rows.append({
            "t": int(t),
            "timestamp": g["timestamp"].iloc[0],

            "price_mean": float(np.nanmean(price_vals)),
            "price_std": float(np.nanstd(price_vals)),
            "price_cvar": cvar_of_array(price_vals, config.cvar_alpha),

            "emission_mean": float(np.nanmean(emission_vals)),
            "emission_std": float(np.nanstd(emission_vals)),
            "emission_cvar": cvar_of_array(emission_vals, config.cvar_alpha),

            "congestion_mean": float(np.nanmean(congestion_vals)),
            "congestion_std": float(np.nanstd(congestion_vals)),
            "congestion_cvar": cvar_of_array(congestion_vals, config.cvar_alpha),

            "capacity_mean_kw": cap_mean,
            "capacity_std_kw": cap_std,
            "robust_capacity_kw": robust_capacity,
        })

    robust_df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)

    robust_df["price_risk"] = (
        robust_df["price_mean"]
        + config.lambda_std * robust_df["price_std"]
        + config.lambda_cvar * robust_df["price_cvar"]
    )
    robust_df["emission_risk"] = (
        robust_df["emission_mean"]
        + config.lambda_std * robust_df["emission_std"]
        + config.lambda_cvar * robust_df["emission_cvar"]
    )
    robust_df["congestion_risk"] = (
        robust_df["congestion_mean"]
        + config.lambda_std * robust_df["congestion_std"]
        + config.lambda_cvar * robust_df["congestion_cvar"]
    )

    robust_df["price_norm"] = normalize_series(robust_df["price_risk"])
    robust_df["emission_norm"] = normalize_series(robust_df["emission_risk"])
    robust_df["congestion_norm"] = normalize_series(robust_df["congestion_risk"])

    robust_df["base_score"] = (
        config.w_price * robust_df["price_norm"]
        + config.w_emission * robust_df["emission_norm"]
        + config.w_congestion * robust_df["congestion_norm"]
    )

    if config.verbose_debug:
        print(
            "[debug] robust capacity min/mean/max:",
            float(robust_df["robust_capacity_kw"].min()),
            float(robust_df["robust_capacity_kw"].mean()),
            float(robust_df["robust_capacity_kw"].max()),
        )

    return robust_df


def schedule_one_scenario_with_robust_signals(
    scenario_dir: Path,
    robust_df: pd.DataFrame,
    out_dir: Path,
    config: StochasticOptimizationConfig,
) -> Dict[str, Any]:
    grid_df = pd.read_csv(scenario_dir / "optimizer_grid.csv")
    ev_df = pd.read_csv(scenario_dir / "optimizer_ev_sessions.csv")

    grid_df["timestamp"] = pd.to_datetime(grid_df["timestamp"], errors="coerce")
    for c in ["price_kwh", "emission_ton_per_kwh", "congestion_intensity", "ev_capacity_limit_kw"]:
        if c in grid_df.columns:
            grid_df[c] = pd.to_numeric(grid_df[c], errors="coerce")

    if len(ev_df) > 0:
        if "arrival_time" in ev_df.columns:
            ev_df["arrival_time"] = pd.to_datetime(ev_df["arrival_time"], errors="coerce")
        if "departure_time" in ev_df.columns:
            ev_df["departure_time"] = pd.to_datetime(ev_df["departure_time"], errors="coerce")

    T = len(robust_df)
    dt = config.time_step_hours

    # blend robust capacity with actual scenario capacity to avoid over-conservatism
    actual_capacity = pd.to_numeric(
        grid_df.get("ev_capacity_limit_kw", pd.Series(np.full(T, np.nan))),
        errors="coerce"
    ).fillna(config.capacity_floor_kw).values.astype(float)

    robust_capacity = pd.to_numeric(
        robust_df["robust_capacity_kw"], errors="coerce"
    ).fillna(config.capacity_floor_kw).values.astype(float)

    remaining_capacity = (
        (1.0 - config.capacity_blend_actual) * robust_capacity
        + config.capacity_blend_actual * np.maximum(actual_capacity, config.capacity_floor_kw)
    )
    remaining_capacity = np.maximum(remaining_capacity, config.capacity_floor_kw)

    schedule_rows: List[Dict[str, Any]] = []
    session_rows: List[Dict[str, Any]] = []

    if ev_df.empty:
        ev_df = pd.DataFrame(columns=[
            "ev_id", "arrival_step", "departure_step", "required_energy_kwh",
            "max_charging_power_kw", "battery_capacity_kwh", "init_soc"
        ])

    for _, ev in ev_df.iterrows():
        ev_id = ev["ev_id"]
        arr = max(int(ev["arrival_step"]), 0)
        dep = min(int(ev["departure_step"]), T)

        required_energy = max(float(ev["required_energy_kwh"]), 0.0)
        max_power_kw = float(ev["max_charging_power_kw"])
        capacity_kwh = float(ev["battery_capacity_kwh"])
        init_soc = float(ev["init_soc"])

        if dep <= arr:
            session_rows.append({
                "ev_id": ev_id,
                "required_energy_kwh": required_energy,
                "served_energy_kwh": 0.0,
                "unmet_energy_kwh": required_energy,
                "service_ratio": 0.0,
                "energy_cost": 0.0,
                "emission_ton": 0.0,
                "health_penalty_total": 0.0,
            })
            continue

        feasible_hours = list(range(arr, dep))
        ranked = []

        for t in feasible_hours:
            nominal_charge = min(max_power_kw, max(0.0, remaining_capacity[t]))
            hp = health_penalty_proxy(
                init_soc=init_soc,
                max_power_kw=max_power_kw,
                battery_capacity_kwh=capacity_kwh,
                charge_kw=max(nominal_charge, 0.0),
                cumulative_energy_before_kwh=0.0,
                dt_hours=dt,
                config=config,
            )
            score = float(robust_df.loc[t, "base_score"]) + config.w_health * hp
            ranked.append((t, score))

        ranked_hours = [t for t, _ in sorted(ranked, key=lambda x: x[1])]

        served_energy = 0.0
        total_cost = 0.0
        total_emission = 0.0
        total_health = 0.0

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
                max_power_kw=max_power_kw,
                battery_capacity_kwh=capacity_kwh,
                charge_kw=charge_kw,
                cumulative_energy_before_kwh=served_energy,
                dt_hours=dt,
                config=config,
            )

            energy_kwh = charge_kw * dt
            served_energy += energy_kwh
            remaining_capacity[t] -= charge_kw

            price_kwh = float(grid_df.loc[t, "price_kwh"])
            emission_kwh = float(grid_df.loc[t, "emission_ton_per_kwh"])
            congestion = float(grid_df.loc[t, "congestion_intensity"])

            cost = price_kwh * energy_kwh
            emission = emission_kwh * energy_kwh

            total_cost += cost
            total_emission += emission
            total_health += hp

            schedule_rows.append({
                "ev_id": ev_id,
                "t": t,
                "timestamp": grid_df.loc[t, "timestamp"],
                "charge_kw": charge_kw,
                "energy_kwh": energy_kwh,
                "actual_price_kwh": price_kwh,
                "actual_emission_ton_per_kwh": emission_kwh,
                "actual_congestion_intensity": congestion,
                "robust_score": float(robust_df.loc[t, "base_score"]),
                "health_penalty": hp,
                "cost": cost,
                "emission_ton": emission,
            })

        unmet = max(required_energy - served_energy, 0.0)
        session_rows.append({
            "ev_id": ev_id,
            "arrival_step": arr,
            "departure_step": dep,
            "required_energy_kwh": required_energy,
            "served_energy_kwh": served_energy,
            "unmet_energy_kwh": unmet,
            "service_ratio": 0.0 if required_energy <= 1e-9 else served_energy / required_energy,
            "energy_cost": total_cost,
            "emission_ton": total_emission,
            "health_penalty_total": total_health,
        })

    schedule_df = pd.DataFrame(schedule_rows)
    session_df = pd.DataFrame(session_rows)

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
        hourly_results = grid_df[["t", "timestamp", "load_mw", "price_mwh", "emission_ton_per_mwh", "congestion_intensity"]].merge(
            hourly_sched, on=["t", "timestamp"], how="left"
        )
        for c in ["scheduled_ev_power_kw", "scheduled_ev_energy_kwh", "scheduled_ev_cost", "scheduled_ev_emission_ton", "scheduled_ev_health_penalty"]:
            hourly_results[c] = pd.to_numeric(hourly_results[c], errors="coerce").fillna(0.0)

    total_required = float(session_df["required_energy_kwh"].sum()) if len(session_df) else 0.0
    total_served = float(session_df["served_energy_kwh"].sum()) if len(session_df) else 0.0
    total_unmet = float(session_df["unmet_energy_kwh"].sum()) if len(session_df) else 0.0
    unmet_penalty = config.unmet_penalty_per_kwh * total_unmet if config.allow_unmet else 0.0

    objective = float(
        hourly_results["scheduled_ev_cost"].sum()
        + config.w_emission * hourly_results["scheduled_ev_emission_ton"].sum()
        + config.w_health * hourly_results["scheduled_ev_health_penalty"].sum()
        + unmet_penalty
    )

    summary = {
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
        "weighted_objective_proxy": objective,
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
            "objective_summary": summary,
        }, f, indent=2, ensure_ascii=False, default=str)

    fig_path = scenario_out / "stochastic_ev_scheduling_plot.png"
    plot_stochastic_schedule_for_one_scenario(
        hourly_results=hourly_results,
        session_results=session_df,
        save_path=fig_path,
        scenario_name=scenario_dir.name,
    )

    return summary


def build_risk_summary(summary_df: pd.DataFrame, config: StochasticOptimizationConfig) -> Dict[str, Any]:
    obj = pd.to_numeric(summary_df["weighted_objective_proxy"], errors="coerce").values
    srv = pd.to_numeric(summary_df["service_ratio"], errors="coerce").values
    unmet = pd.to_numeric(summary_df["total_unmet_energy_kwh"], errors="coerce").values

    alpha = config.cvar_alpha

    risk_summary = {
        "n_scenarios": int(len(summary_df)),
        "expected_objective": float(np.nanmean(obj)),
        "worst_case_objective": float(np.nanmax(obj)),
        "cvar_objective": cvar_of_array(obj, alpha),

        "expected_service_ratio": float(np.nanmean(srv)),
        "worst_case_service_ratio": float(np.nanmin(srv)),
        "service_ratio_cvar_bad_tail": cvar_of_array(1.0 - srv, alpha),

        "expected_unmet_energy_kwh": float(np.nanmean(unmet)),
        "worst_case_unmet_energy_kwh": float(np.nanmax(unmet)),
        "cvar_unmet_energy_kwh": cvar_of_array(unmet, alpha),
    }
    return risk_summary


def apply_plot_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.2,
    })


def plot_robust_hourly_signals(
    robust_df: pd.DataFrame,
    save_path: Path,
) -> None:
    apply_plot_style()

    df = robust_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    fig.suptitle("Robust Hourly Signals for Stochastic EV Scheduling", fontsize=15, y=0.98)

    panels = [
        ("price_mean", "price_std", "price_cvar", "price_risk", "Price Signal (kWh-based)", "Risk-adjusted Price"),
        ("emission_mean", "emission_std", "emission_cvar", "emission_risk", "Emission Signal", "Risk-adjusted Emission"),
        ("congestion_mean", "congestion_std", "congestion_cvar", "congestion_risk", "Congestion Signal", "Risk-adjusted Congestion"),
    ]

    for ax, (mcol, scol, ccol, rcol, ylabel, title) in zip(axes[:3], panels):
        ax.plot(df["timestamp"], df[mcol], label="Mean")
        ax.plot(df["timestamp"], df[ccol], linestyle="--", label="CVaR")
        ax.fill_between(
            df["timestamp"],
            df[mcol] - df[scol],
            df[mcol] + df[scol],
            alpha=0.15,
            label="Mean ± Std"
        )
        ax.plot(df["timestamp"], df[rcol], linewidth=2.6, label="Risk-adjusted")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax = axes[3]
    ax.plot(df["timestamp"], df["robust_capacity_kw"], linewidth=2.5, label="Robust Capacity")
    ax.plot(df["timestamp"], df["capacity_mean_kw"], linewidth=1.8, linestyle="--", label="Capacity Mean")
    ax.fill_between(
        df["timestamp"],
        np.maximum(df["capacity_mean_kw"] - df["capacity_std_kw"], 0.0),
        df["capacity_mean_kw"] + df["capacity_std_kw"],
        alpha=0.12,
        label="Capacity Mean ± Std"
    )
    ax.set_ylabel("Capacity (kW)")
    ax.set_title("Robust EV Charging Capacity")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(df) // 8)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.965])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_stochastic_summary(
    summary_df: pd.DataFrame,
    risk_summary: Dict[str, Any],
    save_path: Path,
) -> None:
    apply_plot_style()

    df = summary_df.copy()
    df = df.sort_values("weighted_objective_proxy").reset_index(drop=True)

    fig = plt.figure(figsize=(13, 8), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(df["scenario"], df["weighted_objective_proxy"])
    ax1.axhline(risk_summary["expected_objective"], linestyle="--", linewidth=2, label="Expected")
    ax1.axhline(risk_summary["worst_case_objective"], linestyle=":", linewidth=2, label="Worst-case")
    ax1.axhline(risk_summary["cvar_objective"], linestyle="-.", linewidth=2, label="CVaR")
    ax1.set_title("Scenario Objective")
    ax1.set_ylabel("Objective")
    ax1.tick_params(axis="x", rotation=35)
    ax1.legend(frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(df["scenario"], df["service_ratio"])
    ax2.axhline(risk_summary["expected_service_ratio"], linestyle="--", linewidth=2, label="Expected")
    ax2.axhline(risk_summary["worst_case_service_ratio"], linestyle=":", linewidth=2, label="Worst-case")
    ax2.set_title("Service Ratio")
    ax2.set_ylabel("Ratio")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="x", rotation=35)
    ax2.legend(frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(df["scenario"], df["total_unmet_energy_kwh"])
    ax3.axhline(risk_summary["expected_unmet_energy_kwh"], linestyle="--", linewidth=2, label="Expected")
    ax3.axhline(risk_summary["worst_case_unmet_energy_kwh"], linestyle=":", linewidth=2, label="Worst-case")
    ax3.axhline(risk_summary["cvar_unmet_energy_kwh"], linestyle="-.", linewidth=2, label="CVaR")
    ax3.set_title("Unmet Energy")
    ax3.set_ylabel("kWh")
    ax3.tick_params(axis="x", rotation=35)
    ax3.legend(frameon=False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    text = (
        f"Expected objective: {risk_summary['expected_objective']:.3f}\n"
        f"Worst-case objective: {risk_summary['worst_case_objective']:.3f}\n"
        f"CVaR objective: {risk_summary['cvar_objective']:.3f}\n\n"
        f"Expected service ratio: {risk_summary['expected_service_ratio']:.3f}\n"
        f"Worst-case service ratio: {risk_summary['worst_case_service_ratio']:.3f}\n\n"
        f"Expected unmet energy: {risk_summary['expected_unmet_energy_kwh']:.3f}\n"
        f"Worst-case unmet energy: {risk_summary['worst_case_unmet_energy_kwh']:.3f}\n"
        f"CVaR unmet energy: {risk_summary['cvar_unmet_energy_kwh']:.3f}"
    )
    ax4.text(
        0.03, 0.95, text,
        ha="left", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", alpha=0.08)
    )
    ax4.set_title("Risk Summary", pad=10)

    fig.suptitle("Stochastic Optimization Performance Across Scenarios", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_stochastic_schedule_for_one_scenario(
    hourly_results: pd.DataFrame,
    session_results: pd.DataFrame,
    save_path: Path,
    scenario_name: str,
) -> None:
    apply_plot_style()

    hr = hourly_results.copy()
    sr = session_results.copy()

    hr["timestamp"] = pd.to_datetime(hr["timestamp"], errors="coerce")
    for c in [
        "scheduled_ev_power_kw",
        "scheduled_ev_energy_kwh",
        "scheduled_ev_cost",
        "scheduled_ev_emission_ton",
        "scheduled_ev_health_penalty",
        "price_mwh",
        "emission_ton_per_mwh",
        "congestion_intensity",
    ]:
        if c in hr.columns:
            hr[c] = pd.to_numeric(hr[c], errors="coerce")

    for c in ["served_energy_kwh", "unmet_energy_kwh", "required_energy_kwh", "service_ratio"]:
        if c in sr.columns:
            sr[c] = pd.to_numeric(sr[c], errors="coerce")

    fig = plt.figure(figsize=(13, 9), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.0, 1.15], hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(hr["timestamp"], hr["scheduled_ev_power_kw"], linewidth=2.6, label="Scheduled EV Power")
    ax1.set_title(f"{scenario_name}: Stochastic EV Charging Schedule")
    ax1.set_ylabel("Power (kW)")
    ax1.legend(frameon=False, loc="upper right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    l1 = ax2.plot(hr["timestamp"], hr["price_mwh"], linewidth=2.2, label="Price")[0]
    ax2.set_ylabel("Price ($/MWh)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax2b = ax2.twinx()
    l2 = ax2b.plot(hr["timestamp"], hr["emission_ton_per_mwh"], linewidth=2.0, linestyle="--", label="Emission")[0]
    ax2b.set_ylabel("Emission (ton/MWh)")
    ax2b.spines["top"].set_visible(False)

    ax2.legend([l1, l2], ["Price", "Emission"], frameon=False, loc="upper right")
    ax2.set_title("Actual Scenario Signals")

    ax3 = fig.add_subplot(gs[2, 0])
    if len(sr) > 0:
        sr_plot = sr.sort_values("required_energy_kwh", ascending=False).head(15).copy()
        x = np.arange(len(sr_plot))
        ax3.bar(x, sr_plot["served_energy_kwh"], width=0.75, label="Served")
        ax3.bar(x, sr_plot["unmet_energy_kwh"], width=0.75, bottom=sr_plot["served_energy_kwh"], label="Unmet")
        ax3.set_xticks(x)
        ax3.set_xticklabels(sr_plot["ev_id"], rotation=35, ha="right")
        ax3.set_ylabel("Energy (kWh)")
        ax3.set_title("Session-Level Charging Service (Top 15 EVs)")
        ax3.legend(frameon=False, loc="upper right")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
    else:
        ax3.text(0.5, 0.5, "No EV sessions scheduled", ha="center", va="center", fontsize=12)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title("Session-Level Charging Service")

    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(hr) // 8)))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle("Stochastic Scenario-Based EV Scheduling", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def run_all(config: StochasticOptimizationConfig) -> pd.DataFrame:
    scenario_dirs = list_scenario_dirs(config.input_dir, config.scenario_prefix)
    if not scenario_dirs:
        raise FileNotFoundError(f"No scenario folders found under {config.input_dir}")

    config.out_dir.mkdir(parents=True, exist_ok=True)

    grid_scenarios = load_all_grid_scenarios(scenario_dirs)
    robust_df = build_robust_hourly_signals(grid_scenarios, config)
    robust_df.to_csv(config.out_dir / "robust_hourly_signals.csv", index=False)

    rows = []
    for scenario_dir in scenario_dirs:
        rows.append(
            schedule_one_scenario_with_robust_signals(
                scenario_dir=scenario_dir,
                robust_df=robust_df,
                out_dir=config.out_dir,
                config=config,
            )
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(config.out_dir / "stochastic_results_summary.csv", index=False)

    risk_summary = build_risk_summary(summary_df, config)
    with open(config.out_dir / "stochastic_risk_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": asdict(config),
            "risk_summary": risk_summary,
        }, f, indent=2, ensure_ascii=False, default=str)

    plot_robust_hourly_signals(
        robust_df=robust_df,
        save_path=config.out_dir / "robust_hourly_signals_plot.png",
    )

    plot_stochastic_summary(
        summary_df=summary_df,
        risk_summary=risk_summary,
        save_path=config.out_dir / "stochastic_summary_plot.png",
    )

    return summary_df


def main() -> None:
    config = StochasticOptimizationConfig()
    summary_df = run_all(config)

    print(f"[ok] stochastic optimization results built -> {config.out_dir}")
    print(f"[ok] scenarios processed                   -> {len(summary_df)}")
    print(f"[ok] robust signals saved                  -> {config.out_dir / 'robust_hourly_signals.csv'}")
    print(f"[ok] summary saved                         -> {config.out_dir / 'stochastic_results_summary.csv'}")
    print(f"[ok] risk summary saved                    -> {config.out_dir / 'stochastic_risk_summary.json'}")
    print("\nScenario-level summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()