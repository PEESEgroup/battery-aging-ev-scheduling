#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_lambda_sweep_stochastic_optimization.py

Run stochastic EV scheduling under multiple risk-aversion levels lambda.

This script extends the existing stochastic optimization workflow by:
1) reading one input scenario set (e.g., optimization_inputs_joint or optimization_inputs_stress)
2) building robust hourly signals once
3) evaluating a lambda grid for mean-CVaR style risk preference
4) re-running scheduling for each lambda
5) saving lambda-specific results and a cross-lambda comparison

Expected input:
    optimization_inputs_joint/
        scenario_00/
            optimizer_grid.csv
            optimizer_ev_sessions.csv
            optimizer_ev_hourly.csv
        ...

Outputs:
    lambda_sweep_results/
        lambda_0_0/
            robust_hourly_signals.csv
            scenario_00/
                charging_schedule.csv
                hourly_results.csv
                session_results.csv
                objective_summary.json
            ...
            stochastic_results_summary.csv
            stochastic_risk_summary.json
        lambda_0_3/
            ...
        lambda_0_7/
            ...
        lambda_sweep_summary.csv
        lambda_sweep_tradeoff.png
        lambda_sweep_service_risk.png
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class LambdaSweepConfig:
    input_dir = Path("optimization_inputs_stress")
    out_dir: Path = Path("lambda_sweep_results")
    scenario_prefix: str = "scenario_"
    time_step_hours: float = 1.0

    # risk preference sweep
    lambda_values: tuple = (0.0, 0.3, 0.7)
    cvar_alpha: float = 0.9

    # base objective weights
    w_price: float = 1.0
    w_emission: float = 30.0
    w_congestion: float = 0.2
    w_health: float = 0.8

    # risk signal construction
    lambda_std: float = 0.5
    lambda_cvar: float = 1.0

    # robust capacity
    capacity_floor_kw: float = 1500.0
    capacity_std_factor: float = 0.2
    capacity_blend_actual: float = 0.3

    # health proxy
    soc_threshold: float = 0.65
    alpha_throughput: float = 1.0
    beta_high_soc: float = 4.0
    gamma_high_power: float = 2.5

    # unmet service
    allow_unmet: bool = True
    unmet_penalty_per_kwh: float = 2.0

    # extra mean-CVaR evaluation weights
    eta_unmet_cvar: float = 2.0
    eta_peak_cvar: float = 0.2
    eta_congestion_cvar: float = 0.5

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
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def health_penalty_proxy(
    init_soc: float,
    max_power_kw: float,
    battery_capacity_kwh: float,
    charge_kw: float,
    cumulative_energy_before_kwh: float,
    dt_hours: float,
    config: LambdaSweepConfig,
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
    config: LambdaSweepConfig,
    lambda_risk: float,
) -> pd.DataFrame:
    all_grid = pd.concat(grid_scenarios, ignore_index=True)

    for c in ["price_kwh", "emission_ton_per_kwh", "congestion_intensity", "ev_capacity_limit_kw"]:
        all_grid[c] = pd.to_numeric(all_grid[c], errors="coerce")

    rows = []
    for t, g in all_grid.groupby("t", as_index=False):
        price_vals = g["price_kwh"].values
        emission_vals = g["emission_ton_per_kwh"].values
        congestion_vals = g["congestion_intensity"].values
        capacity_vals = g["ev_capacity_limit_kw"].values

        cap_mean = float(np.nanmean(capacity_vals))
        cap_std = float(np.nanstd(capacity_vals))
        robust_capacity = max(cap_mean - config.capacity_std_factor * cap_std, config.capacity_floor_kw)

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

    # lambda_risk controls mean-CVaR weighting in risk signals
    robust_df["price_risk"] = (
        (1.0 - lambda_risk) * robust_df["price_mean"]
        + lambda_risk * robust_df["price_cvar"]
        + config.lambda_std * robust_df["price_std"]
    )
    robust_df["emission_risk"] = (
        (1.0 - lambda_risk) * robust_df["emission_mean"]
        + lambda_risk * robust_df["emission_cvar"]
        + config.lambda_std * robust_df["emission_std"]
    )
    robust_df["congestion_risk"] = (
        (1.0 - lambda_risk) * robust_df["congestion_mean"]
        + lambda_risk * robust_df["congestion_cvar"]
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

    if config.verbose_debug:
        print(
            f"[debug][lambda={lambda_risk}] robust capacity min/mean/max:",
            float(robust_df["robust_capacity_kw"].min()),
            float(robust_df["robust_capacity_kw"].mean()),
            float(robust_df["robust_capacity_kw"].max()),
        )

    return robust_df


def schedule_one_scenario(
    scenario_dir: Path,
    robust_df: pd.DataFrame,
    out_dir: Path,
    config: LambdaSweepConfig,
    lambda_risk: float,
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

    peak_ev_power = float(hourly_results["scheduled_ev_power_kw"].max()) if len(hourly_results) else 0.0
    congestion_exposure = float(
        (pd.to_numeric(hourly_results["scheduled_ev_energy_kwh"], errors="coerce").fillna(0.0)
         * pd.to_numeric(hourly_results["congestion_intensity"], errors="coerce").fillna(0.0)).sum()
    ) if len(hourly_results) else 0.0

    summary = {
        "scenario": scenario_dir.name,
        "lambda": float(lambda_risk),
        "n_ev_sessions": int(len(session_df)),
        "horizon_hours": int(T),
        "total_required_energy_kwh": total_required,
        "total_served_energy_kwh": total_served,
        "total_unmet_energy_kwh": total_unmet,
        "service_ratio": 0.0 if total_required <= 1e-9 else total_served / total_required,
        "total_energy_cost": float(hourly_results["scheduled_ev_cost"].sum()),
        "total_emission_ton": float(hourly_results["scheduled_ev_emission_ton"].sum()),
        "total_health_penalty": float(hourly_results["scheduled_ev_health_penalty"].sum()),
        "peak_ev_power_kw": peak_ev_power,
        "congestion_exposure": congestion_exposure,
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
            "lambda": float(lambda_risk),
            "input_dir": str(config.input_dir),
            "config": asdict(config),
            "objective_summary": summary,
        }, f, indent=2, ensure_ascii=False, default=str)

    return summary


def build_risk_summary(summary_df: pd.DataFrame, config: LambdaSweepConfig, lambda_risk: float) -> Dict[str, Any]:
    obj = pd.to_numeric(summary_df["weighted_objective_proxy"], errors="coerce").values
    srv = pd.to_numeric(summary_df["service_ratio"], errors="coerce").values
    unmet = pd.to_numeric(summary_df["total_unmet_energy_kwh"], errors="coerce").values
    peak = pd.to_numeric(summary_df["peak_ev_power_kw"], errors="coerce").values
    cong = pd.to_numeric(summary_df["congestion_exposure"], errors="coerce").values

    EJ = float(np.nanmean(obj))
    CVJ = cvar_of_array(obj, config.cvar_alpha)
    CVU = cvar_of_array(unmet, config.cvar_alpha)
    CVP = cvar_of_array(peak, config.cvar_alpha)
    CVG = cvar_of_array(cong, config.cvar_alpha)

    total_mean_cvar_objective = float(
        (1.0 - lambda_risk) * EJ
        + lambda_risk * CVJ
        + config.eta_unmet_cvar * CVU
        + config.eta_peak_cvar * CVP
        + config.eta_congestion_cvar * CVG
    )

    return {
        "input_dir": Path(config.input_dir).name,
        "lambda": float(lambda_risk),
        "alpha": float(config.cvar_alpha),
        "n_scenarios": int(len(summary_df)),
        "expected_objective": EJ,
        "cvar_objective": CVJ,
        "worst_case_objective": float(np.nanmax(obj)),
        "expected_service_ratio": float(np.nanmean(srv)),
        "worst_case_service_ratio": float(np.nanmin(srv)),
        "expected_unmet_energy_kwh": float(np.nanmean(unmet)),
        "cvar_unmet_energy_kwh": CVU,
        "expected_peak_ev_power_kw": float(np.nanmean(peak)),
        "cvar_peak_ev_power_kw": CVP,
        "expected_congestion_exposure": float(np.nanmean(cong)),
        "cvar_congestion_exposure": CVG,
        "total_mean_cvar_objective": total_mean_cvar_objective,
    }


def run_one_lambda(config: LambdaSweepConfig, lambda_risk: float) -> Dict[str, Any]:
    scenario_dirs = list_scenario_dirs(config.input_dir, config.scenario_prefix)
    if not scenario_dirs:
        raise FileNotFoundError(f"No scenario folders found under {config.input_dir}")

    lambda_tag = str(lambda_risk).replace(".", "_")
    lambda_out_dir = config.out_dir / f"lambda_{lambda_tag}"
    lambda_out_dir.mkdir(parents=True, exist_ok=True)

    grid_scenarios = load_all_grid_scenarios(scenario_dirs)
    robust_df = build_robust_hourly_signals(grid_scenarios, config, lambda_risk)
    robust_df.to_csv(lambda_out_dir / "robust_hourly_signals.csv", index=False)

    rows = []
    for scenario_dir in scenario_dirs:
        rows.append(
            schedule_one_scenario(
                scenario_dir=scenario_dir,
                robust_df=robust_df,
                out_dir=lambda_out_dir,
                config=config,
                lambda_risk=lambda_risk,
            )
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(lambda_out_dir / "stochastic_results_summary.csv", index=False)

    risk_summary = build_risk_summary(summary_df, config, lambda_risk)
    with open(lambda_out_dir / "stochastic_risk_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": asdict(config),
            "risk_summary": risk_summary,
        }, f, indent=2, ensure_ascii=False, default=str)

    return risk_summary


def plot_lambda_tradeoff(summary_df: pd.DataFrame, save_path: Path) -> None:
    apply_plot_style()
    df = summary_df.sort_values("lambda").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor="white")
    ax.plot(df["lambda"], df["expected_objective"], marker="o", label="Expected objective")
    ax.plot(df["lambda"], df["cvar_objective"], marker="s", label="CVaR objective")
    ax.plot(df["lambda"], df["total_mean_cvar_objective"], marker="^", label="Total mean-CVaR objective")

    for _, row in df.iterrows():
        ax.text(row["lambda"], row["total_mean_cvar_objective"], f"{row['total_mean_cvar_objective']:.1f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Risk aversion λ")
    ax.set_ylabel("Objective value")
    ax.set_title("Lambda sweep: mean–CVaR trade-off")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_lambda_service_risk(summary_df: pd.DataFrame, save_path: Path) -> None:
    apply_plot_style()
    df = summary_df.sort_values("lambda").reset_index(drop=True)

    fig = plt.figure(figsize=(12, 6.2), facecolor="white")
    gs = fig.add_gridspec(1, 2, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df["lambda"], df["expected_service_ratio"], marker="o", label="Expected service ratio")
    ax1.plot(df["lambda"], df["worst_case_service_ratio"], marker="s", label="Worst-case service ratio")
    ax1.set_xlabel("Risk aversion λ")
    ax1.set_ylabel("Service ratio")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Service robustness across λ")
    ax1.legend(frameon=False, loc="lower left")
    ax1.grid(True, axis="y")
    ax1.grid(False, axis="x")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df["lambda"], df["cvar_unmet_energy_kwh"], marker="o", label="CVaR unmet")
    ax2.plot(df["lambda"], df["cvar_peak_ev_power_kw"], marker="s", label="CVaR peak power")
    ax2.plot(df["lambda"], df["cvar_congestion_exposure"], marker="^", label="CVaR congestion exposure")
    ax2.set_xlabel("Risk aversion λ")
    ax2.set_ylabel("Risk metric value")
    ax2.set_title("Tail-risk of critical constraints")
    ax2.legend(frameon=False, loc="upper left")
    ax2.grid(True, axis="y")
    ax2.grid(False, axis="x")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def main() -> None:
    config = LambdaSweepConfig()
    config.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for lam in config.lambda_values:
        print(f"[run] lambda = {lam}")
        rows.append(run_one_lambda(config, float(lam)))

    lambda_summary_df = pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)
    lambda_summary_df.to_csv(config.out_dir / "lambda_sweep_summary.csv", index=False)

    plot_lambda_tradeoff(
        summary_df=lambda_summary_df,
        save_path=config.out_dir / "lambda_sweep_tradeoff.png",
    )
    plot_lambda_service_risk(
        summary_df=lambda_summary_df,
        save_path=config.out_dir / "lambda_sweep_service_risk.png",
    )

    print(f"[ok] lambda sweep results saved -> {config.out_dir}")
    print(f"[ok] summary saved              -> {config.out_dir / 'lambda_sweep_summary.csv'}")
    print(f"[ok] figure saved               -> {config.out_dir / 'lambda_sweep_tradeoff.png'}")
    print(f"[ok] figure saved               -> {config.out_dir / 'lambda_sweep_service_risk.png'}")
    print("\\nLambda sweep summary:")
    print(lambda_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
