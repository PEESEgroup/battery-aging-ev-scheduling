#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_mean_cvar_objective.py

Evaluate mean-CVaR objectives from existing stochastic optimization results.

Expected input structure:
    stochastic_results_joint/
        stochastic_results_summary.csv
        scenario_00/hourly_results.csv
        scenario_01/hourly_results.csv
        ...
or
    stochastic_results_stress/
        stochastic_results_summary.csv
        scenario_00/hourly_results.csv
        ...

Outputs:
    mean_cvar_evaluation/
        lambda_sweep_evaluation.csv
        scenario_risk_metrics.csv
        mean_cvar_summary.json
        lambda_tradeoff_plot.png
        risk_components_plot.png
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class MeanCVARConfig:
    input_dir: Path = Path("stochastic_results_stress")
    out_dir: Path = Path("mean_cvar_evaluation")

    scenario_prefix: str = "scenario_"
    cvar_alpha: float = 0.9

    # lambda sweep
    lambda_values: tuple = (0.0, 0.3, 0.7)

    # weights for total scenario objective
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


def load_base_summary(input_dir: Path) -> pd.DataFrame:
    summary_path = input_dir / "stochastic_results_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing file: {summary_path}")
    return pd.read_csv(summary_path)


def compute_congestion_exposure(hourly_df: pd.DataFrame) -> float:
    df = hourly_df.copy()
    for c in ["scheduled_ev_energy_kwh", "scheduled_ev_power_kw", "congestion_intensity"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if "scheduled_ev_energy_kwh" in df.columns:
        return float((df["scheduled_ev_energy_kwh"] * df["congestion_intensity"]).sum())
    return float((df["scheduled_ev_power_kw"] * df["congestion_intensity"]).sum())


def build_scenario_risk_metrics(config: MeanCVARConfig) -> pd.DataFrame:
    summary_df = load_base_summary(config.input_dir)
    scenario_dirs = list_scenario_dirs(config.input_dir, config.scenario_prefix)

    metrics_rows: List[Dict[str, Any]] = []

    summary_lookup = {row["scenario"]: row for _, row in summary_df.iterrows()}

    for sdir in scenario_dirs:
        scenario = sdir.name
        hourly_path = sdir / "hourly_results.csv"
        session_path = sdir / "session_results.csv"

        if not hourly_path.exists():
            raise FileNotFoundError(f"Missing file: {hourly_path}")
        if not session_path.exists():
            raise FileNotFoundError(f"Missing file: {session_path}")

        hourly_df = pd.read_csv(hourly_path)
        session_df = pd.read_csv(session_path)

        peak_ev_power_kw = float(pd.to_numeric(hourly_df.get("scheduled_ev_power_kw"), errors="coerce").fillna(0.0).max())
        congestion_exposure = compute_congestion_exposure(hourly_df)

        lookup = summary_lookup.get(scenario, {})
        metrics_rows.append({
            "scenario": scenario,
            "weighted_objective_proxy": float(lookup.get("weighted_objective_proxy", np.nan)),
            "service_ratio": float(lookup.get("service_ratio", np.nan)),
            "total_unmet_energy_kwh": float(lookup.get("total_unmet_energy_kwh", np.nan)),
            "total_energy_cost": float(lookup.get("total_energy_cost", np.nan)),
            "total_emission_ton": float(lookup.get("total_emission_ton", np.nan)),
            "total_health_penalty": float(lookup.get("total_health_penalty", np.nan)),
            "peak_ev_power_kw": peak_ev_power_kw,
            "congestion_exposure": congestion_exposure,
            "n_sessions": int(len(session_df)),
        })

    metrics_df = pd.DataFrame(metrics_rows).sort_values("scenario").reset_index(drop=True)

    if config.verbose_debug and len(metrics_df):
        print("[debug] scenario metrics preview:")
        print(metrics_df.head().to_string(index=False))

    return metrics_df


def evaluate_lambda_sweep(metrics_df: pd.DataFrame, config: MeanCVARConfig) -> pd.DataFrame:
    J = pd.to_numeric(metrics_df["weighted_objective_proxy"], errors="coerce").values
    U = pd.to_numeric(metrics_df["total_unmet_energy_kwh"], errors="coerce").values
    P = pd.to_numeric(metrics_df["peak_ev_power_kw"], errors="coerce").values
    G = pd.to_numeric(metrics_df["congestion_exposure"], errors="coerce").values
    S = pd.to_numeric(metrics_df["service_ratio"], errors="coerce").values

    EJ = float(np.nanmean(J))
    CVJ = cvar_of_array(J, config.cvar_alpha)

    EU = float(np.nanmean(U))
    CVU = cvar_of_array(U, config.cvar_alpha)

    EP = float(np.nanmean(P))
    CVP = cvar_of_array(P, config.cvar_alpha)

    EG = float(np.nanmean(G))
    CVG = cvar_of_array(G, config.cvar_alpha)

    ES = float(np.nanmean(S))
    WS = float(np.nanmin(S))

    rows: List[Dict[str, Any]] = []
    for lam in config.lambda_values:
        total_mean_cvar_objective = (
            (1.0 - lam) * EJ
            + lam * CVJ
            + config.eta_unmet_cvar * CVU
            + config.eta_peak_cvar * CVP
            + config.eta_congestion_cvar * CVG
        )

        rows.append({
            "lambda": float(lam),
            "alpha": float(config.cvar_alpha),
            "expected_objective": EJ,
            "cvar_objective": CVJ,
            "expected_unmet_energy_kwh": EU,
            "cvar_unmet_energy_kwh": CVU,
            "expected_peak_ev_power_kw": EP,
            "cvar_peak_ev_power_kw": CVP,
            "expected_congestion_exposure": EG,
            "cvar_congestion_exposure": CVG,
            "expected_service_ratio": ES,
            "worst_case_service_ratio": WS,
            "eta_unmet_cvar": float(config.eta_unmet_cvar),
            "eta_peak_cvar": float(config.eta_peak_cvar),
            "eta_congestion_cvar": float(config.eta_congestion_cvar),
            "total_mean_cvar_objective": float(total_mean_cvar_objective),
        })

    return pd.DataFrame(rows)


def plot_lambda_tradeoff(lambda_df: pd.DataFrame, save_path: Path) -> None:
    apply_plot_style()

    df = lambda_df.copy().sort_values("lambda").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5.8), facecolor="white")
    ax.plot(df["lambda"], df["expected_objective"], marker="o", label="Expected objective")
    ax.plot(df["lambda"], df["cvar_objective"], marker="s", label="CVaR objective")
    ax.plot(df["lambda"], df["total_mean_cvar_objective"], marker="^", label="Total mean-CVaR objective")

    for _, row in df.iterrows():
        ax.text(row["lambda"], row["total_mean_cvar_objective"], f"{row['total_mean_cvar_objective']:.1f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Risk aversion λ")
    ax.set_ylabel("Objective value")
    ax.set_title("Mean–CVaR trade-off across risk aversion levels")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_risk_components(lambda_df: pd.DataFrame, save_path: Path) -> None:
    apply_plot_style()

    df = lambda_df.copy().sort_values("lambda").reset_index(drop=True)

    fig = plt.figure(figsize=(12, 6.2), facecolor="white")
    gs = fig.add_gridspec(1, 2, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df["lambda"], df["cvar_unmet_energy_kwh"], marker="o", label="CVaR unmet")
    ax1.plot(df["lambda"], df["cvar_peak_ev_power_kw"], marker="s", label="CVaR peak power")
    ax1.set_xlabel("Risk aversion λ")
    ax1.set_ylabel("Risk metric value")
    ax1.set_title("Risk of unmet demand and peak power")
    ax1.legend(frameon=False, loc="upper left")
    ax1.grid(True, axis="y")
    ax1.grid(False, axis="x")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df["lambda"], df["cvar_congestion_exposure"], marker="^", label="CVaR congestion exposure")
    ax2.plot(df["lambda"], df["expected_service_ratio"], marker="d", label="Expected service ratio")
    ax2.set_xlabel("Risk aversion λ")
    ax2.set_ylabel("Metric value")
    ax2.set_title("Congestion tail-risk and service quality")
    ax2.legend(frameon=False, loc="upper left")
    ax2.grid(True, axis="y")
    ax2.grid(False, axis="x")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def main() -> None:
    config = MeanCVARConfig()
    config.out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = build_scenario_risk_metrics(config)
    lambda_df = evaluate_lambda_sweep(metrics_df, config)

    metrics_df.to_csv(config.out_dir / "scenario_risk_metrics.csv", index=False)
    lambda_df.to_csv(config.out_dir / "lambda_sweep_evaluation.csv", index=False)

    summary_payload = {
        "input_dir": str(config.input_dir),
        "config": asdict(config),
        "n_scenarios": int(len(metrics_df)),
        "alpha": float(config.cvar_alpha),
        "lambda_values": list(config.lambda_values),
        "expected_objective": float(lambda_df["expected_objective"].iloc[0]) if len(lambda_df) else None,
        "cvar_objective": float(lambda_df["cvar_objective"].iloc[0]) if len(lambda_df) else None,
        "expected_unmet_energy_kwh": float(lambda_df["expected_unmet_energy_kwh"].iloc[0]) if len(lambda_df) else None,
        "cvar_unmet_energy_kwh": float(lambda_df["cvar_unmet_energy_kwh"].iloc[0]) if len(lambda_df) else None,
        "expected_peak_ev_power_kw": float(lambda_df["expected_peak_ev_power_kw"].iloc[0]) if len(lambda_df) else None,
        "cvar_peak_ev_power_kw": float(lambda_df["cvar_peak_ev_power_kw"].iloc[0]) if len(lambda_df) else None,
        "expected_congestion_exposure": float(lambda_df["expected_congestion_exposure"].iloc[0]) if len(lambda_df) else None,
        "cvar_congestion_exposure": float(lambda_df["cvar_congestion_exposure"].iloc[0]) if len(lambda_df) else None,
    }

    with open(config.out_dir / "mean_cvar_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False, default=str)

    plot_lambda_tradeoff(
        lambda_df=lambda_df,
        save_path=config.out_dir / "lambda_tradeoff_plot.png",
    )
    plot_risk_components(
        lambda_df=lambda_df,
        save_path=config.out_dir / "risk_components_plot.png",
    )

    print(f"[ok] evaluation saved        -> {config.out_dir}")
    print(f"[ok] scenario metrics saved  -> {config.out_dir / 'scenario_risk_metrics.csv'}")
    print(f"[ok] lambda sweep saved      -> {config.out_dir / 'lambda_sweep_evaluation.csv'}")
    print(f"[ok] summary saved           -> {config.out_dir / 'mean_cvar_summary.json'}")
    print(f"[ok] figure saved            -> {config.out_dir / 'lambda_tradeoff_plot.png'}")
    print(f"[ok] figure saved            -> {config.out_dir / 'risk_components_plot.png'}")
    print("\\nLambda sweep evaluation:")
    print(lambda_df.to_string(index=False))


if __name__ == "__main__":
    main()
