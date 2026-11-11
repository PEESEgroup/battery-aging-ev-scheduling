#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_stress_modes.py

Compare stochastic optimization results across multiple stress modes.

Expected inputs:
    stochastic_results_evening_peak_cluster/
        stochastic_results_summary.csv
        stochastic_risk_summary.json

    stochastic_results_sustained_high_price/
        stochastic_results_summary.csv
        stochastic_risk_summary.json

    stochastic_results_capacity_shortage/
        stochastic_results_summary.csv
        stochastic_risk_summary.json

    stochastic_results_combined_extreme/
        stochastic_results_summary.csv
        stochastic_risk_summary.json

Outputs:
    stress_mode_comparison/
        stress_mode_risk_summary.csv
        stress_mode_scenario_summary.csv
        compare_objective_risk.png
        compare_service_unmet.png
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class CompareStressConfig:
    result_dirs: Dict[str, Path] = None
    out_dir: Path = Path("stress_mode_comparison")

    def __post_init__(self):
        if self.result_dirs is None:
            self.result_dirs = {
                "evening_peak_cluster": Path("stochastic_results_evening_peak_cluster"),
                "sustained_high_price": Path("stochastic_results_sustained_high_price"),
                "capacity_shortage": Path("stochastic_results_capacity_shortage"),
                "combined_extreme": Path("stochastic_results_combined_extreme"),
            }


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def collect_results(config: CompareStressConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    risk_rows: List[Dict[str, Any]] = []
    scenario_rows: List[pd.DataFrame] = []

    for mode, result_dir in config.result_dirs.items():
        risk_path = result_dir / "stochastic_risk_summary.json"
        scenario_path = result_dir / "stochastic_results_summary.csv"

        if not risk_path.exists():
            raise FileNotFoundError(f"Missing file: {risk_path}")
        if not scenario_path.exists():
            raise FileNotFoundError(f"Missing file: {scenario_path}")

        risk_payload = load_json(risk_path)
        risk_summary = risk_payload.get("risk_summary", {})
        risk_summary["stress_mode"] = mode
        risk_rows.append(risk_summary)

        scenario_df = pd.read_csv(scenario_path)
        scenario_df["stress_mode"] = mode
        scenario_rows.append(scenario_df)

    risk_df = pd.DataFrame(risk_rows)
    scenario_df = pd.concat(scenario_rows, ignore_index=True)

    return risk_df, scenario_df


def plot_objective_risk_comparison(risk_df: pd.DataFrame, save_path: Path) -> None:
    apply_plot_style()

    df = risk_df.copy()
    order = [
        "evening_peak_cluster",
        "sustained_high_price",
        "capacity_shortage",
        "combined_extreme",
    ]
    df["stress_mode"] = pd.Categorical(df["stress_mode"], categories=order, ordered=True)
    df = df.sort_values("stress_mode").reset_index(drop=True)

    x = np.arange(len(df))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
    ax.bar(x - width, df["expected_objective"], width=width, label="Expected")
    ax.bar(x, df["worst_case_objective"], width=width, label="Worst-case")
    ax.bar(x + width, df["cvar_objective"], width=width, label="CVaR")

    ax.set_xticks(x)
    ax.set_xticklabels(df["stress_mode"], rotation=20, ha="right")
    ax.set_ylabel("Objective")
    ax.set_title("Objective Risk Comparison Across Stress Modes")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_service_unmet_comparison(risk_df: pd.DataFrame, save_path: Path) -> None:
    apply_plot_style()

    df = risk_df.copy()
    order = [
        "evening_peak_cluster",
        "sustained_high_price",
        "capacity_shortage",
        "combined_extreme",
    ]
    df["stress_mode"] = pd.Categorical(df["stress_mode"], categories=order, ordered=True)
    df = df.sort_values("stress_mode").reset_index(drop=True)

    fig = plt.figure(figsize=(12, 7), facecolor="white")
    gs = fig.add_gridspec(1, 2, wspace=0.3)

    x = np.arange(len(df))
    width = 0.36

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x - width / 2, df["expected_service_ratio"], width=width, label="Expected")
    ax1.bar(x + width / 2, df["worst_case_service_ratio"], width=width, label="Worst-case")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["stress_mode"], rotation=20, ha="right")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Service Ratio")
    ax1.set_title("Service Ratio Comparison")
    ax1.legend(frameon=False, loc="upper left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x - width / 2, df["expected_unmet_energy_kwh"], width=width, label="Expected")
    ax2.bar(x + width / 2, df["cvar_unmet_energy_kwh"], width=width, label="CVaR")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["stress_mode"], rotation=20, ha="right")
    ax2.set_ylabel("Unmet Energy (kWh)")
    ax2.set_title("Unmet Energy Comparison")
    ax2.legend(frameon=False, loc="upper left")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def main() -> None:
    config = CompareStressConfig()
    config.out_dir.mkdir(parents=True, exist_ok=True)

    risk_df, scenario_df = collect_results(config)

    risk_df.to_csv(config.out_dir / "stress_mode_risk_summary.csv", index=False)
    scenario_df.to_csv(config.out_dir / "stress_mode_scenario_summary.csv", index=False)

    plot_objective_risk_comparison(
        risk_df=risk_df,
        save_path=config.out_dir / "compare_objective_risk.png",
    )
    plot_service_unmet_comparison(
        risk_df=risk_df,
        save_path=config.out_dir / "compare_service_unmet.png",
    )

    print(f"[ok] comparison results saved -> {config.out_dir}")
    print(f"[ok] risk summary saved       -> {config.out_dir / 'stress_mode_risk_summary.csv'}")
    print(f"[ok] scenario summary saved   -> {config.out_dir / 'stress_mode_scenario_summary.csv'}")
    print(f"[ok] figure saved             -> {config.out_dir / 'compare_objective_risk.png'}")
    print(f"[ok] figure saved             -> {config.out_dir / 'compare_service_unmet.png'}")
    print("\\nRisk summary:")
    print(risk_df.to_string(index=False))


if __name__ == "__main__":
    main()
