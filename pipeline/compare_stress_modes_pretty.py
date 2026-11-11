#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_stress_modes_pretty.py

Create cleaner, publication-ready comparison figures for stress-mode results.
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
    out_dir: Path = Path("stress_mode_comparison_pretty")

    def __post_init__(self):
        if self.result_dirs is None:
            self.result_dirs = {
                "Evening peak\ncluster": Path("stochastic_results_evening_peak_cluster"),
                "Sustained\nhigh price": Path("stochastic_results_sustained_high_price"),
                "Capacity\nshortage": Path("stochastic_results_capacity_shortage"),
                "Combined\nextreme": Path("stochastic_results_combined_extreme"),
            }


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_plot_style() -> None:
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
        "lines.linewidth": 2.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def collect_results(config: CompareStressConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    risk_rows: List[Dict[str, Any]] = []
    scenario_rows: List[pd.DataFrame] = []

    for label, result_dir in config.result_dirs.items():
        risk_path = result_dir / "stochastic_risk_summary.json"
        scenario_path = result_dir / "stochastic_results_summary.csv"

        if not risk_path.exists():
            raise FileNotFoundError(f"Missing file: {risk_path}")
        if not scenario_path.exists():
            raise FileNotFoundError(f"Missing file: {scenario_path}")

        risk_payload = load_json(risk_path)
        risk_summary = risk_payload.get("risk_summary", {})
        risk_summary["stress_mode"] = label
        risk_rows.append(risk_summary)

        scenario_df = pd.read_csv(scenario_path)
        scenario_df["stress_mode"] = label
        scenario_rows.append(scenario_df)

    risk_df = pd.DataFrame(risk_rows)
    scenario_df = pd.concat(scenario_rows, ignore_index=True)
    return risk_df, scenario_df


def _bar_labels(ax, values, xpos, fmt="{:.2f}", ypad_frac=0.015):
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    ymax = finite.max() if finite.size else 1.0
    pad = ymax * ypad_frac
    for x, v in zip(xpos, vals):
        if np.isfinite(v):
            ax.text(x, v + pad, fmt.format(v), ha="center", va="bottom", fontsize=9)


def plot_objective_risk_pretty(risk_df: pd.DataFrame, save_path: Path) -> None:
    apply_plot_style()

    df = risk_df.copy().reset_index(drop=True)
    labels = df["stress_mode"].tolist()
    x = np.arange(len(df))
    width = 0.22

    fig, ax = plt.subplots(figsize=(11.5, 6.2), facecolor="white")

    b1 = ax.bar(x - width, df["expected_objective"], width=width, label="Expected", alpha=0.95)
    b2 = ax.bar(x, df["cvar_objective"], width=width, label="CVaR", alpha=0.95)
    b3 = ax.bar(x + width, df["worst_case_objective"], width=width, label="Worst-case", alpha=0.95)

    _bar_labels(ax, df["expected_objective"], x - width, fmt="{:.1f}")
    _bar_labels(ax, df["cvar_objective"], x, fmt="{:.1f}")
    _bar_labels(ax, df["worst_case_objective"], x + width, fmt="{:.1f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Objective value")
    ax.set_title("Risk-sensitive objective across stress modes")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_service_unmet_pretty(risk_df: pd.DataFrame, save_path: Path) -> None:
    apply_plot_style()

    df = risk_df.copy().reset_index(drop=True)
    labels = df["stress_mode"].tolist()
    x = np.arange(len(df))
    width = 0.34

    fig = plt.figure(figsize=(12.2, 6.4), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x - width / 2, df["expected_service_ratio"], width=width, label="Expected", alpha=0.95)
    ax1.bar(x + width / 2, df["worst_case_service_ratio"], width=width, label="Worst-case", alpha=0.95)
    _bar_labels(ax1, df["expected_service_ratio"], x - width / 2, fmt="{:.2f}", ypad_frac=0.01)
    _bar_labels(ax1, df["worst_case_service_ratio"], x + width / 2, fmt="{:.2f}", ypad_frac=0.01)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Service ratio")
    ax1.set_title("Charging service robustness")
    ax1.legend(frameon=False, ncol=2, loc="upper left")
    ax1.grid(True, axis="y")
    ax1.grid(False, axis="x")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x - width / 2, df["expected_unmet_energy_kwh"], width=width, label="Expected unmet", alpha=0.95)
    ax2.bar(x + width / 2, df["cvar_unmet_energy_kwh"], width=width, label="CVaR unmet", alpha=0.95)
    _bar_labels(ax2, df["expected_unmet_energy_kwh"], x - width / 2, fmt="{:.1f}")
    _bar_labels(ax2, df["cvar_unmet_energy_kwh"], x + width / 2, fmt="{:.1f}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Unmet energy (kWh)")
    ax2.set_title("Tail-risk of unmet charging demand")
    ax2.legend(frameon=False, ncol=2, loc="upper left")
    ax2.grid(True, axis="y")
    ax2.grid(False, axis="x")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_scenario_distribution_pretty(scenario_df: pd.DataFrame, save_path: Path) -> None:
    apply_plot_style()

    df = scenario_df.copy()
    # preserve display order from config labels
    order = list(dict.fromkeys(df["stress_mode"].tolist()))
    data = [pd.to_numeric(df.loc[df["stress_mode"] == m, "weighted_objective_proxy"], errors="coerce").dropna().values for m in order]

    fig, ax = plt.subplots(figsize=(11.5, 5.8), facecolor="white")
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        labels=order,
        showfliers=True,
        medianprops=dict(linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        boxprops=dict(linewidth=1.5),
    )

    for patch in bp["boxes"]:
        patch.set_alpha(0.45)

    ax.set_ylabel("Scenario objective")
    ax.set_title("Scenario-level objective distribution across stress modes")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")

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

    plot_objective_risk_pretty(
        risk_df=risk_df,
        save_path=config.out_dir / "compare_objective_risk_pretty.png",
    )
    plot_service_unmet_pretty(
        risk_df=risk_df,
        save_path=config.out_dir / "compare_service_unmet_pretty.png",
    )
    plot_scenario_distribution_pretty(
        scenario_df=scenario_df,
        save_path=config.out_dir / "compare_scenario_distribution_pretty.png",
    )

    print(f"[ok] comparison results saved -> {config.out_dir}")
    print(f"[ok] risk summary saved       -> {config.out_dir / 'stress_mode_risk_summary.csv'}")
    print(f"[ok] scenario summary saved   -> {config.out_dir / 'stress_mode_scenario_summary.csv'}")
    print(f"[ok] figure saved             -> {config.out_dir / 'compare_objective_risk_pretty.png'}")
    print(f"[ok] figure saved             -> {config.out_dir / 'compare_service_unmet_pretty.png'}")
    print(f"[ok] figure saved             -> {config.out_dir / 'compare_scenario_distribution_pretty.png'}")
    print("\\nRisk summary:")
    print(risk_df.to_string(index=False))


if __name__ == "__main__":
    main()
