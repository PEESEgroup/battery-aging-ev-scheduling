#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_one_shot_vs_receding_horizon.py

Compare one-shot stochastic optimization with receding-horizon optimization.

Expected inputs:
    stochastic_results_joint/
        stochastic_results_summary.csv
    receding_horizon_results/
        receding_horizon_summary.csv

Outputs:
    compare_one_shot_vs_rh/
        comparison_summary.csv
        merged_scenario_metrics.csv
        compare_objective_service.png
        compare_unmet_peak_congestion.png
        compare_distribution_boxplot.png
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class CompareOSRHConfig:
    one_shot_summary_path: Path = Path("stochastic_results_stress/stochastic_results_summary.csv")
    rh_summary_path: Path = Path("receding_horizon_results/receding_horizon_summary.csv")
    out_dir: Path = Path("compare_one_shot_vs_rh")

    one_shot_label: str = "One-shot stochastic"
    rh_label: str = "Receding horizon"


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


def cvar_of_array(x: np.ndarray, alpha: float = 0.9) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    q = np.quantile(x, alpha)
    tail = x[x >= q]
    if len(tail) == 0:
        return float(q)
    return float(np.mean(tail))


def load_and_standardize(one_shot_path: Path, rh_path: Path):
    if not one_shot_path.exists():
        raise FileNotFoundError(f"Missing file: {one_shot_path}")
    if not rh_path.exists():
        raise FileNotFoundError(f"Missing file: {rh_path}")

    os_df = pd.read_csv(one_shot_path)
    rh_df = pd.read_csv(rh_path)

    needed = [
        "scenario",
        "total_required_energy_kwh",
        "total_served_energy_kwh",
        "total_unmet_energy_kwh",
        "service_ratio",
        "total_energy_cost",
        "total_emission_ton",
        "total_health_penalty",
        "peak_ev_power_kw",
        "congestion_exposure",
        "weighted_objective_proxy",
    ]

    for col in needed:
        if col not in os_df.columns:
            os_df[col] = np.nan
        if col not in rh_df.columns:
            rh_df[col] = np.nan

    os_df = os_df[needed].copy()
    rh_df = rh_df[needed].copy()

    os_df["method"] = "One-shot stochastic"
    rh_df["method"] = "Receding horizon"

    return os_df, rh_df


def build_summary_table(os_df: pd.DataFrame, rh_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, df in [("One-shot stochastic", os_df), ("Receding horizon", rh_df)]:
        obj = pd.to_numeric(df["weighted_objective_proxy"], errors="coerce").values
        srv = pd.to_numeric(df["service_ratio"], errors="coerce").values
        unmet = pd.to_numeric(df["total_unmet_energy_kwh"], errors="coerce").values
        peak = pd.to_numeric(df["peak_ev_power_kw"], errors="coerce").values
        cong = pd.to_numeric(df["congestion_exposure"], errors="coerce").values
        cost = pd.to_numeric(df["total_energy_cost"], errors="coerce").values
        emission = pd.to_numeric(df["total_emission_ton"], errors="coerce").values

        rows.append({
            "method": label,
            "n_scenarios": int(len(df)),
            "expected_objective": float(np.nanmean(obj)),
            "worst_case_objective": float(np.nanmax(obj)),
            "cvar_objective": cvar_of_array(obj),
            "expected_service_ratio": float(np.nanmean(srv)),
            "worst_case_service_ratio": float(np.nanmin(srv)),
            "expected_unmet_energy_kwh": float(np.nanmean(unmet)),
            "cvar_unmet_energy_kwh": cvar_of_array(unmet),
            "expected_peak_ev_power_kw": float(np.nanmean(peak)),
            "cvar_peak_ev_power_kw": cvar_of_array(peak),
            "expected_congestion_exposure": float(np.nanmean(cong)),
            "cvar_congestion_exposure": cvar_of_array(cong),
            "expected_energy_cost": float(np.nanmean(cost)),
            "expected_emission_ton": float(np.nanmean(emission)),
        })
    return pd.DataFrame(rows)


def build_merged_metrics(os_df: pd.DataFrame, rh_df: pd.DataFrame) -> pd.DataFrame:
    return os_df.merge(
        rh_df,
        on="scenario",
        how="outer",
        suffixes=("_one_shot", "_rh")
    )


def _annotate_bars(ax, bars, fmt="{:.2f}", frac=0.015):
    heights = [b.get_height() for b in bars]
    finite = [h for h in heights if np.isfinite(h)]
    ymax = max(finite) if finite else 1.0
    pad = ymax * frac
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + pad,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=9,
            )


def plot_objective_service(summary_df: pd.DataFrame, save_path: Path) -> None:
    apply_plot_style()

    df = summary_df.copy()
    methods = df["method"].tolist()
    x = np.arange(len(methods))
    width = 0.26

    fig = plt.figure(figsize=(12, 6.6), facecolor="white")
    gs = fig.add_gridspec(1, 2, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    b1 = ax1.bar(x - width, df["expected_objective"], width=width, label="Expected", alpha=0.95)
    b2 = ax1.bar(x, df["cvar_objective"], width=width, label="CVaR", alpha=0.95)
    b3 = ax1.bar(x + width, df["worst_case_objective"], width=width, label="Worst-case", alpha=0.95)
    _annotate_bars(ax1, b1, fmt="{:.1f}")
    _annotate_bars(ax1, b2, fmt="{:.1f}")
    _annotate_bars(ax1, b3, fmt="{:.1f}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=12, ha="right")
    ax1.set_ylabel("Objective value")
    ax1.set_title("Objective robustness")
    ax1.legend(frameon=False, ncol=3, loc="upper left")
    ax1.grid(True, axis="y")
    ax1.grid(False, axis="x")

    ax2 = fig.add_subplot(gs[0, 1])
    b4 = ax2.bar(x - width / 2, df["expected_service_ratio"], width=width, label="Expected", alpha=0.95)
    b5 = ax2.bar(x + width / 2, df["worst_case_service_ratio"], width=width, label="Worst-case", alpha=0.95)
    _annotate_bars(ax2, b4, fmt="{:.2f}", frac=0.01)
    _annotate_bars(ax2, b5, fmt="{:.2f}", frac=0.01)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=12, ha="right")
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("Service ratio")
    ax2.set_title("Charging service robustness")
    ax2.legend(frameon=False, ncol=2, loc="upper left")
    ax2.grid(True, axis="y")
    ax2.grid(False, axis="x")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_unmet_peak_congestion(summary_df: pd.DataFrame, save_path: Path) -> None:
    apply_plot_style()

    df = summary_df.copy()
    methods = df["method"].tolist()
    x = np.arange(len(methods))
    width = 0.32

    fig = plt.figure(figsize=(13.4, 7.2), facecolor="white")
    gs = fig.add_gridspec(1, 3, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    b1 = ax1.bar(x - width / 2, df["expected_unmet_energy_kwh"], width=width, label="Expected unmet", alpha=0.95)
    b2 = ax1.bar(x + width / 2, df["cvar_unmet_energy_kwh"], width=width, label="CVaR unmet", alpha=0.95)
    _annotate_bars(ax1, b1, fmt="{:.1f}")
    _annotate_bars(ax1, b2, fmt="{:.1f}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=12, ha="right")
    ax1.set_ylabel("Unmet energy (kWh)")
    ax1.set_title("Unmet charging demand")
    ax1.legend(frameon=False, loc="upper left")
    ax1.grid(True, axis="y")
    ax1.grid(False, axis="x")

    ax2 = fig.add_subplot(gs[0, 1])
    b3 = ax2.bar(x - width / 2, df["expected_peak_ev_power_kw"], width=width, label="Expected peak", alpha=0.95)
    b4 = ax2.bar(x + width / 2, df["cvar_peak_ev_power_kw"], width=width, label="CVaR peak", alpha=0.95)
    _annotate_bars(ax2, b3, fmt="{:.1f}")
    _annotate_bars(ax2, b4, fmt="{:.1f}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=12, ha="right")
    ax2.set_ylabel("Peak EV power (kW)")
    ax2.set_title("Charging peak risk")
    ax2.legend(frameon=False, loc="upper left")
    ax2.grid(True, axis="y")
    ax2.grid(False, axis="x")

    ax3 = fig.add_subplot(gs[0, 2])
    b5 = ax3.bar(x - width / 2, df["expected_congestion_exposure"], width=width, label="Expected", alpha=0.95)
    b6 = ax3.bar(x + width / 2, df["cvar_congestion_exposure"], width=width, label="CVaR", alpha=0.95)
    _annotate_bars(ax3, b5, fmt="{:.1f}")
    _annotate_bars(ax3, b6, fmt="{:.1f}")
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=12, ha="right")
    ax3.set_ylabel("Congestion exposure")
    ax3.set_title("Congestion tail-risk")
    ax3.legend(frameon=False, loc="upper left")
    ax3.grid(True, axis="y")
    ax3.grid(False, axis="x")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_distribution_boxplot(os_df: pd.DataFrame, rh_df: pd.DataFrame, save_path: Path) -> None:
    apply_plot_style()

    data = [
        pd.to_numeric(os_df["weighted_objective_proxy"], errors="coerce").dropna().values,
        pd.to_numeric(rh_df["weighted_objective_proxy"], errors="coerce").dropna().values,
    ]
    labels = ["One-shot stochastic", "Receding horizon"]

    fig, ax = plt.subplots(figsize=(10.8, 5.8), facecolor="white")
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.5,
        labels=labels,
        showfliers=True,
        medianprops=dict(linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        boxprops=dict(linewidth=1.5),
    )

    for patch in bp["boxes"]:
        patch.set_alpha(0.45)

    means = [np.nanmean(d) if len(d) else np.nan for d in data]
    for i, m in enumerate(means, start=1):
        if np.isfinite(m):
            ax.scatter(i, m, marker="D", s=42, zorder=3, label="Mean" if i == 1 else None)

    ax.set_ylabel("Scenario objective")
    ax.set_title("Scenario-level objective distribution")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def main() -> None:
    config = CompareOSRHConfig()
    config.out_dir.mkdir(parents=True, exist_ok=True)

    os_df, rh_df = load_and_standardize(config.one_shot_summary_path, config.rh_summary_path)
    summary_df = build_summary_table(os_df, rh_df)
    merged_df = build_merged_metrics(os_df, rh_df)

    summary_df.to_csv(config.out_dir / "comparison_summary.csv", index=False)
    merged_df.to_csv(config.out_dir / "merged_scenario_metrics.csv", index=False)

    plot_objective_service(summary_df, config.out_dir / "compare_objective_service.png")
    plot_unmet_peak_congestion(summary_df, config.out_dir / "compare_unmet_peak_congestion.png")
    plot_distribution_boxplot(os_df, rh_df, config.out_dir / "compare_distribution_boxplot.png")

    print(f"[ok] comparison saved        -> {config.out_dir}")
    print(f"[ok] summary saved           -> {config.out_dir / 'comparison_summary.csv'}")
    print(f"[ok] merged metrics saved    -> {config.out_dir / 'merged_scenario_metrics.csv'}")
    print(f"[ok] figure saved            -> {config.out_dir / 'compare_objective_service.png'}")
    print(f"[ok] figure saved            -> {config.out_dir / 'compare_unmet_peak_congestion.png'}")
    print(f"[ok] figure saved            -> {config.out_dir / 'compare_distribution_boxplot.png'}")
    print("\\nComparison summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
