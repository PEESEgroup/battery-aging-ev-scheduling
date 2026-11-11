import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. Basic utilities
# =========================================================

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def minmax_normalize(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    valid = x[np.isfinite(x)]
    if len(valid) == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    mn = valid.min()
    mx = valid.max()
    if abs(mx - mn) < 1e-12:
        return pd.Series(np.ones(len(x)) * 0.5, index=x.index)
    return (x - mn) / (mx - mn)


def compute_cvar(values, alpha=0.95):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan

    var = np.quantile(arr, alpha)
    tail = arr[arr >= var]

    if len(tail) == 0:
        return float(var)

    return float(np.mean(tail))


def apply_publication_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "font.size": 11,
        "axes.grid": False,
        "lines.linewidth": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    })


def scenario_display_name(name: str) -> str:
    mapping = {
        "joint": "Baseline",
        "evening_peak_cluster": "Evening peak cluster",
        "sustained_high_price": "Sustained high price",
        "capacity_shortage": "Capacity shortage",
    }
    return mapping.get(name, name.replace("_", " ").title())


def compute_overestimation_df(scenario_summary: pd.DataFrame) -> pd.DataFrame:
    df = scenario_summary.copy()
    df["flexibility_gap"] = df["flexibility_naive"] - df["flexibility_realized"]
    df["overestimation_pct"] = np.where(
        pd.to_numeric(df["flexibility_naive"], errors="coerce") > 0,
        100.0 * df["flexibility_gap"] / df["flexibility_naive"],
        np.nan
    )
    return df


# =========================================================
# 2. Figure 1 utilities
# =========================================================

def build_radar_ready_df(
    scenario_summary: pd.DataFrame,
    clip_lower: float = 0.0,
    clip_upper: float = 1.2
) -> pd.DataFrame:
    """
    Build baseline-relative radar / heatmap data.

    Expected columns in scenario_summary:
        service_ratio_rel
        cost_per_served_kwh_rel
        carbon_per_served_kwh_rel
        degradation_per_served_kwh_rel
        peak_load_rel
        cvar_unmet_demand_rel

    Baseline should be 1.0 on all relative metrics.
    Values < 1 indicate worse-than-baseline performance.
    Values > 1 indicate better-than-baseline performance.
    """
    df = scenario_summary.copy()

    metric_map = {
        "service_ratio_rel": "service_ratio_norm",
        "cost_per_served_kwh_rel": "cost_per_served_kwh_norm",
        "carbon_per_served_kwh_rel": "carbon_per_served_kwh_norm",
        "degradation_per_served_kwh_rel": "degradation_per_served_kwh_norm",
        "peak_load_rel": "peak_load_norm",
        "cvar_unmet_demand_rel": "cvar_unmet_demand_norm",
    }

    for src_col, dst_col in metric_map.items():
        if src_col not in df.columns:
            df[dst_col] = np.nan
            continue

        vals = pd.to_numeric(df[src_col], errors="coerce").astype(float)
        vals = vals.clip(lower=clip_lower, upper=clip_upper)
        df[dst_col] = vals

    return df


# =========================================================
# 3. Figure 1a - Radar as 2x2 subplots
# =========================================================

def plot_figure1_radar(
    scenario_summary: pd.DataFrame,
    save_path: str,
    title: str = "Figure 1a. Structural system differences across stress scenarios"
):
    apply_publication_style()

    df = build_radar_ready_df(scenario_summary, clip_lower=0.0, clip_upper=1.2)

    metric_cols = [
        "service_ratio_norm",
        "cost_per_served_kwh_norm",
        "carbon_per_served_kwh_norm",
        "degradation_per_served_kwh_norm",
        "peak_load_norm",
        "cvar_unmet_demand_norm",
    ]
    labels = ["Service", "Cost", "Carbon", "Degradation", "Peak", "Risk"]

    scenario_order = ["joint", "evening_peak_cluster", "sustained_high_price", "capacity_shortage"]
    df = df.set_index("scenario").reindex(scenario_order).reset_index()

    baseline_row = df[df["scenario"] == "joint"].iloc[0]
    baseline_vals = baseline_row[metric_cols].tolist()
    baseline_vals += baseline_vals[:1]

    angles = np.linspace(0, 2 * np.pi, len(metric_cols), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 9),
        subplot_kw=dict(polar=True)
    )
    axes = axes.flatten()

    for ax, scenario in zip(axes, scenario_order):
        row = df[df["scenario"] == scenario].iloc[0]
        vals = row[metric_cols].tolist()
        vals += vals[:1]

        ax.plot(angles, baseline_vals, color="lightgray", linewidth=2, linestyle="--")
        ax.fill(angles, baseline_vals, color="lightgray", alpha=0.08)

        ax.plot(angles, vals, linewidth=2.5)
        ax.fill(angles, vals, alpha=0.16)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0", "1.2"], fontsize=8)
        ax.set_title(scenario_display_name(scenario), pad=18, fontsize=12)

    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 4. Figure 1b - Heatmap
# =========================================================

def plot_figure1_heatmap(
    scenario_summary: pd.DataFrame,
    save_path: str,
    title: str = "Figure 1b. Heatmap of baseline-relative system performance"
):
    apply_publication_style()

    df = build_radar_ready_df(scenario_summary, clip_lower=0.0, clip_upper=1.2)

    scenario_order = ["joint", "evening_peak_cluster", "sustained_high_price", "capacity_shortage"]
    df = df.set_index("scenario").reindex(scenario_order).reset_index()

    plot_cols = [
        "service_ratio_norm",
        "cost_per_served_kwh_norm",
        "carbon_per_served_kwh_norm",
        "degradation_per_served_kwh_norm",
        "peak_load_norm",
        "cvar_unmet_demand_norm",
    ]
    plot_labels = ["Service", "Cost", "Carbon", "Degradation", "Peak", "Risk"]

    data = df[plot_cols].to_numpy()

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    im = ax.imshow(data, aspect="auto", vmin=0.0, vmax=1.2)

    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels([scenario_display_name(x) for x in df["scenario"]])
    ax.set_xticks(np.arange(len(plot_labels)))
    ax.set_xticklabels(plot_labels, rotation=25, ha="right")
    ax.set_title(title, pad=10)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.set_ylabel("Relative to baseline", rotation=90)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 5. Figure 2 - Dual heatmaps instead of Pareto
# =========================================================

def plot_figure2_pareto(
    pareto_df: pd.DataFrame,
    save_path: str,
    title: str = "Figure 2. Scenario- and lambda-dependent performance landscape"
):
    apply_publication_style()

    if pareto_df.empty:
        print("[warn] pareto_df is empty. Figure 2 not generated.")
        return

    df = pareto_df.copy()
    for col in ["lambda_risk", "service_ratio", "degradation_per_served_kwh", "cvar_unmet_demand"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    scenario_order = ["joint", "evening_peak_cluster", "sustained_high_price", "capacity_shortage"]
    lambda_order = sorted(df["lambda_risk"].dropna().unique())

    service_mat = (
        df.pivot(index="scenario", columns="lambda_risk", values="service_ratio")
        .reindex(index=scenario_order, columns=lambda_order)
    )

    risk_mat = (
        df.pivot(index="scenario", columns="lambda_risk", values="cvar_unmet_demand")
        .reindex(index=scenario_order, columns=lambda_order)
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))

    im1 = axes[0].imshow(service_mat.values, aspect="auto")
    axes[0].set_title("Service ratio")
    axes[0].set_xticks(np.arange(len(lambda_order)))
    axes[0].set_xticklabels([f"λ={x:.1f}" for x in lambda_order])
    axes[0].set_yticks(np.arange(len(scenario_order)))
    axes[0].set_yticklabels([scenario_display_name(x) for x in scenario_order])

    for i in range(service_mat.shape[0]):
        for j in range(service_mat.shape[1]):
            val = service_mat.iloc[i, j]
            if pd.notna(val):
                axes[0].text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9)

    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.ax.set_ylabel("Service ratio", rotation=90)

    im2 = axes[1].imshow(risk_mat.values, aspect="auto")
    axes[1].set_title("CVaR unmet demand")
    axes[1].set_xticks(np.arange(len(lambda_order)))
    axes[1].set_xticklabels([f"λ={x:.1f}" for x in lambda_order])
    axes[1].set_yticks(np.arange(len(scenario_order)))
    axes[1].set_yticklabels([scenario_display_name(x) for x in scenario_order])

    for i in range(risk_mat.shape[0]):
        for j in range(risk_mat.shape[1]):
            val = risk_mat.iloc[i, j]
            if pd.notna(val):
                axes[1].text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9)

    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.ax.set_ylabel("CVaR unmet demand", rotation=90)

    fig.suptitle(title, fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 6. Figure 3 - Baseline-relative flexibility loss bar
# =========================================================

def plot_figure3_flexibility_overestimation(
    scenario_summary: pd.DataFrame,
    save_path: str,
    title: str = "Figure 3. Flexibility loss relative to baseline"
):
    apply_publication_style()

    df = scenario_summary.copy()
    baseline_val = float(df.loc[df["scenario"] == "joint", "flexibility_realized"].iloc[0])

    df = df[df["scenario"] != "joint"].copy()
    df["flexibility_loss_pct"] = 100.0 * (baseline_val - df["flexibility_realized"]) / baseline_val
    df = df.sort_values("flexibility_loss_pct", ascending=False).reset_index(drop=True)

    x = np.arange(len(df))

    # lighter, harmonious palette
    bar_color = "#8FB6D8"   # light mist blue
    edge_color = "#5E84A6"  # soft blue-gray edge
    text_color = "#3E566B"

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.bar(
        x,
        df["flexibility_loss_pct"],
        color=bar_color,
        edgecolor=edge_color,
        linewidth=0.8,
        width=0.58
    )

    for i, row in df.iterrows():
        ax.text(
            i,
            row["flexibility_loss_pct"] + 0.05,
            f"{row['flexibility_loss_pct']:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            color=text_color
        )

    ax.axhline(0, color="gray", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([scenario_display_name(s) for s in df["scenario"]], rotation=15, ha="right")
    ax.set_ylabel("Flexibility loss (%)")
    ax.set_title(title, pad=10)
    ax.grid(True, axis="y", alpha=0.14)
    ax.grid(False, axis="x")

    ymax = max(df["flexibility_loss_pct"].max() * 1.18, 0.3)
    ax.set_ylim(0, ymax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 7. Figure 4 - Absolute stacked bar + percentage labels
# =========================================================

def plot_figure4_contribution_decomposition(
    contrib_df: pd.DataFrame,
    save_path: str,
    title: str = "Figure 4. Absolute and relative composition of flexibility-related losses"
):
    apply_publication_style()

    if contrib_df.empty:
        print("[warn] contrib_df is empty. Figure 4 not generated.")
        return

    df = contrib_df.copy()
    scenario_order = ["joint", "evening_peak_cluster", "sustained_high_price", "capacity_shortage"]
    df = df.set_index("scenario").reindex(scenario_order).reset_index()

    for col in ["degradation_loss", "congestion_loss", "behavior_loss"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["total_loss"] = df["degradation_loss"] + df["congestion_loss"] + df["behavior_loss"]

    x = np.arange(len(df))
    width = 0.60

    # lighter and coordinated palette
    color_deg = "#9EBBD3"   # light blue-gray
    color_cong = "#E6B089"  # light warm orange
    color_beh = "#B8CDA3"   # light olive green

    edge_deg = "#6E8EA8"
    edge_cong = "#C58E67"
    edge_beh = "#8FAA7C"

    fig, ax = plt.subplots(figsize=(9.6, 6.0))

    ax.bar(
        x, df["degradation_loss"],
        width=width,
        label="Degradation-induced loss",
        color=color_deg,
        edgecolor=edge_deg,
        linewidth=0.6
    )
    ax.bar(
        x, df["congestion_loss"],
        width=width,
        bottom=df["degradation_loss"],
        label="Congestion-induced loss",
        color=color_cong,
        edgecolor=edge_cong,
        linewidth=0.6
    )
    ax.bar(
        x,
        df["behavior_loss"],
        width=width,
        bottom=df["degradation_loss"] + df["congestion_loss"],
        label="Behavior-induced loss",
        color=color_beh,
        edgecolor=edge_beh,
        linewidth=0.6
    )

    for i, row in df.iterrows():
        total = row["total_loss"]
        if total <= 0:
            continue

        ax.text(
            i, total * 1.02, f"{total:.1f}",
            ha="center", va="bottom", fontsize=11, color="#3F3F3F"
        )

        d_pct = 100 * row["degradation_loss"] / total
        c_pct = 100 * row["congestion_loss"] / total
        b_pct = 100 * row["behavior_loss"] / total

        ax.text(
            i, row["degradation_loss"] / 2,
            f"{d_pct:.0f}%",
            ha="center", va="center", fontsize=9, color="#2F4A5E"
        )
        ax.text(
            i,
            row["degradation_loss"] + row["congestion_loss"] / 2,
            f"{c_pct:.0f}%",
            ha="center",
            va="center",
            fontsize=9,
            color="#5A402C"
        )
        ax.text(
            i,
            row["degradation_loss"] + row["congestion_loss"] + row["behavior_loss"] / 2,
            f"{b_pct:.0f}%",
            ha="center",
            va="center",
            fontsize=9,
            color="#46593D"
        )

    ax.set_xticks(x)
    ax.set_xticklabels([scenario_display_name(s) for s in df["scenario"]], rotation=15, ha="right")
    ax.set_ylabel("Loss contribution")
    ax.set_title(title, pad=10)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(True, axis="y", alpha=0.14)
    ax.grid(False, axis="x")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 8. Figure 5 - Violin + median + CVaR
# =========================================================

def plot_figure5_risk_distribution(
    risk_df: pd.DataFrame,
    save_path: str,
    metric: str = "unmet_demand",
    alpha: float = 0.95,
    title: str = "Figure 5. Distribution and tail risk of unmet demand"
):
    apply_publication_style()

    if risk_df.empty:
        print("[warn] risk_df is empty. Figure 5 not generated.")
        return

    df = risk_df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    scenario_order = ["joint", "evening_peak_cluster", "sustained_high_price", "capacity_shortage"]
    scenarios = [s for s in scenario_order if s in df["scenario"].unique()]

    data = []
    cvar_vals = []
    medians = []

    for scenario in scenarios:
        vals = df.loc[df["scenario"] == scenario, metric].dropna().values.astype(float)
        data.append(vals)
        cvar_vals.append(compute_cvar(vals, alpha=alpha) if len(vals) > 0 else np.nan)
        medians.append(np.median(vals) if len(vals) > 0 else np.nan)

    fig, ax = plt.subplots(figsize=(9.5, 6.2))

    vp = ax.violinplot(
        data,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for body in vp["bodies"]:
        body.set_alpha(0.35)

    x = np.arange(1, len(scenarios) + 1)

    ax.scatter(x, medians, s=55, zorder=3, label="Median")
    ax.plot(x, cvar_vals, marker="D", linewidth=2.2, label=f"CVaR{int(alpha*100)}", zorder=4)

    for i, val in enumerate(cvar_vals, start=1):
        if np.isfinite(val):
            ax.text(i + 0.03, val, f"{val:.2f}", fontsize=10, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([scenario_display_name(s) for s in scenarios], rotation=15, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title, pad=10)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, axis="y", alpha=0.18)
    ax.grid(False, axis="x")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 9. Main wrapper
# =========================================================

def generate_all_figures(
    scenario_summary: pd.DataFrame,
    pareto_df: pd.DataFrame,
    contrib_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    output_dir: str = "figures_final"
):
    ensure_dir(output_dir)

    save1 = os.path.join(output_dir, "Figure1a_radar.png")
    save2 = os.path.join(output_dir, "Figure1b_heatmap.png")
    save3 = os.path.join(output_dir, "Figure2_pareto.png")
    save4 = os.path.join(output_dir, "Figure3_flexibility_overestimation.png")
    save5 = os.path.join(output_dir, "Figure4_contribution_decomposition.png")
    save6 = os.path.join(output_dir, "Figure5_risk_distribution.png")

    plot_figure1_radar(
        scenario_summary=scenario_summary,
        save_path=save1
    )

    plot_figure1_heatmap(
        scenario_summary=scenario_summary,
        save_path=save2
    )

    plot_figure2_pareto(
        pareto_df=pareto_df,
        save_path=save3
    )

    plot_figure3_flexibility_overestimation(
        scenario_summary=scenario_summary,
        save_path=save4
    )

    plot_figure4_contribution_decomposition(
        contrib_df=contrib_df,
        save_path=save5
    )

    plot_figure5_risk_distribution(
        risk_df=risk_df,
        save_path=save6,
        metric="unmet_demand",
        alpha=0.95
    )

    print("[saved]", save1)
    print("[saved]", save2)
    print("[saved]", save3)
    print("[saved]", save4)
    print("[saved]", save5)
    print("[saved]", save6)


# =========================================================
# 10. Example: load csv outputs and plot
# =========================================================

if __name__ == "__main__":
    plot_data_dir = Path("plot_data")

    scenario_summary = pd.read_csv(plot_data_dir / "scenario_summary.csv")
    pareto_df = pd.read_csv(plot_data_dir / "pareto_df.csv")
    contrib_df = pd.read_csv(plot_data_dir / "contrib_df.csv")
    risk_df = pd.read_csv(plot_data_dir / "risk_df.csv")

    generate_all_figures(
        scenario_summary=scenario_summary,
        pareto_df=pareto_df,
        contrib_df=contrib_df,
        risk_df=risk_df,
        output_dir="figures_final"
    )

    print("[ok] all figures saved to figures_final/")