
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 输入路径（根据你实际修改）
# =========================
WITHOUT_DEG = Path("stochastic_results_joint/stochastic_results_summary.csv")
WITH_DEG = Path("stochastic_results_joint_with_deg/stochastic_results_summary.csv")

OUT_DIR = Path("compare_degradation")
OUT_DIR.mkdir(exist_ok=True)


# =========================
# 统一风格
# =========================
def set_style():
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
        "grid.alpha": 0.2,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# =========================
# 读取数据
# =========================
def load_data():
    df_no = pd.read_csv(WITHOUT_DEG)
    df_deg = pd.read_csv(WITH_DEG)

    def summarize(df):
        return {
            "service_ratio": np.nanmean(df["service_ratio"]),
            "unmet": np.nanmean(df["total_unmet_energy_kwh"]),
            "peak": np.nanmean(df["peak_ev_power_kw"]),
            "delta_soh": np.nanmean(df.get("total_delta_soh", 0)),
            "final_soh": np.nanmean(df.get("mean_final_soh", 1)),
        }

    return summarize(df_no), summarize(df_deg)


# =========================
# 绘图函数（论文级）
# =========================
def plot_comparison(no_deg, with_deg):

    set_style()

    labels = ["Without Deg", "With Deg"]
    x = np.arange(len(labels))
    width = 0.35

    # =========================
    # 图1：Service + Unmet
    # =========================
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.3)

    # Service
    ax1 = fig.add_subplot(gs[0, 0])
    vals = [no_deg["service_ratio"], with_deg["service_ratio"]]
    bars = ax1.bar(x, vals)

    for i, v in enumerate(vals):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha="center")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Service Ratio")
    ax1.set_ylabel("Ratio")

    # Unmet
    ax2 = fig.add_subplot(gs[0, 1])
    vals = [no_deg["unmet"], with_deg["unmet"]]
    bars = ax2.bar(x, vals)

    for i, v in enumerate(vals):
        ax2.text(i, v + 1, f"{v:.1f}", ha="center")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title("Unmet Energy (kWh)")

    plt.savefig(OUT_DIR / "service_unmet.png", dpi=300)
    plt.close()

    # =========================
    # 图2：Peak + SOH
    # =========================
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 3, wspace=0.35)

    # Peak power
    ax1 = fig.add_subplot(gs[0, 0])
    vals = [no_deg["peak"], with_deg["peak"]]
    ax1.bar(x, vals)
    for i, v in enumerate(vals):
        ax1.text(i, v + 2, f"{v:.1f}", ha="center")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Peak EV Power (kW)")

    # Delta SOH
    ax2 = fig.add_subplot(gs[0, 1])
    vals = [no_deg["delta_soh"], with_deg["delta_soh"]]
    ax2.bar(x, vals)
    for i, v in enumerate(vals):
        ax2.text(i, v + 1e-5, f"{v:.4f}", ha="center")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title("Total Degradation (ΔSOH)")

    # Final SOH
    ax3 = fig.add_subplot(gs[0, 2])
    vals = [no_deg["final_soh"], with_deg["final_soh"]]
    ax3.bar(x, vals)
    for i, v in enumerate(vals):
        ax3.text(i, v + 0.001, f"{v:.3f}", ha="center")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_title("Final SOH")

    plt.savefig(OUT_DIR / "peak_soh.png", dpi=300)
    plt.close()


# =========================
# 主函数
# =========================
def main():
    no_deg, with_deg = load_data()

    print("WITHOUT DEG:", no_deg)
    print("WITH DEG:", with_deg)

    plot_comparison(no_deg, with_deg)

    print(f"\nPlots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
