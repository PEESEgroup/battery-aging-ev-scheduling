import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.dates as mdates

# =========================
# 配置
# =========================
SCENARIOS = {
    "Baseline": "joint_scenarios",
    "Capacity Shortage": "stress_scenarios_capacity_shortage",
    "High Price": "stress_scenarios_sustained_high_price",
    "Evening Peak": "stress_scenarios_evening_peak_cluster",
}

OUT_DIR = Path("scenario_plots")
OUT_DIR.mkdir(exist_ok=True)


# =========================
# 统一风格（很重要）
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
# 核心绘图函数
# =========================
def plot_one_scenario(name, folder):

    set_style()

    # 默认用 scenario_00（你可以改成循环多个）
    path = Path(folder) / "scenario_00"

    grid = pd.read_csv(path / "grid_hourly.csv")
    ev = pd.read_csv(path / "ev_hourly.csv")

    grid["timestamp"] = pd.to_datetime(grid["timestamp"])

    # =========================
    # 图结构
    # =========================
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.0, 1.2], hspace=0.35)

    # =========================
    # 1. EV demand / load
    # =========================
    ax1 = fig.add_subplot(gs[0, 0])

    ax1.plot(
        grid["timestamp"],
        ev["ev_total_energy_kwh"],
        linewidth=2.5,
        label="EV Charging Demand",
    )

    ax1.set_ylabel("Energy (kWh)")
    ax1.set_title(f"{name}: EV Charging Demand")
    ax1.legend(frameon=False)

    # =========================
    # 2. Grid signals
    # =========================
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    l1 = ax2.plot(
        grid["timestamp"],
        grid["weighted_lbmp_mwh"],
        linewidth=2.2,
        label="Price",
    )[0]

    ax2.set_ylabel("Price ($/MWh)")

    ax2b = ax2.twinx()
    l2 = ax2b.plot(
        grid["timestamp"],
        grid["weighted_imer_co2_ton_per_mwh"],
        linestyle="--",
        linewidth=2,
        label="Emission",
    )[0]

    ax2b.set_ylabel("Emission")

    ax2.legend([l1, l2], ["Price", "Emission"], frameon=False)

    # =========================
    # 3. Congestion + Sessions
    # =========================
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

    ax3.plot(
        grid["timestamp"],
        grid["congestion_intensity"],
        linewidth=2.2,
        label="Congestion",
    )

    ax3.bar(
        grid["timestamp"],
        ev["ev_n_sessions"],
        alpha=0.3,
        label="# EV Sessions",
    )

    ax3.set_ylabel("Congestion / Sessions")
    ax3.set_title("System Stress & EV Arrival")
    ax3.legend(frameon=False)

    # =========================
    # 时间轴美化
    # =========================
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax3.get_xticklabels(), rotation=30)

    fig.suptitle(f"{name} Scenario", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT_DIR / f"{name.replace(' ', '_')}.png", dpi=300)
    plt.close()


# =========================
# 主程序
# =========================
def main():
    for name, folder in SCENARIOS.items():
        print(f"Plotting {name}...")
        plot_one_scenario(name, folder)

    print(f"\nAll plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()