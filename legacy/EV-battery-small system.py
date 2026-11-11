##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

# =========================
# 1. Basic settings
# =========================
np.random.seed(42)

TIME_HORIZON = 24
N_EV = 260
DT = 1.0
P_MAX_EV = 7.0

# Health model weights
ALPHA_THROUGHPUT = 0.8
BETA_HIGH_SOC = 6.0
GAMMA_HIGH_POWER = 3.5
LAMBDA_HEALTH = 2.8

# Flexibility penalty coefficient:
# maps health burden into effective flexibility loss
ALPHA_FLEX = 0.02

hours = np.arange(TIME_HORIZON)

# =========================
# 2. Time-varying signals
# =========================
# price: 夜间便宜，傍晚高
price = np.array([
    0.07, 0.07, 0.07, 0.08, 0.09, 0.11,
    0.14, 0.19, 0.26, 0.30, 0.32, 0.28,
    0.22, 0.18, 0.15, 0.17, 0.22, 0.34,
    0.40, 0.38, 0.26, 0.17, 0.10, 0.08
], dtype=float)

# carbon: 中午更低，傍晚和夜间更高
carbon = np.array([
    0.64, 0.62, 0.60, 0.54, 0.44, 0.30,
    0.20, 0.12, 0.07, 0.05, 0.04, 0.04,
    0.05, 0.06, 0.08, 0.14, 0.24, 0.40,
    0.56, 0.62, 0.66, 0.68, 0.68, 0.66
], dtype=float)

# baseline capacity profile
base_available_cap = np.array([
    145, 145, 145, 145, 150, 155,
    170, 185, 215, 225, 225, 220,
    210, 210, 205, 195, 170, 145,
    125, 120, 120, 120, 125, 140
], dtype=float)

# =========================
# 3. EV data structure
# =========================
@dataclass
class EV:
    ev_id: int
    arrival: int
    departure: int
    init_soc: float
    target_soc: float
    battery_kwh: float
    max_power: float

    @property
    def required_energy(self):
        return max(0.0, (self.target_soc - self.init_soc) * self.battery_kwh)

# =========================
# 4. Generate synthetic EV fleet
# =========================
def generate_evs(n_ev: int):
    evs = []
    for i in range(n_ev):
        # 50% daytime / 50% evening
        if np.random.rand() < 0.50:
            arrival = int(np.clip(np.random.normal(loc=8.5, scale=1.8), 0, 21))
            stay = int(np.clip(np.random.normal(loc=8.5, scale=1.6), 4, 11))
        else:
            arrival = int(np.clip(np.random.normal(loc=17.8, scale=2.0), 0, 22))
            stay = int(np.clip(np.random.normal(loc=7.0, scale=1.8), 4, 11))

        departure = min(arrival + stay, TIME_HORIZON)

        init_soc = float(np.clip(np.random.normal(loc=0.30, scale=0.12), 0.05, 0.65))
        target_soc = float(np.clip(np.random.normal(loc=0.88, scale=0.05), 0.75, 0.95))
        battery_kwh = float(np.clip(np.random.normal(loc=65, scale=10), 45, 90))

        evs.append(
            EV(
                ev_id=i,
                arrival=arrival,
                departure=departure,
                init_soc=init_soc,
                target_soc=target_soc,
                battery_kwh=battery_kwh,
                max_power=P_MAX_EV
            )
        )
    return evs

# =========================
# 5. Health cost model
# =========================
def marginal_health_cost(curr_soc, charge_power, max_power):
    throughput_term = ALPHA_THROUGHPUT

    high_soc_term = 0.0
    if curr_soc > 0.65:
        high_soc_term = BETA_HIGH_SOC * ((curr_soc - 0.65) / 0.35) ** 2

    power_ratio = charge_power / max_power if max_power > 0 else 0.0
    high_power_term = GAMMA_HIGH_POWER * (power_ratio ** 2)

    return throughput_term + high_soc_term + high_power_term

# =========================
# 6. Strategy-specific slot signal
# =========================
def slot_signal(strategy, t, ev, curr_soc, remaining_energy):
    """
    Lower signal = more preferred slot
    """
    slack = max(ev.departure - t, 1)
    urgency = (remaining_energy / slack) / max(ev.max_power, 1e-6)

    if strategy == "cost":
        return price[t] - 0.16 * urgency

    elif strategy == "carbon":
        return carbon[t] - 0.16 * urgency

    elif strategy == "health":
        proxy_power = 2.0 if curr_soc >= 0.65 else 2.5
        hour_bias = 0.03 * (t - ev.arrival)
        return (
            0.10 * price[t]
            + LAMBDA_HEALTH * marginal_health_cost(curr_soc, proxy_power, ev.max_power)
            + hour_bias
            - 0.06 * urgency
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

# =========================
# 7. Core simulation
# =========================
def run_simulation(evs, strategy="cost", available_cap=None, alpha_flex=ALPHA_FLEX):
    if available_cap is None:
        available_cap = base_available_cap.copy()

    n = len(evs)

    remaining_energy = np.array([ev.required_energy for ev in evs], dtype=float)
    requested_energy = remaining_energy.copy()
    soc = np.array([ev.init_soc for ev in evs], dtype=float)

    charge_matrix = np.zeros((n, TIME_HORIZON))
    energy_matrix = np.zeros((n, TIME_HORIZON))
    health_matrix = np.zeros((n, TIME_HORIZON))

    cap_left = available_cap.copy()

    for _ in range(40):
        progress = False

        order = sorted(
            range(n),
            key=lambda i: (
                evs[i].departure,
                -(remaining_energy[i] / max(evs[i].departure - evs[i].arrival, 1))
            )
        )

        for i in order:
            if remaining_energy[i] <= 1e-6:
                continue

            ev = evs[i]
            feasible_hours = [t for t in range(ev.arrival, ev.departure) if cap_left[t] > 1e-6]
            if not feasible_hours:
                continue

            ranked_hours = sorted(
                feasible_hours,
                key=lambda t: (slot_signal(strategy, t, ev, soc[i], remaining_energy[i]), t)
            )

            for t in ranked_hours:
                if remaining_energy[i] <= 1e-6:
                    break
                if cap_left[t] <= 1e-6:
                    continue

                if strategy == "health":
                    power_cap = 2.5 if soc[i] < 0.65 else 1.5
                else:
                    power_cap = ev.max_power

                feasible_power = min(power_cap, cap_left[t], remaining_energy[i] / DT)
                if feasible_power <= 1e-6:
                    continue

                delivered = feasible_power * DT

                charge_matrix[i, t] += feasible_power
                energy_matrix[i, t] += delivered
                cap_left[t] -= feasible_power
                remaining_energy[i] -= delivered

                prev_soc = soc[i]
                soc[i] = min(1.0, soc[i] + delivered / ev.battery_kwh)

                h_cost = marginal_health_cost(prev_soc, feasible_power, ev.max_power) * delivered
                health_matrix[i, t] += h_cost

                progress = True

        if not progress:
            break

    agg_power = charge_matrix.sum(axis=0)
    agg_energy = energy_matrix.sum(axis=0)

    total_cost = float(np.sum(agg_energy * price))
    total_carbon = float(np.sum(agg_energy * carbon))
    total_health = float(np.sum(health_matrix))

    served_energy = float(agg_energy.sum())
    requested_energy_total = float(requested_energy.sum())
    unserved_energy = float(remaining_energy.sum())

    # ---- New flexibility metrics ----
    naive_flexibility = served_energy
    sustainable_flexibility = max(0.0, served_energy - alpha_flex * total_health)
    flexibility_overestimation = naive_flexibility - sustainable_flexibility
    overestimation_ratio = (
        flexibility_overestimation / naive_flexibility
        if naive_flexibility > 0 else np.nan
    )

    results = {
        "strategy": strategy,
        "agg_power": agg_power,
        "agg_energy": agg_energy,
        "charge_matrix": charge_matrix,
        "energy_matrix": energy_matrix,
        "health_matrix": health_matrix,
        "total_cost": total_cost,
        "total_carbon": total_carbon,
        "total_health": total_health,
        "peak_load": float(agg_power.max()),
        "load_variance": float(np.var(agg_power)),
        "served_energy": served_energy,
        "unserved_energy": unserved_energy,
        "service_ratio": served_energy / requested_energy_total if requested_energy_total > 0 else 0.0,
        "cost_per_served_kWh": total_cost / served_energy if served_energy > 0 else np.nan,
        "carbon_per_served_kWh": total_carbon / served_energy if served_energy > 0 else np.nan,
        "health_per_served_kWh": total_health / served_energy if served_energy > 0 else np.nan,
        "naive_flexibility": naive_flexibility,
        "sustainable_flexibility": sustainable_flexibility,
        "flexibility_overestimation": flexibility_overestimation,
        "overestimation_ratio": overestimation_ratio,
        "ev_health": health_matrix.sum(axis=1),
        "ev_energy": energy_matrix.sum(axis=1),
        "available_cap": available_cap.copy(),
        "alpha_flex": alpha_flex,
    }
    return results

# =========================
# 8. Run all strategies
# =========================
def run_all(cap_scale=1.0, alpha_flex=ALPHA_FLEX):
    evs = generate_evs(N_EV)
    available_cap = base_available_cap * cap_scale

    out_cost = run_simulation(evs, strategy="cost", available_cap=available_cap, alpha_flex=alpha_flex)
    out_carbon = run_simulation(evs, strategy="carbon", available_cap=available_cap, alpha_flex=alpha_flex)
    out_health = run_simulation(evs, strategy="health", available_cap=available_cap, alpha_flex=alpha_flex)

    return evs, {
        "cost": out_cost,
        "carbon": out_carbon,
        "health": out_health
    }

# =========================
# 9. Multi-scenario analysis
# =========================
def run_congestion_scenarios(cap_scales=(0.85, 1.0, 1.15), alpha_flex=ALPHA_FLEX):
    scenario_outputs = {}
    for cap_scale in cap_scales:
        _, outputs = run_all(cap_scale=cap_scale, alpha_flex=alpha_flex)
        scenario_outputs[f"cap_x_{cap_scale:.2f}"] = outputs
    return scenario_outputs

# =========================
# 10. Summaries
# =========================
def summarize(outputs):
    rows = []
    for name, out in outputs.items():
        rows.append({
            "strategy": name,
            "total_cost_$": out["total_cost"],
            "total_carbon_kg": out["total_carbon"],
            "total_health_index": out["total_health"],
            "served_energy_kWh": out["served_energy"],
            "unserved_energy_kWh": out["unserved_energy"],
            "service_ratio": out["service_ratio"],
            "cost_per_served_kWh": out["cost_per_served_kWh"],
            "carbon_per_served_kWh": out["carbon_per_served_kWh"],
            "health_per_served_kWh": out["health_per_served_kWh"],
            "naive_flexibility": out["naive_flexibility"],
            "sustainable_flexibility": out["sustainable_flexibility"],
            "flexibility_overestimation": out["flexibility_overestimation"],
            "overestimation_ratio": out["overestimation_ratio"],
            "peak_power_kW": out["peak_load"],
        })
    df = pd.DataFrame(rows)
    return df.sort_values("strategy").reset_index(drop=True)

def summarize_scenarios(scenario_outputs):
    rows = []
    for scen_name, outputs in scenario_outputs.items():
        for strat_name, out in outputs.items():
            rows.append({
                "scenario": scen_name,
                "strategy": strat_name,
                "served_energy_kWh": out["served_energy"],
                "total_health_index": out["total_health"],
                "naive_flexibility": out["naive_flexibility"],
                "sustainable_flexibility": out["sustainable_flexibility"],
                "overestimation_ratio": out["overestimation_ratio"],
                "cost_per_served_kWh": out["cost_per_served_kWh"],
                "carbon_per_served_kWh": out["carbon_per_served_kWh"],
                "health_per_served_kWh": out["health_per_served_kWh"],
            })
    return pd.DataFrame(rows)

# =========================
# 11. Plotting (publication-ready)
# =========================
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

COLORS = {
    "cost": "#1f3c88",      # deep blue
    "carbon": "#2a9d8f",    # teal
    "health": "#c76d3a",    # muted orange
    "naive": "#8da0cb",
    "sustainable": "#66c2a5",
    "cap": "#7f7f7f",
    "price": "#4c78a8",
    "carbon_signal": "#59a14f",
}

def set_pub_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def save_figure(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")

def style_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, linestyle="--", alpha=0.22)
    ax.set_axisbelow(True)

def annotate_bars(ax, bars, fmt="{:.3f}", rotation=0, fontsize=8, y_offset_frac=0.01):
    ymin, ymax = ax.get_ylim()
    y_offset = (ymax - ymin) * y_offset_frac
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + y_offset,
            fmt.format(h),
            ha="center",
            va="bottom",
            rotation=rotation,
            fontsize=fontsize
        )

def plot_inputs_and_profiles(outputs):
    """
    Fig. 1: external signals + aggregate charging profiles
    """
    set_pub_style()
    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 5.8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.15]}
    )

    # Panel (a): external signals
    ax = axes[0]
    ax2 = ax.twinx()

    l1 = ax.plot(hours, price, color=COLORS["price"], marker="o", linewidth=2, label="Electricity price")
    l2 = ax2.plot(hours, carbon, color=COLORS["carbon_signal"], marker="s", linewidth=2, label="Carbon intensity")
    l3 = ax.bar(hours, base_available_cap, color=COLORS["cap"], alpha=0.18, width=0.8, label="Available capacity")

    ax.set_ylabel("Price ($/kWh)")
    ax2.set_ylabel("Carbon (kg CO$_2$/kWh)")
    ax.set_title("Input signals and charging outcomes", loc="left", fontweight="bold")
    style_axis(ax, grid_axis="y")
    ax2.spines["top"].set_visible(False)

    handles = l1 + l2 + [l3]
    labels = ["Electricity price", "Carbon intensity", "Available capacity"]
    ax.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))

    # Panel (b): aggregate charging profiles
    ax = axes[1]
    for s in ["cost", "carbon", "health"]:
        ax.plot(
            hours,
            outputs[s]["agg_power"],
            color=COLORS[s],
            linewidth=2.2,
            marker="o",
            label=s.capitalize()
        )
    ax.plot(hours, base_available_cap, color="black", linestyle="--", linewidth=1.4, label="Capacity limit")

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Aggregate charging power (kW)")
    style_axis(ax, grid_axis="both")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.set_xticks(np.arange(0, 24, 2))

    fig.tight_layout()
    save_figure(fig, "fig1_inputs_and_profiles")
    plt.show()

def plot_unit_metrics(outputs):
    """
    Fig. 2: unit-normalized metrics, better for paper than total metrics
    """
    set_pub_style()

    metrics = [
        ("cost_per_served_kWh", "Cost per served kWh"),
        ("carbon_per_served_kWh", "Carbon per served kWh"),
        ("health_per_served_kWh", "Health per served kWh"),
        ("overestimation_ratio", "Overestimation ratio"),
    ]
    strategies = ["cost", "carbon", "health"]

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2))
    for ax, (metric, title) in zip(axes, metrics):
        vals = [outputs[s][metric] for s in strategies]
        bars = ax.bar(
            np.arange(len(strategies)),
            vals,
            color=[COLORS[s] for s in strategies],
            width=0.65
        )
        ax.set_xticks(np.arange(len(strategies)))
        ax.set_xticklabels([s.capitalize() for s in strategies], rotation=0)
        ax.set_title(title, pad=8)
        style_axis(ax, grid_axis="y")

        if metric == "overestimation_ratio":
            annotate_bars(ax, bars, fmt="{:.2%}", fontsize=8, y_offset_frac=0.02)
        else:
            annotate_bars(ax, bars, fmt="{:.3f}", fontsize=8, y_offset_frac=0.02)

    fig.suptitle("Comparison of strategy-level performance metrics", y=1.05, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig2_unit_metrics")
    plt.show()

def plot_health_distribution(outputs):
    """
    Fig. 3: ECDF-style health burden plot, cleaner than sorted raw line
    """
    set_pub_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))

    for s in ["cost", "carbon", "health"]:
        x = np.sort(outputs[s]["ev_health"])
        y = np.linspace(0, 1, len(x))
        ax.plot(x, y, color=COLORS[s], linewidth=2.2, label=s.capitalize())

    ax.set_xlabel("Vehicle-level health burden")
    ax.set_ylabel("Cumulative fraction of EVs")
    ax.set_title("Distribution of battery health burden", loc="left", fontweight="bold")
    style_axis(ax, grid_axis="both")
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    save_figure(fig, "fig3_health_distribution")
    plt.show()

def plot_cost_health_carbon_scatter(outputs):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.4))

    strategies = ["cost", "carbon", "health"]
    x_vals = [outputs[s]["health_per_served_kWh"] for s in strategies]
    y_vals = [outputs[s]["cost_per_served_kWh"] for s in strategies]

    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    x_range = x_max - x_min
    y_range = y_max - y_min

    # 先画点
    for s in strategies:
        x = outputs[s]["health_per_served_kWh"]
        y = outputs[s]["cost_per_served_kWh"]

        ax.scatter(
            x, y,
            s=180,
            color=COLORS[s],
            edgecolor="black",
            linewidth=0.7,
            zorder=3
        )

    # 给坐标轴留白，避免标签贴边
    ax.set_xlim(x_min - 0.22 * x_range, x_max + 0.28 * x_range)
    ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.18 * y_range)

    # 为每个点单独设定标注位置（相对点）
    label_specs = {
        "health": {
            "dx": -0.12 * x_range,
            "dy":  0.08 * y_range,
            "ha": "right",
            "va": "center",
        },
        "cost": {
            "dx":  0.05 * x_range,
            "dy": -0.02 * y_range,
            "ha": "left",
            "va": "center",
        },
        "carbon": {
            "dx":  0.05 * x_range,
            "dy":  0.10 * y_range,
            "ha": "left",
            "va": "bottom",
        },
    }

    # 再画文字标注
    for s in strategies:
        x = outputs[s]["health_per_served_kWh"]
        y = outputs[s]["cost_per_served_kWh"]
        c = outputs[s]["carbon_per_served_kWh"]

        spec = label_specs[s]
        tx = x + spec["dx"]
        ty = y + spec["dy"]

        ax.annotate(
            f"{s.capitalize()}\nC={c:.3f}",
            xy=(x, y),
            xytext=(tx, ty),
            textcoords="data",
            ha=spec["ha"],
            va=spec["va"],
            fontsize=9,
            arrowprops=dict(
                arrowstyle="-",
                lw=0.8,
                color="0.35",
                shrinkA=4,
                shrinkB=4
            ),
            zorder=4
        )

    ax.set_xlabel("Health burden per served kWh")
    ax.set_ylabel("Cost per served kWh ($/kWh)")
    ax.set_title("Trade-off between cost and battery health", loc="left", fontweight="bold")
    style_axis(ax, grid_axis="both")

    fig.tight_layout()
    save_figure(fig, "fig4_tradeoff_scatter_fixed_v2")
    plt.show()

def plot_flexibility_misestimation(outputs):
    """
    Fig. 5: naive vs sustainable flexibility + overestimation ratio
    """
    set_pub_style()
    strategies = ["cost", "carbon", "health"]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), gridspec_kw={"width_ratios": [1.4, 1]})

    # Panel (a): paired bars
    ax = axes[0]
    x = np.arange(len(strategies))
    width = 0.34

    naive_vals = [outputs[s]["naive_flexibility"] for s in strategies]
    sust_vals = [outputs[s]["sustainable_flexibility"] for s in strategies]

    bars1 = ax.bar(x - width/2, naive_vals, width=width, color=COLORS["naive"], label="Naive flexibility")
    bars2 = ax.bar(x + width/2, sust_vals, width=width, color=COLORS["sustainable"], label="Sustainable flexibility")

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in strategies])
    ax.set_ylabel("Flexibility (kWh-equivalent)")
    ax.set_title("Flexibility misestimation", loc="left", fontweight="bold")
    style_axis(ax, grid_axis="y")
    ax.legend(frameon=False, loc="upper right")

    # only annotate ratio above sustainable bars
    ratios = [outputs[s]["overestimation_ratio"] for s in strategies]
    for xi, r, y in zip(x, ratios, sust_vals):
        ax.text(xi, y + 18, f"{r:.1%}", ha="center", va="bottom", fontsize=8)

    # Panel (b): overestimation ratio
    ax = axes[1]
    bars = ax.bar(
        np.arange(len(strategies)),
        ratios,
        color=[COLORS[s] for s in strategies],
        width=0.65
    )
    ax.set_xticks(np.arange(len(strategies)))
    ax.set_xticklabels([s.capitalize() for s in strategies])
    ax.set_ylabel("Overestimation ratio")
    style_axis(ax, grid_axis="y")
    annotate_bars(ax, bars, fmt="{:.1%}", fontsize=8, y_offset_frac=0.02)

    fig.tight_layout()
    save_figure(fig, "fig5_flexibility_misestimation")
    plt.show()

def plot_congestion_effects(scenario_outputs):
    set_pub_style()

    scenario_names = list(scenario_outputs.keys())
    scenario_labels = [s.replace("cap_x_", "×") for s in scenario_names]
    strategies = ["cost", "carbon", "health"]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    for s in strategies:
        vals = [scenario_outputs[sc][s]["flexibility_overestimation"] for sc in scenario_names]
        ax.plot(
            scenario_labels,
            vals,
            marker="o",
            linewidth=2.2,
            color=COLORS[s],
            label=s.capitalize()
        )
        for xi, yi in zip(scenario_labels, vals):
            ax.text(xi, yi, f"{yi:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Capacity scaling factor")
    ax.set_ylabel("Flexibility overestimation (kWh-equivalent)")
    ax.set_title("Congestion increases the overestimation of EV fleet flexibility",
                 loc="left", fontweight="bold")
    style_axis(ax, grid_axis="y")
    ax.legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, "fig6_congestion_effects_overestimation")
    plt.show()

def plot_results(outputs):
    plot_inputs_and_profiles(outputs)
    plot_unit_metrics(outputs)
    plot_health_distribution(outputs)
    plot_cost_health_carbon_scatter(outputs)
    plot_flexibility_misestimation(outputs)

# =========================
# 12. Main
# =========================
if __name__ == "__main__":
    # Base scenario
    evs, outputs = run_all(cap_scale=1.0, alpha_flex=ALPHA_FLEX)

    summary_df = summarize(outputs)
    print("\n=== Base Scenario Summary ===")
    print(summary_df.round(4))

    plot_results(outputs)

    # Congestion scenarios
    scenario_outputs = run_congestion_scenarios(
        cap_scales=(0.60, 0.80, 1.00, 1.20, 1.40),
        alpha_flex=ALPHA_FLEX
    )
    scenario_df = summarize_scenarios(scenario_outputs)

    print("\n=== Congestion Scenario Summary ===")
    print(scenario_df.round(4))

    plot_congestion_effects(scenario_outputs)