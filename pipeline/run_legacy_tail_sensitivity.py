"""Additional CVaR sensitivity with fresh draws from the archived GMM/RF models.

Run with the project's py39 environment (scikit-learn 1.2.1 and pandas 1.5).
These are synthetic draws from legacy models and not from a CVAE–diffusion model.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from build_optimization_inputs_flexible import (
    OptimizationInputConfig, build_ev_session_input, build_grid_input,
)
from build_stress_scenarios import (
    StressScenarioConfig, build_ev_hourly_from_sessions, build_joint_hourly,
    stress_ev_sessions, stress_grid_hourly,
)
from nonanticipative_scheduler import Fleet, Forecast, cvar, reference_replay
from run_section_iv_replay import GRID_COLS, GROUPS, ROOT, load, plt, save
from train_ev_generator import EVGenerator
from train_grid_generator import GridGenerator


def generate(job):
    seed, sample_size = job
    ev_model = EVGenerator.load(ROOT / "processed_ev" / "ev_generator.pkl")
    grid_model = GridGenerator.load(ROOT / "processed_grid" / "grid_generator.pkl")
    for model in grid_model.models.values():
        model.n_jobs = 1
    grid_model.config.data_path = ROOT / "processed_grid" / "grid_generator_ready.csv"
    ev_model.rng = np.random.default_rng(seed)
    grid_model.rng = np.random.default_rng(seed + 1)
    seed_row = grid_model.load_data().iloc[-1].copy()
    forecasts = {
        label: Forecast([case[2] for case in load(group, 0)[:4]])
        for label, group in GROUPS.items()
    }
    input_config = OptimizationInputConfig()
    results = []
    for j in range(sample_size):
        ev_base = ev_model.sample_sessions_for_horizon(str(seed_row.timestamp), 24, "acn")
        grid_base = grid_model.sample_trajectory(seed_row, 24, True)
        for label, group in GROUPS.items():
            stress = StressScenarioConfig(stress_mode=group, random_seed=seed + 10 + j)
            ev_data = stress_ev_sessions(ev_base, stress)
            grid_data = stress_grid_hourly(grid_base, stress)
            ev_hourly = build_ev_hourly_from_sessions(ev_data, grid_data)
            opt_grid = build_grid_input(build_joint_hourly(grid_data, ev_hourly), input_config)
            opt_ev = build_ev_session_input(ev_data, opt_grid, input_config)
            grid = opt_grid[GRID_COLS].to_numpy(dtype=float)
            if group == "capacity_shortage":
                grid[17:21, 3] = np.minimum(grid[17:21, 3], 15.0)
            fleet = Fleet(opt_ev.arrival_step.to_numpy(int), opt_ev.departure_step.to_numpy(int),
                          opt_ev.required_energy_kwh.to_numpy(float), opt_ev.max_charging_power_kw.to_numpy(float),
                          opt_ev.battery_capacity_kwh.to_numpy(float), opt_ev.init_soc.to_numpy(float))
            executed, *_ = reference_replay(fleet, grid, forecasts[label], 4)
            unmet = np.maximum(fleet.demand - executed.sum(axis=1), 0).sum()
            results.append({"seed": seed, "draw": j, "group": label, "unmet_kwh": unmet})
        if (j + 1) % 10 == 0:
            print(f"Legacy tail seed {seed}: {j + 1}/{sample_size}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, args.draws) for seed in (101, 211, 307)]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            batches = list(pool.map(generate, jobs))
    else:
        batches = list(map(generate, jobs))
    df = pd.DataFrame([row for batch in batches for row in batch])
    df.to_csv(args.output / "legacy_tail_replays.csv", index=False)
    stats = []
    for (seed, group), g in df.groupby(["seed", "group"], sort=False):
        for n in sorted(set([min(args.draws, 10), min(args.draws, 20), args.draws])):
            for alpha in (0.8, 0.9, 0.95):
                stats.append({"seed": seed, "group": group, "n": n, "alpha": alpha,
                              "cvar_kwh": float(cvar(g.unmet_kwh.to_numpy()[:n], alpha)),
                              "tail_mass_trajectories": (1-alpha)*n})
    stats = pd.DataFrame(stats)
    stats.to_csv(args.output / "legacy_tail_seed_sensitivity.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(8.4, 2.9), sharey=True)
    for ax, group in zip(axes, GROUPS):
        sub = stats[(stats.group == group) & (stats.alpha == 0.95)]
        for seed, data in sub.groupby("seed"):
            ax.plot(data.n, data.cvar_kwh, marker="o", markersize=3, label=f"Seed {seed}")
        ax.set_title(group, fontsize=9)
        ax.set_xlabel("Synthetic replays", fontsize=8)
        ax.set_xticks(sorted(sub.n.unique()))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("CVaR95 unmet energy (kWh)", fontsize=8)
    axes[-1].legend(frameon=False, fontsize=7)
    fig.suptitle("Tail sensitivity under the archived GMM and random forest generators", fontsize=10)
    fig.tight_layout()
    save(fig, args.output / "fig_legacy_tail_seed_sensitivity")


if __name__ == "__main__":
    main()

