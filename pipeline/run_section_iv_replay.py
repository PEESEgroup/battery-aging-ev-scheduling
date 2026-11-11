"""Run the available-data Section IV replay and export manuscript figures."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from battery_degradation_model import BatteryDegradationModel
from nonanticipative_scheduler import (
    AGING, Fleet, Forecast, activate, aging, cvar, reference_replay, solve_plan,
)

ROOT = Path(__file__).resolve().parent
GROUPS = {
    "Normal": "joint",
    "Evening peak": "evening_peak_cluster",
    "High price": "sustained_high_price",
    "Capacity shortage": "capacity_shortage",
}
STARTS = (15, 17)
DURATIONS = (1, 2, 4)
COMMITMENTS = (0, 2, 5, 10, 20, 40)
RELIABILITY = 0.95
GRID_COLS = ["price_kwh", "emission_ton_per_kwh", "congestion_intensity", "ev_capacity_limit_kw"]


def load(group, floor, shortage_control=False):
    cases = []
    for directory in sorted((ROOT / f"optimization_inputs_{group}").glob("scenario_*")):
        ev = pd.read_csv(directory / "optimizer_ev_sessions.csv")
        grid = pd.read_csv(directory / "optimizer_grid.csv")[GRID_COLS].to_numpy(dtype=float)
        fleet = Fleet(ev.arrival_step.to_numpy(int), ev.departure_step.to_numpy(int),
                      ev.required_energy_kwh.to_numpy(float), ev.max_charging_power_kw.to_numpy(float),
                      ev.battery_capacity_kwh.to_numpy(float), ev.init_soc.to_numpy(float))
        assert len(grid) == 24 and np.isfinite(grid).all()
        assert np.all(fleet.arrival >= 0) and np.all(fleet.departure <= len(grid))
        assert np.all(fleet.departure > fleet.arrival)
        grid[:, 3] = np.maximum(grid[:, 3], floor)
        if shortage_control:
            grid[17:21, 3] = np.minimum(grid[17:21, 3], 15.0)
        cases.append((directory.name, fleet, grid))
    assert len(cases) == 10
    return cases


def run_case(name, split, case_id, fleet, grid, forecast, pairing, capacity_mode, count, offline, delivery, service):
    baseline = reference_replay(fleet, grid, forecast, count)
    ref, losses, plans, histories = baseline
    context = dict(group=name, split=split, pairing=pairing, capacity=capacity_mode)
    unmet = float(np.maximum(fleet.demand - ref.sum(axis=1), 0.0).sum())
    service.append({**context, "scenario": case_id, "planning_paths": count,
                    "unmet_kwh": unmet, "served_ratio": 1 - unmet / fleet.demand.sum(),
                    "zero_capacity_hours": int(np.count_nonzero(grid[:, 3] == 0))})
    if count != 4:
        return
    for first in STARTS:
        prefix = ref.copy()
        prefix[:, first:] = 0
        for duration in DURATIONS:
            fixed = plans[first].sum(axis=0)[first:first+duration]
            for direction in (1, -1):
                for aging_on in (True, False):
                    row = {**context, "scenario": case_id, "start_hour": first,
                           "duration_h": duration, "direction": "Up" if direction == 1 else "Down",
                           "aging_budget": "Zero additional" if aging_on else "No incremental limit"}
                    result = solve_plan(
                        fleet, first, grid[first:, 3], np.zeros(len(grid)-first),
                        prefix, histories[first], target=ref.sum(axis=1),
                        budget=losses if aging_on else None, window=(first, fixed, direction, None),
                        offline=True,
                    )
                    potential = np.nan if result is None else max(0.0, result[1])
                    source = "solver" if result is not None else "unresolved"
                    reference_deviation = np.min(direction * (
                        ref[:, first:first+duration].sum(axis=0) - fixed))
                    if result is None and reference_deviation >= -1e-7:
                        potential = max(0.0, reference_deviation)
                        source = "reference_witness"
                    for r in COMMITMENTS:
                        ok, reason = activate(
                            fleet, grid, forecast, count, baseline, first, direction, r, fixed, aging_on,
                        )
                        if ok and (not np.isfinite(potential) or r > potential + 1e-5):
                            # A verified online trajectory also certifies offline feasibility.
                            potential = float(r)
                            source = "online_witness"
                        delivery.append({**row, "commitment_kw": r, "delivered": ok, "outcome": reason})
                    offline.append({**row, "offline_kw": potential, "feasible": np.isfinite(potential),
                                    "offline_source": source})


def summarize(offline, delivery, service, output):
    keys = ["capacity", "pairing", "group", "start_hour", "duration_h", "direction", "aging_budget"]
    test = offline[offline.split == "test"]
    bounds = test.groupby(keys, sort=False).offline_kw.agg(
        offline_feasible="count", offline_median_kw="median",
        offline_min_kw="min", offline_max_kw="max").reset_index()
    rates = delivery.groupby(keys + ["split", "commitment_kw"], sort=False).delivered.agg(
        successes="sum", trajectories="size", delivery_rate="mean").reset_index()
    rates.to_csv(output / "delivery_rates.csv", index=False)
    eligible = rates[rates.delivery_rate >= RELIABILITY]
    commitments = eligible.groupby(keys + ["split"]).commitment_kw.max().unstack("split").rename(
        columns={"test": "test_empirical_kw", "validation": "validation_selected_kw"}).reset_index()
    for col in ("test_empirical_kw", "validation_selected_kw"):
        if col not in commitments:
            commitments[col] = np.nan
    table = bounds.merge(commitments, on=keys, how="left")
    if "offline_source" in test:
        solved = (test.assign(offline_solver=test.offline_source.eq("solver"))
                  .groupby(keys, sort=False).offline_solver.sum().reset_index())
        table = table.merge(solved, on=keys, how="left")
    test_rates = rates[rates.split == "test"].rename(columns={"commitment_kw": "validation_selected_kw"})
    table = table.merge(test_rates[keys + ["validation_selected_kw", "successes", "trajectories", "delivery_rate"]],
                        on=keys + ["validation_selected_kw"], how="left")
    table.to_csv(output / "table_flexibility.csv", index=False)

    tail = []
    for keys_, subset in service.groupby(["capacity", "pairing", "group", "planning_paths"], sort=False):
        cap, pair, group, paths = keys_
        values = subset.unmet_kwh.to_numpy()
        for sample_size in (2, 3, len(values)):
            for alpha in (0.8, 0.9, 0.95):
                estimates = [float(cvar(np.array(sample), alpha)) for sample in combinations(values, sample_size)]
                tail.append(dict(capacity=cap, pairing=pair, group=group, planning_paths=paths,
                                 evaluation_size=sample_size, alpha=alpha, subsets=len(estimates),
                                 median_cvar_kwh=np.median(estimates), min_cvar_kwh=min(estimates),
                                 max_cvar_kwh=max(estimates)))
    pd.DataFrame(tail).drop_duplicates().to_csv(output / "tail_sensitivity.csv", index=False)
    return table, rates, pd.DataFrame(tail).drop_duplicates()


def figures(table, rates, tail, output):
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    blue, orange = "#0072B2", "#D55E00"
    for capacity in ("Raw", "1500 kW floor"):
        key = "raw" if capacity == "Raw" else "capacity_floor"
        subset = rates[(rates.capacity == capacity) & (rates.pairing == "Original")
                       & (rates.aging_budget == "Zero additional") & (rates.start_hour == 17)
                       & (rates.split == "test")]
        comparison = table[(table.capacity == capacity) & (table.pairing == "Original")
                           & (table.aging_budget == "Zero additional") & (table.start_hour == 17)]
        fig, axes = plt.subplots(2, 4, figsize=(9, 4.6), sharex="row", sharey="row")
        for col, group in enumerate(GROUPS):
            top, bottom = axes[:, col]
            top.set_title(group)
            for direction, color, offset in (("Up", blue, -0.09), ("Down", orange, 0.09)):
                data = comparison[(comparison.group == group) & (comparison.direction == direction)].sort_values("duration_h")
                x = np.arange(len(data)) + offset
                top.scatter(x, data.offline_median_kw, color=color, s=14)
                top.vlines(x, data.offline_min_kw, data.offline_max_kw, color=color, linewidth=1.1)
                top.scatter(x, data.test_empirical_kw, marker="x", color=color, s=26)
                absent = data.offline_feasible == 0
                for x0 in x[absent]:
                    top.plot(x0, 0.03, marker="|", color=color, transform=top.get_xaxis_transform())
                group_rates = subset[(subset.group == group) & (subset.direction == direction)
                                     & (subset.duration_h == 1)].sort_values("commitment_kw")
                bottom.plot(group_rates.commitment_kw, group_rates.delivery_rate,
                            linestyle="-" if direction == "Up" else "--",
                            marker="o" if direction == "Up" else "s",
                            markerfacecolor=color if direction == "Up" else "white",
                            markersize=3, color=color)
            top.set_xticks(range(len(DURATIONS)))
            top.set_xticklabels(DURATIONS)
            top.set_xlabel("Duration (h)")
            bottom.axhline(RELIABILITY, color="0.45", ls="--", linewidth=0.8)
            bottom.set_ylim(-0.05, 1.08)
            bottom.set_xlabel("Commitment (kW)")
        if capacity == "Raw" and not (comparison.offline_max_kw > 0).any():
            axes[0, 0].set_ylim(-0.1, 0.1)
            axes[0, 0].set_yticks([0])
        else:
            axes[0, 0].set_ylim(bottom=-0.5)
        axes[0, 0].set_ylabel("Flexibility (kW)")
        axes[1, 0].set_ylabel("1 h delivery fraction")
        fig.suptitle(f"Offline potential and rolling delivery at 17:00 | {capacity}", fontsize=10)
        fig.text(0.5, 0.026, "Blue: upward   Orange: downward   Dot and line: offline median and range\n"
                 "×: empirical online threshold met   |: no feasible offline plan found   Dashed line: 95% target",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.11, 1, 0.95))
        save(fig, output / f"fig_flexibility_{key}")

    shown = tail[(tail.capacity == "Raw") & (tail.pairing == "Original")
                 & (tail.planning_paths == 4) & (tail.alpha == 0.95)]
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    for i, group in enumerate(GROUPS):
        data = shown[shown.group == group].sort_values("evaluation_size")
        x = data.evaluation_size.to_numpy() + (i - 1.5) * 0.09
        ax.plot(x, data.median_cvar_kwh, marker="o", markersize=4, label=group)
        ax.vlines(x, data.min_cvar_kwh, data.max_cvar_kwh, color=ax.lines[-1].get_color())
    ax.set_xlabel("Number of replay trajectories in subset")
    ax.set_ylabel("CVaR95 of unmet energy (kWh)")
    ax.set_xticks(sorted(shown.evaluation_size.unique()))
    ax.legend(frameon=False, ncol=2, loc="center")
    fig.tight_layout()
    save(fig, output / "fig_tail_sensitivity")


def save(fig, path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(fig)


def manuscript_outputs(table, rates, service, tail, output):
    focus = table[(table.start_hour == 17) & (table.duration_h == 1)].copy()
    rows = focus[(focus.pairing == "Original") & (focus.aging_budget == "Zero additional")]
    report = [
        "# Section IV results from the available scenario archive",
        "",
        "## Offline potential and rolling delivery at 17:00 for one-hour activation",
        "",
        "| Capacity | Scenario group | Direction | Feasible offline plans found | Offline median [range] (kW) | Validation-selected commitment (kW) | Independent test deliveries | Test empirical 95% commitment (kW) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]

    def number(value):
        return "—" if pd.isna(value) else f"{value:.2f}"

    for row in rows.itertuples():
        interval = "—" if pd.isna(row.offline_median_kw) else (
            f"{row.offline_median_kw:.2f} [{row.offline_min_kw:.2f}, {row.offline_max_kw:.2f}]")
        successes = "—" if pd.isna(row.successes) else f"{int(row.successes)}/{int(row.trajectories)}"
        report.append(f"| {row.capacity} | {row.group} | {row.direction} | "
                      f"{row.offline_feasible}/3 | {interval} | {number(row.validation_selected_kw)} | "
                      f"{successes} | {number(row.test_empirical_kw)} |")
    report += [
        "",
        "An em dash indicates that the corresponding result was unavailable. A value of 0 kW is a tested value. "
        "The offline median and range are conditional on a feasible solution being found. Cases where the cutting-plane "
        "solve did not produce a feasible solution use a verified reference or online witness when available. "
        "The offline source is recorded for every case. Three of three deliveries "
        "are required to clear the 95% empirical threshold with three test trajectories. This threshold is descriptive "
        "and does not establish 95% population reliability. Even 3/3 has a two-sided exact 95% binomial lower bound of 29.2% under independent Bernoulli sampling.",
        "",
        "## Main findings",
        "",
    ]
    paired_raw = service[(service.capacity == "Raw") & (service.pairing == "Original")
                         & (service.split == "test") & (service.planning_paths == 4)]
    paired_floor = service[(service.capacity == "1500 kW floor") & (service.pairing == "Original")
                           & (service.split == "test") & (service.planning_paths == 4)]
    for cap, data in (("raw", paired_raw), ("1,500 kW floor", paired_floor)):
        zero = data.zero_capacity_hours
        report.append(f"- Under the {cap} setting, the test trajectories had {zero.min()}–{zero.max()} zero-capacity "
                      f"hours per day. Mean served-energy ratios across test cases were "
                      f"{100*data.served_ratio.mean():.1f}%.")
    primary = table[(table.capacity == "Raw") & (table.pairing == "Original")
                    & (table.aging_budget == "Zero additional")]
    report.append(f"- Among {len(primary)} raw-capacity combinations of group, start, duration, and direction, "
                  f"{int((primary.test_empirical_kw > 0).sum())} met the empirical threshold at a positive "
                  "tested commitment. See the complete table for both start times and all durations.")
    evening = rows[(rows.capacity == "1500 kW floor") & (rows.group == "Evening peak")
                   & (rows.direction == "Down")].iloc[0]
    report.append(f"- With the floor, the evening-peak offline median for a one-hour reduction at 17:00 was "
                  f"{evening.offline_median_kw:.2f} kW. The validation-selected {evening.validation_selected_kw:g} kW "
                  f"commitment delivered on {int(evening.successes)}/{int(evening.trajectories)} test trajectories. "
                  f"The largest tested commitment that delivered on all test trajectories was "
                  f"{evening.test_empirical_kw:g} kW.")
    report.append("- The archived capacity-shortage traces have the same hourly capacity as the paired normal traces. "
                  "The separately constructed shortage control caps charging at 15 kW from 17:00 to 21:00 "
                  "on top of the 1,500 kW floor input and leaves other hours unchanged. Its outputs are stored separately.")
    report += [
        "",
        "## Methods and scope",
        "",
        "- Scenario IDs 00–03 supplied the planning paths, 04–06 were used for commitment selection, and "
        "07–09 were used for test replay in each group. Scenario IDs are paired across stress regimes. Each "
        "online solve uses one common forecast plan across the planning paths and dispatches its first action. "
        "Later actions are recomputed after observing new data. The planning representation includes already "
        "observed EVs, uses the minimum forecast capacity, and shares even future planned actions across forecast paths. The four grid signals are clipped to their planning-set min–max scale, "
        "and their relative weights are normalized to sum to one. The departure shortfall penalty is 2 per kWh.",
        "- The primary input is the archived capacity proxy with zero capacity preserved. The sensitivity arm "
        "applies the legacy 1,500 kW floor to both the planning and evaluation inputs. This is a distinct "
        "assumed capacity regime. It is not evidence that the archived grid had that headroom.",
        "- A fixed power reference is set at commitment. The solver uses causal projections of the shadow reference "
        "and tests terminal energy and aging against its realized trajectory. Windows are 1, 2 and 4 h, beginning "
        "at hour 15 or 17. The commitment grid is 0, 2, 5, 10, 20 and 40 kW. The archive truncates sessions at "
        "the end of the 24 h grid trace. Departure validation uses the supplied truncated indices.",
        "- The archived semi-empirical aging model supplies both the objective and the per-vehicle incremental "
        "constraint. The relaxed arm removes the incremental constraint and retains the aging cost. The grid-pairing "
        "control cycles complete grid paths within each regime and data split, keeping the EV sessions fixed. "
        "This is an archive-level dependence sensitivity. It does not verify a learned CVAE–diffusion joint distribution.",
        "- The archive CVaR sensitivity compares 1, 2 and 4 planning paths and exhaustively evaluates two- and "
        "three-trajectory subsets of the same available test cases. It is a small-sample diagnostic. A calibrated "
        "PINN checkpoint, independent cell validation and the manuscript's CVAE–diffusion model are required "
        "for the full Section IV claims. "
        "Deadline-policy and equal-budget LLM/BO comparisons are outside this replay.",
        "",
        "## Figure captions",
        "",
        "**Offline potential and online delivery.** Median and range of feasible scenario-wise offline potentials "
        "(top) and the empirical success fractions for a one-hour activation (bottom), at 17:00 under zero additional "
        "aging. The two directions are reported separately. A cross marks the largest tested magnitude meeting "
        "the empirical threshold. A separate panel set uses the legacy 1,500 kW floor.",
        "",
        "**Aging and pairing sensitivity.** Number of successful positive-commitment replays across the two starts, "
        "three durations, five positive magnitudes, and three test trajectories. The factorial comparison crosses "
        "the original versus permuted grid paths with the presence versus absence of the incremental aging budget. "
        "Counts are descriptive because the same trajectories contribute to multiple commitments.",
        "",
        "**Tail sensitivity.** Median and range of empirical CVaR95 of unmet energy across all test subsets of "
        "the indicated size, using four planning paths. These ranges describe subset sensitivity.",
    ]
    control_path = output / "constructed_shortage" / "table_flexibility.csv"
    if control_path.exists():
        control = pd.read_csv(control_path)
        original = focus[(focus.group == "Capacity shortage") & (focus.capacity == "1500 kW floor")]
        comparison = pd.concat([original, control[(control.start_hour == 17) & (control.duration_h == 1)]],
                               ignore_index=True)
        comparison.to_csv(output / "table_capacity_control.csv", index=False)
        ctrl_rates = pd.read_csv(control_path.with_name("delivery_rates.csv"))
        combined = pd.concat([rates[rates.capacity == "1500 kW floor"], ctrl_rates], ignore_index=True)
        combined = combined[(combined.group == "Capacity shortage") & (combined.pairing == "Original")
                            & (combined.aging_budget == "Zero additional") & (combined.start_hour == 17)
                            & (combined.duration_h == 1) & (combined.split == "test")]
        fig, axes = plt.subplots(1, 2, figsize=(6, 2.8), sharey=True)
        for direction, ax in zip(("Up", "Down"), axes):
            for cap, data in combined[combined.direction == direction].groupby("capacity", sort=False):
                data = data.sort_values("commitment_kw")
                ax.plot(data.commitment_kw, data.delivery_rate, marker="o", markersize=4, label=cap)
            ax.axhline(RELIABILITY, ls="--", color="0.45", linewidth=0.8)
            ax.set_title(direction)
            ax.set_ylim(-0.04, 1.08)
            ax.set_xlabel("Commitment (kW)")
        axes[0].set_ylabel("Delivery fraction (3 test trajectories)")
        handles, legend_labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, legend_labels, frameon=False, fontsize=7, loc="lower center", ncol=2)
        fig.suptitle("Constructed capacity shortage | 1 h at 17:00", fontsize=10)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        save(fig, output / "fig_capacity_control")
        report += ["", "**Constructed capacity shortage.** Replay delivery with the 1,500 kW floor and with "
                   "an additional 15 kW cap during 17:00–21:00. The paired runs use the same archived EV and grid "
                   "inputs, reference policy, and evaluation split. Reference trajectories are recomputed under each cap."]
    legacy_tail = output / "legacy_tail" / "legacy_tail_seed_sensitivity.csv"
    if legacy_tail.exists():
        values = pd.read_csv(legacy_tail)
        report += ["", "**Additional synthetic tail sensitivity.** Fresh draws from the archived GMM and random "
                   "forest models at three random seeds. The four regimes use the same parent draw within each seed. "
                   "The capacity-shortage regime adds a 15 kW cap to the raw capacity proxy during 17:00–21:00. "
                   "These draws provide a sensitivity check for the available generators."]
        for group, case in values[values.alpha.eq(0.95)].groupby("group", sort=False):
            largest = case[case.n == case.n.max()]
            report.append(f"- {group}, n = {int(case.n.max())} per seed: CVaR95 ranged from "
                          f"{largest.cvar_kwh.min():.1f} to {largest.cvar_kwh.max():.1f} kWh across seeds.")
    (output / "manuscript_results.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    factor_keys = ["capacity", "pairing", "group", "direction", "aging_budget"]
    factor = (rates[(rates.split == "test") & (rates.commitment_kw > 0)]
              .groupby(factor_keys, sort=False)[["successes", "trajectories"]].sum().reset_index())
    replay_count = len(STARTS) * len(DURATIONS) * (len(COMMITMENTS) - 1) * 3
    factor.to_csv(output / "table_aging_pairing.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.8), sharex=True, sharey=True)
    for ax, (cap, direction) in zip(axes.flat, (
            ("Raw", "Up"), ("Raw", "Down"), ("1500 kW floor", "Up"), ("1500 kW floor", "Down"))):
        frame = factor[(factor.capacity == cap) & (factor.direction == direction)]
        labels = []
        for index, (pair, aging_budget) in enumerate((
                ("Original", "Zero additional"), ("Original", "No incremental limit"),
                ("Permuted", "Zero additional"), ("Permuted", "No incremental limit"))):
            values = frame[(frame.pairing == pair) & (frame.aging_budget == aging_budget)]
            values = values.set_index("group").reindex(list(GROUPS))
            labels.append(values.successes.to_numpy())
        ax.imshow(np.array(labels).T, vmin=0, vmax=replay_count, cmap="Blues", aspect="auto")
        for i in range(4):
            for j in range(4):
                v = labels[j][i]
                ax.text(j, i, f"{int(v)}/{replay_count}", ha="center", va="center",
                        color="white" if v > replay_count / 2 else "black")
        ax.set_title(f"{cap} | {direction}")
        ax.set_yticks(range(4))
        ax.set_yticklabels(GROUPS)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Original\nzero", "Original\nrelaxed", "Permuted\nzero", "Permuted\nrelaxed"])
    fig.suptitle("Aging and EV–grid pairing | positive commitments", fontsize=10)
    fig.tight_layout()
    save(fig, output / "fig_aging_pairing")


def check():
    model = BatteryDegradationModel(AGING)
    values = np.array([4.0, 2.0, 3.0])
    soc, cap = 0.68, 60.0
    losses, grad = aging(values, soc, cap)
    for i, kw in enumerate(values):
        start = soc + values[:i].sum() / cap
        assert np.isclose(losses[i], model.compute_increment(start, kw, cap)["delta_soh_total"])
        direction = np.eye(len(values))[i] * 1e-5
        assert np.isclose((aging(values+direction, soc, cap)[0].sum() - losses.sum()) / 1e-5,
                          grad[i], atol=1e-8)
    assert np.isclose(cvar(np.arange(10), 0.95), 9)
    assert np.isclose(cvar(np.arange(10), 0.85), (9 + 0.5 * 8) / 1.5)
    fleet = Fleet(np.array([0]), np.array([2]), np.array([7.]), np.array([7.]),
                  np.array([60.]), np.array([.3]))
    executed = np.zeros((1, 2))
    result = solve_plan(fleet, 0, np.array([7., 7.]), np.zeros(2), executed, np.zeros(1),
                        target=np.array([7.]), window=(0, np.array([7.]), -1, None), offline=True)
    assert np.isclose(result[1], 7) and np.allclose(result[0], [[0, 7]])
    # Alter unrevealed grid/EV inputs. The dispatch before either change is identical.
    grid = np.tile([0.1, 0.001, 1., 7.], (2, 1))
    forecast = Forecast([grid])
    other = grid.copy()
    other[1] = [2., 0.05, 8., 0.]
    two = Fleet(np.array([0, 1]), np.array([2, 2]), np.array([7., 7.]), np.array([7., 7.]),
                np.array([60., 60.]), np.array([.3, .3]))
    a = reference_replay(fleet, grid, forecast, 1)[0]
    b = reference_replay(two, other, forecast, 1)[0]
    assert np.allclose(a[:, 0], b[:1, 0]) and b[1, 0] == 0


def evaluate_group(job):
    capacity_mode, floor, name, group, pairing, shortage_only = job
    cases = load(group, floor, shortage_only)
    forecast = Forecast([case[2] for case in cases[:4]])
    offline, delivery, service = [], [], []
    for split, selected in (("validation", cases[4:7]), ("test", cases[7:])):
        for index, (case_id, fleet, grid) in enumerate(selected):
            if pairing == "Permuted":
                grid = selected[(index + 1) % len(selected)][2]
            run_case(name, split, case_id, fleet, grid, forecast, pairing,
                     capacity_mode, 4, offline, delivery, service)
            if split == "test" and capacity_mode == "Raw" and pairing == "Original":
                for n in (1, 2):
                    run_case(name, split, case_id, fleet, grid, forecast, pairing,
                             capacity_mode, n, offline, delivery, service)
    print(f"Finished {capacity_mode} | {name} | {pairing}", flush=True)
    return offline, delivery, service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "section_iv_replay_results")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--shortage-only", action="store_true",
                        help="Run the constructed 15 kW capacity control in a separate directory")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.shortage_only and args.output == ROOT / "section_iv_replay_results":
        args.output = args.output / "constructed_shortage"
    if args.report_only:
        offline, delivery, service = (pd.read_csv(args.output / name) for name in
                                      ("offline_cases.csv", "replay_cases.csv", "reference_service.csv"))
        table, rates, tail = summarize(offline, delivery, service[service.split == "test"], args.output)
        figures(table, rates, tail, args.output)
        manuscript_outputs(table, rates, service, tail, args.output)
        return
    check()
    if args.check_only:
        print("Passed the aging, CVaR, offline and information-prefix checks.")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    offline, delivery, service = [], [], []
    settings = (("Constructed 15 kW shortage", 1500),) if args.shortage_only else (
        ("Raw", 0), ("1500 kW floor", 1500))
    groups = {"Capacity shortage": "capacity_shortage"} if args.shortage_only else GROUPS
    jobs = [(capacity_mode, floor, name, group, pairing, args.shortage_only)
            for capacity_mode, floor in settings for name, group in groups.items()
            for pairing in ("Original", "Permuted")]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(evaluate_group, jobs))
    else:
        results = list(map(evaluate_group, jobs))
    for result in results:
        for rows, new_rows in zip((offline, delivery, service), result):
            rows.extend(new_rows)
    offline, delivery, service = (pd.DataFrame(rows) for rows in (offline, delivery, service))
    offline.to_csv(args.output / "offline_cases.csv", index=False)
    delivery.to_csv(args.output / "replay_cases.csv", index=False)
    service.to_csv(args.output / "reference_service.csv", index=False)
    table, rates, tail = summarize(offline, delivery, service[service.split == "test"], args.output)
    if not args.shortage_only:
        figures(table, rates, tail, args.output)
        manuscript_outputs(table, rates, service, tail, args.output)


if __name__ == "__main__":
    main()

