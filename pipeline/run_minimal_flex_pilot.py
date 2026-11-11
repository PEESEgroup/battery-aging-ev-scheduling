"""Replay one-hour downward EV commitments and screen nine local configurations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from nonanticipative_scheduler import (
    Forecast, TOL, reference_replay, replay_aging, solve_plan, update,
)
from run_aging_capacity_diagnostics import controlled
from run_section_iv_replay import ROOT, load, plt

START = 17
RECOVERY = slice(18, 21)
COMMITMENTS = (0, 5, 10, 15, 20, 25, 30)
BUDGETS = (("zero", 0.0), ("positive", 1e-5), ("unlimited", None))
CONFIGS = tuple(product((0.0, 0.25, 0.5), (0.0, 1e-6, 5e-6)))
STRESSES = ("00", "10", "01", "11")


class PilotForecast(Forecast):
    """Keep the existing stress score and change only future capacity quantiles."""

    def capacities(self, hour, observed, quantile):
        caps, score = self.build(hour, observed, 4)
        if quantile:
            history = self.signals(observed)
            distance = (((self.signals(self.donors[:, :hour+1]) - history)
                         / self.scale) ** 2).mean(axis=(1, 2))
            paths = self.donors[np.argsort(distance, kind="stable")[:4], hour:, 3]
            caps[1:] = np.quantile(paths[:, 1:], quantile, axis=0)
        return caps, score


def stressed(grid, cap):
    changed = grid.copy()
    changed[RECOVERY, 3] = np.minimum(changed[RECOVERY, 3], cap)
    return changed


def replay(fleet, grid, forecast, baseline, fixed, q, allowance, quantile, margin):
    ref, ref_loss, plans, history = baseline
    executed = ref.copy()
    executed[:, START:] = 0.0
    losses = history[START].copy()
    for hour in range(START, len(grid)):
        caps, score = forecast.capacities(hour, grid[:hour+1], quantile)
        projected = plans[hour].copy()
        projected[:, :hour] = ref[:, :hour]
        budget = None if allowance is None else replay_aging(fleet, projected) + allowance - margin
        detail = {}
        solution = solve_plan(
            fleet, hour, caps, score, executed, losses,
            target=projected.sum(axis=1), budget=budget,
            window=(START, fixed, -1, q), diagnostics=detail,
        )
        if solution is None:
            return dict(outcome="planning_infeasible", failure_hour=hour,
                        solver_status=detail["status"], delivered=False,
                        common_health_pass=False, complete_delivery=False)
        update(fleet, executed, losses, hour, solution[0][:, hour])

    served = executed.sum(axis=1)
    extra = losses - ref_loss
    aggregate = executed.sum(axis=0)
    gap = max(0.0, float(aggregate[START] - fixed[0] + q))
    grid_ok = np.all(aggregate <= grid[:, 3] + 1e-6)
    service_ok = np.allclose(served, ref.sum(axis=1), atol=1e-6, rtol=0)
    full_service = bool(np.all(served >= fleet.demand - 1e-6))
    health_ok = allowance is None or np.all(extra <= allowance + TOL)
    common_health = bool(np.all(extra <= 1e-5 + TOL))
    outcome = "delivered"
    for failed, reason in (
        (gap > 1e-6 or not grid_ok, "power_or_capacity"),
        (not service_ok, "departure_energy"), (not health_ok, "aging"),
    ):
        if failed:
            outcome = reason
            break
    return dict(
        outcome=outcome, delivered=outcome == "delivered",
        complete_delivery=outcome == "delivered" and full_service,
        service_matched=bool(service_ok), full_service=full_service,
        unmet_kwh=float(np.maximum(fleet.demand - served, 0).sum()),
        max_extra_soh=float(extra.max(initial=-np.inf)) if len(extra) else np.nan,
        max_extra_over_external_soh=float(np.maximum(extra - 1e-5, 0).max(initial=0)),
        common_health_pass=common_health,
        power_gap_kw=gap, recovery_peak_kw=float(aggregate[RECOVERY].max()),
        power_17_kw=float(aggregate[START]),
    )


def evaluate(job):
    split, case, group, fleet, grid, ref_forecast, forecast, study = job
    base = reference_replay(fleet, grid, ref_forecast, 4)
    ref, losses, plans, history = base
    fixed = plans[START].sum(axis=0)[START:START+1]
    prefix = ref.copy()
    prefix[:, START:] = 0
    target = ref.sum(axis=1)
    context = dict(split=split, scenario_id=case, pressure=group,
                   reference_kw=float(fixed[0]),
                   reference_unmet_kwh=float(np.maximum(fleet.demand-target, 0).sum()),
                   recovery_cap_kw=float(grid[RECOVERY, 3].min()))
    offline = []
    rows = []
    if study == "factorial":
        for label, budget in BUDGETS:
            detail = {}
            solution = solve_plan(
                fleet, START, grid[START:, 3], np.zeros(len(grid)-START),
                prefix, history[START], target=target,
                budget=None if budget is None else losses + budget,
                window=(START, fixed, -1, None), offline=True,
                allow_signed=True, diagnostics=detail,
            )
            offline.append(dict(
                **context, budget=label, signed_offline_kw=solution[1] if solution else np.nan,
                offline_status=detail["status"], cuts=detail["cuts"],
                max_extra_soh=float((replay_aging(fleet, prefix+solution[0])-losses).max())
                if solution and len(fleet.arrival) else np.nan,
            ))
        settings = [(label, budget, 0.0, 0.0) for label, budget in BUDGETS]
    else:
        # The default grid point is already included in the factorial pass.
        settings = [("positive", 1e-5, quantile, margin)
                    for quantile, margin in CONFIGS if quantile or margin]
    for label, budget, quantile, margin in settings:
        for q in COMMITMENTS:
            result = replay(fleet, grid, forecast, base, fixed, q, budget, quantile, margin)
            rows.append(dict(**context, study=study, budget=label,
                             budget_soh=budget, quantile=quantile, margin_soh=margin,
                             commitment_kw=q, **result))
    print(f"{study} {split} {group} {case}", flush=True)
    return rows, offline


def run(jobs, workers):
    if workers == 1:
        results = list(map(evaluate, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(evaluate, jobs))
    return ([row for rows, _ in results for row in rows],
            [row for _, rows in results for row in rows])


def plots(control, offline, events, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), constrained_layout=True)
    subset = control[(control.direction == "Down") & control.capacity_kw.isin((4, 5))]
    for cap, part in subset.groupby("capacity_kw"):
        x = pd.to_numeric(part.allowance, errors="coerce")
        axes[0].plot(x.fillna(4e-5)*1e6, part.numeric_kw, "o-", label=f"{cap:g} kW cap")
    axes[0].set(xticks=[0, 5, 10, 20, 30, 40],
                xticklabels=["0", "5", "10", "20", "30", "None"],
                xlabel="Extra SOH budget (millionths)", ylabel="Offline reduction (kW)",
                title="Controlled single EV")
    axes[0].legend()
    pairs = offline.pivot(index=["pressure", "scenario_id", "split"],
                          columns="budget", values="signed_offline_kw")
    for group in STRESSES:
        part = pairs.xs(group, level="pressure")
        valid = part[["zero", "unlimited"]].dropna()
        valid = valid[(valid.zero >= -1e-5) & (valid.unlimited >= -1e-5)]
        delta = valid.unlimited.clip(lower=0) - valid.zero.clip(lower=0)
        axes[1].scatter([group]*len(delta), delta, alpha=0.55)
        axes[1].text(group, .95, f"n={len(delta)}", ha="center",
                     transform=axes[1].get_xaxis_transform())
    axes[1].axhline(0, color="gray", linewidth=1)
    axes[1].set(xlabel="EV stress / grid stress", ylabel="Increase after removing budget (kW)",
                title="Archive pairs with nonnegative feasible bounds")
    fig.savefig(output/"aging_contrast.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), constrained_layout=True)
    chosen = events[(events.study == "factorial") & (events.budget == "positive")
                    & (events.split == "regression")]
    for group, part in chosen.groupby("pressure"):
        counts = part.groupby("commitment_kw").delivered.sum()
        axes[0].plot(counts.index, counts, "o-", label=group)
    axes[0].set(xlabel="Commitment (kW)", ylabel="Events delivered out of 3",
                yticks=range(4), title="Four pressure combinations")
    axes[0].legend(title="EV / grid")
    grid = grid_scores(events)
    gains = grid.groupby(["pressure", "scenario_id", "split"]).agg(
        best=("max_delivered_kw", "max"),
        default=("default_kw", "first"),
    ).reset_index()
    for j, split in enumerate(("validation", "regression")):
        part = gains[gains.split == split]
        for k, group in enumerate(STRESSES):
            delta = part[part.pressure == group].eval("best - default").dropna()
            axes[1].scatter([k + .12*(j-.5)]*len(delta), delta,
                            label=split if k == 0 else None, alpha=.7)
    axes[1].axhline(5, color="gray", linestyle="--", linewidth=1)
    axes[1].set(xticks=range(4), xticklabels=STRESSES, xlabel="EV stress / grid stress",
                ylabel="Best grid point minus default (kW)", title="Configuration headroom")
    axes[1].legend()
    fig.savefig(output/"joint_stress_and_config.png", dpi=170)
    plt.close(fig)


def grid_scores(events):
    keys = ["split", "scenario_id", "pressure", "quantile", "margin_soh"]
    positive = events[events.budget == "positive"].copy()
    positive["max_delivered_kw"] = positive.commitment_kw.where(positive.delivered)
    scores = positive.groupby(keys, as_index=False).max_delivered_kw.max()
    default = scores[(scores["quantile"] == 0) & (scores.margin_soh == 0)]
    return scores.merge(default[keys[:3]+["max_delivered_kw"]].rename(
        columns={"max_delivered_kw": "default_kw"}), on=keys[:3], validate="many_to_one")


def report(events, offline, cap, output):
    grid = grid_scores(events)
    groups = ["split", "pressure", "budget"]
    factor = events[events.study == "factorial"]
    rates = factor.groupby(groups+["commitment_kw"]).delivered.sum()
    selected = rates.loc["validation"].reset_index()
    selected = selected[selected.delivered == 3].groupby(["pressure", "budget"]).commitment_kw.max()
    lines = [
        "# One-hour downward charging pilot", "",
        f"Constructed base capacity floor: 1500 kW. Recovery cap: {cap:.2f} kW, 18:00–21:00.",
        "Cap is the greater of 15 kW and the 75th percentile of eight planning-only reference recovery peaks.",
        "Scenario 04–06 select commitments; previously viewed 07–09 are regression checks, not independent confirmation.",
        "No-activation reference uses four uncapped grid donors; candidate planning uses four nearest paths from eight paired uncapped/capped donors.",
        "A failed rolling solve ends that replay. Actual power gaps and aging are only available for completed trajectories.",
        "Budget variants are evaluated against their own budget here; the positive-budget configuration grid has one shared external standard.",
        "", "## Validation-selected commitments and regression delivery", "",
        "| Pressure | Budget | Selected kW | Matched-service deliveries / 3 | Full-service deliveries / 3 |",
        "|---|---|---:|---:|---:|",
    ]
    for group, budget in product(STRESSES, [label for label, _ in BUDGETS]):
        q = selected.get((group, budget))
        part = factor[(factor.split == "regression") & (factor.pressure == group)
                      & (factor.budget == budget) & (factor.commitment_kw == q)]
        row = (f"{q:g} | {part.delivered.sum()}/3 | {part.complete_delivery.sum()}/3"
               if q is not None else "— | — | —")
        lines.append(f"| {group} | {budget} | {row} |")
    pairs = offline.pivot(index=["split", "scenario_id", "pressure"],
                          columns="budget", values="signed_offline_kw")
    eligible = pairs[["zero", "unlimited"]].dropna()
    eligible = eligible[(eligible.zero >= -1e-5) & (eligible.unlimited >= -1e-5)]
    affected = int(((eligible.unlimited - eligible.zero) > 1e-4).sum())
    gains = grid.groupby(["split", "scenario_id", "pressure"], as_index=False).agg(
        best=("max_delivered_kw", "max"), default=("default_kw", "first"))
    gains["gain"] = gains.best - gains.default
    hits = gains[gains.gain >= 5]
    paired = factor[factor.budget == "positive"].pivot(
        index=["split", "scenario_id", "commitment_kw"],
        columns="pressure", values="delivered",
    )
    grid_flips = int((paired["00"] != paired["01"]).sum()
                     + (paired["10"] != paired["11"]).sum())
    extra_failures = offline.offline_status.value_counts()
    negative = int(((pairs.zero < -1e-5) | (pairs.unlimited < -1e-5)).sum())
    # Require distinct scenario IDs in both partitions before spending effort on LLM.
    counts = hits.groupby("split").scenario_id.nunique()
    proceed = all(counts.get(split, 0) >= 2 for split in ("validation", "regression"))
    lines.extend([
        "", "## Decision", "",
        f"- Aging changed {affected}/{len(eligible)} nonnegative comparable offline bounds by more than 0.0001 kW.",
        f"- {negative} paired blocks had a negative direction bound. "
        f"{extra_failures.get('cut_limit', 0)} individual offline solves reached the aging cut limit.",
        f"- The recovery cap changed {grid_flips}/{2 * len(paired)} paired delivery outcomes at fixed kW under the positive budget.",
        f"- Reference unmet demand ranged from {factor.reference_unmet_kwh.min():.2f} to "
        f"{factor.reference_unmet_kwh.max():.2f} kWh per event. Across all replay attempts, "
        f"{events.complete_delivery.sum()} met both the commitment and full demand.",
        f"- Grid configurations improved on the default by at least 5 kW in {len(hits)}/{len(gains)} event/pressure blocks with "
        f"{counts.get('validation', 0)} distinct validation and {counts.get('regression', 0)} distinct regression scenario IDs.",
        f"- {int((gains.default.isna() & gains.best.notna()).sum())} blocks moved from no successful tested default commitment to a successful grid commitment. These are reported separately from numerical gains.",
        f"- LLM gate: {'promising enough for a rule/BO comparison on new events' if proceed else 'defer LLM tuning under this pilot design'}. "
        "The grid maximum is an event-wise screening upper envelope, not the performance of a deployable selector.",
        "- No LLM was called. The sample supports descriptive comparisons only.",
    ])
    (output/"conclusions.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"minimal_flex_pilot_results")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    normal = load("joint", 1500)
    evening = load("evening_peak_cluster", 1500)
    assert [row[0] for row in normal] == [row[0] for row in evening]
    donors = [row[2] for row in normal[:4]]
    reference = Forecast(donors)
    planning_peaks = [
        reference_replay(fleet, grid, reference, 4)[0].sum(axis=0)[RECOVERY].max()
        for (_, normal_fleet, grid), (_, evening_fleet, _) in zip(normal[:4], evening[:4])
        for fleet in (normal_fleet, evening_fleet)
    ]
    cap = max(15.0, float(np.quantile(planning_peaks, 0.75)))
    forecasts = PilotForecast([path for grid in donors for path in (grid, stressed(grid, cap))])
    jobs = [
        ("validation" if i < 7 else "regression", case, group, fleet,
         stressed(grid, cap) if group[1] == "1" else grid, reference, forecasts)
        for i, ((case, normal_fleet, grid), (_, evening_fleet, _))
        in enumerate(zip(normal, evening)) if i >= 4
        for group, fleet in (("00", normal_fleet), ("10", evening_fleet),
                             ("01", normal_fleet), ("11", evening_fleet))
    ]
    print(f"Recovery cap {cap:.4f} kW; running {len(jobs)} event/pressure blocks", flush=True)
    control = controlled()
    rows, bounds = run([(*job, "factorial") for job in jobs], args.workers)
    offline = pd.DataFrame(bounds)
    offline.to_csv(args.output/"offline_aging_bounds.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output/"event_replays.csv", index=False)
    grid_rows, _ = run([(*job, "grid") for job in jobs], args.workers)
    events = pd.DataFrame(rows + grid_rows)
    events.to_csv(args.output/"event_replays.csv", index=False)
    grid_scores(events).to_csv(args.output/"configuration_grid.csv", index=False)
    control.to_csv(args.output/"controlled_aging.csv", index=False)
    plots(control, offline, events, args.output)
    report(events, offline, cap, args.output)


if __name__ == "__main__":
    main()
