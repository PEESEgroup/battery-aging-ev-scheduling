"""Controlled full-service extension of the one-hour EV flexibility pilot."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from nonanticipative_scheduler import AGING, Fleet, replay_aging, solve_plan, update
from run_minimal_flex_pilot import (
    BUDGETS, CONFIGS, RECOVERY, START, STRESSES, PilotForecast, grid_scores, replay, stressed,
)
from run_section_iv_replay import ROOT, plt

N = 8
STEPS = 21
REQUESTS = tuple(range(0, 43, 2))
# Each case shares demand and capacity between its four paired treatments.
CALIBRATION = ((13.6, 39.0), (13.9, 40.0), (14.2, 40.5), (14.4, 39.5))
TEST = ((13.7, 38.8), (14.0, 39.8), (14.3, 41.0), (14.4, 40.2))


def reference_case(demand, pressure, recovery_cap):
    hours = 3 if pressure[0] == "1" else 4
    stop = START + hours
    fleet = Fleet(
        np.full(N, START), np.full(N, stop), np.full(N, demand),
        np.full(N, 7.0), np.full(N, 60.0), np.full(N, 0.30),
    )
    grid = np.zeros((STEPS, 4))
    grid[:, 3] = N * 7.0
    if pressure[1] == "1":
        grid = stressed(grid, recovery_cap)
    ref = np.zeros((N, STEPS))
    ref[:, START:stop] = demand / hours
    assert np.allclose(ref.sum(axis=1), fleet.demand)
    assert np.all(ref.sum(axis=0) <= grid[:, 3] + 1e-9)
    assert np.all(ref <= fleet.power[:, None])
    assert np.all(fleet.soc + fleet.demand / fleet.capacity < AGING.soc_threshold)
    prefix = np.zeros_like(ref)
    losses = np.zeros(N)
    history = []
    for hour in range(STEPS):
        history.append(losses.copy())
        update(fleet, prefix, losses, hour, ref[:, hour])
    np.testing.assert_allclose(losses, replay_aging(fleet, ref), atol=1e-12)
    # No-activation plan is the flat, feasible dispatch announced at 17:00.
    return fleet, grid, (ref, losses, [ref] * STEPS, history), hours


def retry_linprog(*args, **kwargs):
    result = linprog(*args, **kwargs)
    if result.status == 4:
        result = linprog(*args, **{**kwargs, "method": "highs-ds",
                                 "options": {"presolve": False}})
    if result.status == 4:
        result = linprog(*args, **{**kwargs, "method": "highs-ipm"})
    return result


def evaluate(job):
    split, case, demand, recovery_cap, pressure, forecast = job
    fleet, grid, baseline, hours = reference_case(demand, pressure, recovery_cap)
    ref, losses, _, history = baseline
    fixed = ref.sum(axis=0)[START:START+1]
    cap = grid[RECOVERY, 3].min()
    physical = min(fixed[0], (hours-1) * (cap-fixed[0]))
    factor = AGING.k_power / AGING.p_ref_kw**2 * (1 + 1/(hours-1))
    context = dict(split=split, scenario_id=case, pressure=pressure, demand_per_ev_kwh=demand,
                   recovery_cap_kw=cap, departure_hour=START+hours,
                   reference_kw=fixed[0], reference_unmet_kwh=0.0)
    # The EVs have identical windows and states. Averaging their schedules
    # preserves feasibility under the convex aging limit and linear grid cap.
    single = Fleet(fleet.arrival[:1], fleet.departure[:1], fleet.demand[:1],
                   fleet.power[:1], fleet.capacity[:1], fleet.soc[:1])
    single_grid = grid.copy()
    single_grid[:, 3] /= N
    single_ref = ref[:1]
    single_base = (single_ref, losses[:1], [single_ref] * STEPS,
                   [row[:1] for row in history])
    fixed_single = fixed / N
    offline = []
    for label, allowance in BUDGETS:
        analytic = physical if allowance is None else min(physical, N*np.sqrt(allowance/factor))
        diagnostics = {}
        solution = solve_plan(
            single, START, single_grid[START:, 3], np.zeros(STEPS-START),
            np.zeros_like(single_ref), history[START][:1], target=single.demand,
            budget=None if allowance is None else losses[:1] + allowance,
            window=(START, fixed_single, -1, None), offline=True,
            diagnostics=diagnostics, cut_limit=100, aging_tol=1e-10,
        )
        if solution is None or abs(N*solution[1]-analytic) > 0.06:
            raise RuntimeError((context, label, analytic, solution, diagnostics))
        offline.append(dict(**context, budget=label, analytic_kw=analytic,
                            offline_kw=N*solution[1], solver_status=diagnostics["status"]))
    rows = []
    settings = [(label, allowance, 0., 0., "factorial") for label, allowance in BUDGETS]
    settings += [("positive", 1e-5, quantile, margin, "grid")
                 for quantile, margin in CONFIGS if quantile or margin]
    for label, allowance, quantile, margin, study in settings:
        for q in REQUESTS:
            try:
                with patch("nonanticipative_scheduler.linprog", retry_linprog):
                    result = replay(single, single_grid, forecast, single_base, fixed_single, q/N,
                                    allowance, quantile, margin)
            except RuntimeError as error:
                if "HiGHS" not in str(error):
                    raise
                result = dict(outcome="solver_error", solver_status=str(error),
                              delivered=False, complete_delivery=False)
            for key in ("unmet_kwh", "power_gap_kw", "recovery_peak_kw", "power_17_kw"):
                if key in result:
                    result[key] *= N
            if result["delivered"] and not result["complete_delivery"]:
                raise AssertionError("Matched service must imply full service in this experiment")
            rows.append(dict(**context, study=study, budget=label, budget_soh=allowance,
                             quantile=quantile, margin_soh=margin, commitment_kw=q, **result))
    print(f"{split} {case} {pressure}: physical {physical:.2f} kW", flush=True)
    return rows, offline


def summarize(events, offline, output):
    factor = events[events.study == "factorial"]
    feasible = factor.groupby(["split", "pressure", "budget", "commitment_kw"]).complete_delivery.sum()
    selected = feasible.loc["validation"].reset_index()
    selected = selected[selected.complete_delivery == len(CALIBRATION)]
    selected = selected.groupby(["pressure", "budget"]).commitment_kw.max().rename("selected_kw").reset_index()
    validation = feasible.loc["regression"].rename("test_deliveries").reset_index()
    summary = selected.merge(validation, left_on=["pressure", "budget", "selected_kw"],
                             right_on=["pressure", "budget", "commitment_kw"])
    summary.drop(columns="commitment_kw").to_csv(output/"commitment_summary.csv", index=False)
    effects = offline.pivot(index=["split", "scenario_id", "pressure"], columns="budget",
                            values="analytic_kw").reset_index()
    effects["aging_loss_kw"] = effects.unlimited - effects.positive
    effect_rows = []
    for (split, case), part in effects.groupby(["split", "scenario_id"]):
        by = part.set_index("pressure")
        for budget in ("positive", "unlimited"):
            q = by[budget]
            effect_rows.append(dict(split=split, scenario_id=case, budget=budget,
                                    ev_loss_kw=q["00"]-q["10"],
                                    grid_loss_kw=q["00"]-q["01"],
                                    joint_loss_kw=q["00"]-q["11"],
                                    interaction_kw=q["10"]+q["01"]-q["11"]-q["00"],
                                    grid_loss_under_ev_stress_kw=q["10"]-q["11"]))
    impacts = pd.DataFrame(effect_rows)
    impacts.to_csv(output/"paired_pressure_effects.csv", index=False)

    grid = grid_scores(events)
    grid.to_csv(output/"configuration_grid.csv", index=False)
    best = grid.groupby(["split", "scenario_id", "pressure"], as_index=False).agg(
        best=("max_delivered_kw", "max"), default=("default_kw", "first"))
    best["gain"] = best.best - best.default
    # Choose a single grid point on calibration only, maximizing the sum of four
    # pressure-specific commitments that deliver in all calibration cases.
    config = events[events.budget == "positive"]
    counts = config.groupby(["split","pressure","quantile","margin_soh","commitment_kw"]).complete_delivery.sum()
    choices = counts.loc["validation"].reset_index()
    choices = choices[choices.complete_delivery == len(CALIBRATION)]
    chosen = choices.groupby(["quantile","margin_soh","pressure"]).commitment_kw.max()
    scores = chosen.unstack("pressure").reindex(columns=STRESSES).fillna(0).sum(axis=1)
    winner = scores.idxmax()
    fixed_rows = []
    for quantile, margin in dict.fromkeys(((0.0,0.0), winner)):
        for group in STRESSES:
            q = chosen.get((quantile, margin, group))
            delivered = counts.get(("regression",group,quantile,margin,q)) if q is not None else None
            fixed_rows.append(dict(quantile=quantile, margin_soh=margin, pressure=group,
                                   selected_kw=q, test_deliveries=delivered))
    pd.DataFrame(fixed_rows).to_csv(output/"frozen_configuration_comparison.csv", index=False)
    test_best = best[best.split == "regression"]
    hits = test_best[test_best.gain >= 2]
    test_effects = effects[effects.split == "regression"].groupby("pressure").mean(numeric_only=True)
    test_impacts = impacts[impacts.split == "regression"].groupby("budget").mean(numeric_only=True)
    lines = [
        "# Full-service one-hour flexibility pilot", "",
        "Controlled synthetic matched fleets, eight EVs per event, 17:00–18:00 downward commitment.",
        "EV pressure shortens available parking from four to three hours. Grid pressure caps 18:00–21:00 capacity.",
        "Every no-response reference fully charges every EV and meets the same realized capacity path.",
        "The reference is announced flat charging. No actual future capacity enters the current action.",
        "Identical EVs are solved as a symmetric representative schedule, then replicated eight times.",
        "Four cases set commitments and choose the configuration. Four distinct constructed cases test the frozen choices.",
        "Pressure-specific commitments are conditional diagnostics. Capped and uncapped cases with the same EV roster are indistinguishable at 17:00.",
        "", "## Mean test-case offline bounds (kW)", "",
        "| Pressure | Unlimited budget | Positive budget | Aging loss |",
        "|---|---:|---:|---:|",
    ]
    for group in STRESSES:
        row = test_effects.loc[group]
        lines.append(f"| {group} | {row.unlimited:.2f} | {row.positive:.2f} | {row.aging_loss_kw:.2f} |")
    lines += ["", "## Paired mean test-case impacts (kW)", "",
              "| Budget | EV-only loss | Grid-only loss | Joint loss | Joint interaction | Grid loss under EV stress |",
              "|---|---:|---:|---:|---:|---:|"]
    for budget, row in test_impacts.iterrows():
        lines.append(f"| {budget} | {row.ev_loss_kw:.2f} | {row.grid_loss_kw:.2f} | "
                     f"{row.joint_loss_kw:.2f} | {row.interaction_kw:.2f} | {row.grid_loss_under_ev_stress_kw:.2f} |")
    lines += ["", "## Frozen commitments and full-service test deliveries", "",
              "| Pressure | Budget | Selected kW | Deliveries / 4 |",
              "|---|---|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.pressure} | {row.budget} | {row.selected_kw:g} | {row.test_deliveries}/4 |")
    lines += ["", "## Nine-point screen", "",
              f"- Best calibration configuration: capacity quantile {winner[0]:g}, SOH margin {winner[1]:g}. "
              f"Sum of four calibrated commitments: {scores.loc[winner]:g} kW versus {scores.loc[(0.,0.)]:g} kW for default.",
              f"- Event-wise gain of at least one 2 kW step in {len(hits)}/{len(test_best)} test blocks. "
              f"Maximum gain: {test_best.gain.max():g} kW.",
              f"- {int((test_best.default.isna() & test_best.best.notna()).sum())} test blocks recover from no successful default commitment.",
              f"- Pressure groups with no common successful calibration request under the best fixed configuration: "
              f"{', '.join(group for group in STRESSES if (*winner, group) not in chosen.index) or 'none'}.",
              "- The summed calibration score ranks configurations for screening only. It treats an unavailable group as zero and is not a fleet commitment.",
              "- Event-wise grid maxima are screening bounds, not performance of a causal configuration selector.",
              "- No LLM called. The small constructed set is a mechanism check, not a reliability estimate."]
    (output/"conclusions.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    test_effects.loc[list(STRESSES), ["unlimited","positive"]].plot.bar(ax=axes[0], rot=0)
    axes[0].set(xlabel="EV / grid pressure", ylabel="Mean offline bound (kW)",
                title="Aging budget and recovery capacity")
    positive = factor[(factor.split == "regression") & (factor.budget == "positive")]
    for group, part in positive.groupby("pressure"):
        series = part.groupby("commitment_kw").complete_delivery.sum()
        axes[1].plot(series.index, series, "o-", label=group)
    axes[1].set(xlabel="Commitment (kW)", ylabel="Full-service deliveries out of 4",
                yticks=range(5), title="Rolling replay under the default configuration")
    axes[1].legend(title="EV / grid")
    fig.savefig(output/"full_service_flexibility.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"full_service_recovery_pilot_results")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Two capped and two uncapped paths, using only calibration capacities.
    _, uncapped, _, _ = reference_case(CALIBRATION[0][0], "00", CALIBRATION[0][1])
    donors = np.array([stressed(uncapped, CALIBRATION[0][1]), uncapped,
                       stressed(uncapped, CALIBRATION[2][1]), uncapped])
    donors[:, :, 3] /= N
    forecast = PilotForecast(donors)
    jobs = [(split, f"{split}_{i+1:02d}", demand, cap, pressure, forecast)
            for split, cases in (("validation", CALIBRATION), ("regression", TEST))
            for i, (demand, cap) in enumerate(cases) for pressure in STRESSES]
    if args.workers == 1:
        results = list(map(evaluate, jobs))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(evaluate, jobs))
    events = pd.DataFrame([row for rows, _ in results for row in rows])
    offline = pd.DataFrame([row for _, rows in results for row in rows])
    assert len(events) == len(jobs) * (len(BUDGETS) + len(CONFIGS)-1) * len(REQUESTS)
    events.to_csv(args.output/"event_replays.csv", index=False)
    offline.to_csv(args.output/"offline_bounds.csv", index=False)
    summarize(events, offline, args.output)


if __name__ == "__main__":
    main()
