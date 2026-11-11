"""Explain fine-grid delivery failures without modifying the fallback policy."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_service_fallback_pilot as fallback
from nonanticipative_scheduler import AGING, Fleet
from run_full_service_recovery_pilot import CALIBRATION, N, TEST, reference_case, retry_linprog
from run_minimal_flex_pilot import PilotForecast, START, STRESSES, stressed
from run_section_iv_replay import ROOT, plt

REQUESTS = np.arange(9) / 2
EPS = 1e-6


def diagnosis(row):
    q = row["request_kw"]
    if row["delivered"]:
        return "delivered"
    if q > row["physical_bound_kw"] + EPS:
        return "physical_capacity"
    if q > row["health_bound_kw"] + EPS:
        return "offline_aging_budget"
    if not row["tracking"]:
        if q > row["initial_forecast_bound_kw"] + EPS:
            return "rolling_forecast"
        if "17:service_guard" in row["fallback_modes"]:
            return "rolling_service_guard"
        return "rolling_dispatch"
    if not row["full_service"]:
        return "rolling_recovery_service"
    return "rolling_aging_after_dispatch"


def evaluate(split, index, demand, cap, group, forecast):
    fleet, grid, baseline, hours = reference_case(demand, group, cap)
    ref, losses, _, _ = baseline
    unit = Fleet(fleet.arrival[:1], fleet.departure[:1], fleet.demand[:1],
                 fleet.power[:1], fleet.capacity[:1], fleet.soc[:1])
    grid[:, 3] /= N
    departure = START + hours
    reference = ref[:, START].sum()
    factor = AGING.k_power / AGING.p_ref_kw**2 * (1 + 1/(hours-1))
    aging_bound = N*np.sqrt(fallback.ALLOWANCE/factor)
    real_cap = grid[START+1:departure, 3].sum()*N
    physical_bound = min(reference, real_cap - (hours-1)*reference)
    minimum, _ = forecast.capacities(START, grid[:START+1], 0.0)
    forecast_bound = min(reference, minimum[1:hours].sum()*N - (hours-1)*reference)
    guard_bound = min(reference, np.minimum(minimum[1:hours], fallback.RECOVERY_FLOOR).sum()*N
                      - (hours-1)*reference)

    events = []
    traces = []
    for policy in fallback.POLICIES:
        for q in REQUESTS:
            actions = {}

            def trace_update(model, executed, sohloss, hour, action):
                actions[hour] = float(action.sum()*N)
                fallback_update(model, executed, sohloss, hour, action)

            def trace_solve(model, hour, caps, score, executed, sohloss, **kwargs):
                info = {}
                result = fallback_solve(model, hour, caps, score, executed, sohloss,
                                        diagnostics=info, **kwargs)
                traces.append(dict(**keys, hour=hour, attempt=len(traces),
                                   tracking_constraint=kwargs.get("window") is not None,
                                   incremental_budget=kwargs.get("budget") is not None,
                                   fixed_service=kwargs.get("target") is not None,
                                   projected_recovery_capacity_kwh=caps[1:departure-hour].sum()*N,
                                   solver_status=info["status"],
                                   proposed_power_kw=float(result[0][:,hour].sum()*N)
                                   if result else np.nan))
                return result

            keys = dict(split=split, event_id=f"{split}_{index:02d}", pressure=group,
                        policy=policy, request_kw=q)
            with (patch.object(fallback, "update", trace_update),
                  patch.object(fallback, "solve_plan", trace_solve),
                  patch("nonanticipative_scheduler.linprog", retry_linprog)):
                result = fallback.replay(unit, grid, ref[:1], losses[:1], forecast, q/N, policy)
            row = dict(
                **keys, **result, reference_kw=reference, departure_hour=departure,
                recovery_capacity_kwh=real_cap, physical_bound_kw=physical_bound,
                aging_only_bound_kw=aging_bound, health_bound_kw=min(physical_bound, aging_bound),
                initial_forecast_bound_kw=forecast_bound, service_guard_bound_kw=guard_bound,
                offline_min_extra_soh=factor*(q/N)**2 if q <= physical_bound+EPS else np.nan,
                remaining_after_17_kwh=N*demand-actions[START],
                actual_recovery_margin_after_17_kwh=real_cap-(N*demand-actions[START]),
                **{f"dispatch_{hour}_kw": actions.get(hour, np.nan) for hour in range(17, 21)},
            )
            row["cause"] = diagnosis(row)
            events.append(row)
    return events, traces


fallback_update = fallback.update
fallback_solve = fallback.solve_plan


def summarize(events, output):
    test = events[events.split == "test"]
    counts = test.groupby(["policy", "pressure", "request_kw"])[
        ["full_service", "tracking", "aging_pass", "delivered"]
    ].sum().reset_index()
    counts.to_csv(output/"test_delivery_counts.csv", index=False)
    joint = test[test.pressure == "11"]
    causes = joint.groupby(["policy", "request_kw", "cause"]).size().unstack(fill_value=0)
    causes.to_csv(output/"joint_failure_counts.csv")
    calibration = events[events.split == "calibration"].copy()
    calibration["ev_type"] = calibration.pressure.str[0]
    pooled = calibration.groupby(["policy","ev_type","request_kw"]).delivered.sum()
    frozen = pooled[pooled == 2*len(CALIBRATION)].reset_index()
    frozen = frozen.groupby(["policy","ev_type"]).request_kw.max()
    frozen_rows = []
    for (policy, ev_type), q in frozen.items():
        for group in (ev_type+"0", ev_type+"1"):
            row = counts[(counts.policy == policy) & (counts.pressure == group)
                         & (counts.request_kw == q)].iloc[0]
            frozen_rows.append(dict(policy=policy, pressure=group, request_kw=q,
                                    service_cases=row.full_service, delivered_cases=row.delivered))
    pd.DataFrame(frozen_rows).to_csv(output/"frozen_commitments.csv", index=False)

    rule = joint[joint.policy == "visible_slack_rule"]
    lines = [
        "# Fine-grid service fallback diagnosis", "",
        "Four calibration and four separate synthetic test events. Four paired conditions and two unchanged policies.",
        "Requests run from 0 through 4 kW at 0.5 kW spacing. All no-activation references deliver full service.",
        "Failure attribution first tests physical feasibility and minimum possible aging using realized capacity offline.",
        "When both pass, attribution uses the executed trajectory and the initial forecast and guard bounds.",
        "Actual capacity after 17:00 is used for retrospective diagnosis only.",
        "", "## Joint-stress test outcomes", "",
        "| Request kW | Fixed service | Fixed delivery | Rule service | Rule delivery |",
        "|---|---:|---:|---:|---:|",
    ]
    for q in REQUESTS:
        vals = []
        for policy in fallback.POLICIES:
            part = joint[(joint.policy == policy) & (joint.request_kw == q)]
            vals += [part.full_service.sum(), part.delivered.sum()]
        lines.append(f"| {q:g} | {vals[0]}/4 | {vals[1]}/4 | {vals[2]}/4 | {vals[3]}/4 |")
    lines += ["", "## Visible-rule failures at 2 kW", "",
              "| Event | Physical kW | Forecast kW | Guard kW | Actual kW | Attribution |",
              "|---|---:|---:|---:|---:|---|"]
    for row in rule[rule.request_kw == 2].itertuples():
        lines.append(f"| {row.event_id} | {row.physical_bound_kw:.2f} | {row.initial_forecast_bound_kw:.2f} | "
                     f"{row.service_guard_bound_kw:.2f} | {row.actual_reduction_kw:.2f} | {row.cause} |")
    positive = rule[rule.request_kw > 0].groupby("request_kw").delivered.sum()
    stable = positive[positive == len(TEST)]
    cause_counts = rule[~rule.delivered].cause.value_counts()
    lines += ["", f"Joint test rule failures by cause: {cause_counts.to_dict()}.",
              f"Minimum offline aging at 2 kW is {rule[rule.request_kw == 2].offline_min_extra_soh.max():.3g} SOH per EV "
              f"against the {fallback.ALLOWANCE:g} SOH budget.",
              f"The tightest 17:00 service-guard bound is {rule.service_guard_bound_kw.min():.2f} kW. "
              "The requested grid contains no positive value at or below that bound.",
              f"Positive rule requests delivered in every joint test event: "
              f"{', '.join(f'{q:g} kW' for q in stable.index) or 'none on the tested grid'}.",
              "No LLM was called. These constructed-event results are descriptive."]
    (output/"conclusions.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for policy in fallback.POLICIES:
        sub = counts[(counts.pressure == "11") & (counts.policy == policy)]
        axes[0].plot(sub.request_kw, sub.full_service, "o-", label=policy)
        axes[1].plot(sub.request_kw, sub.delivered, "o-", label=policy)
    for ax, title in zip(axes, ("Full departure service", "Commitment, service and aging")):
        ax.set(xlabel="Request (kW)", ylabel="Joint-stress events out of 4",
               xticks=REQUESTS, yticks=range(5), title=title)
        ax.legend(fontsize=8)
    fig.savefig(output/"fine_grid_joint_delivery.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"fine_grid_fallback_diagnostics_results")
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    _, grid, _, _ = reference_case(CALIBRATION[0][0], "00", CALIBRATION[0][1])
    donors = np.array([stressed(grid, CALIBRATION[0][1]), grid,
                       stressed(grid, CALIBRATION[2][1]), grid])
    donors[:, :, 3] /= N
    forecast = PilotForecast(donors)
    rows, traces = [], []
    for split, cases in (("calibration", CALIBRATION), ("test", TEST)):
        for index, (demand, cap) in enumerate(cases, 1):
            for group in STRESSES:
                events, attempts = evaluate(split, index, demand, cap, group, forecast)
                rows.extend(events)
                traces.extend(attempts)
            print(f"Finished {split} {index}", flush=True)
    events = pd.DataFrame(rows)
    assert len(events) == 576
    first = events.pivot(index=["event_id","policy","request_kw"], columns="pressure",
                         values="dispatch_17_kw")
    np.testing.assert_allclose(first["00"], first["01"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(first["10"], first["11"], atol=1e-9, rtol=0)
    events.to_csv(output/"event_diagnostics.csv", index=False)
    pd.DataFrame(traces).to_csv(output/"rolling_solver_trace.csv", index=False)
    summarize(events, output)


if __name__ == "__main__":
    main()
