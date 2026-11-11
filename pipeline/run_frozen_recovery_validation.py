"""Validate frozen recovery estimates on previously unused synthetic events."""
import argparse
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_fine_grid_fallback_diagnostics as diagnostic
import run_service_fallback_pilot as fallback
from run_calibrated_recovery_floor import verify
from run_full_service_recovery_pilot import CALIBRATION, N, TEST, reference_case
from run_minimal_flex_pilot import PilotForecast, STRESSES, stressed
from run_section_iv_replay import ROOT, plt

REQUESTS = (0.2, 0.7)
ESTIMATES = (("existing", 38.50), ("improved", 38.75))
RESERVE_KW = 0.25
DEMANDS = (14.15, 14.28, 14.37, 14.40)
CAPS = (38.45, 38.55, 38.65, 38.74, 38.75, 38.76, 38.85, 39.05)
STRATA = ("below", "near", "above")


def stratum(cap):
    if cap < 38.70:
        return "below"
    if cap <= 38.80:
        return "near"
    return "above"


def service_cause(row):
    if row["full_service"]:
        return "none"
    if row["actual_recovery_margin_after_17_kwh"] < -1e-6:
        return "recovery_energy_insufficient_after_17_dispatch"
    spare_last = row["actual_recovery_margin_after_17_kwh"]
    # Any unfilled charger capacity in an earlier recovery hour is not
    # available after that hour has passed.
    for hour in range(18, row["departure_hour"] - 1):
        spare_last -= row["actual_slot_cap_kw"] - row[f"dispatch_{hour}_kw"]
    if spare_last < -1e-6:
        return "insufficient_energy_after_later_dispatch"
    return "remaining_dispatch_or_health_limit"


def run(events, forecast):
    rows, traces = [], []
    with (patch.object(diagnostic, "REQUESTS", REQUESTS),
          patch.object(fallback, "POLICIES", ("visible_slack_rule",))):
        for estimate, floor in ESTIMATES:
            with patch.object(fallback, "RECOVERY_FLOOR", floor/N):
                for index, (demand, cap) in enumerate(events, 1):
                    for pressure in STRESSES:
                        part, attempts = diagnostic.evaluate(
                            "new_test", index, demand, cap, pressure, forecast)
                        actual_cap = cap if pressure[1] == "1" else 7*N
                        for row in part:
                            row.update(estimate=estimate, recovery_floor_kw=floor,
                                       demand_per_ev_kwh=demand, paired_cap_kw=cap,
                                       actual_slot_cap_kw=actual_cap, cap_stratum=stratum(cap))
                            row["service_failure_cause"] = service_cause(row)
                        rows.extend(part)
                        traces.extend(dict(**attempt, estimate=estimate) for attempt in attempts)
            print(f"Completed {estimate}: {len(events)*len(STRESSES)*len(REQUESTS)} replays", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(traces)


def summarize(events, output):
    group_keys = ["estimate", "pressure", "request_kw"]
    metrics = ["full_service", "tracking", "aging_pass", "delivered"]
    totals = events.groupby(group_keys).agg(cases=("event_id", "size"), **{
        key: (key, "sum") for key in metrics}).reset_index()
    for key in metrics:
        totals[key+"_rate"] = totals[key]/totals.cases
    totals.to_csv(output/"group_outcomes.csv", index=False)
    joint = events[events.pressure == "11"]
    strata = joint.groupby(group_keys+["cap_stratum"]).agg(
        cases=("event_id","size"), full_service=("full_service","sum"),
        delivered=("delivered","sum")).reset_index()
    strata.to_csv(output/"joint_capacity_strata.csv", index=False)
    joint.groupby(["estimate","request_kw","cause","service_failure_cause"]).size().rename(
        "count").reset_index().to_csv(output/"joint_failure_causes.csv", index=False)
    pairs = events.pivot(index=["event_id","pressure","request_kw"], columns="estimate",
                         values=["full_service","delivered"])
    pairs.columns = [f"{metric}_{estimate}" for metric, estimate in pairs.columns]
    pairs["service_regression"] = pairs.full_service_existing & ~pairs.full_service_improved
    pairs["delivery_gain"] = ~pairs.delivered_existing & pairs.delivered_improved
    pairs["delivery_loss"] = pairs.delivered_existing & ~pairs.delivered_improved
    pairs.to_csv(output/"paired_outcomes.csv")

    lines = [
        "# Frozen recovery floor validation", "",
        f"{len(DEMANDS)*len(CAPS)} new synthetic paired events with eight identical EVs each.",
        "The fixed requests are 0.20 and 0.70 kW during 17:00–18:00.",
        "Existing floor 38.50 kW, improved floor 38.75 kW. The original calibration minimum remains 39.00 kW, with a frozen 0.25 kW reserve.",
        "New recovery caps cross 38.75 kW. All no-response reference plans meet each departure requirement and realized capacity.",
        "Recovery cap labels and realized future capacities are absent from the 17:00 action.",
        "The grid is a controlled stress design, not a sample for estimating population reliability.",
        "", "## Four paired pressure groups", "",
        "| Estimate | Pressure | Request (kW) | Full service | Delivered |",
        "|---|---|---:|---:|---:|",
    ]
    for row in totals.itertuples():
        lines.append(f"| {row.estimate} | {row.pressure} | {row.request_kw:.2f} | "
                     f"{row.full_service}/{row.cases} | {row.delivered}/{row.cases} |")
    lines += ["", "## Joint pressure by recovery cap", "",
              "Below is under 38.70 kW. Near is 38.70 through 38.80 kW. Above is over 38.80 kW.",
              "", "| Estimate | Request (kW) | Stratum | Full service | Delivered |",
              "|---|---:|---|---:|---:|"]
    for row in strata.itertuples():
        lines.append(f"| {row.estimate} | {row.request_kw:.2f} | {row.cap_stratum} | "
                     f"{row.full_service}/{row.cases} | {row.delivered}/{row.cases} |")
    lines += ["", "## Paired changes and failure causes", "",
              f"Across {len(pairs)} paired requests, service regressions: {pairs.service_regression.sum()}, "
              f"delivery gains: {pairs.delivery_gain.sum()}, delivery losses: {pairs.delivery_loss.sum()}.",
              "Delivery requires full service, event tracking and the same per-vehicle aging budget.",
              "The failure-cause table separates offline infeasibility from online forecast, guard and recovery failures.",
              "No LLM was called."]
    (output/"conclusions.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, q in zip(axes, REQUESTS):
        part = strata[strata.request_kw == q]
        for index, (estimate, _) in enumerate(ESTIMATES):
            sub = part[part.estimate == estimate].set_index("cap_stratum").loc[list(STRATA)]
            xpos = np.arange(len(STRATA)) + (index-0.5)*0.3
            ax.bar(xpos, sub.delivered/sub.cases, width=0.28, label=estimate)
            ax.scatter(xpos, sub.full_service/sub.cases, marker="_", s=95, c="black",
                       label="full service" if index == 0 else None, zorder=3)
        ax.set(xticks=range(len(STRATA)), xticklabels=STRATA, ylim=(0, 1.1),
               ylabel="Fraction of cases", xlabel="Recovery capacity stratum",
               title=f"Joint stress at {q:.2f} kW")
        ax.legend(fontsize=8, loc="lower right")
    fig.savefig(output/"frozen_floor_validation.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"frozen_recovery_validation_results")
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    new = tuple(product(DEMANDS, CAPS))
    assert set(new).isdisjoint((*CALIBRATION, *TEST))
    assert ESTIMATES[1][1] == min(cap for _, cap in CALIBRATION)-RESERVE_KW
    _, grid, _, _ = reference_case(CALIBRATION[0][0], "00", CALIBRATION[0][1])
    donors = np.array([stressed(grid, CALIBRATION[0][1]), grid,
                       stressed(grid, CALIBRATION[2][1]), grid])
    donors[:, :, 3] /= N
    forecast = PilotForecast(donors)
    events, traces = run(new, forecast)
    assert len(events) == len(new)*len(STRESSES)*len(REQUESTS)*len(ESTIMATES)
    verify(events)
    events.to_csv(output/"event_diagnostics.csv", index=False)
    traces.to_csv(output/"rolling_solver_trace.csv", index=False)
    summarize(events, output)


if __name__ == "__main__":
    main()
