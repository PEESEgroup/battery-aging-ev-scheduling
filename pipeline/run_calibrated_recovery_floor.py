"""Test the 0.2 kW boundary and a calibration-only recovery capacity floor."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_fine_grid_fallback_diagnostics as diagnostic
import run_service_fallback_pilot as fallback
from run_full_service_recovery_pilot import CALIBRATION, N, TEST, reference_case
from run_minimal_flex_pilot import PilotForecast, STRESSES, stressed
from run_section_iv_replay import ROOT, plt

REQUESTS = np.arange(21) / 10
RESERVE_KW = 0.25
ESTIMATES = ("existing_floor", "calibrated_floor")


def evaluate(estimate, floor, forecast):
    rows = []
    with (patch.object(diagnostic, "REQUESTS", REQUESTS),
          patch.object(fallback, "POLICIES", ("visible_slack_rule",)),
          patch.object(fallback, "RECOVERY_FLOOR", floor / N)):
        for split, cases in (("calibration", CALIBRATION), ("test", TEST)):
            for index, (demand, cap) in enumerate(cases, 1):
                for group in STRESSES:
                    events, _ = diagnostic.evaluate(split, index, demand, cap, group, forecast)
                    rows.extend(dict(**row, estimate=estimate, recovery_floor_kw=floor)
                                for row in events)
                print(f"{estimate} {split} {index}", flush=True)
    return pd.DataFrame(rows)


def verify(events):
    first = events.pivot(index=["event_id", "estimate", "request_kw"], columns="pressure",
                         values="dispatch_17_kw")
    np.testing.assert_allclose(first["00"], first["01"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(first["10"], first["11"], atol=1e-9, rtol=0)
    assert not events.duplicated(["event_id","estimate","request_kw","pressure"]).any()
    assert not events.solver_errors.astype(bool).any()
    assert (events.delivered == (events.full_service & events.tracking & events.aging_pass)).all()


def summarize(events, fine, calibrated, output):
    metrics = ["full_service", "tracking", "aging_pass", "delivered"]
    test = events[events.split == "test"]
    counts = test.groupby(["estimate","pressure","request_kw"])[metrics].sum().reset_index()
    counts.to_csv(output/"test_delivery_counts.csv", index=False)
    validation = events[events.split == "calibration"].copy()
    validation["ev_type"] = validation.pressure.str[0]
    pooled = validation.groupby(["estimate","ev_type","request_kw"]).delivered.sum()
    selected = pooled[pooled == 2*len(CALIBRATION)].reset_index()
    selected = selected.groupby(["estimate","ev_type"]).request_kw.max()
    frozen = []
    for (estimate, ev_type), q in selected.items():
        for group in (ev_type+"0", ev_type+"1"):
            row = counts[(counts.estimate == estimate) & (counts.pressure == group)
                         & (counts.request_kw == q)].iloc[0]
            frozen.append(dict(estimate=estimate, pressure=group, request_kw=q,
                               service_cases=row.full_service, delivered_cases=row.delivered))
    frozen = pd.DataFrame(frozen)
    frozen.to_csv(output/"frozen_commitments.csv", index=False)

    keys = ["event_id", "pressure", "request_kw"]
    paired = test.pivot(index=keys, columns="estimate", values=metrics)
    paired.columns = [f"{metric}_{estimate}" for metric, estimate in paired.columns]
    paired["service_regression"] = paired.full_service_existing_floor & ~paired.full_service_calibrated_floor
    paired["delivery_gain"] = ~paired.delivered_existing_floor & paired.delivered_calibrated_floor
    paired["delivery_loss"] = paired.delivered_existing_floor & ~paired.delivered_calibrated_floor
    paired.to_csv(output/"paired_test_outcomes.csv")

    lines = [
        "# Recovery capacity calibration and fine commitments", "",
        f"The existing floor is {fallback.RECOVERY_FLOOR*N:g} kW. The calibrated floor is {calibrated:g} kW.",
        f"The calibrated floor is the minimum stressed recovery capacity in the four calibration events minus a {RESERVE_KW:g} kW reserve.",
        "Only the floor changes. Both methods use the same online forecasts, health budget, observable states, and service fallback.",
        "At 17:00 the calibrated floor is shared by capped and uncapped cases. It never uses realized test capacity.",
        "The reserve is a design choice. This floor is an empirical estimate and is not a guaranteed lower bound.",
        "Commitments pool capacity treatments at calibration, conditional only on the observable parking duration.",
        "", "## Existing-rule joint-stress check", "",
        "| Request kW | Test full service / 4 | Test delivery / 4 |", "|---|---:|---:|",
    ]
    for row in fine.itertuples():
        lines.append(f"| {row.request_kw:g} | {row.full_service}/4 | {row.delivered}/4 |")
    lines += ["", "## Frozen test outcomes", "",
              "| Estimate | Pressure | Request kW | Full service / 4 | Delivery / 4 |",
              "|---|---|---:|---:|---:|"]
    for row in frozen.itertuples():
        lines.append(f"| {row.estimate} | {row.pressure} | {row.request_kw:g} | "
                     f"{row.service_cases}/4 | {row.delivered_cases}/4 |")
    lines += ["", f"Across {len(paired)} paired test requests, service regressions: {paired.service_regression.sum()}, "
              f"new deliveries: {paired.delivery_gain.sum()}, lost deliveries: {paired.delivery_loss.sum()}.",
              "The comparison uses previously inspected synthetic test events. The frozen calculation does not read their capacities.",
              "No LLM was called."]
    (output/"conclusions.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for estimate in ESTIMATES:
        part = counts[(counts.pressure == "11") & (counts.estimate == estimate)]
        for ax, metric, title in zip(axes, ("full_service", "delivered"),
                                     ("Full departure service", "Commitment with service and aging")):
            ax.plot(part.request_kw, part[metric], "o-", markersize=3, label=estimate)
            ax.set(xlabel="Request (kW)", ylabel="Joint-stress events out of 4",
                   yticks=range(5), title=title)
            ax.legend(fontsize=8)
    fig.savefig(output/"recovery_estimate_comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"calibrated_recovery_floor_results")
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    _, grid, _, _ = reference_case(CALIBRATION[0][0], "00", CALIBRATION[0][1])
    donors = np.array([stressed(grid, CALIBRATION[0][1]), grid,
                       stressed(grid, CALIBRATION[2][1]), grid])
    donors[:, :, 3] /= N
    forecast = PilotForecast(donors)
    original = evaluate(ESTIMATES[0], fallback.RECOVERY_FLOOR*N, forecast)
    subset = original[(original.split == "test") & (original.pressure == "11")
                      & (original.request_kw <= 0.5)]
    fine = subset.groupby("request_kw", as_index=False)[["full_service","delivered"]].sum()
    subset.to_csv(output/"original_rule_joint_0_to_0p5.csv", index=False)
    print(fine.to_string(index=False), flush=True)
    calibrated = min(cap for _, cap in CALIBRATION) - RESERVE_KW
    updated = evaluate(ESTIMATES[1], calibrated, forecast)
    events = pd.concat([original, updated], ignore_index=True)
    assert len(events) == 1344
    verify(events)
    events.to_csv(output/"event_diagnostics.csv", index=False)
    summarize(events, fine, calibrated, output)


if __name__ == "__main__":
    main()
