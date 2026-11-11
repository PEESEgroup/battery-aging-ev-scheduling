"""Compare causal charging rules with service-first recovery after a failed solve."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from nonanticipative_scheduler import Fleet, TOL, solve_plan, update
from run_full_service_recovery_pilot import (
    CALIBRATION, N, REQUESTS, TEST, reference_case, retry_linprog,
)
from run_minimal_flex_pilot import PilotForecast, START, STRESSES, stressed
from run_section_iv_replay import ROOT, plt

ALLOWANCE = 1e-5
# Design-time recovery floor, below all scenario caps and above the largest
# flat reference. It is supplied to the rule, not read from a test trajectory.
RECOVERY_FLOOR = 38.5 / N
POLICIES = ("fixed_min", "visible_slack_rule")


def future_energy(fleet, caps, hour):
    last = int(fleet.departure[0]) - hour
    return np.minimum(caps[1:last], fleet.power[0]).sum()


def replay(fleet, grid, ref, reference_loss, forecast, q, policy):
    executed = np.zeros_like(ref)
    losses = np.zeros_like(reference_loss)
    target = fleet.demand
    budget = reference_loss + ALLOWANCE
    reference = ref.sum(axis=0)[START:START+1]
    window = START, reference, -1, q
    fallback_steps = []
    fallback_modes = []
    median_steps = []
    errors = []
    departure = int(fleet.departure[0])

    def attempt(hour, caps, score, **kwargs):
        try:
            return solve_plan(fleet, hour, caps, score, executed, losses, **kwargs)
        except RuntimeError as error:
            if "HiGHS" not in str(error):
                raise
            errors.append(f"{hour}:{error}")
            return None

    for hour in range(START, departure):
        observed = grid[:hour+1]
        minimum, score = forecast.capacities(hour, observed, 0.0)
        safe = minimum.copy()
        if hour == START:
            # The declared floor applies only to the future recovery hours.
            safe[1:] = np.minimum(safe[1:], RECOVERY_FLOOR)
        elif hour < departure:
            # After recovery starts, use the capacity just observed to
            # update the short-term service fallback projection.
            safe[1:] = min(observed[-1, 3], fleet.power[0])
        caps = minimum
        can_use_median = False
        if policy == "visible_slack_rule":
            remaining = target[0] - executed.sum()
            if hour == START:
                current_if_tracking = max(0.0, reference[0] - q)
                can_use_median = remaining - current_if_tracking <= future_energy(fleet, safe, hour) + TOL
            else:
                can_use_median = observed[-1, 3] >= fleet.power[0] - TOL
            if can_use_median:
                caps, score = forecast.capacities(hour, observed, 0.5)
                median_steps.append(hour)
        planned = attempt(hour, caps, score, target=target, budget=budget, window=window)
        if planned is not None and policy == "visible_slack_rule":
            # A more optimistic forecast may select a lower action than the
            # requested power ceiling. Check the actual candidate action.
            remaining_after = target[0] - executed.sum() - planned[0][0, hour]
            if remaining_after > future_energy(fleet, safe, hour) + TOL:
                planned = None
                fallback_modes.append(f"{hour}:service_guard")
        if planned is None:
            fallback_steps.append(hour)
            service_score = np.zeros_like(safe)
            # Retain tracking and the incremental budget if both fit the
            # observed-state service replan. Drop tracking before health.
            for mode, aging_limit, tracking in (
                ("service_health_tracking", budget, window),
                ("service_health", budget, None),
                ("service_absolute_health", None, None),
            ):
                planned = attempt(hour, safe, service_score, target=target,
                                  budget=aging_limit, window=tracking)
                if planned is not None:
                    fallback_modes.append(f"{hour}:{mode}")
                    break
            if planned is None:
                planned = attempt(hour, safe, service_score)
                fallback_modes.append(f"{hour}:best_effort")
        action = np.zeros(len(fleet.arrival)) if planned is None else planned[0][:, hour]
        if np.any(action < -TOL) or action.sum() > observed[-1, 3] + TOL:
            raise AssertionError("Unsafe charging action")
        update(fleet, executed, losses, hour, action)

    served = executed.sum(axis=1)
    service = bool(np.all(served >= target - 1e-6))
    aging_ok = bool(np.all(losses <= budget + TOL))
    reduction = float((reference[0] - executed[:, START].sum()) * N)
    tracking = reduction >= q*N - 1e-6
    return dict(
        full_service=service, tracking=tracking, aging_pass=aging_ok,
        delivered=service and tracking and aging_ok,
        actual_reduction_kw=reduction,
        unmet_kwh=float(np.maximum(target-served, 0).sum()*N),
        max_extra_soh=float((losses-reference_loss).max()),
        fallback_steps=len(fallback_steps), fallback_hours=";".join(map(str, fallback_steps)),
        fallback_modes=";".join(fallback_modes), median_hours=";".join(map(str, median_steps)),
        solver_errors=";".join(errors),
    )


def evaluate(job):
    split, case, demand, cap, pressure, forecast = job
    fleet, grid, baseline, hours = reference_case(demand, pressure, cap)
    ref, losses, _, _ = baseline
    single = Fleet(fleet.arrival[:1], fleet.departure[:1], fleet.demand[:1],
                   fleet.power[:1], fleet.capacity[:1], fleet.soc[:1])
    grid[:, 3] /= N
    rows = []
    with patch("nonanticipative_scheduler.linprog", retry_linprog):
        for policy in POLICIES:
            for q in REQUESTS:
                result = replay(single, grid, ref[:1], losses[:1], forecast, q/N, policy)
                rows.append(dict(split=split, event_id=case, pressure=pressure, policy=policy,
                                 departure_hour=START+hours, reference_kw=ref[:, START].sum(),
                                 recovery_cap_kw=grid[18, 3]*N, request_kw=q, **result))
    print(f"{split} {case} {pressure}", flush=True)
    return rows


def summarize(events, output):
    keys = ["policy", "pressure", "request_kw"]
    counts = events.groupby(["split"]+keys)[
        ["full_service", "tracking", "aging_pass", "delivered", "fallback_steps"]
    ].sum()
    counts.to_csv(output/"delivery_by_request.csv")
    calibration = counts.loc["calibration"].reset_index()
    # Capacity-stress labels are unavailable at 17:00. Pool both grid
    # treatments before choosing the single request for each visible EV type.
    calibration["ev_type"] = calibration.pressure.str[0]
    pooled = calibration.groupby(["policy", "ev_type", "request_kw"]).delivered.sum()
    selected = pooled[pooled == 2*len(CALIBRATION)].reset_index()
    selected = selected.groupby(["policy", "ev_type"]).request_kw.max()
    rows = []
    for policy in POLICIES:
        for group in STRESSES:
            q = selected.get((policy, group[0]))
            if q is None:
                rows.append(dict(policy=policy, pressure=group))
                continue
            result = counts.loc[("test", policy, group, q)]
            rows.append(dict(policy=policy, pressure=group, request_kw=q,
                             full_service=result.full_service, tracking=result.tracking,
                             aging_pass=result.aging_pass, delivered=result.delivered,
                             fallback_steps=result.fallback_steps))
    frozen = pd.DataFrame(rows)
    frozen.to_csv(output/"frozen_commitments.csv", index=False)
    test = events[events.split == "test"]
    detail = test.groupby(["policy", "pressure", "request_kw"])[["full_service", "delivered"]].sum()
    lines = [
        "# Causal service fallback experiment", "",
        "17:00–18:00 downward commitments under four paired synthetic conditions.",
        "Four calibration events and four distinct test events. Identical EVs use the symmetric representative schedule.",
        "The external budget is 0.00001 SOH per vehicle. Full service and commitment delivery are counted separately.",
        "The visible-slack rule uses the declared 38.5 kW recovery floor before dispatch and only the current capacity observation during recovery.",
        "The shared fallback retries from executed state, prioritizes departure service, and records any incremental budget breach.",
        "Commitments are frozen by visible EV deadline group, pooling capped and uncapped cases.", "",
        "## Frozen test outcomes", "",
        "| Policy | Pressure | Request kW | Full service / 4 | Delivered / 4 | Fallback steps |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in frozen.itertuples():
        if pd.isna(row.request_kw):
            lines.append(f"| {row.policy} | {row.pressure} | — | — | — | — |")
        else:
            lines.append(f"| {row.policy} | {row.pressure} | {row.request_kw:g} | "
                         f"{row.full_service:g}/4 | {row.delivered:g}/4 | {row.fallback_steps:g} |")
    lines += ["", "## Positive-request sweep on the four test events", "",
              "| Policy | Pressure | Largest positive request delivered 4/4 (kW) | Full service across positive requests |",
              "|---|---|---:|---:|"]
    for policy in POLICIES:
        for group in STRESSES:
            part = detail.loc[(policy, group)]
            positive = part[part.index > 0]
            best = positive.index[positive.delivered == len(TEST)]
            lines.append(f"| {policy} | {group} | {max(best) if len(best) else '—'} | "
                         f"{positive.full_service.sum()}/{len(positive)*len(TEST)} |")
    lines += ["", "The positive-request sweep is descriptive and is not used to select frozen commitments.",
              "A positive request counts as delivered only when service, tracking, and the incremental aging budget all pass.",
              "No LLM was called. Constructed events do not establish population reliability."]
    (output/"conclusions.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True, sharey=True,
                             constrained_layout=True)
    for column, group in enumerate(STRESSES):
        for policy in POLICIES:
            sub = detail.loc[(policy, group)]
            for row, metric in enumerate(("full_service", "delivered")):
                axes[row, column].plot(sub.index, sub[metric], "o-", markersize=3, label=policy)
        axes[0, column].set_title(f"EV/grid pressure {group}")
        axes[1, column].set_xlabel("Request (kW)")
    axes[0, 0].set_ylabel("Full service / 4")
    axes[1, 0].set_ylabel("Delivered / 4")
    axes[0, 0].set_yticks(range(5))
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output/"fallback_delivery.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"service_fallback_pilot_results")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    _, uncapped, _, _ = reference_case(CALIBRATION[0][0], "00", CALIBRATION[0][1])
    donors = np.array([stressed(uncapped, CALIBRATION[0][1]), uncapped,
                       stressed(uncapped, CALIBRATION[2][1]), uncapped])
    donors[:, :, 3] /= N
    forecast = PilotForecast(donors)
    jobs = [(split, f"{split}_{i+1:02d}", demand, cap, group, forecast)
            for split, cases in (("calibration", CALIBRATION), ("test", TEST))
            for i, (demand, cap) in enumerate(cases) for group in STRESSES]
    if args.workers == 1:
        results = list(map(evaluate, jobs))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(evaluate, jobs))
    events = pd.DataFrame([row for part in results for row in part])
    assert len(events) == len(jobs)*len(POLICIES)*len(REQUESTS)
    events.to_csv(args.output/"event_replays.csv", index=False)
    summarize(events, args.output)


if __name__ == "__main__":
    main()
