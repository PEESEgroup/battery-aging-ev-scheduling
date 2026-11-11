"""Paired single-factor ablation of the fixed-reference directional replay."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nonanticipative_scheduler import Forecast
from run_directional_pinn_validation import Aging, GROUPS, START, TOL, read_case, reference, replay


def run(source, output, *, replan_risk=False):
    aging = Aging(source)
    parent = output.parent / "online.csv"
    prior = pd.read_csv(parent).query("split == 'test' and request_kw == 2")
    records = []
    for group in GROUPS:
        paths = sorted((source / f"optimization_inputs_{group}").glob("scenario_*"))
        assert len(paths) == 10
        cases = [read_case(path) for path in paths]
        for _, grid, _ in cases:
            grid[:, 3] = np.maximum(grid[:, 3], 1500.)
            if group == "capacity_shortage":
                grid[17:21, 3] = np.minimum(grid[17:21, 3], 15.)
        forecast = Forecast([case[1] for case in cases[:4]])
        for k in range(7, 10):
            fleet, grid, temp = cases[k]
            ref, proposal, caps = reference(fleet, grid, forecast)
            baseline_loss = aging.losses(fleet, ref, temp)
            assert np.all(baseline_loss <= aging.limit + TOL)
            full_caps, full_score = forecast.build(START, grid[:START+1], 4)
            mean_caps, mean_score = forecast.build(START, grid[:START+1], 4, risk_terms=False)
            np.testing.assert_array_equal(full_caps, caps)
            np.testing.assert_array_equal(mean_caps, caps)
            full_reference = ref, proposal, caps, baseline_loss
            if replan_risk:
                neutral_ref, neutral_proposal, neutral_caps = reference(fleet, grid, forecast, risk_terms=False)
                np.testing.assert_array_equal(neutral_caps, caps)
                neutral = neutral_ref, neutral_proposal, neutral_caps, aging.losses(fleet, neutral_ref, temp)
            else:
                neutral = full_reference
            for direction, label in ((1, "up"), (-1, "down")):
                full_action = None
                for variant in (("full", "no_risk") if replan_risk else ("full", "no_aging", "no_risk")):
                    ref, proposal, caps, baseline_loss = neutral if variant == "no_risk" else full_reference
                    digest = hashlib.sha256(np.ascontiguousarray(ref).tobytes()).hexdigest()
                    action, deviation, service, health, physical = replay(
                        fleet, grid, ref, proposal, caps, direction, 2., aging,
                        temp, baseline_loss, enforce_aging=variant != "no_aging")
                    tracking = deviation >= 2 - TOL
                    delivered = tracking and service and health and physical
                    if variant == "full":
                        full_action = action
                        archived = prior[(prior.group == group) & (prior.scenario == paths[k].name)
                                         & (prior.direction == label)]
                        assert len(archived) == 1
                        old = archived.iloc[0]
                        assert np.isclose(deviation, old.actual_kw, atol=TOL, rtol=0)
                        assert (service, health, physical, delivered) == (
                            old.service_pass, old.health_pass, old.physical_pass, old.delivered)
                    if variant == "no_risk" and not replan_risk:
                        np.testing.assert_array_equal(action, full_action)
                    records.append(dict(group=group, scenario=paths[k].name, direction=label,
                                        variant=variant, request_kw=2., actual_kw=deviation,
                                        tracking=tracking, service_pass=service, health_pass=health,
                                        physical_pass=physical, delivered=delivered,
                                        dispatch_changed=not np.array_equal(action, full_action),
                                        forecast_risk_score_max_delta=float(np.max(abs(full_score-mean_score))),
                                        reference_sha256=digest,
                                        reference_service_ratio=ref.sum()/fleet.demand.sum(),
                                        actual_service_ratio=action.sum()/fleet.demand.sum(),
                                        reference_mean_soh_loss=baseline_loss.mean(),
                                        actual_mean_soh_loss=aging.losses(fleet, action, temp).mean(),
                                        reference_meets_soh_floor=bool(np.all(baseline_loss <= aging.limit + TOL)),
                                        reference_changed=not np.array_equal(ref, full_reference[0])))
    events = pd.DataFrame(records)
    assert len(events) == (48 if replan_risk else 72) and not events.duplicated(["group","scenario","direction","variant"]).any()
    if not replan_risk:
        assert (events.groupby(["group","scenario","direction"]).reference_sha256.nunique() == 1).all()
    summary = (events.groupby(["group","direction","variant"], sort=False)
               .agg(trajectories=("scenario","size"), actual_median_kw=("actual_kw","median"),
                    tracking=("tracking","sum"), service_pass=("service_pass","sum"),
                    health_pass=("health_pass","sum"), delivered=("delivered","sum"),
                    dispatch_changed=("dispatch_changed","sum"), reference_changed=("reference_changed","sum"), reference_service_ratio=("reference_service_ratio","mean"), actual_service_ratio=("actual_service_ratio","mean"), actual_mean_soh_loss=("actual_mean_soh_loss","mean")).reset_index())
    output.mkdir(parents=True, exist_ok=True)
    events.to_csv(output/"paired_events.csv", index=False)
    summary.to_csv(output/"summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replan-risk", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.source, args.output, replan_risk=args.replan_risk)


