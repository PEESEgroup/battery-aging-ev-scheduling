"""Paired replay sensitivity on crossed demand and price perturbations."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nonanticipative_scheduler import Fleet, Forecast
from run_directional_pinn_validation import Aging, GROUPS, TOL, read_case, reference, replay

LEVELS = (0.9, 1.0, 1.1)
VARIANTS = ("full", "no_aging", "no_risk")


def run(source, output):
    aging = Aging(source)
    records = []
    archived = pd.read_csv(output.parent / "ablation" / "paired_events.csv")
    risk_archive = pd.read_csv(output.parent / "risk_replan" / "paired_events.csv")
    for group in GROUPS:
        paths = sorted((source / f"optimization_inputs_{group}").glob("scenario_*"))
        assert len(paths) == 10
        cases = [read_case(path) for path in paths]
        for _, grid, _ in cases:
            grid[:, 3] = np.maximum(grid[:, 3], 1500.)
            if group == "capacity_shortage":
                grid[17:21, 3] = np.minimum(grid[17:21, 3], 15.)
        forecast = Forecast([case[1] for case in cases[:4]])
        for index in range(7, 10):
            original, original_grid, temp = cases[index]
            for demand, price in product(LEVELS, repeat=2):
                fleet = Fleet(original.arrival, original.departure, original.demand * demand,
                              original.power, original.capacity, original.soc)
                grid = original_grid.copy()
                grid[:, 0] *= price
                refs = {}
                for risk in (True, False):
                    ref, proposal, caps = reference(fleet, grid, forecast, risk_terms=risk)
                    baseline = aging.losses(fleet, ref, temp)
                    refs[risk] = ref, proposal, caps, baseline
                np.testing.assert_array_equal(refs[True][2], refs[False][2])
                for direction, label in ((1, "up"), (-1, "down")):
                    for variant in VARIANTS:
                        ref, proposal, caps, baseline = refs[variant != "no_risk"]
                        action, deviation, service, health, physical = replay(
                            fleet, grid, ref, proposal, caps, direction, 2., aging,
                            temp, baseline, enforce_aging=variant != "no_aging")
                        tracking = deviation >= 2 - TOL
                        floor = bool(np.all(baseline <= aging.limit + TOL))
                        delivered = tracking and service and health and physical and floor
                        if demand == price == 1.:
                            prior = risk_archive if variant == "no_risk" else archived
                            old = prior[(prior.group == group) & (prior.scenario == paths[index].name)
                                        & (prior.direction == label) & (prior.variant == variant)].iloc[0]
                            assert np.isclose(deviation, old.actual_kw, rtol=0, atol=TOL)
                            assert (tracking, service, health, physical, delivered) == (
                                old.tracking, old.service_pass, old.health_pass, old.physical_pass, old.delivered)
                        records.append(dict(group=group, parent=paths[index].name, demand_scale=demand,
                                            price_scale=price, direction=label, variant=variant, actual_kw=deviation,
                                            tracked=tracking, service=service, health=health, physical=physical,
                                            reference_floor=floor, delivered=delivered,
                                            served_ratio=action.sum()/fleet.demand.sum(),
                                            mean_soh_loss=aging.losses(fleet, action, temp).mean()))
            print(group, paths[index].name, flush=True)
    events = pd.DataFrame(records)
    keys = ["group", "parent", "demand_scale", "price_scale", "direction", "variant"]
    assert len(events) == 648 and not events.duplicated(keys).any()
    output.mkdir(parents=True, exist_ok=True)
    events.to_csv(output/"events.csv", index=False)
    measures = {"tracked":"mean", "delivered":"mean", "service":"mean", "health":"mean",
                "mean_soh_loss":"mean", "served_ratio":"mean", "reference_floor":"mean"}
    for suffix, columns in (("pooled", ["direction","variant"]),
                            ("regime", ["group","direction","variant"]),
                            ("perturbation", ["demand_scale","price_scale","direction","variant"])):
        summary = events.groupby(columns, sort=False).agg(measures).reset_index()
        summary.to_csv(output/f"{suffix}.csv", index=False)
        if suffix == "pooled":
            print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.source, args.output)

