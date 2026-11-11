"""Build a local paired recovery-margin by departure-pressure screening design."""
import argparse
import hashlib
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from run_directional_pinn_validation import Aging, GROUPS, TOL, read_case
from run_llm_timing_pilot import (
    DEFAULTS, RECOVERY, REQUEST, START, TimingForecast, dump, rolling_reference, simulate,
)


MARGINS = {"ample": 6., "critical": 2.25}
RESERVES = (0., 1.5, 2.5, 4., 6.)
PENALTIES = (.5, 2., 5., 10.)
SERVICE_TOL = .001
CAPACITY_FLOOR = 1500.


def adjacent_bound(fleet, grid, ref):
    """Exact two-slot physical bound for a fixed reference, before PINN."""
    eligible = (fleet.arrival <= START) & (fleet.departure > RECOVERY)
    overlap = np.minimum(ref[:, START], fleet.power - ref[:, RECOVERY])[eligible]
    return max(0., min(float(overlap.sum()), grid[RECOVERY, 3] - ref[:, RECOVERY].sum()))


def parent_cases(source):
    seen = set()
    for group in GROUPS:
        entries = []
        for path in sorted((source / f"optimization_inputs_{group}").glob("scenario_*")):
            ev_hash = hashlib.sha256((path / "optimizer_ev_sessions.csv").read_bytes()).hexdigest()
            if ev_hash in seen:
                continue
            seen.add(ev_hash)
            entries.append((ev_hash, path, read_case(path)))
        entries.sort(key=lambda row: row[0])
        n_test = max(1, round(.3 * len(entries)))
        for i, (ev_hash, path, case) in enumerate(entries):
            yield group, path.name, ev_hash, path, case, "holdout" if i < n_test else "development"


def run(source, output):
    output.mkdir(parents=True, exist_ok=True)
    parents = list(parent_cases(source))
    aging = Aging(source)
    def background(grid):
        grid = grid.copy()
        grid[:, 3] = np.maximum(grid[:, 3], CAPACITY_FLOOR)
        return grid

    forecast = TimingForecast([background(case[1]) for _, _, _, _, case, split in parents
                               if split == "development"])
    native_forecast = TimingForecast([case[1] for _, _, _, _, case, split in parents
                                      if split == "development"])
    nominal = []
    for group, parent, digest, path, (fleet, original, temp), split in parents:
        grid = background(original)
        native, _, _ = rolling_reference((fleet, original, temp, None), native_forecast, DEFAULTS)
        reference, _, _ = rolling_reference((fleet, grid, temp, None), forecast, DEFAULTS)
        remaining = np.maximum(0., fleet.demand - reference[:, :START].sum(axis=1))
        connected = (fleet.arrival <= START) & (fleet.departure > START)
        urgent = connected & (fleet.departure <= RECOVERY + 1)
        urgent_overlap = urgent & (fleet.departure > RECOVERY)
        demand = max(1e-9, remaining[connected].sum())
        state = dict(group=group, parent=parent, parent_sha256=digest, split=split, source=str(path),
                     urgent_share=float(remaining[urgent].sum()/demand),
                     urgent_overlap_share=float(remaining[urgent_overlap].sum()/demand),
                     native_activation_cap_kw=original[START, 3], native_recovery_cap_kw=original[RECOVERY, 3],
                     native_physical_kw=adjacent_bound(fleet, original, native),
                     demand_kwh=float(fleet.demand.sum()), connected_kw=float(fleet.power[connected].sum()))
        nominal.append((state, (fleet, grid, temp), reference))
    threshold = float(np.median([s["urgent_share"] for s, _, _ in nominal if s["split"] == "development"]))
    manifest, cases = [], []
    for state, (fleet, original, temp), reference in nominal:
        pressure = "high" if state["urgent_share"] >= threshold else "low"
        for margin, requested_room in MARGINS.items():
            grid = original.copy()
            grid[RECOVERY, 3] = min(grid[RECOVERY, 3], reference[:, RECOVERY].sum() + requested_room)
            case = fleet, grid, temp, RECOVERY
            ref, _, _ = rolling_reference(case, forecast, DEFAULTS)
            row = dict(**state, cell=f"{margin}_{pressure}", margin=margin, pressure=pressure,
                       cap_hour=RECOVERY, original_cap_kw=original[RECOVERY, 3], cap_kw=grid[RECOVERY, 3],
                       nominal_grid_room_kw=grid[RECOVERY, 3]-reference[:, RECOVERY].sum(),
                       nominal_physical_kw=adjacent_bound(fleet, grid, reference),
                       default_physical_kw=adjacent_bound(fleet, grid, ref),
                       default_grid_room_kw=grid[RECOVERY, 3]-ref[:, RECOVERY].sum())
            manifest.append(row)
            cases.append((row, case))
    design = pd.DataFrame(manifest)
    assert design.groupby("parent_sha256").split.nunique().eq(1).all()
    assert design.groupby("parent_sha256").size().eq(len(MARGINS)).all()
    design.to_csv(output / "manifest.csv", index=False)
    dump(output / "protocol.json", dict(request_kw=REQUEST, activation_hour=START, recovery_hour=RECOVERY,
         target_room_kw=MARGINS, urgency_dev_median=threshold, reserves_kw=RESERVES, penalties=PENALTIES,
         fixed_weights={k: DEFAULTS[k] for k in ("lambda_risk", "w_price")},
         service_tolerance_fraction=SERVICE_TOL,
         split="stratified by archived group; exact EV file hashes deduplicated; lowest hashes held out",
         donor_pool="development trajectories only with common background capacity; 18:00 hard cap announced",
         capacity_background_floor_kw=CAPACITY_FLOOR,
         physical_bound="exact for adjacent two-slot shift around a fixed reference, before PINN",
         sensitivity="any grid point qualifies at no more than 0.1 pp service loss vs default, when default fails",
         holdout="construction and default physical diagnostics only; no search or PINN outcomes"))
    print(f"Constructed {len(cases)} cases from {len(parents)} unique EV trajectories.", flush=True)

    scan = []
    for row, case in cases:
        if row["split"] != "development":
            continue
        for reserve, penalty in product(RESERVES, PENALTIES):
            params = dict(DEFAULTS, recovery_reserve_kw=reserve, unmet_penalty_per_kwh=penalty)
            reference = rolling_reference(case, forecast, params)
            physical = adjacent_bound(case[0], case[1], reference[0])
            metrics = simulate(case, forecast, aging, params, reference=reference)
            assert not metrics["qualified"] or physical >= REQUEST - TOL
            scan.append(dict(parent_sha256=row["parent_sha256"], cell=row["cell"], reserve_kw=reserve,
                             penalty=penalty, physical_kw=physical, **metrics))
    results = pd.DataFrame(scan)
    assert len(results) == len(RESERVES)*len(PENALTIES)*sum(r["split"] == "development" for r in manifest)
    results.to_csv(output / "development_scan.csv", index=False)
    outcomes = []
    for (cell, parent), rows in results.groupby(["cell", "parent_sha256"]):
        baseline = rows[(rows.reserve_kw == 0.) & (rows.penalty == DEFAULTS["unmet_penalty_per_kwh"])].iloc[0]
        preserves_service = rows.served_ratio >= baseline.served_ratio - SERVICE_TOL - TOL
        outcomes.append(dict(cell=cell, parent_sha256=parent, default_qualified=baseline.qualified,
                             any_physical=bool((rows.physical_kw >= REQUEST - TOL).any()),
                             any_qualified=rows.qualified.any(), varies=rows.qualified.nunique() > 1,
                             service_preserving_gain=bool(not baseline.qualified and
                                                          (rows.qualified & preserves_service).any())))
    outcomes = pd.DataFrame(outcomes)
    outcomes.to_csv(output / "development_sensitivity.csv", index=False)
    summary = design.assign(
        physical=design.default_physical_kw >= REQUEST - TOL,
        nominal_physical=design.nominal_physical_kw >= REQUEST - TOL,
        native_physical=design.native_physical_kw >= REQUEST - TOL,
    ).groupby(["split", "cell"]).agg(
        n=("parent_sha256", "size"), default_physical=("physical", "sum"),
        nominal_physical=("nominal_physical", "sum"), mean_pressure=("urgent_share", "mean"),
        native_physical=("native_physical", "sum"),
        mean_default_physical_kw=("default_physical_kw", "mean"),
    ).reset_index()
    summary.merge(outcomes.groupby("cell").agg(
        default_qualified=("default_qualified", "sum"), any_qualified=("any_qualified", "sum"),
        any_physical=("any_physical", "sum"),
        weight_sensitive=("varies", "sum"), service_preserving_gain=("service_preserving_gain", "sum"),
    ).reset_index().assign(split="development"), on=["split", "cell"], how="left").to_csv(
        output / "summary.csv", index=False)
    results.groupby(["cell", "reserve_kw", "penalty"])[
        ["qualified", "served_ratio", "mean_soh_loss"]
    ].mean().reset_index().to_csv(output / "development_grid_summary.csv", index=False)
    print(f"Completed {len(results)} development replays.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())
