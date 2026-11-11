"""Local crossed departure-pressure and recovery-price screen."""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lock_recovery_holdout import digest, fingerprint
from run_directional_pinn_validation import Aging, TOL, read_case
from run_llm_timing_pilot import DEFAULTS, RECOVERY, REQUEST, START, TimingForecast, dump, rolling_reference, simulate
from screen_recovery_departure import SERVICE_TOL, adjacent_bound, parent_cases


RESERVES = (0., 1.5, 2.5, 4., 6.)
PENALTIES = (.5, 2., 5., 10.)
PRICE_WEIGHTS = (.4, 1., 2.2)
BASELINE = (0., 2., 1.)
KEYS = ["reserve_kw", "unmet_penalty", "price_weight"]
METRICS = ["qualified", "served_ratio", "mean_soh_loss", "tracked", "service_match", "aging_pass"]


def run(source, factorial, locked, output):
    output.mkdir(parents=True, exist_ok=False)
    prior = json.loads((factorial / "protocol.json").read_text(encoding="utf-8"))
    floor = prior["capacity_background_floor_kw"]
    room = prior["target_room_kw"]["critical"]
    lock = json.loads((locked / "lock.json").read_text(encoding="utf-8"))
    assert digest(locked / "locked_manifest.csv") == lock["manifest_sha256"]

    def background(grid):
        grid = grid.copy()
        grid[:, 3] = np.maximum(grid[:, 3], floor)
        return grid

    parents = []
    dev = pd.read_csv(factorial / "manifest.csv")
    dev = set(dev.loc[dev.split == "development", "parent_sha256"])
    for _, _, key, path, (fleet, grid, temp), split in parent_cases(source):
        if key in dev:
            assert split == "development"
            parents.append(("development", fingerprint(fleet), path, fleet, background(grid), temp))
    assert len(parents) == 21
    forecast = TimingForecast([p[4] for p in parents])
    prices = np.array([p[4][[START, RECOVERY], 0] for p in parents]).ravel()
    lo, hi = np.quantile(prices, [.25, .75])
    spread = float(hi-lo)
    assert spread > 0

    locked_rows = pd.read_csv(locked / "locked_manifest.csv")
    for row in locked_rows[locked_rows.margin == "critical"].itertuples():
        path = locked / row.path
        assert digest(path / "optimizer_grid.csv") == row.grid_sha256
        assert digest(path / "optimizer_ev_sessions.csv") == row.ev_sha256
        fleet, grid, temp = read_case(path)
        assert fingerprint(fleet) == row.parent_fingerprint
        parents.append(("holdout", row.parent_fingerprint, path, fleet, background(grid), temp))
    assert len(parents) == len({p[1] for p in parents}) == 77

    contexts = []
    for split, parent, source_path, fleet, grid, temp in parents:
        ref, _, _ = rolling_reference((fleet, grid, temp, None), forecast, DEFAULTS)
        eligible = (fleet.arrival <= START) & (fleet.departure > RECOVERY)
        urgent = eligible & (fleet.departure <= RECOVERY + 1)
        remaining = np.maximum(0., fleet.demand - ref[:, :START].sum(axis=1))
        pressure = float(remaining[urgent].sum()/max(1e-9, remaining[eligible].sum()))
        cap = min(grid[RECOVERY, 3], ref[:, RECOVERY].sum() + room)
        contexts.append((split, parent, source_path, fleet, grid, temp, pressure, cap))
    threshold = float(np.median([c[6] for c in contexts if c[0] == "development"]))
    dump(output / "protocol.json", dict(
        development_parents=21, holdout_parents=56, urgency_cutoff=threshold,
        urgency_definition="share of remaining energy due by end of recovery among EVs connected in both slots",
        pooled_development_price_quartiles=[float(lo), float(hi)],
        price_design="parent-specific pair mean plus/minus half the development interquartile spread; half spread clipped at mean to avoid negative intervention prices",
        cap_design="same 18:00 cap per parent; original-price unbound rolling reference at 18:00 plus 2.25 kW",
        background_floor_kw=floor, request_kw=REQUEST, risk_fixed=DEFAULTS["lambda_risk"],
        grid={k: v for k, v in zip(KEYS, (RESERVES, PENALTIES, PRICE_WEIGHTS))},
        selection="cell-average service within 0.1 percentage points of cell default; maximize qualified rate then minimize SOH loss then maximize service",
        primary_candidate_cell="high_pressure_high_price",
        development_source_sha256=digest(factorial / "manifest.csv"),
        holdout_source_sha256=lock["manifest_sha256"], api_calls=0,
        holdout_use="default physical diagnostics only, no PINN replay or tuning"))
    cases, manifest = {}, []
    for split, parent, source_path, fleet, grid, temp, pressure, cap in contexts:
        level = "high" if pressure >= threshold else "low"
        midpoint = float(grid[[START, RECOVERY], 0].mean())
        half = min(spread/2, midpoint)
        for price, sign in (("low", -1), ("high", 1)):
            cell = f"{level}_pressure_{price}_price"
            variant = grid.copy()
            variant[START, 0], variant[RECOVERY, 0] = midpoint-sign*half, midpoint+sign*half
            variant[RECOVERY, 3] = cap
            case = fleet, variant, temp, RECOVERY
            reference = rolling_reference(case, forecast, DEFAULTS)
            ref = reference[0]
            bound = adjacent_bound(fleet, variant, ref)
            actual_room = float(cap-ref[:, RECOVERY].sum())
            manifest.append(dict(split=split, cell=cell, parent=parent, source=str(source_path),
                pressure_share=pressure, activation_price=variant[START, 0], recovery_price=variant[RECOVERY, 0],
                cap_kw=cap, default_room_kw=actual_room, near_target=abs(actual_room-room)<=TOL,
                physical_kw=bound, physical=bound>=REQUEST-TOL))
            if split == "development":
                cases[cell, parent] = case, reference
    design = pd.DataFrame(manifest)
    assert design.groupby("parent").split.nunique().eq(1).all()
    assert design.groupby("parent").cap_kw.nunique().eq(1).all()
    assert design.groupby("parent").size().eq(2).all()
    design.to_csv(output / "manifest.csv", index=False)
    preflight = design.groupby(["split", "cell"]).agg(
        parents=("parent", "size"), physical=("physical", "sum"), near_target=("near_target", "sum"),
        mean_actual_room_kw=("default_room_kw", "mean"), mean_physical_kw=("physical_kw", "mean"),
        mean_pressure_share=("pressure_share", "mean")).reset_index()
    preflight.to_csv(output / "physical_preflight.csv", index=False)
    print(preflight.to_string(index=False), flush=True)

    aging = Aging(source)
    results = []
    for cell in sorted(design.loc[design.split == "development", "cell"].unique()):
        for (case_cell, parent), (case, default_ref) in cases.items():
            if case_cell != cell:
                continue
            for point in product(RESERVES, PENALTIES, PRICE_WEIGHTS):
                reserve, penalty, price = point
                params = dict(DEFAULTS, recovery_reserve_kw=reserve, unmet_penalty_per_kwh=penalty, w_price=price)
                ref = default_ref if point == BASELINE else rolling_reference(case, forecast, params)
                metrics = simulate(case, forecast, aging, params, reference=ref)
                results.append(dict(cell=cell, parent=parent, **dict(zip(KEYS, point)), **metrics))
        print(f"{cell} scanned", flush=True)
    scan = pd.DataFrame(results)
    assert len(scan) == 42*len(RESERVES)*len(PENALTIES)*len(PRICE_WEIGHTS)
    scan.to_csv(output / "development_scan.csv", index=False)
    grid = scan.groupby(["cell", *KEYS])[METRICS].mean().reset_index()
    grid.to_csv(output / "development_grid_summary.csv", index=False)
    is_default = (grid[KEYS] == BASELINE).all(axis=1)
    defaults = grid[is_default].set_index("cell").served_ratio
    grid["service_ok"] = grid.served_ratio >= grid.cell.map(defaults)-SERVICE_TOL-TOL
    ranked = grid[grid.service_ok].sort_values(
        ["qualified", "mean_soh_loss", "served_ratio", *KEYS],
        ascending=[False, True, False, True, True, True])
    winners = ranked.drop_duplicates("cell").copy()
    winners.to_csv(output / "cell_winners.csv", index=False)
    cross = winners[["cell", *KEYS]].rename(columns={"cell": "source_cell"}).merge(grid, on=KEYS)
    cross = cross.rename(columns={"cell": "target_cell"})
    cross["qualified_gap_to_own"] = cross.target_cell.map(winners.set_index("cell").qualified)-cross.qualified
    cross.to_csv(output / "cross_replay.csv", index=False)

    price_effect = scan.groupby(["cell", "parent", "reserve_kw", "unmet_penalty"]).agg(
        qualification_changes=("qualified", "nunique"), service_span=("served_ratio", lambda a: a.max()-a.min()),
        health_span=("mean_soh_loss", lambda a: a.max()-a.min())).reset_index()
    price_effect["qualification_changes"] = price_effect.qualification_changes > 1
    price_effect.groupby(["cell", "parent"])[["qualification_changes", "service_span", "health_span"]].max().reset_index().to_csv(
        output / "price_weight_sensitivity.csv", index=False)
    dump(output / "completion.json", dict(development_replays=len(scan), cross_replays=len(cross),
        holdout_pinn_replays=0, pinn_sha256=digest(aging.path), manifest_sha256=digest(output/"manifest.csv")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "factorial", "locked", "output"):
        parser.add_argument(name, type=Path)
    torch.set_num_threads(1)
    run(*(path.resolve() for path in vars(parser.parse_args()).values()))
