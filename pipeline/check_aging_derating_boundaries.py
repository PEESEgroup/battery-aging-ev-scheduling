"""Expand the development grid, freeze configurations, then cross-replay holdout."""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lock_recovery_holdout import digest
from run_directional_pinn_validation import TOL
from run_llm_timing_pilot import DEFAULTS, REQUEST, dump, simulate
from screen_aging_derating_overlap import DERATING_KW, RESERVES, RISKS, construct, summarize
from screen_recovery_departure import SERVICE_TOL


KEYS = ["risk", "reserve_kw"]
DEFAULT_POINT = (.3, 0.)
OLD_WINNER = (.9, 6.)
EXTENDED_RISKS = (1.2, 1.5, 2.)


def replay(row, case, forecast, reference, point, aging):
    risk, reserve = point
    params = dict(DEFAULTS, lambda_risk=risk, recovery_reserve_kw=reserve)
    reference = reference if point == DEFAULT_POINT else None
    metrics = simulate(case, forecast, aging, params, reference=reference)
    return {k: row[k] for k in ("parent", "regime", "event", "probability")} | dict(
        risk=risk, reserve_kw=reserve, **metrics)


def summarize_grid(scan):
    rows = []
    for (regime, risk, reserve), part in scan.groupby(["regime", *KEYS]):
        rows.append(dict(regime=regime, risk=risk, reserve_kw=reserve, **summarize(part).to_dict()))
    grid = pd.DataFrame(rows)
    default = grid[(grid.risk == .3) & (grid.reserve_kw == 0.)].set_index("regime").served_ratio
    grid["service_ok"] = grid.served_ratio >= grid.regime.map(default)-SERVICE_TOL-TOL
    return grid


def run(source, factorial, locked, prior, output):
    aging, design, cases = construct(source, factorial, locked, output, soc_stress=True)
    old_design = pd.read_csv(prior/"manifest.csv")
    pd.testing.assert_frame_equal(design, old_design, check_exact=False, atol=TOL, rtol=0.)
    dev = design[(design.split == "development") & design.high_exposure]
    held = design[design.split == "holdout"]
    assert dev.parent.nunique() == 15 and held.parent.nunique() == 56
    # Even with a derating, retain at least one request's worth of 18:00
    # capacity for the reference plan on every high-exposure dev parent.
    reserve_ceiling = int(np.floor(dev.nominal_cap_18_kw.min()-DERATING_KW-REQUEST))
    assert reserve_ceiling > max(RESERVES)
    new_reserves = tuple(float(x) for x in range(8, reserve_ceiling+1, 2))
    assert new_reserves and new_reserves[-1] == reserve_ceiling
    prior_points = set(product(RISKS, RESERVES))
    points = set(product((*RISKS, *EXTENDED_RISKS), (*RESERVES, *new_reserves)))
    new_points = sorted(points-prior_points)
    dump(output/"extension_protocol.json", dict(
        risk_grid=sorted({p[0] for p in points}), reserve_grid_kw=sorted({p[1] for p in points}),
        reserve_ceiling_kw=reserve_ceiling, minimum_nominal_recovery_cap_kw=float(dev.nominal_cap_18_kw.min()),
        reserve_ceiling_basis="minimum development nominal recovery cap minus 1.25 kW derating and 2 kW retained reference capacity",
        risk_bound_basis="dimensionless sensitivity through 2x the normalized mean term coefficient; no physical upper bound",
        selection="within 0.1 pp of default mean service; maximize qualification, minimize unmet CVaR then SOH loss",
        old_grid_sha256=digest(prior/"development_scan.csv"),
        prior_holdout_lock_sha256=digest(locked/"lock.json"),
        holdout="evaluate only locked old/default and two new winners; full set and physical paired subset",
        api_calls=0))
    print(f"Extension: {len(new_points)} new points, recovery-reserve ceiling {reserve_ceiling} kW.", flush=True)
    old = pd.read_csv(prior/"development_scan.csv")
    old = old[old.high_exposure].drop(columns="high_exposure")
    assert len(old) == len(dev)*len(prior_points)
    dev_cases = [entry for entry in cases if entry[0]["split"] == "development" and entry[0]["high_exposure"]]
    records = []
    for i, (row, case, forecast, ref) in enumerate(dev_cases, 1):
        for point in new_points:
            records.append(replay(row, case, forecast, ref, point, aging))
        if i % 15 == 0:
            print(f"Development branches complete: {i}/{len(dev_cases)}", flush=True)
    scan = pd.concat([old, pd.DataFrame(records)], ignore_index=True)
    assert len(scan) == len(dev)*len(points)
    assert not scan.duplicated(["regime", "parent", "event", *KEYS]).any()
    scan.to_csv(output/"development_scan.csv", index=False)
    grid = summarize_grid(scan)
    grid.to_csv(output/"development_grid_summary.csv", index=False)
    winners = grid[grid.service_ok].sort_values(
        ["qualified_rate", "cvar_unmet_kwh", "mean_soh_loss", *KEYS],
        ascending=[False, True, True, True, True]).drop_duplicates("regime")
    winners.to_csv(output/"development_winners.csv", index=False)
    configs = {"default": DEFAULT_POINT, "old_boundary": OLD_WINNER}
    configs.update({row.regime+"_winner": (row.risk, row.reserve_kw) for row in winners.itertuples()})
    lock = dict(configurations=configs, manifest_sha256=digest(output/"manifest.csv"),
                development_grid_sha256=digest(output/"development_grid_summary.csv"))
    dump(output/"configuration_lock.json", lock)
    print(f"Locked configurations before holdout replay: {configs}", flush=True)

    # Each mother trajectory contributes both event timings and both event branches.
    physical = held[held.event].groupby("parent").physical.all()
    physical = set(physical[physical].index)
    assert len(physical) == 53
    rows = []
    test_cases = [entry for entry in cases if entry[0]["split"] == "holdout"]
    for i, (row, case, forecast, ref) in enumerate(test_cases, 1):
        for point in sorted(set(configs.values())):
            rows.append(replay(row, case, forecast, ref, point, aging))
        if i % 56 == 0:
            print(f"Holdout branches complete: {i}/{len(test_cases)}", flush=True)
    holdout = pd.DataFrame(rows)
    assert len(holdout) == len(held)*len(set(configs.values()))
    assert not holdout.duplicated(["regime", "parent", "event", *KEYS]).any()
    holdout.to_csv(output/"holdout_replays.csv", index=False)
    summaries = []
    for panel, part in (("all", holdout), ("both_event_physical", holdout[holdout.parent.isin(physical)])):
        for name, (risk, reserve) in configs.items():
            subset = part[(part.risk == risk) & (part.reserve_kw == reserve)]
            for regime, group in subset.groupby("regime"):
                summaries.append(dict(panel=panel, config=name, risk=risk, reserve_kw=reserve,
                                      target_regime=regime, **summarize(group).to_dict()))
    pd.DataFrame(summaries).to_csv(output/"holdout_cross_replay.csv", index=False)
    dump(output/"completion.json", dict(**lock, new_development_replays=len(records),
         reused_development_replays=len(old), holdout_replays=len(holdout), physical_holdout_parents=len(physical)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "factorial", "locked", "prior", "output"):
        parser.add_argument(name, type=Path)
    torch.set_num_threads(1)
    run(*(p.resolve() for p in vars(parser.parse_args()).values()))
