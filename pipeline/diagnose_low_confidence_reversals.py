"""Verify state-rank reversals and locate active constraints on the locked panel."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from check_low_confidence_risk_boundary import ROW_ID, OLD_WINNER, reconstruct, replay as score_replay, stats
from lock_recovery_holdout import digest
from run_directional_pinn_validation import replay, swap, TOL
from run_llm_timing_pilot import DEFAULTS, RECOVERY, REQUEST, START, dump, rolling_reference
from screen_adaptive_recovery_reserve import construct
from screen_forecast_adaptive_reserve import BASELINE, KEYS


MIN_EFFECT = .01
EDGES = ((0, 1), (2, 3), (0, 2), (1, 3))
MECHANISM_POINTS = (BASELINE, (1.1, 8.), OLD_WINNER, (1.25, 8.))


def cell_table(raw):
    return pd.DataFrame([dict(bucket=b, risk=risk, reserve_kw=reserve, **stats(group))
        for (b, risk, reserve), group in raw.groupby(["bucket", *KEYS])])


def find_reversals(grid):
    q = grid.pivot(index=KEYS, columns="bucket", values="qualified_rate")
    ok = grid.pivot(index=KEYS, columns="bucket", values="cvar_ok")
    rows = []
    for x, y in EDGES:
        for a, b in combinations(q.index, 2):
            d = q.loc[a, x]-q.loc[b, x]
            e = q.loc[a, y]-q.loc[b, y]
            if d*e < 0 and min(abs(d), abs(e)) >= MIN_EFFECT and all(ok.loc[p, k] for p in (a,b) for k in (x,y)):
                rows.append(dict(state_a=x, state_b=y, risk_a=a[0], reserve_a=a[1], risk_b=b[0],
                                 reserve_b=b[1], dev_gap_state_a=d, dev_gap_state_b=e))
    return pd.DataFrame(rows)


def diagnose(entries, aging):
    records = []
    for row, case, forecast in entries:
        fleet, grid, temp, _ = case
        for risk, reserve in MECHANISM_POINTS:
            params = dict(DEFAULTS, lambda_risk=risk, recovery_reserve_kw=reserve)
            reference, proposal, caps = rolling_reference(case, forecast, params)
            _, planning, _ = forecast.build_case(START, grid[:START+1], params, {})
            p, q = proposal[:, START], proposal[:, RECOVERY]
            connected = (fleet.arrival <= START) & (fleet.departure > RECOVERY)
            car_room = np.maximum(0., np.minimum(p, fleet.power-q))*connected
            pre_grid_room = max(0., float(caps[RECOVERY-START]-q.sum()))
            if abs(caps[RECOVERY-START]-2.25) > TOL:
                raise AssertionError("Minimum recovery path cap changed")
            forecast_temp = np.r_[temp[:START+1], np.repeat(temp[START], 23-START)]
            shifts = swap(fleet, proposal, caps, -1, REQUEST, aging, forecast_temp)
            no_pinn = min(REQUEST, pre_grid_room, float(car_room.sum()))
            no_pinn_swap = swap(fleet, proposal, caps, -1, REQUEST, aging, forecast_temp, enforce_aging=False)
            assert abs(no_pinn_swap.sum()-no_pinn) <= TOL
            reallocated = not np.allclose(shifts, no_pinn_swap, atol=TOL, rtol=0.)
            relaxed = caps.copy()
            relaxed[RECOVERY-START] = forecast.scenario_paths(START, grid[:START+1])[:, RECOVERY-START, 3].max()
            base = aging.losses(fleet, reference, temp)
            _, delivered, service, health, physical = replay(fleet, grid, reference, proposal, caps, -1, REQUEST,
                                                             aging, temp, base)
            _, _, wide_service, wide_health, wide_physical = replay(fleet, grid, reference, proposal, relaxed,
                                                                    -1, REQUEST, aging, temp, base)
            assert physical and wide_physical
            if reallocated:
                _, _, service_without_screen, health_without_screen, off_physical = replay(
                    fleet, grid, reference, proposal, caps, -1, REQUEST, aging, temp, base, enforce_aging=False)
                assert off_physical
            else:
                service_without_screen, health_without_screen = service, health
            assert abs(delivered-shifts.sum()) <= TOL
            after_grid_room = max(0., float(grid[RECOVERY, 3]-reference[:, RECOVERY].sum()))
            # Replay first clips by each vehicle's residual charger capacity, then by the actual grid cap.
            residual_by_car = np.minimum(shifts, fleet.power-reference[:, RECOVERY])
            car_clip = float(shifts.sum()-residual_by_car.sum())
            grid_clip = max(0., float(residual_by_car.sum()-after_grid_room))
            records.append({k: row[k] for k in ("split", *ROW_ID, "probability")} | dict(
                risk=risk, reserve_kw=reserve, forecast_room_kw=pre_grid_room,
                forecast_planning_room_after_reserve_kw=planning[RECOVERY-START],
                reserve_saturated=planning[RECOVERY-START]<=TOL,
                car_room_kw=float(car_room.sum()), no_pinn_track=no_pinn>=REQUEST-TOL,
                pinned_track=delivered>=REQUEST-TOL,
                forecast_cap_blocks=pre_grid_room<REQUEST-TOL and car_room.sum()>=REQUEST-TOL,
                pinn_blocks=no_pinn>=REQUEST-TOL and delivered<REQUEST-TOL,
                pinn_reallocates=reallocated, service_without_screen=service_without_screen,
                health_without_screen=health_without_screen,
                post_recovery_grid_room_kw=after_grid_room, clipped_by_car_kw=car_clip,
                clipped_by_actual_grid_kw=grid_clip, service_ok=service, health_ok=health,
                wider_forecast_service=wide_service, wider_forecast_health=wide_health,
                qualified=bool(delivered>=REQUEST-TOL and service and health and np.all(base<=aging.limit+TOL))))
    return pd.DataFrame(records)


def run(source, factorial, fresh, prior_stress, panel, previous, output):
    output.mkdir(parents=True, exist_ok=False)
    lock = json.loads((panel/"scenario_lock.json").read_text(encoding="utf-8"))
    assert digest(panel/"locked_manifest.csv") == lock["manifest_sha256"]
    manifest = pd.read_csv(panel/"locked_manifest.csv")
    assert manifest.groupby(["parent", "branch"]).recovery_hard_cap_kw.nunique().eq(1).all()
    assert manifest.groupby(["parent", "high_exposure"]).soc_values.nunique().eq(1).all()
    assert manifest.groupby("parent").split.nunique().eq(1).all()
    assert manifest.groupby("split").parent.nunique().to_dict() == {"development": 8, "holdout": 14}
    prior = json.loads((previous/"configuration_lock.json").read_text(encoding="utf-8"))
    assert prior["panel_sha256"] == lock["manifest_sha256"]
    assert digest(previous/"development_replays.csv") == prior["development_replay_sha256"]
    dev_grid = pd.read_csv(previous/"development_grid.csv")
    reversals = find_reversals(dev_grid)
    points = {tuple(p) for p in MECHANISM_POINTS}
    if not reversals.empty:
        points |= {tuple(p) for p in reversals[["risk_a","reserve_a"]].itertuples(index=False, name=None)}
        points |= {tuple(p) for p in reversals[["risk_b","reserve_b"]].itertuples(index=False, name=None)}
    dump(output/"protocol.json", dict(
        min_forecast_cap_kw=2.25, scenario_manifest_sha256=lock["manifest_sha256"],
        development_replay_sha256=prior["development_replay_sha256"],
        reversal="opposite sign and at least 1 percentage point in both states; both candidates meet +0.5 kWh cell CVaR limits",
        one_factor_edges=EDGES, development_reversals=len(reversals), holdout_points=sorted(points),
        mechanism_points=MECHANISM_POINTS, actual_capacity_constraints="enforced in all replays, including mechanism diagnostics",
        api_calls=0))
    print(f"Found {len(reversals)} development reversal pairs; holdout checks {len(points)} points.", flush=True)
    aging, _, source_cases = construct(source, factorial, fresh, prior_stress, output/"source_design", audit_physical=False)
    cases = reconstruct(source_cases, manifest, 2.25)
    held_cases = [x for x in cases if x[0]["split"]=="holdout"]
    old = pd.read_csv(previous/"holdout_replays.csv")
    existing = set(old[KEYS].itertuples(index=False, name=None))
    held = pd.concat([old[old.apply(lambda r: (r.risk,r.reserve_kw) in points, axis=1)],
        score_replay(held_cases, aging, lambda _: sorted(points-existing), "New holdout candidates")], ignore_index=True)
    assert len(held) == len(held_cases)*len(points) and not held.duplicated(ROW_ID+KEYS).any()
    held.to_csv(output/"holdout_replays.csv", index=False)
    hgrid = cell_table(held)
    hgrid.to_csv(output/"holdout_grid.csv", index=False)
    indexed = hgrid.set_index([*KEYS, "bucket"])
    result = []
    for row in reversals.itertuples(index=False):
        a, b = (row.risk_a, row.reserve_a), (row.risk_b, row.reserve_b)
        x, y = row.state_a, row.state_b
        d, e = (indexed.loc[(*a,k),"qualified_rate"]-indexed.loc[(*b,k),"qualified_rate"] for k in (x,y))
        passed = all(indexed.loc[(*p,k),"cvar_unmet_kwh"] <=
                     indexed.loc[(*BASELINE,k),"cvar_unmet_kwh"]+.5+TOL for p in (a,b) for k in (x,y))
        matches = d*row.dev_gap_state_a > 0 and e*row.dev_gap_state_b > 0 and min(abs(d),abs(e))>=MIN_EFFECT
        result.append(row._asdict() | dict(holdout_gap_state_a=d, holdout_gap_state_b=e,
                       holdout_cvar_ok=passed, replicated=bool(passed and matches)))
    pd.DataFrame(result).to_csv(output/"reversals.csv", index=False)
    print("Holdout ranking checked; auditing activation and recovery constraints.", flush=True)
    diag = diagnose(cases, aging)
    diag.to_csv(output/"constraint_diagnostics.csv", index=False)
    dump(output/"completion.json", dict(development_reversals=len(reversals),
        replicated_reversals=sum(x["replicated"] for x in result), holdout_parents=14,
        diagnostic_replays=len(diag)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "factorial", "fresh", "prior_stress", "panel", "previous", "output"):
        parser.add_argument(name, type=Path)
    torch.set_num_threads(1)
    run(*(p.resolve() for p in vars(parser.parse_args()).values()))
