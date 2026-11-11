"""Lock a forecast-state risk/reserve policy under a development CVaR guardrail."""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lock_recovery_holdout import digest
from run_directional_pinn_validation import TOL
from run_llm_timing_pilot import DEFAULTS, RECOVERY, START, dump, rolling_reference, simulate
from screen_adaptive_recovery_reserve import STATE, construct
from screen_aging_derating_overlap import summarize, weighted_cvar


POINTS = tuple(product((.6, .9, 1.2), (6., 8., 10.)))
BASELINE = (.9, 6.)
MARGIN = .5
KEYS = ["risk", "reserve_kw"]
ID = ["parent", "event", *STATE]


def forecast_state(case, forecast, threshold):
    fleet = case[0]
    _, proposal, caps = rolling_reference(case, forecast, dict(DEFAULTS, lambda_risk=.9))
    connected = (fleet.arrival <= START) & (fleet.departure > RECOVERY)
    transferable = np.maximum(0., np.minimum(proposal[:, START], fleet.power-proposal[:, RECOVERY]))*connected
    soc = fleet.soc + proposal[:, :RECOVERY].sum(axis=1)/fleet.capacity
    exposure = transferable[soc >= threshold].sum()/max(transferable.sum(), 1e-9)
    return float(max(0., caps[RECOVERY-START]-proposal[:, RECOVERY].sum())), float(exposure)


def balanced_cut(values):
    unique = np.unique(np.asarray(values).round(8))
    if len(unique) == 1:
        return float(unique[0])
    mid = (unique[1:]+unique[:-1])/2
    return float(min(mid, key=lambda x: abs(np.mean(values > x)-.5)))


def state_bin(room, exposure, cuts):
    return 2*int(room > cuts[0]+TOL)+int(exposure > cuts[1]+TOL)


def build_state(entries, threshold, cuts=None):
    states = {}
    for row, case, forecast in entries:
        key = (row["parent"], *(row[k] for k in STATE))
        if key not in states:
            assert not row["event"]
            states[key] = forecast_state(case, forecast, threshold)
    values = np.array(list(states.values()))
    if cuts is None:
        cuts = tuple(balanced_cut(v) for v in values.T)
    return {k: (*v, state_bin(*v, cuts)) for k, v in states.items()}, cuts


def replay(entries, states, aging, choose, label):
    rows, refs = [], {}
    active_parent = None
    for i, (row, case, forecast) in enumerate(entries, 1):
        if row["parent"] != active_parent:
            refs.clear()
            active_parent = row["parent"]
        room, exposure, bucket = states[(row["parent"], *(row[k] for k in STATE))]
        for point in choose(bucket):
            params = dict(DEFAULTS, lambda_risk=point[0], recovery_reserve_kw=point[1])
            # The LP depends on fleet energy targets, which the SOC intervention preserves.
            key = (row["event_probability"], row["target_remaining_room_kw"], row["event"], point)
            if key not in refs:
                refs[key] = rolling_reference(case, forecast, params)
            metrics = simulate(case, forecast, aging, params, reference=refs[key])
            rows.append({k: row[k] for k in [*ID, "probability"]} | dict(
                predicted_room_kw=room, predicted_high_soc_share=exposure, bucket=bucket,
                risk=point[0], reserve_kw=point[1], **metrics))
        if i % 160 == 0 or i == len(entries):
            print(f"{label} branches complete: {i}/{len(entries)}", flush=True)
    df = pd.DataFrame(rows)
    assert not df.duplicated([*ID, *KEYS]).any()
    return df


def select(dev, cuts, output):
    cases = dev.sort_values(ID).drop_duplicates(ID)
    indices = cases.set_index(ID).index
    buckets = cases.bucket.to_numpy()
    observed = sorted(set(buckets))
    weights = cases.probability.to_numpy()
    by_point = {
        point: rows.set_index(ID).reindex(indices)
        for point, rows in dev.groupby(KEYS)
    }
    columns = ["unmet_kwh", "qualified", "served_ratio", "mean_soh_loss"]
    cube = np.array([by_point[point][columns].to_numpy(dtype=float) for point in POINTS])
    assert not np.isnan(cube).any()
    cap = weighted_cvar(cube[POINTS.index(BASELINE), :, 0], weights)+MARGIN

    def rank(mapping):
        chosen = np.array([mapping[b] for b in buckets])
        values = cube[chosen, np.arange(len(buckets))]
        cvar = weighted_cvar(values[:, 0], weights)
        if cvar > cap+TOL:
            return None
        qual, service, soh = np.average(values[:, 1:], weights=weights, axis=0)
        # Round only for ordering so numerical noise cannot outweigh a genuine metric tie.
        return (-round(float(qual), 10), round(cvar, 9), round(float(soh), 10),
                -round(float(service), 10), len(set(mapping.values())))

    fixed, adaptive = [], []
    for idx in range(len(POINTS)):
        mapping = dict.fromkeys(observed, idx)
        score = rank(mapping)
        if score is not None:
            fixed.append((score, idx))
    best_fixed = POINTS[min(fixed)[1]]
    for idxs in product(range(len(POINTS)), repeat=len(observed)):
        mapping = dict(zip(observed, idxs))
        score = rank(mapping)
        if score is not None:
            adaptive.append((score, idxs))
    best = dict(zip(observed, (POINTS[i] for i in min(adaptive)[1])))
    for b in set(range(4))-set(best):
        best[b] = best_fixed
    lock = dict(forecast_room_cut_kw=cuts[0], forecast_high_soc_share_cut=cuts[1],
                baseline=BASELINE, fixed_best=best_fixed, adaptive={str(k): v for k, v in sorted(best.items())},
                development_cvar_cap_kwh=cap, baseline_development_cvar_kwh=cap-MARGIN,
                development_replay_sha256=digest(output/"development_replays.csv"))
    dump(output/"configuration_lock.json", lock)
    return best_fixed, best, lock


def metrics(rows):
    return {k: float(v) for k, v in summarize(rows).items()}


def summarize_policies(dev, held, fixed, adaptive, output):
    tables = []
    for split, panel in (("development", dev), ("holdout", held)):
        parts = {
            "fixed_reference": panel[(panel.risk == BASELINE[0]) & (panel.reserve_kw == BASELINE[1])],
            "fixed_best": panel[(panel.risk == fixed[0]) & (panel.reserve_kw == fixed[1])],
            "adaptive": panel.merge(pd.DataFrame([dict(bucket=b, risk=p[0], reserve_kw=p[1])
                                                  for b, p in adaptive.items()]), on=["bucket", *KEYS]),
        }
        baseline = metrics(parts["fixed_reference"])
        assert all(len(p) == len(parts["fixed_reference"]) for p in parts.values())
        for name, part in parts.items():
            pooled = metrics(part)
            tables.append(dict(split=split, policy=name, bucket="all", **pooled,
                               cvar_delta=pooled["cvar_unmet_kwh"]-baseline["cvar_unmet_kwh"]))
            for bucket, rows in part.groupby("bucket"):
                tables.append(dict(split=split, policy=name, bucket=bucket, **metrics(rows)))
    pd.DataFrame(tables).to_csv(output/"policy_summary.csv", index=False)


def run(source, factorial, locked, previous, output):
    aging, design, cases = construct(source, factorial, locked, previous, output, audit_physical=False)
    dump(output/"forecast_policy_protocol.json", dict(
        grid=POINTS, baseline=BASELINE, cvar_increment_limit_kwh=MARGIN,
        state="17:00 reference forecast minimum 18:00 hard-cap headroom and projected transferable-power-weighted high SOC share",
        state_binning="development-only balanced cut for each feature; one joint risk/reserve choice per bin",
        fitting="pooled qualification maximized under CVaR guardrail; ties favor lower CVaR, lower SOH, higher service, fewer distinct settings",
        paired_event_state="derive one state before event; identical across both realized branches",
        api_calls=0, manifest_sha256=digest(output/"manifest.csv")))
    development = [e for e in cases if e[0]["split"] == "development"]
    holdout = [e for e in cases if e[0]["split"] == "holdout"]
    assert len(development) == 240 and len(holdout) == 896
    states, cuts = build_state(development, aging.model.config.soc_threshold)
    print(f"Development forecast cuts: {cuts}.", flush=True)
    dev = replay(development, states, aging, lambda _: POINTS, "Development")
    assert len(dev) == len(development)*len(POINTS)
    dev.to_csv(output/"development_replays.csv", index=False)
    grid = [dict(risk=p[0], reserve_kw=p[1], **metrics(rows)) for p, rows in dev.groupby(KEYS)]
    pd.DataFrame(grid).to_csv(output/"development_fixed_grid.csv", index=False)
    fixed, adaptive, lock = select(dev, cuts, output)
    print(f"Locked before holdout: fixed {fixed}, adaptive {adaptive}.", flush=True)

    states, _ = build_state(holdout, aging.model.config.soc_threshold, cuts)
    held = replay(holdout, states, aging, lambda b: sorted({BASELINE, fixed, adaptive[b]}), "Holdout")
    held.to_csv(output/"holdout_replays.csv", index=False)
    summarize_policies(dev, held, fixed, adaptive, output)
    dump(output/"completion.json", dict(development_replays=len(dev), holdout_replays=len(held),
         lock_sha256=digest(output/"configuration_lock.json"), holdout_parents=design.query("split == 'holdout'").parent.nunique()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "factorial", "locked", "previous", "output"):
        parser.add_argument(name, type=Path)
    torch.set_num_threads(1)
    run(*(p.resolve() for p in vars(parser.parse_args()).values()))
