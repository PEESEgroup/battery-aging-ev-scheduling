"""Compare fixed and state-dependent reserves on paired derating/SOC scenarios."""
import argparse
from dataclasses import replace
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lock_recovery_holdout import digest, fingerprint
from run_directional_pinn_validation import Aging, TOL, read_case
from run_llm_timing_pilot import DEFAULTS, RECOVERY, REQUEST, START, TimingForecast, dump, rolling_reference, simulate
from screen_aging_derating_overlap import DERATING_KW, HOURS, summarize
from screen_recovery_departure import adjacent_bound, parent_cases


PROBABILITIES = (.25, .5)
ROOMS = (2.25, 4.25)
SHARES = (.25, .75)
RESERVES = (6., 10.)
RISK = .9
STATE = ["event_probability", "target_remaining_room_kw", "target_high_soc_share"]
LIMITS = {"served_ratio": .001, "cvar_unmet_kwh": .5, "mean_soh_loss": .0002}


class ScenarioForecast(TimingForecast):
    def __init__(self, base, caps, probability):
        self.donors, self.center, self.scale = base.donors, base.center, base.scale
        self.caps = caps
        self.events = round(4*probability)

    def scenario_paths(self, t, observed):
        paths = super().scenario_paths(t, observed)
        assert len(paths) == 4
        for hour, cap in zip(HOURS, self.caps):
            if hour > t:
                paths[:, hour-t, 3] = cap
        if RECOVERY > t:
            paths[-self.events:, RECOVERY-t, 3] -= DERATING_KW
        return paths


def make_fleet(fleet, ref, high_share, rng, threshold):
    eligible = (fleet.arrival <= START) & (fleet.departure > RECOVERY)
    potential = np.maximum(0., np.minimum(ref[:, START], fleet.power-ref[:, RECOVERY]))*eligible
    soc = fleet.soc + ref[:, :RECOVERY].sum(axis=1)/fleet.capacity
    inc = np.minimum(np.maximum(0., threshold+.02-soc),
                     np.maximum(0., 1.-fleet.demand/fleet.capacity-fleet.soc))
    high = soc >= threshold
    candidates = rng.permutation(np.flatnonzero((potential > TOL) & ~high & (soc+inc >= threshold)))
    total = max(potential.sum(), 1e-9)
    attained = np.r_[0., potential[candidates].cumsum()] + potential[high].sum()
    count = np.argmin(np.abs(attained/total-high_share))
    adjusted = fleet.soc.copy()
    idx = candidates[:count]
    adjusted[idx] += inc[idx]
    new_fleet = replace(fleet, soc=adjusted)
    assert np.allclose(new_fleet.target, fleet.target, atol=TOL, rtol=0.)
    active = (potential > TOL)
    headcount = np.count_nonzero(active & (soc+adjusted-fleet.soc >= threshold))/max(1, active.sum())
    return new_fleet, float(attained[count]/total), float(headcount), int(len(idx))


def score_table(rows):
    return {k: float(v) for k, v in summarize(rows).items()}


def eligible(candidate, baseline):
    return (candidate["served_ratio"] >= baseline["served_ratio"]-LIMITS["served_ratio"]-TOL
            and candidate["cvar_unmet_kwh"] <= baseline["cvar_unmet_kwh"]+LIMITS["cvar_unmet_kwh"]+TOL
            and candidate["mean_soh_loss"] <= baseline["mean_soh_loss"]+LIMITS["mean_soh_loss"]+TOL)


def learn(dev, output):
    """Train a single state lookup, with a cell-level guard and pooled guard."""
    low = dev[dev.reserve_kw == 6.]
    base = score_table(low)
    chosen = []
    for state, rows in dev.groupby(STATE, sort=True):
        a, b = (score_table(rows[rows.reserve_kw == k]) for k in RESERVES)
        pick = 10. if eligible(b, a) and b["qualified_rate"] > a["qualified_rate"]+TOL else 6.
        chosen.append(dict(zip(STATE, state)) | dict(reserve_kw=pick,
                      higher_reserve_qual_gain=b["qualified_rate"]-a["qualified_rate"],
                      higher_reserve_cvar_delta=b["cvar_unmet_kwh"]-a["cvar_unmet_kwh"],
                      higher_reserve_soh_delta=b["mean_soh_loss"]-a["mean_soh_loss"],
                      higher_reserve_service_delta=b["served_ratio"]-a["served_ratio"],
                      higher_reserve_pass=eligible(b, a)))
    policy = pd.DataFrame(chosen)
    adaptive = dev.merge(policy[STATE+["reserve_kw"]], on=STATE+["reserve_kw"])
    if not eligible(score_table(adaptive), base):
        policy["reserve_kw"] = 6.
    policy.to_csv(output/"locked_state_rule.csv", index=False)
    dump(output/"configuration_lock.json", dict(
        rule_sha256=digest(output/"locked_state_rule.csv"),
        development_replay_sha256=digest(output/"development_replays.csv"),
        manifest_sha256=digest(output/"manifest.csv")))
    return policy


def construct(source, factorial, locked, previous, output, audit_physical=True):
    output.mkdir(parents=True, exist_ok=False)
    prior = json.loads((factorial/"protocol.json").read_text(encoding="utf-8"))
    floor = prior["capacity_background_floor_kw"]
    old = pd.read_csv(previous/"manifest.csv")
    admitted = set(old.loc[(old.split == "development") & old.high_exposure, "parent"])
    assert len(admitted) == 15
    lock = json.loads((locked/"lock.json").read_text(encoding="utf-8"))
    assert digest(locked/"locked_manifest.csv") == lock["manifest_sha256"]

    def background(grid):
        grid = grid.copy()
        grid[:, 3] = np.maximum(grid[:, 3], floor)
        return grid

    parents, donors = [], []
    for _, _, _, path, (fleet, grid, _), split in parent_cases(source):
        if split != "development":
            continue
        grid = background(grid)
        donors.append(grid)
        key = fingerprint(fleet)
        if key in admitted:
            parents.append((split, key, path, fleet, grid))
    assert len(parents) == 15 and len(donors) == 21
    for row in pd.read_csv(locked/"locked_manifest.csv").query("margin == 'critical'").itertuples():
        path = locked/row.path
        assert digest(path/"optimizer_ev_sessions.csv") == row.ev_sha256
        assert digest(path/"optimizer_grid.csv") == row.grid_sha256
        fleet, grid, _ = read_case(path)
        assert fingerprint(fleet) == row.parent_fingerprint
        parents.append(("holdout", row.parent_fingerprint, path, fleet, background(grid)))
    expected_parents = 15+sum(lock["locked_parents"].values())
    assert len(parents) == len({p[1] for p in parents}) == expected_parents
    aging = Aging(source)
    threshold = aging.model.config.soc_threshold
    temp = np.full(24, 25.)
    base = TimingForecast(donors)
    contexts, scale_donors = [], []
    for split, key, path, fleet, grid in parents:
        ref = rolling_reference((fleet, grid, temp, None), base, DEFAULTS)[0]
        for room in ROOMS:
            caps = np.minimum(grid[list(HOURS), 3], ref[:, list(HOURS)].sum(axis=0)+
                              np.array([room+DERATING_KW, 3.5]))
            contexts.append((split, key, path, fleet, grid, ref, room, caps))
            if split == "development":
                stressed = grid.copy()
                stressed[list(HOURS), 3] = caps
                stressed[RECOVERY, 3] -= DERATING_KW
                scale_donors.append(stressed)
    scale = TimingForecast(scale_donors)
    base.center, base.scale = scale.center, scale.scale
    dump(output/"protocol.json", dict(
        state_grid=dict(zip(STATE, (PROBABILITIES, ROOMS, SHARES))),
        reserve_grid_kw=RESERVES, risk_fixed=RISK, request_kw=REQUEST, derating_kw=DERATING_KW,
        temperature_c=25., high_soc_threshold=threshold, soc_recovery_target=threshold+.02,
        exposure="target share of transferable power at high SOC; record realized power and vehicle shares",
        guardrails_relative_to_fixed_6=LIMITS,
        rule="choose 10 in a state only if development qualification improves and service/CVaR/SOH guardrails hold; apply pooled guard",
        forecast="original four nearest paths, one or two carrying the 18:00 event; observed capacity always hard",
        fixed_cap_behavior="reserve applies to reference only; planning capacity clips at zero",
        prior_lock_sha256=digest(locked/"lock.json"), pinn_sha256=digest(aging.path), api_calls=0))

    manifest, cases = [], []
    for split, key, path, fleet, grid, ref, room, caps in contexts:
        for probability, share in product(PROBABILITIES, SHARES):
            forecast = ScenarioForecast(base, caps, probability)
            candidate, power_share, car_share, changed = make_fleet(
                fleet, ref, share, np.random.default_rng(int(key[:12], 16)), threshold)
            for event in (False, True):
                actual = grid.copy()
                actual[list(HOURS), 3] = caps
                if event:
                    actual[RECOVERY, 3] -= DERATING_KW
                assert np.all(actual[:, 3] >= 0)
                case = candidate, actual, temp, None
                row = dict(split=split, parent=key, source=str(path), event=event,
                           probability=probability if event else 1.-probability,
                           event_probability=probability, target_remaining_room_kw=room,
                           target_high_soc_share=share, realized_high_soc_power_share=power_share,
                           realized_high_soc_vehicle_share=car_share, adjusted_vehicles=changed,
                           cap_18_kw=actual[RECOVERY, 3], cap_19_kw=actual[RECOVERY+1, 3])
                if audit_physical:
                    check = rolling_reference(case, forecast, dict(DEFAULTS, lambda_risk=RISK))[0]
                    row["physical_kw"] = adjacent_bound(candidate, actual, check)
                    row["physical"] = row["physical_kw"] >= REQUEST-TOL
                    row["realized_reference_room_kw"] = actual[RECOVERY, 3]-check[:, RECOVERY].sum()
                manifest.append(row)
                cases.append((row, case, forecast))
    design = pd.DataFrame(manifest)
    assert design.groupby("parent").split.nunique().eq(1).all() and design.groupby("parent").size().eq(16).all()
    design.to_csv(output/"manifest.csv", index=False)
    if not audit_physical:
        return aging, design, cases
    design.groupby(["split", *STATE, "event"]).agg(
        parents=("parent", "nunique"), physical=("physical", "sum"),
        actual_room_kw=("realized_reference_room_kw", "mean"),
        high_soc_power_share=("realized_high_soc_power_share", "mean"),
        high_soc_vehicle_share=("realized_high_soc_vehicle_share", "mean"),
    ).reset_index().to_csv(output/"physical_preflight.csv", index=False)
    print("Built 8 state cells and completed physical preflight.", flush=True)
    return aging, design, cases


def run(source, factorial, locked, previous, output):
    aging, design, cases = construct(source, factorial, locked, previous, output)
    # Each fixed policy is evaluated once; the adaptive replay uses the chosen fixed row for each state.
    tables = []
    for split in ("development", "holdout"):
        rows = []
        for i, (row, case, forecast) in enumerate((c for c in cases if c[0]["split"] == split), 1):
            for reserve in RESERVES:
                params = dict(DEFAULTS, lambda_risk=RISK, recovery_reserve_kw=reserve)
                metrics = simulate(case, forecast, aging, params)
                rows.append({k: row[k] for k in ["parent", "event", "probability", *STATE]} |
                            dict(reserve_kw=reserve, **metrics))
            if i % 160 == 0:
                print(f"{split} branches complete: {i}", flush=True)
        panel = pd.DataFrame(rows)
        assert len(panel) == 2*int(design.split.eq(split).sum())
        assert not panel.duplicated(["parent", "event", *STATE, "reserve_kw"]).any()
        panel.to_csv(output/f"{split}_replays.csv", index=False)
        if split == "development":
            policy = learn(panel, output)
            print(f"Locked state rule: {int(policy.reserve_kw.eq(10).sum())}/8 cells use 10 kW.", flush=True)
        adaptive = panel.merge(policy[STATE+["reserve_kw"]], on=STATE+["reserve_kw"])
        assert len(adaptive)*2 == len(panel)
        for name, part in (("fixed_6", panel[panel.reserve_kw == 6.]),
                           ("fixed_10", panel[panel.reserve_kw == 10.]), ("adaptive", adaptive)):
            tables.append(dict(split=split, policy=name, cell="all", **score_table(part)))
            for state, cell in part.groupby(STATE, sort=True):
                tables.append(dict(split=split, policy=name, cell="state", **dict(zip(STATE, state)),
                                   **score_table(cell)))
    pd.DataFrame(tables).to_csv(output/"policy_summary.csv", index=False)
    dump(output/"completion.json", dict(development_replays=480, holdout_replays=1792,
         lock_sha256=digest(output/"configuration_lock.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "factorial", "locked", "previous", "output"):
        parser.add_argument(name, type=Path)
    torch.set_num_threads(1)
    run(*(p.resolve() for p in vars(parser.parse_args()).values()))
