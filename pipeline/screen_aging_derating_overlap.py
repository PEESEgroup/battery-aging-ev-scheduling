"""Screen paired recovery-window and post-recovery derating with the calibrated PINN."""
import argparse
from dataclasses import replace
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


HOURS = (RECOVERY, RECOVERY+1)
EVENT_HOUR = {"overlap": RECOVERY, "separated": RECOVERY+1}
PROBABILITY = .25
DERATING_KW = 1.25
NOMINAL_ROOM_KW = 3.5
RISKS = (0., .3, .6, .9)
RESERVES = (0., 1.5, 2.5, 4., 6.)
ALPHA = .95


class DeratingForecast(TimingForecast):
    """Keep the original nearest-path forecast; one of four paths carries the event."""
    def __init__(self, base, caps, hour):
        self.donors, self.center, self.scale = base.donors, base.center, base.scale
        self.caps, self.hour = caps, hour

    def scenario_paths(self, t, observed):
        paths = super().scenario_paths(t, observed)
        assert len(paths) == 4
        for hour, cap in zip(HOURS, self.caps):
            if hour > t:
                paths[:, hour-t, 3] = cap
        if self.hour > t:
            paths[-1, self.hour-t, 3] -= DERATING_KW
        return paths


def weighted_cvar(loss, weight):
    idx = np.argsort(-loss)
    p = weight[idx]/weight.sum()
    mass = np.minimum(p, np.maximum(0., (1-ALPHA)-np.r_[0., np.cumsum(p)[:-1]]))
    return float(mass @ loss[idx] / (1-ALPHA))


def summarize(group):
    p = group.probability.to_numpy()
    return pd.Series(dict(
        parents=group.parent.nunique(),
        qualified_rate=np.average(group.qualified, weights=p),
        served_ratio=np.average(group.served_ratio, weights=p),
        cvar_unmet_kwh=weighted_cvar(group.unmet_kwh.to_numpy(), p),
        mean_soh_loss=np.average(group.mean_soh_loss, weights=p),
        event_qualified=group.loc[group.event, "qualified"].mean(),
        event_soh_loss=group.loc[group.event, "mean_soh_loss"].mean(),
    ))


def construct(source, factorial, locked, output, soc_stress=False):
    output.mkdir(parents=True, exist_ok=False)
    prior = json.loads((factorial/"protocol.json").read_text(encoding="utf-8"))
    floor = prior["capacity_background_floor_kw"]
    lock = json.loads((locked/"lock.json").read_text(encoding="utf-8"))
    assert digest(locked/"locked_manifest.csv") == lock["manifest_sha256"]
    calibration = source/"real_aging_temporal_pinn/real_aging_temporal_pinn_finetune_interval_dual_head.csv"
    measured = pd.read_csv(calibration)
    temp_values = measured.loc[measured.is_real_data == 1, "temperature_c"].unique()
    assert len(temp_values) == 1
    temp_c = float(temp_values[0])

    def background(grid):
        grid = grid.copy()
        grid[:, 3] = np.maximum(grid[:, 3], floor)
        return grid

    parents = []
    dev = pd.read_csv(factorial/"manifest.csv")
    dev = set(dev.loc[dev.split == "development", "parent_sha256"])
    for _, _, key, path, (fleet, grid, _), split in parent_cases(source):
        if key in dev:
            assert split == "development"
            parents.append(("development", fingerprint(fleet), path, fleet, background(grid)))
    assert len(parents) == 21
    base = TimingForecast([p[4] for p in parents])
    heldout = pd.read_csv(locked/"locked_manifest.csv")
    for row in heldout[heldout.margin == "critical"].itertuples():
        path = locked/row.path
        assert digest(path/"optimizer_grid.csv") == row.grid_sha256
        assert digest(path/"optimizer_ev_sessions.csv") == row.ev_sha256
        fleet, grid, _ = read_case(path)
        assert fingerprint(fleet) == row.parent_fingerprint
        parents.append(("holdout", row.parent_fingerprint, path, fleet, background(grid)))
    assert len(parents) == len({p[1] for p in parents}) == 77
    temp = np.full(24, temp_c)
    aging = Aging(source)
    soc_threshold = float(aging.model.config.soc_threshold)
    stress_target = soc_threshold+.02
    assert stress_target <= measured.loc[measured.is_real_data == 1, "soc"].max()
    contexts, scale_donors, soc_changes = [], [], []
    for split, parent, path, fleet, grid in parents:
        ref, _, _ = rolling_reference((fleet, grid, temp, None), base, DEFAULTS)
        eligible = (fleet.arrival <= START) & (fleet.departure > RECOVERY)
        potential = np.maximum(0., np.minimum(ref[:, START], fleet.power-ref[:, RECOVERY]))*eligible
        # SOC before the recovery slot under the common reference.
        recovery_soc = fleet.soc+ref[:, :RECOVERY].sum(axis=1)/fleet.capacity
        if soc_stress:
            # Raise SOC only where the original energy request still fits the battery.
            room = np.maximum(0., 1.-fleet.demand/fleet.capacity-fleet.soc)
            increment = np.minimum(np.maximum(0., stress_target-recovery_soc), room)*(potential > TOL)
            changed = replace(fleet, soc=fleet.soc+increment)
            assert np.allclose(changed.target, fleet.target, atol=TOL, rtol=0.)
            soc_changes.extend(dict(split=split, parent=parent, vehicle_index=int(i),
                                    original_soc=fleet.soc[i], stressed_soc=changed.soc[i])
                               for i in np.flatnonzero(increment > TOL))
            fleet = changed
            recovery_soc = recovery_soc+increment
        exposure = float(potential @ recovery_soc/max(potential.sum(), 1e-9))
        exposed_share = float(potential[recovery_soc >= soc_threshold].sum()/max(potential.sum(), 1e-9))
        caps = np.minimum(grid[list(HOURS), 3], ref[:, list(HOURS)].sum(axis=0)+NOMINAL_ROOM_KW)
        contexts.append((split, parent, path, fleet, grid, exposure, exposed_share, caps))
        if split == "development":
            for hour in HOURS:
                stress = grid.copy()
                stress[list(HOURS), 3] = caps
                stress[hour, 3] -= DERATING_KW
                scale_donors.append(stress)
    cutoff = float(np.median([c[5] for c in contexts if c[0] == "development" and c[5] > 0]))
    norm = TimingForecast(scale_donors)
    base.center, base.scale = norm.center, norm.scale
    dump(output/"protocol.json", dict(
        nominal_room_kw=NOMINAL_ROOM_KW, derating_kw=DERATING_KW, probability=PROBABILITY,
        event_hours=EVENT_HOUR, request_kw=REQUEST, alpha=ALPHA, temperature_c=temp_c,
        age_exposure="common-reference pre-recovery SOC weighted by physically transferable same-car power",
        high_exposure_cutoff=cutoff, pinn_high_soc_threshold=soc_threshold,
        controlled_soc_stress=soc_stress, recovery_soc_target=stress_target if soc_stress else None,
        stress_high_definition="at least 50% of transferable power associated with SOC at or above PINN threshold" if soc_stress else None,
        forecast="existing nearest-four paths and hard min cap; one future path derated; actual cap only observed at event hour",
        normalization="shared across timings, from development caps only",
        risk_grid=RISKS, reserve_grid_kw=RESERVES,
        selection="within 0.1 pp of default mean service, maximize qualification then minimize unmet CVaR then SOH loss",
        holdout="physical diagnostics only; no PINN replay or tuning",
        holdout_source_sha256=lock["manifest_sha256"], calibration_sha256=digest(calibration),
        pinn_sha256=digest(aging.path), api_calls=0))

    cases, manifests = [], []
    for split, parent, path, fleet, original, exposure, exposed_share, caps in contexts:
        high = exposed_share >= .5 if soc_stress else exposure >= cutoff
        for regime, hour in EVENT_HOUR.items():
            forecast = DeratingForecast(base, caps, hour)
            for event, probability in ((False, 1-PROBABILITY), (True, PROBABILITY)):
                grid = original.copy()
                grid[list(HOURS), 3] = caps
                if event:
                    grid[hour, 3] -= DERATING_KW
                assert np.all(grid[:, 3] >= 0)
                case = fleet, grid, temp, None
                ref = rolling_reference(case, forecast, DEFAULTS)
                bound = adjacent_bound(fleet, grid, ref[0])
                row = dict(split=split, parent=parent, source=str(path), regime=regime, event=event,
                    probability=probability, high_exposure=high, exposure_soc=exposure,
                    pinn_high_soc_share=exposed_share, nominal_cap_18_kw=caps[0], nominal_cap_19_kw=caps[1],
                    actual_cap_18_kw=grid[RECOVERY, 3], actual_cap_19_kw=grid[RECOVERY+1, 3],
                    actual_recovery_room_kw=grid[RECOVERY, 3]-ref[0][:, RECOVERY].sum(),
                    physical_kw=bound, physical=bound>=REQUEST-TOL)
                manifests.append(row)
                cases.append((row, case, forecast, ref))
    design = pd.DataFrame(manifests)
    assert design.groupby("parent").size().eq(4).all()
    assert design.groupby("parent").split.nunique().eq(1).all()
    design.to_csv(output/"manifest.csv", index=False)
    if soc_stress:
        pd.DataFrame(soc_changes).to_csv(output/"soc_stress_changes.csv", index=False)
    preflight = design.groupby(["split", "high_exposure", "regime", "event"]).agg(
        parents=("parent", "size"), physical=("physical", "sum"),
        mean_recovery_room_kw=("actual_recovery_room_kw", "mean"),
        mean_physical_kw=("physical_kw", "mean"), mean_exposure_soc=("exposure_soc", "mean"),
        mean_pinn_high_soc_share=("pinn_high_soc_share", "mean")).reset_index()
    preflight.to_csv(output/"physical_preflight.csv", index=False)
    print(preflight.to_string(index=False), flush=True)
    return aging, design, cases


def run(source, factorial, locked, output, soc_stress=False):
    aging, _, cases = construct(source, factorial, locked, output, soc_stress)
    records = []
    for regime in EVENT_HOUR:
        for row, case, forecast, ref in cases:
            if row["regime"] != regime or row["split"] != "development":
                continue
            for risk, reserve in product(RISKS, RESERVES):
                params = dict(DEFAULTS, lambda_risk=risk, recovery_reserve_kw=reserve)
                reference = ref if (risk, reserve) == (.3, 0.) else None
                metrics = simulate(case, forecast, aging, params, reference=reference)
                records.append(dict(parent=row["parent"], high_exposure=row["high_exposure"], regime=regime,
                    event=row["event"], probability=row["probability"], risk=risk, reserve_kw=reserve, **metrics))
        print(f"{regime} scan complete", flush=True)
    scan = pd.DataFrame(records)
    assert len(scan) == 21*4*len(RISKS)*len(RESERVES)
    scan.to_csv(output/"development_scan.csv", index=False)
    results = []
    for panel, part in (("all", scan), ("high_exposure", scan[scan.high_exposure])):
        for (regime, risk, reserve), rows in part.groupby(["regime", "risk", "reserve_kw"]):
            results.append(dict(panel=panel, regime=regime, risk=risk, reserve_kw=reserve, **summarize(rows).to_dict()))
    grid = pd.DataFrame(results)
    key = ["panel", "regime"]
    default = grid[(grid.risk == .3) & (grid.reserve_kw == 0.)][key+["served_ratio"]]
    grid = grid.merge(default.rename(columns={"served_ratio": "default_service"}), on=key)
    grid["service_ok"] = grid.served_ratio >= grid.default_service-SERVICE_TOL-TOL
    grid.to_csv(output/"development_grid_summary.csv", index=False)
    winners = grid[grid.service_ok].sort_values(
        ["qualified_rate", "cvar_unmet_kwh", "mean_soh_loss", "risk", "reserve_kw"],
        ascending=[False, True, True, True, True]).drop_duplicates(key)
    winners.to_csv(output/"winners.csv", index=False)
    cross = winners[["panel", "regime", "risk", "reserve_kw"]].rename(columns={"regime": "source_regime"}).merge(
        grid, on=["panel", "risk", "reserve_kw"]).rename(columns={"regime": "target_regime"})
    cross.to_csv(output/"cross_replay.csv", index=False)
    dump(output/"completion.json", dict(replays=len(scan), holdout_pinn_replays=0,
        cross_rows=len(cross), manifest_sha256=digest(output/"manifest.csv")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "factorial", "locked", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--soc-stress", action="store_true")
    torch.set_num_threads(1)
    args = parser.parse_args()
    run(args.source.resolve(), args.factorial.resolve(), args.locked.resolve(), args.output.resolve(), args.soc_stress)
