"""Build a paired pilot that varies the timing of capacity and price stress."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

HOURS = (17, 18)
SEED = 42


def energy_bound(ev, caps):
    a, d = (ev[k].to_numpy(int) for k in ("arrival_step", "departure_step"))
    demand = ev.required_energy_kwh.to_numpy(float)
    power = ev.max_charging_power_kw.to_numpy(float)
    headroom = (1 - ev.init_soc.to_numpy(float)) * ev.battery_capacity_kwh.to_numpy(float)
    target = np.minimum(demand, headroom)
    ids, ts = np.where((np.arange(24) >= a[:, None]) & (np.arange(24) < d[:, None]))
    j = np.arange(len(ids))
    ones = np.ones(len(j))
    hourly = coo_matrix((ones, (ts, j)), shape=(24, len(j)))
    vehicle = coo_matrix((ones, (ids, j)), shape=(len(ev), len(j)))
    fit = linprog(-ones, A_ub=vstack((hourly, vehicle)), b_ub=np.r_[caps, target],
                  bounds=list(zip(np.zeros(len(j)), power[ids])), method="highs")
    if not fit.success:
        raise RuntimeError(fit.message)
    return -float(fit.fun), target


def write_cases(source, dest):
    paths = sorted(source.glob("scenario_*"))
    assert [p.name for p in paths] == [f"scenario_{i:02d}" for i in range(10)]
    manifest = []
    for idx, folder in enumerate(paths):
        evfile = folder/"optimizer_ev_sessions.csv"
        gridfile = folder/"optimizer_grid.csv"
        ev, grid = pd.read_csv(evfile), pd.read_csv(gridfile)
        assert list(grid.t) == list(range(24))
        caps = np.maximum(grid.ev_capacity_limit_kw.to_numpy(float), 1500.)
        caps[list(HOURS)] = 1500.
        no_cap, target = energy_bound(ev, caps)
        potential = [np.minimum(ev.max_charging_power_kw, target)[
            (ev.arrival_step <= t) & (ev.departure_step > t)].sum() for t in HOURS]
        cap = round(.25 * min(potential), 3)
        assert 0 < cap < min(potential)
        qlo, qhi = np.quantile(grid.price_kwh, [.25, .75])
        assert qhi > qlo
        design = [("control", None, None)]
        design += [(f"cap_{c}_price_{p}", c, p) for c,p in itertools.product(HOURS, repeat=2)]
        order = np.random.default_rng(SEED+idx).permutation(len(design))
        for position in order:
            cell, c, p = design[position]
            data = grid.copy()
            data["ev_capacity_limit_kw"] = caps
            if c is not None:
                data.loc[data.t == c,"ev_capacity_limit_kw"] = cap
                for hour in HOURS:
                    data.loc[data.t == hour, "price_kwh"] = qhi if p == hour else qlo
                data["price_mwh"] = 1000*data.price_kwh
                if "price_emission_score" in data:
                    data["price_emission_score"] = data.price_mwh * data.emission_ton_per_mwh
            out = dest/cell/folder.name
            out.mkdir(parents=True,exist_ok=True)
            copyfile(evfile,out/evfile.name)
            data.to_csv(out/gridfile.name,index=False)
            bound,_ = energy_bound(ev,data.ev_capacity_limit_kw.to_numpy())
            assert bound <= no_cap + 1e-6
            manifest.append(dict(cell=cell,parent=folder.name,split="development" if idx<7 else "pilot_test",
                                 run_order=int(np.where(order == position)[0][0])+1,
                                 cap_hour=c,high_price_hour=p,cap_kw=cap if c else None,
                                 price_low_per_kwh=qlo if p else None,price_high_per_kwh=qhi if p else None,
                                 fleet_power_at_17_kw=potential[0],fleet_power_at_18_kw=potential[1],
                                 offline_service_ceiling=bound/ev.required_energy_kwh.sum(),
                                 offline_capacity_service_loss_kwh=max(0.,no_cap-bound),
                                 ev_source_sha256=hashlib.sha256(evfile.read_bytes()).hexdigest(),
                                 grid_source_sha256=hashlib.sha256(gridfile.read_bytes()).hexdigest()))
    result = pd.DataFrame(manifest)
    assert len(result)==50 and result.groupby("parent").size().eq(5).all()
    result.to_csv(dest/"manifest.csv",index=False)
    print(result.groupby("cell").agg(parent_cases=("parent","size"),min_cap=("cap_kw","min"),
          max_cap=("cap_kw","max"),avg_ceiling=("offline_service_ceiling","mean"),
          mean_capacity_loss=("offline_capacity_service_loss_kwh","mean")).to_string())


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source",type=Path)
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    write_cases(args.source,args.output)



