"""Replay the four manuscript input groups with the frozen dual-head PINN."""
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
import torch

from nonanticipative_scheduler import Fleet, Forecast

GROUPS = ("joint", "evening_peak_cluster", "sustained_high_price", "capacity_shortage")
SIGNALS = ["price_kwh", "emission_ton_per_kwh", "congestion_intensity", "ev_capacity_limit_kw"]
START, RECOVERY = 17, 18
REQUESTS = (0, 0.25, 0.5, 1, 2, 5, 10, 20, 40)
TOL = 1e-7


def read_case(path):
    ev = pd.read_csv(path / "optimizer_ev_sessions.csv")
    grid = pd.read_csv(path / "optimizer_grid.csv")
    fleet = Fleet(*(ev[col].to_numpy(dtype) for col, dtype in (
        ("arrival_step", int), ("departure_step", int), ("required_energy_kwh", float),
        ("max_charging_power_kw", float), ("battery_capacity_kwh", float), ("init_soc", float))))
    assert len(grid) == 24 and (fleet.arrival >= 0).all() and (fleet.departure <= 24).all()
    assert (fleet.departure > fleet.arrival).all()
    return fleet, grid[SIGNALS].to_numpy(), grid.get("temperature_c", pd.Series(25., index=grid.index)).to_numpy()


def plan(fleet, first, caps, score, prefix, target=None, direction=None):
    hours = np.arange(24)
    active = ((hours >= first) & (hours < fleet.departure[:, None])
              & (hours >= fleet.arrival[:, None]))
    if target is None:
        active &= fleet.arrival[:, None] <= first
    ids, ts = np.where(active)
    n = len(fleet.arrival)
    positions = np.arange(len(ids))
    hourly = coo_matrix((np.ones(len(ids)), (ts-first, positions)), shape=(24-first, len(ids))).tocsc()
    vehicle = coo_matrix((np.ones(len(ids)), (ids, positions)), shape=(n, len(ids))).tocsc()
    output = prefix.copy()
    if not len(ids):
        return output
    remaining = (fleet.target if target is None else target) - prefix.sum(axis=1)
    assert remaining.min() >= -TOL
    remaining = np.maximum(remaining, 0)
    objective = score[ts-first] - 2 if direction is None else -direction * (ts == first).astype(float)
    if target is None:
        from scipy.sparse import vstack
        sol = linprog(objective, A_ub=vstack((hourly, vehicle)), b_ub=np.r_[caps, remaining],
                      bounds=list(zip(np.zeros(len(ids)), fleet.power[ids])), method="highs")
    else:
        sol = linprog(objective, A_ub=hourly, b_ub=caps, A_eq=vehicle, b_eq=remaining,
                      bounds=list(zip(np.zeros(len(ids)), fleet.power[ids])), method="highs")
    if not sol.success:
        raise RuntimeError(sol.message)
    output[ids, ts] = sol.x
    return output


class Aging:
    def __init__(self, source):
        sys.path.insert(0, str(source))
        from train_battery_degradation_pinn_temporal_dual_head import TemporalBatteryDegradationPINN
        self.path = source / "processed/battery_degradation_pinn_temporal_dual_head_real_calibrated.pt"
        self.model = TemporalBatteryDegradationPINN.load(self.path)
        self.model.cycling_head.eval()
        self.model.calendar_head.eval()
        self.mean = torch.as_tensor(self.model.scaler.mean_, dtype=torch.float32)
        self.std = torch.as_tensor(self.model.scaler.std_, dtype=torch.float32)
        self.limit = 1 - self.model.config.min_soh
        assert self.model.config.dt_hours == 1
        test = np.array([[.35, 7, 60, 25, 0, 0, .35, 7/60]], np.float32)
        assert np.isclose(self.predict(test)[0], self.model.predict_increment(.35, 7, 60)["delta_soh_total"])

    @torch.no_grad()
    def predict(self, features):
        x = self.model.scaler.transform(features.astype(np.float32))
        return self.model._forward_from_scaled(torch.as_tensor(x), self.mean, self.std)["delta_soh_total"].numpy().ravel()

    def losses(self, fleet, power, temp):
        n = len(fleet.arrival)
        soc = fleet.soc[:, None] + (np.cumsum(power, axis=1) - power) / fleet.capacity[:, None]
        rate = power / fleet.capacity[:, None]
        high = np.zeros(n)
        last = fleet.arrival - 1
        features, owners = [], []
        for t in range(24):
            ids = np.flatnonzero((fleet.arrival <= t) & (fleet.departure > t))
            rest = np.maximum(0, t - np.maximum(last[ids] + 1, fleet.arrival[ids]))
            dwell = high[ids] + rest * (soc[ids, t] >= self.model.config.soc_threshold)
            mean_soc = soc[ids, t].copy()
            mean_rate = rate[ids, t].copy()
            for i, ev in enumerate(ids):
                first = max(fleet.arrival[ev], t-3)
                if first < t:
                    mean_soc[i] = soc[ev, first:t].mean()
                    mean_rate[i] = rate[ev, first:t].mean()
            features.append(np.column_stack((soc[ids,t], power[ids,t], fleet.capacity[ids],
                                             np.full(len(ids), temp[t]), rest, dwell, mean_soc, mean_rate)))
            owners.append(ids)
            high[ids] = dwell + ((soc[ids,t] + rate[ids,t]) >= self.model.config.soc_threshold)
            last[ids] = t
        return np.bincount(np.concatenate(owners), weights=self.predict(np.concatenate(features)), minlength=n)


def reference(fleet, grid, forecast, *, risk_terms=True):
    power = np.zeros((len(fleet.arrival), 24))
    planned = None
    forecast_caps = None
    for t in range(24):
        caps, score = forecast.build(t, grid[:t+1], 4, risk_terms=risk_terms)
        proposal = plan(fleet, t, caps, score, power)
        if t == START:
            planned, forecast_caps = proposal, caps
        power[:,t] = proposal[:,t]
    return power, planned, forecast_caps


def swap(fleet, planned, caps, direction, request, aging, temp, *, enforce_aging=True):
    # Preserve projected per-vehicle energy by exchanging power between adjacent slots.
    p, q = planned[:,START], planned[:,RECOVERY]
    room = np.minimum(fleet.power-p, q) if direction == 1 else np.minimum(p, fleet.power-q)
    room *= (fleet.arrival <= START) & (fleet.departure > RECOVERY)
    if enforce_aging:
        base_loss = aging.losses(fleet, planned, temp)
        room[base_loss > aging.limit + TOL] = 0
    grid_room = caps[0]-p.sum() if direction == 1 else caps[1]-q.sum()
    remaining = max(0, min(request, grid_room))
    shifts = np.zeros(len(room))
    for i in np.argsort(-room):
        amount = min(remaining, room[i])
        if amount <= TOL:
            break
        candidate = planned[i:i+1].copy()
        single = Fleet(*(x[i:i+1] for x in vars(fleet).values()))
        candidate[0,START] += direction * amount
        candidate[0,RECOVERY] -= direction * amount
        if not enforce_aging or aging.losses(single, candidate, temp)[0] <= base_loss[i] + TOL:
            shifts[i] = amount
            remaining -= amount
    return shifts


def replay(fleet, grid, reference_power, proposal, caps, direction, request, aging, temp, baseline_loss, *, enforce_aging=True):
    forecast_temp = np.r_[temp[:START+1], np.repeat(temp[START], 23-START)]
    shifts = swap(fleet, proposal, caps, direction, request, aging, forecast_temp, enforce_aging=enforce_aging)
    actual = reference_power.copy()
    actual[:,START] += direction * shifts
    # At recovery, constrain the paired adjustment to the observed power and grid limits.
    next_power = actual[:,RECOVERY]
    recovery = np.minimum(shifts, next_power if direction == 1 else fleet.power-next_power)
    if direction < 0 and recovery.sum() > 0:
        recovery *= min(1., max(0., grid[RECOVERY,3]-next_power.sum()) / recovery.sum())
    actual[:,RECOVERY] -= direction * recovery
    loss = aging.losses(fleet, actual, temp)
    deviation = direction * (actual[:,START].sum() - proposal[:,START].sum())
    service = np.max(np.abs(actual.sum(axis=1)-reference_power.sum(axis=1))) <= TOL
    health = np.all(loss <= baseline_loss + TOL) and np.all(loss <= aging.limit + TOL)
    physical = np.all(actual >= -TOL) and np.all(actual <= fleet.power[:,None]+TOL) and np.all(actual.sum(axis=0) <= grid[:,3]+TOL)
    return actual, float(deviation), bool(service), bool(health), bool(physical)


def evaluate(source, output, capacity_floor=0., shortage_cap=None, groups=GROUPS):
    aging = Aging(source)
    records, online, provenance = [], [], []
    for group in groups:
        paths = sorted((source / ("optimization_inputs_"+group)).glob("scenario_*"))
        assert len(paths) == 10
        cases = [read_case(path) for path in paths]
        for _, grid, _ in cases:
            grid[:, 3] = np.maximum(grid[:, 3], capacity_floor)
            if shortage_cap is not None and group == "capacity_shortage":
                grid[17:21, 3] = np.minimum(grid[17:21, 3], shortage_cap)
        forecast = Forecast([case[1] for case in cases[:4]])
        for k in range(4,10):
            fleet, grid, temp = cases[k]
            ref, proposal, caps = reference(fleet, grid, forecast)
            assert np.allclose(ref[:,START], proposal[:,START])
            base_loss = aging.losses(fleet, ref, temp)
            eligible = bool(np.all(base_loss <= aging.limit + TOL))
            partition = "calibration" if k < 7 else "test"
            row = dict(group=group, scenario=paths[k].name, split=partition, start_hour=START, duration_h=1, capacity_floor_kw=capacity_floor, shortage_cap_kw=shortage_cap,
                       served_ratio=ref.sum()/fleet.demand.sum(), floor_pass=eligible,
                       reference_max_loss=base_loss.max())
            for direction in (1,-1):
                ctx = dict(**row, direction="up" if direction == 1 else "down")
                upper = lower = np.nan
                if eligible:
                    prefix = ref.copy()
                    prefix[:,START:] = 0
                    extreme = plan(fleet, START, grid[START:,3], np.zeros(24-START), prefix,
                                   target=ref.sum(axis=1), direction=direction)
                    upper = max(0., direction*(extreme[:,START].sum()-ref[:,START].sum()))
                    lower = 0.
                    for fraction in (1., .75, .5, .25, .1, .05, .01):
                        candidate = ref + fraction*(extreme-ref)
                        losses = aging.losses(fleet, candidate, temp)
                        if np.all(losses <= np.minimum(base_loss, aging.limit)+TOL):
                            lower = fraction * upper
                            break
                for request in REQUESTS:
                    actual, deviation, service, health, physical = replay(
                        fleet, grid, ref, proposal, caps, direction, request, aging, temp, base_loss)
                    delivered = eligible and service and health and physical and deviation >= request-TOL
                    if delivered:
                        assert deviation <= upper+1e-6
                        lower = max(lower, deviation)
                    online.append(dict(**ctx, request_kw=request, actual_kw=deviation,
                                       service_pass=service, health_pass=health, physical_pass=physical, delivered=delivered))
                records.append(dict(**ctx, lower_kw=lower, physical_upper_kw=upper))
            print(group, paths[k].name, partition, eligible, round(base_loss.max(),4), flush=True)
        for path in paths:
            for file in ("optimizer_ev_sessions.csv", "optimizer_grid.csv"):
                f = path/file
                provenance.append(dict(path=str(f), sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    provenance.append(dict(path=str(aging.path), sha256=hashlib.sha256(aging.path.read_bytes()).hexdigest()))
    out = pd.DataFrame(records)
    events = pd.DataFrame(online)
    summary = []
    for (group,direction), data in out[out.split=="test"].groupby(["group","direction"],sort=False):
        e = events[(events.group==group)&(events.direction==direction)]
        cal = e[e.split=="calibration"].groupby("request_kw").delivered.sum()
        verified = cal[(cal==3)&(cal.index>0)]
        selected = float(verified.index.max()) if len(verified) else 0.
        test = e[(e.split=="test")&(e.request_kw==selected)]
        probe = e[(e.split=="test")&(e.request_kw==2)]
        summary.append(dict(group=group,direction=direction,eligible=int(data.floor_pass.sum()),
                            offline_lower_median_kw=data.lower_kw.median(), offline_upper_median_kw=data.physical_upper_kw.median(),
                            selected_kw=selected, selected_delivered=int(test.delivered.sum()), selected_actual_median_kw=test.actual_kw.median(),
                            probe_delivered=int(probe.delivered.sum()), probe_actual_median_kw=probe.actual_kw.median()))
    output.mkdir(parents=True,exist_ok=True)
    for name,data in (("offline",out),("online",events),("summary",pd.DataFrame(summary)),("provenance",pd.DataFrame(provenance))):
        data.to_csv(output/(name+".csv"),index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--start", type=int, choices=(15, 17), default=17)
    parser.add_argument("--capacity-floor", type=float, default=0.)
    parser.add_argument("--shortage-cap", type=float)
    parser.add_argument("--group", choices=GROUPS, action="append")
    args = parser.parse_args()
    if args.capacity_floor < 0 or (args.shortage_cap is not None and args.shortage_cap < 0):
        parser.error("Capacity limits must be nonnegative")
    START, RECOVERY = args.start, args.start + 1
    torch.set_num_threads(1)
    evaluate(args.source, args.output, args.capacity_floor, args.shortage_cap, args.group or GROUPS)








