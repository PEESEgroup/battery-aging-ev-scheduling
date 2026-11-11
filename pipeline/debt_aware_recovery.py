"""Recover curtailed vehicle energy through rolling residual-capacity optimization."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from nonanticipative_scheduler import Fleet
from run_directional_pinn_validation import RECOVERY, START, TOL, plan, swap


def recovery_plan(fleet, first, shadow, caps, debt):
    hours = np.arange(24)
    connected = (hours >= first) & (fleet.arrival[:, None] <= hours) & (hours < fleet.departure[:, None])
    room = np.maximum(0., fleet.power[:, None]-shadow)
    ids, ts = np.where(connected & (debt[:, None] > TOL) & (room > TOL))
    recovery = np.zeros_like(shadow)
    if not len(ids):
        return recovery
    j = np.arange(len(ids))
    ones = np.ones(len(ids))
    constraints = coo_matrix((np.r_[ones, ones], (np.r_[ts-first, 24-first+ids], np.r_[j,j])),
                             shape=(24-first+len(debt), len(j))).tocsc()
    available = np.maximum(0., caps-shadow[:, first:].sum(axis=0))
    limits = np.r_[available, debt]
    # First maximize restored energy, then place the same energy as early as possible.
    bounds = list(zip(np.zeros(len(j)), room[ids, ts]))
    optimum = linprog(-ones, A_ub=constraints, b_ub=limits, bounds=bounds, method="highs")
    if not optimum.success:
        raise RuntimeError(optimum.message)
    early = linprog(ts-first, A_ub=constraints, b_ub=limits, A_eq=ones[None, :],
                    b_eq=[max(0., -optimum.fun)], bounds=bounds, method="highs")
    if not early.success:
        raise RuntimeError(early.message)
    recovery[ids, ts] = early.x
    return recovery


def replay_with_debt(case, forecast, params, aging, reference, proposal, activation_caps, request, baseline_loss):
    fleet, grid, temp, cap_hour = case
    announced = {} if cap_hour is None else {cap_hour: float(grid[cap_hour, 3])}
    temperature = np.r_[temp[:START+1], np.repeat(temp[START], 23-START)]
    shifts = swap(fleet, proposal, activation_caps, -1, request, aging, temperature)
    debt = shifts.copy()
    actual = reference.copy()
    actual[:, START] -= shifts
    recovered_at_18 = 0.
    for hour in range(RECOVERY, 24):
        if debt.max(initial=0.) <= TOL:
            break
        caps, planning, score = forecast.build_case(hour, grid[:hour+1], params, announced)
        prefix = reference.copy()
        prefix[:, hour:] = 0.
        shadow = plan(fleet, hour, planning, score+2.-params["unmet_penalty_per_kwh"], prefix)
        assert np.allclose(shadow[:, hour], reference[:, hour], atol=TOL, rtol=0.)
        recovery = recovery_plan(fleet, hour, shadow, caps, debt)
        forecast_temp = np.r_[temp[:hour+1], np.repeat(temp[hour], 23-hour)]
        for i in np.flatnonzero(recovery[:, hour] > TOL):
            vehicle = Fleet(*(x[i:i+1] for x in vars(fleet).values()))
            ref_path = shadow[i:i+1]
            action = ref_path.copy()
            action[:, :hour] = actual[i:i+1, :hour]
            action[:, hour:] += recovery[i:i+1, hour:]
            ref_loss = aging.losses(vehicle, ref_path, forecast_temp)[0]
            if aging.losses(vehicle, action, forecast_temp)[0] > min(ref_loss, aging.limit)+TOL:
                continue
            amount = min(debt[i], recovery[i, hour])
            actual[i, hour] += amount
            debt[i] -= amount
            if hour == RECOVERY:
                recovered_at_18 += amount
    loss = aging.losses(fleet, actual, temp)
    service = np.max(np.abs(actual.sum(axis=1)-reference.sum(axis=1))) <= TOL
    health = bool(np.all(loss <= baseline_loss+TOL) and np.all(loss <= aging.limit+TOL))
    connected = (fleet.arrival[:, None] <= np.arange(24)) & (np.arange(24) < fleet.departure[:, None])
    assert np.all(np.abs(actual[~connected]) <= TOL)
    assert np.all(actual >= -TOL) and np.all(actual <= fleet.power[:, None]+TOL)
    assert np.all(actual.sum(axis=0) <= grid[:, 3]+TOL)
    return actual, float(shifts.sum()), bool(service), health, bool(shifts.sum()-recovered_at_18 <= TOL)
