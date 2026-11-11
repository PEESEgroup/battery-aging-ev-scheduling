"""Non-anticipative, open-loop scenario MPC for the Section IV replay.

A single horizon plan is shared by the forecast scenarios. Replanning after
each observation provides recourse in execution. Only the current slot is sent.
The legacy semi-empirical aging model is used because calibrated PINN weights
are required to reproduce the manuscript's PINN results.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from battery_degradation_model import DegradationConfig


AGING = DegradationConfig()
TOL = 1e-7
HEALTH_WEIGHT = 0.8
UNMET_WEIGHT = 2.0
RISK_WEIGHT = 0.3


def cvar(samples, alpha, axis=0):
    """Equal-weight upper CVaR with fractional mass at the tail boundary."""
    ordered = np.flip(np.sort(np.asarray(samples), axis=axis), axis=axis)
    mass = (1.0 - alpha) * ordered.shape[axis]
    weights = np.clip(mass - np.arange(ordered.shape[axis]), 0.0, 1.0)
    return np.tensordot(weights / mass, np.moveaxis(ordered, axis, 0), axes=1)


@dataclass
class Fleet:
    arrival: np.ndarray
    departure: np.ndarray
    demand: np.ndarray
    power: np.ndarray
    capacity: np.ndarray
    soc: np.ndarray

    @property
    def target(self):
        return np.minimum(self.demand, (1.0 - self.soc) * self.capacity)


def aging(power, soc, capacity):
    """Loss and tangent of the legacy model for one EV's remaining slots."""
    avg_soc = soc + (np.cumsum(power) - 0.5 * power) / capacity
    excess = np.maximum(avg_soc - AGING.soc_threshold, 0.0)
    linear = AGING.k_throughput + AGING.k_soc_swing / capacity
    quad = AGING.k_power / AGING.p_ref_kw**2
    loss = (linear * power + quad * power**2 + AGING.k_calendar
            + AGING.k_high_soc * excess**2 + AGING.k_high_soc_dwell * excess)
    soc_grad = (2 * AGING.k_high_soc * excess
                + AGING.k_high_soc_dwell * (excess > 0))
    grad = (linear + 2 * quad * power
            + (np.cumsum(soc_grad[::-1])[::-1] - 0.5 * soc_grad) / capacity)
    return loss, grad


def replay_aging(fleet, plan):
    return np.array([
        aging(plan[i, start:end], fleet.soc[i], fleet.capacity[i])[0].sum()
        for i, (start, end) in enumerate(zip(fleet.arrival, fleet.departure))
    ])


class Forecast:
    """Forecasts from an exclusive donor set and current observations only."""

    def __init__(self, donors):
        self.donors = np.asarray(donors)
        signals = self.signals(self.donors)
        self.center = signals.min(axis=(0, 1))
        span = signals.max(axis=(0, 1)) - self.center
        self.scale = np.where(span > 1e-12, span, 1.0)

    @staticmethod
    def signals(grid):
        return np.concatenate((grid[..., :3], -grid[..., 3:4]), axis=-1)

    def build(self, t, observed, count, *, risk_terms=True):
        history = self.signals(observed)
        distance = (((self.signals(self.donors[:, :t+1]) - history) / self.scale)**2
                    ).mean(axis=(1, 2))
        paths = self.donors[np.argsort(distance, kind="stable")[:count], t:].copy()
        paths[:, 0] = observed[-1]
        q = np.clip((self.signals(paths) - self.center) / self.scale, 0.0, 1.0)
        stress = q.mean(axis=0)
        if risk_terms:
            stress = stress + 0.5 * q.std(axis=0) + 0.3 * cvar(q, 0.95)
        score = stress @ np.array([1.0, 30.0, 0.2, 1.0]) / 32.2
        return paths[..., 3].min(axis=0), score


def solve_plan(fleet, start, caps, score, executed, losses, *,
               target=None, budget=None, window=None, offline=False,
               diagnostics=None, allow_signed=False, cut_limit=24, aging_tol=TOL):
    """Sparse LP with supporting cuts for convex SOH loss.

    When offline is true the objective maximizes the sustained deviation.
    Otherwise the objective is grid stress plus aging plus departure shortfall.
    Under the shared-plan restriction all forecast scenarios have the same
    Eq. (51) cost, so Eq. (50) multiplies it by 1 + lambda_risk.
    """
    if allow_signed and not offline:
        raise ValueError("Signed deviation is an offline diagnostic")
    if diagnostics is not None:
        diagnostics.update(status="prefix_infeasible", cuts=0, iterations=0)
    n, steps = executed.shape
    end = start + len(caps)
    known = (fleet.arrival <= start) if not offline else np.ones(n, dtype=bool)
    active = known & (fleet.departure > start)
    ids, hours = np.where(active[:, None]
                          & (np.arange(steps) >= np.maximum(fleet.arrival, start)[:, None])
                          & (np.arange(steps) < np.minimum(fleet.departure, end)[:, None]))
    slots = len(ids)
    widths = np.bincount(ids, minlength=n)
    offsets = np.r_[0, np.cumsum(widths)]
    vehicles = np.flatnonzero(widths)
    used = executed.sum(axis=1)
    remaining = (fleet.target if target is None else target) - used
    if target is not None and np.any(remaining[known] < -TOL):
        return None
    if target is not None and np.any(np.abs(remaining[known & ~active]) > TOL):
        return None
    if budget is not None and np.any(losses[known] > budget[known] + aging_tol):
        return None

    # The last variable is sustained kW (offline) or an aging epigraph (online).
    objective = np.r_[np.zeros(slots), -1.0] if offline else np.r_[
        (1 + RISK_WEIGHT) * (score[hours-start] - (UNMET_WEIGHT if target is None else 0)),
        (1 + RISK_WEIGHT) * HEALTH_WEIGHT,
    ]
    bounds = list(zip(np.zeros(slots), fleet.power[ids])) + [(None if allow_signed else 0, None)]
    ub_rows, ub_cols, ub_values, ub_rhs = [], [], [], []
    eq_rows, eq_cols, eq_values, eq_rhs = [], [], [], []

    def add(row_data, cols, vals, rhs):
        rows, columns, values, right = row_data
        rows.extend([len(right)] * len(cols))
        columns.extend(cols)
        values.extend(vals)
        right.append(rhs)

    ub = ub_rows, ub_cols, ub_values, ub_rhs
    eq = eq_rows, eq_cols, eq_values, eq_rhs
    slot_ids = np.arange(slots)
    for t, cap in enumerate(caps, start):
        selected = slot_ids[hours == t]
        add(ub, selected, np.ones(len(selected)), cap)
    for i in vehicles:
        selected = slot_ids[offsets[i]:offsets[i+1]]
        add(eq if target is not None else ub, selected, np.ones(len(selected)),
            max(0.0, remaining[i]))
    if window is not None:
        first, reference, direction, commitment = window
        for t in range(max(start, first), min(end, first + len(reference))):
            selected = slot_ids[hours == t]
            cols = np.r_[selected, slots] if offline else selected
            vals = np.r_[np.full(len(selected), -direction), 1] if offline else np.full(len(selected), -direction)
            rhs = -direction * reference[t-first] - (0 if offline else commitment)
            add(ub, cols, vals, rhs)
    if offline and window is None:
        raise ValueError("An offline solve requires an activation window")

    def sparse(row_data):
        rows, cols, vals, rhs = row_data
        return coo_matrix((vals, (rows, cols)), shape=(len(rhs), slots+1)).tocsc() if rhs else None

    # Supporting hyperplanes give an outer relaxation. Never return a point
    # that violates the original, nonlinear per-vehicle aging constraint.
    best = None
    for iteration in range(cut_limit):
        opt = linprog(objective, A_ub=sparse(ub), b_ub=ub_rhs or None,
                      A_eq=sparse(eq), b_eq=eq_rhs or None, bounds=bounds, method="highs")
        if diagnostics is not None:
            diagnostics.update(iterations=iteration+1, solver_status=int(opt.status))
        if opt.status == 2:
            if diagnostics is not None:
                diagnostics["status"] = "initial_lp_infeasible" if iteration == 0 else "infeasible_after_aging_cuts"
            return None
        if not opt.success:
            raise RuntimeError(opt.message)
        power = opt.x[:-1]
        all_grad = np.zeros(slots)
        aging_total = 0.0
        violated = False
        for i in vehicles:
            selected = slot_ids[offsets[i]:offsets[i+1]]
            soc = fleet.soc[i] + used[i] / fleet.capacity[i]
            loss, grad = aging(power[selected], soc, fleet.capacity[i])
            value = loss.sum()
            all_grad[selected] = grad
            aging_total += value
            health_limit = 1.0 - AGING.min_soh if budget is None else min(budget[i], 1.0 - AGING.min_soh)
            allowable = health_limit - losses[i]
            if value > allowable + aging_tol:
                scale = 10000.0
                add(ub, selected, grad * scale, (allowable - value + grad @ power[selected]) * scale)
                if diagnostics is not None:
                    diagnostics["cuts"] += 1
                violated = True
        if not violated:
            plan = np.zeros_like(executed)
            plan[ids, hours] = power
            best = plan, opt.x[-1]
            if offline or aging_total <= opt.x[-1] + 2e-5:
                if diagnostics is not None:
                    diagnostics["status"] = "solved"
                return best
        if not offline:
            add(ub, np.r_[slot_ids, slots], np.r_[all_grad, -1.0],
                all_grad @ power - aging_total)
    if diagnostics is not None:
        diagnostics["status"] = "cut_limit" if best is None else "feasible_at_cut_limit"
    if best is None:
        return None
    return best


def update(fleet, executed, losses, t, action):
    active = np.flatnonzero((fleet.arrival <= t) & (fleet.departure > t))
    for i in active:
        soc = fleet.soc[i] + executed[i, :t].sum() / fleet.capacity[i]
        losses[i] += aging(np.array([action[i]]), soc, fleet.capacity[i])[0][0]
    executed[:, t] = action


def reference_replay(fleet, grid, forecast, count):
    executed = np.zeros((len(fleet.arrival), len(grid)))
    losses = np.zeros(len(fleet.arrival))
    plans = []
    histories = []
    for t in range(len(grid)):
        caps, score = forecast.build(t, grid[:t+1], count)
        histories.append(losses.copy())
        result = solve_plan(fleet, t, caps, score, executed, losses)
        if result is None:
            raise RuntimeError("No feasible reference action")
        plan, _ = result
        plans.append(plan)
        update(fleet, executed, losses, t, plan[:, t])
    assert np.allclose(losses, replay_aging(fleet, executed))
    return executed, losses, plans, histories


def activate(fleet, grid, forecast, count, baseline, first, direction, r,
             fixed, enforce_aging):
    ref, ref_loss, plans, history = baseline
    executed = ref.copy()
    executed[:, first:] = 0
    losses = history[first].copy()
    window = first, fixed, direction, r
    for t in range(first, len(grid)):
        caps, score = forecast.build(t, grid[:t+1], count)
        # Project the shadow reference from its causal realized prefix plus
        # its current shared forecast plan. Future replay outcomes stay hidden.
        projected = plans[t].copy()
        projected[:, :t] = ref[:, :t]
        projected_target = projected.sum(axis=1)
        projected_budget = replay_aging(fleet, projected) if enforce_aging else None
        result = solve_plan(fleet, t, caps, score, executed, losses,
                            target=projected_target, budget=projected_budget, window=window)
        if result is None:
            return False, "planning_infeasible"
        plan, _ = result
        update(fleet, executed, losses, t, plan[:, t])
    deviation = direction * (executed[:, first:first+len(fixed)].sum(axis=0) - fixed)
    if np.any(deviation < r - 1e-6) or np.any(executed.sum(axis=0) > grid[:, 3] + 1e-6):
        return False, "power_or_capacity"
    if not np.allclose(executed.sum(axis=1), ref.sum(axis=1), atol=1e-6, rtol=0):
        return False, "departure_energy"
    if enforce_aging and np.any(losses > ref_loss + TOL):
        return False, "aging"
    return True, "delivered"



