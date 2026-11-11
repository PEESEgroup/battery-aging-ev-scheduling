"""Generate and physically screen fresh evening-peak development trajectories."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack, vstack
import torch

from lock_recovery_holdout import digest, fingerprint, load_generators
from run_directional_pinn_validation import START, RECOVERY, read_case
from run_llm_timing_pilot import dump


def physical_shift(fleet, caps):
    """Maximize 17-to-18 same-car transfer while preserving maximum feasible service."""
    active = ((np.arange(24) >= fleet.arrival[:, None])
              & (np.arange(24) < fleet.departure[:, None]))
    ids, hours = np.where(active)
    n, m = len(fleet.arrival), len(ids)
    slots = np.arange(m)
    hourly = coo_matrix((np.ones(m), (hours, slots)), shape=(24, m))
    vehicle = coo_matrix((np.ones(m), (ids, slots)), shape=(n, m))
    constraints = vstack((hourly, vehicle)).tocsr()
    limits = np.r_[caps, fleet.target]
    bounds = list(zip(np.zeros(m), fleet.power[ids]))
    service = linprog(-np.ones(m), A_ub=constraints, b_ub=limits, bounds=bounds, method="highs")
    if not service.success:
        raise RuntimeError(service.message)
    eligible = np.flatnonzero(active[:, START] & active[:, RECOVERY])
    if not len(eligible):
        return 0., -service.fun
    k = len(eligible)
    j = np.arange(k)
    index = np.full((n, 24), -1)
    index[ids, hours] = slots
    # d_i <= p_i,17 and p_i,18 + d_i <= P_i; recover against actual hourly capacity.
    power = coo_matrix((np.r_[-np.ones(k), np.ones(k)],
                        (np.arange(2*k), np.r_[index[eligible, START], index[eligible, RECOVERY]])),
                       shape=(2*k, m))
    transfer = coo_matrix((np.ones(2*k), (np.arange(2*k), np.tile(j, 2))), shape=(2*k, k))
    upper = vstack((hstack((constraints, coo_matrix((24+n, k)))),
                    hstack((power, transfer)),
                    hstack((hourly.tocsr()[RECOVERY], np.ones((1, k)))))).tocsr()
    result = linprog(np.r_[np.zeros(m), -np.ones(k)], A_ub=upper,
                     b_ub=np.r_[limits, np.zeros(k), fleet.power[eligible], caps[RECOVERY]],
                     A_eq=np.r_[np.ones(m), np.zeros(k)][None, :], b_eq=[-service.fun],
                     bounds=bounds+[(0, None)]*k, method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return -result.fun, -service.fun


def run(source, grid_data, prior, output, replay_capacity=False, frozen_selection=None):
    output.mkdir(parents=True, exist_ok=False)
    from build_stress_scenarios import (StressScenarioConfig, stress_ev_sessions, stress_grid_hourly,
                                       build_ev_hourly_from_sessions, build_joint_hourly)
    from build_optimization_inputs import OptimizationInputConfig, build_ev_session_input, build_grid_input
    ev, grid, seed_row, hashes = load_generators(source, grid_data)
    first_seed, draws = (942001 if replay_capacity else 941001), 60
    if frozen_selection is not None:
        assert replay_capacity
        first_seed, draws = 943001, 120
    stress = StressScenarioConfig(stress_mode="evening_peak_cluster")
    opts = OptimizationInputConfig()
    excluded = set(pd.read_csv(prior / "manifest.csv").parent)
    for root in {source, prior.parent}:
        for manifest in root.glob("*/locked_manifest.csv"):
            table = pd.read_csv(manifest)
            for key in ("parent_fingerprint", "parent"):
                if key in table:
                    excluded.update(table[key])
    dump(output / "screen_protocol.json", dict(first_seed=first_seed, planned_draws=draws,
        split="holdout" if frozen_selection is not None else "development",
        frozen_selection_sha256=digest(frozen_selection) if frozen_selection is not None else None,
        generator_sha256=hashes, grid_seed_input_sha256=digest(grid_data),
        condition="At least 2 kWh same-vehicle shift from 17 to 18 at maximum physically feasible fleet service. "
                  "Enforce site caps, individual charging limits, dwell windows, and SOC-limited demands.",
        capacity="Original matched-feedback hard cap max(raw, 1500 kW)" if replay_capacity else "Raw site cap",
        screen_tolerance_kwh=1e-7, use_pinn=False, use_method_feedback=False,
        evaluation="Existing replay, fixed archived development forecast donors and capacity convention."))
    start = pd.Timestamp(seed_row["timestamp"])
    manifest, audit = [], []
    for seed in range(first_seed, first_seed+draws):
        ev.rng = np.random.default_rng(seed)
        grid.rng = np.random.default_rng(seed+10000)
        torch.manual_seed(seed)
        sessions = ev.sample_sessions_for_horizon(str(start), hours=24, source="acn", tail_strength=.6,
                                                  use_tail_enhancement=True, include_pinn_context=False)
        if sessions.empty:
            audit.append(dict(seed=seed, status="empty"))
            continue
        torch.manual_seed(seed+10000)
        hourly = grid.sample_trajectory(seed_row, horizon=24, stochastic=True, tail_strength=.6)
        sessions = stress_ev_sessions(sessions, stress)
        hourly = stress_grid_hourly(hourly, stress)
        joint = build_joint_hourly(hourly, build_ev_hourly_from_sessions(sessions, hourly))
        grid_input = build_grid_input(joint, opts)
        ev_input = build_ev_session_input(sessions, grid_input, opts)
        if ev_input.empty:
            audit.append(dict(seed=seed, status="empty"))
            continue
        dest = output / "scenarios" / f"parent_{seed}"
        dest.mkdir(parents=True)
        ev_path, grid_path = dest/"optimizer_ev_sessions.csv", dest/"optimizer_grid.csv"
        ev_input.to_csv(ev_path, index=False)
        grid_input.to_csv(grid_path, index=False)
        fleet, signals, _ = read_case(dest)
        parent = fingerprint(fleet)
        if parent in excluded:
            audit.append(dict(seed=seed, status="duplicate"))
            continue
        excluded.add(parent)
        caps = np.maximum(signals[:, 3], 1500.) if replay_capacity else signals[:, 3]
        shift, service = physical_shift(fleet, caps)
        passed = shift >= 2-1e-7
        audit.append(dict(seed=seed, status="feasible" if passed else "infeasible",
                          shift_kwh=shift, max_service_kwh=service))
        if passed:
            manifest.append(dict(seed=seed, parent=parent, scenario=str(dest.relative_to(output)),
                                 ev_sha256=digest(ev_path), grid_sha256=digest(grid_path)))
    pd.DataFrame(audit).to_csv(output/"screen_audit.csv", index=False)
    assert manifest, "No physically feasible draws"
    pd.DataFrame(manifest).to_csv(output/"locked_manifest.csv", index=False)
    dump(output/"scenario_lock.json", dict(parents=len(manifest),
         sha256=digest(output/"locked_manifest.csv"), protocol_sha256=digest(output/"screen_protocol.json")))
    print(f"Locked {len(manifest)} of {draws} physically screened draws.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "grid_data", "prior", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--replay-capacity", action="store_true")
    parser.add_argument("--frozen-selection", type=Path)
    torch.set_num_threads(1)
    run(**vars(parser.parse_args()))
