"""Draw and lock a physics-screened holdout using archived generator checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import torch

from nonanticipative_scheduler import Fleet
from run_directional_pinn_validation import SIGNALS, TOL
from run_llm_timing_pilot import DEFAULTS, RECOVERY, REQUEST, START, TimingForecast, dump, rolling_reference
from screen_recovery_departure import adjacent_bound, parent_cases


QUOTAS = {"high": 32, "low": 24}
DRAW_LIMIT = 320
SEED = 481703
TAIL = .6
FLEET_COLUMNS = {
    "arrival_step": int, "departure_step": int, "required_energy_kwh": float,
    "max_charging_power_kw": float, "battery_capacity_kwh": float, "init_soc": float,
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(fleet):
    values = np.column_stack(tuple(vars(fleet).values())).round(8)
    return hashlib.sha256(np.ascontiguousarray(values[np.lexsort(values.T)]).tobytes()).hexdigest()


def load_generators(source, grid_data):
    sys.path.insert(0, str(source))
    archive = source / "final_world_model_exact_matched"
    sys.path.insert(0, str(archive))
    from train_ev_generator_f import EVGenerator
    from train_grid_generator_f import GridGenerator

    # The archived EV count table was serialized with pandas 1.x.
    try:
        __import__("pandas.core.indexes.numeric")
    except ModuleNotFoundError:
        numeric = types.ModuleType("pandas.core.indexes.numeric")
        numeric.Int64Index = numeric.UInt64Index = numeric.Float64Index = pd.Index
        sys.modules[numeric.__name__] = numeric
    try:
        __import__("numpy._core.multiarray")
    except ModuleNotFoundError:
        from numpy.core import multiarray
        sys.modules["numpy._core.multiarray"] = multiarray
    ev_path = archive / "processed_ev/ev_generator.pkl"
    grid_path = archive / "processed_grid/grid_generator.pkl"
    ev_gen = EVGenerator.load(ev_path)
    grid_gen = GridGenerator.load(grid_path)
    grid_gen.config.data_path = grid_data
    features = list(grid_gen.feature_cols)
    seed_row = grid_gen.load_data().iloc[-1].copy()
    assert features == list(grid_gen.feature_cols)
    for model in (ev_gen, grid_gen):
        model.cvae.eval()
        model.denoiser.eval()
    return ev_gen, grid_gen, seed_row, {"ev": digest(ev_path), "grid": digest(grid_path)}


def run(source, seed_data, prior, output, first_seed=SEED, quotas=None):
    quotas = QUOTAS if quotas is None else quotas
    output.mkdir(parents=True, exist_ok=False)
    prior_manifest = pd.read_csv(prior / "manifest.csv")
    prior_protocol = json.loads((prior / "protocol.json").read_text(encoding="utf-8"))
    threshold = prior_protocol["urgency_dev_median"]
    margins = prior_protocol["target_room_kw"]
    floor = prior_protocol["capacity_background_floor_kw"]

    def background(grid):
        grid = grid.copy()
        grid[:, 3] = np.maximum(grid[:, 3], floor)
        return grid

    originals = list(parent_cases(source))
    archived_dev = set(prior_manifest.loc[prior_manifest.split == "development", "parent_sha256"])
    assert {row[2] for row in originals if row[-1] == "development"} == archived_dev
    forecast = TimingForecast([background(case[1]) for _, _, key, _, case, _ in originals
                               if key in archived_dev])
    seen = {fingerprint(case[0]) for _, _, _, _, case, _ in originals}
    ev_gen, grid_gen, grid_seed, hashes = load_generators(source, seed_data)
    from build_joint_scenarios_f import build_ev_hourly_features, build_joint_hourly
    from build_optimization_inputs import OptimizationInputConfig, build_ev_session_input, build_grid_input

    start = pd.Timestamp(grid_seed["timestamp"])
    config = OptimizationInputConfig()
    dump(output / "protocol.json", dict(
        quotas=quotas, draw_limit=DRAW_LIMIT, first_rng_seed=first_seed, urgency_threshold=threshold,
        margins_kw=margins, request_kw=REQUEST, activation_step=START, recovery_step=RECOVERY,
        background_capacity_floor_kw=floor, tail_strength=TAIL, ev_source="acn", generator_hashes=hashes,
        generator_modules=["final_world_model_exact_matched/train_ev_generator_f",
                           "final_world_model_exact_matched/train_grid_generator_f"],
        grid_seed_timestamp=str(start), grid_seed_data_sha256=digest(seed_data),
        prior_manifest_sha256=digest(prior / "manifest.csv"), defaults=DEFAULTS,
        acceptance="first quota per pressure in seeded order with fixed-reference two-slot bound >=2 kW in both margins; retain both rows",
        scope="conditional held-out stress panel; archived holdout reported separately; no PINN or warm-start results",
    ))
    print(f"Frozen quotas {quotas}; seed row {start}; urgency cutoff {threshold:.6f}.", flush=True)
    audit, locked = [], []
    counts = dict.fromkeys(quotas, 0)
    for i in range(DRAW_LIMIT):
        seed = first_seed + i
        ev_gen.rng = np.random.default_rng(seed)
        grid_gen.rng = np.random.default_rng(seed + DRAW_LIMIT)
        torch.manual_seed(seed)
        ev = ev_gen.sample_sessions_for_horizon(str(start), hours=24, source="acn",
                                                tail_strength=TAIL, use_tail_enhancement=True,
                                                include_pinn_context=False)
        torch.manual_seed(seed + DRAW_LIMIT)
        grid = grid_gen.sample_trajectory(grid_seed, horizon=24, stochastic=True, tail_strength=TAIL)
        joint = build_joint_hourly(grid, build_ev_hourly_features(ev, start + pd.Timedelta(hours=1), 24))
        hourly = build_grid_input(joint, config)
        sessions = build_ev_session_input(ev, hourly, config)
        if sessions.empty:
            audit.append(dict(seed=seed, status="empty"))
            continue
        fleet = Fleet(*(sessions[col].to_numpy(dtype) for col, dtype in FLEET_COLUMNS.items()))
        key = fingerprint(fleet)
        if key in seen:
            audit.append(dict(seed=seed, status="duplicate", parent_fingerprint=key))
            continue
        seen.add(key)
        hourly["ev_capacity_limit_kw"] = np.maximum(hourly.ev_capacity_limit_kw, floor)
        signal = background(hourly[SIGNALS].to_numpy())
        temp = np.full(24, 25.)  # Same fallback as the archived optimizer export.
        ref, _, _ = rolling_reference((fleet, signal, temp, None), forecast, DEFAULTS)
        connected = (fleet.arrival <= START) & (fleet.departure > START)
        urgent = connected & (fleet.departure <= RECOVERY + 1)
        remaining = np.maximum(0., fleet.demand - ref[:, :START].sum(axis=1))
        pressure_score = float(remaining[urgent].sum()/max(1e-9, remaining[connected].sum()))
        pressure = "high" if pressure_score >= threshold else "low"
        rows = []
        for margin, room in margins.items():
            modified = signal.copy()
            modified[RECOVERY, 3] = min(signal[RECOVERY, 3], ref[:, RECOVERY].sum() + room)
            projected, _, _ = rolling_reference((fleet, modified, temp, RECOVERY), forecast, DEFAULTS)
            bound = adjacent_bound(fleet, modified, projected)
            rows.append(dict(cell=f"{margin}_{pressure}", margin=margin, cap_hour=RECOVERY,
                             cap_kw=modified[RECOVERY, 3], physical_kw=bound,
                             grid_room_kw=modified[RECOVERY, 3]-projected[:, RECOVERY].sum(),
                             physical=bound >= REQUEST - TOL))
        status = "infeasible" if not all(r["physical"] for r in rows) else (
            "quota_full" if counts[pressure] >= quotas[pressure] else "accepted")
        case = dict(seed=seed, grid_seed=seed+DRAW_LIMIT, pressure=pressure, urgent_share=pressure_score,
                    parent_fingerprint=key, status=status)
        audit.extend({**case, **row} for row in rows)
        if status == "accepted":
            counts[pressure] += 1
            for row in rows:
                dest = output / "scenarios" / row["cell"] / f"parent_{seed}"
                dest.mkdir(parents=True)
                grid_export = hourly.copy()
                grid_export.loc[RECOVERY, "ev_capacity_limit_kw"] = row["cap_kw"]
                ev_path, grid_path = dest/"optimizer_ev_sessions.csv", dest/"optimizer_grid.csv"
                sessions.to_csv(ev_path, index=False)
                grid_export.to_csv(grid_path, index=False)
                locked.append(dict(**case, **row, split="holdout", path=str(dest.relative_to(output)),
                                   ev_sha256=digest(ev_path), grid_sha256=digest(grid_path)))
        if (i+1) % 20 == 0 or counts == quotas:
            print(f"Draw {i+1}: {counts}", flush=True)
        if counts == quotas:
            break

    audit = pd.DataFrame(audit)
    lock = pd.DataFrame(locked)
    assert lock.groupby("parent_fingerprint").size().eq(2).all() and lock.physical.all()
    audit.to_csv(output / "candidate_audit.csv", index=False)
    lock.to_csv(output / "locked_manifest.csv", index=False)
    grouped = audit[audit.status.isin(["accepted", "infeasible", "quota_full"])].groupby("cell")
    summary = grouped.agg(candidate_parents=("parent_fingerprint", "nunique"),
                          candidate_feasible=("physical", "sum"))
    summary["locked_new_parents"] = lock.groupby("cell").parent_fingerprint.nunique()
    summary["locked_new_feasible"] = lock.groupby("cell").physical.sum()
    archived = prior_manifest[prior_manifest.split == "holdout"].copy()
    archived["physical"] = archived.default_physical_kw >= REQUEST - TOL
    summary["archived_parents"] = archived.groupby("cell").parent_sha256.nunique()
    summary["archived_feasible"] = archived.groupby("cell").physical.sum()
    summary.to_csv(output / "summary.csv")
    dump(output / "lock.json", dict(locked_parents=counts, quotas_met=counts == quotas,
         manifest_sha256=digest(output / "locked_manifest.csv"), audit_sha256=digest(output / "candidate_audit.csv")))
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "seed_data", "prior", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--first-seed", type=int, default=SEED)
    parser.add_argument("--high", type=int, default=QUOTAS["high"])
    parser.add_argument("--low", type=int, default=QUOTAS["low"])
    args = parser.parse_args()
    run(args.source.resolve(), args.seed_data.resolve(), args.prior.resolve(), args.output.resolve(),
        args.first_seed, {"high": args.high, "low": args.low})
