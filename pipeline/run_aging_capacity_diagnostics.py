"""Trace fixed-service offline solves and vary incremental SOH and grid headroom."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from nonanticipative_scheduler import AGING, TOL, Fleet, Forecast, replay_aging, reference_replay, solve_plan
from run_section_iv_replay import DURATIONS, GROUPS, ROOT, STARTS, load

DEVIATION_TOL = 1e-5


def audit():
    trace, sweep = [], []
    for cap_name, floor in (("Raw", 0), ("1500 kW floor", 1500)):
        for group, directory in GROUPS.items():
            cases = load(directory, floor)
            forecast = Forecast([item[2] for item in cases[:4]])
            test = cases[7:]
            for i, (name, fleet, grid) in enumerate(test):
                donor = test[(i+1) % len(test)][2]
                ref, budget, plans, histories = reference_replay(fleet, grid, forecast, 4)
                target = ref.sum(axis=1)
                for k in STARTS:
                    prefix = ref.copy()
                    prefix[:, k:] = 0
                    free_caps = np.full(len(grid)-k, fleet.power.sum()+target.sum()+1)
                    unlimited_charger = replace(fleet, power=np.full(len(target), target.sum()+1))
                    for duration in DURATIONS:
                        fixed = plans[k].sum(axis=0)[k:k+duration]
                        for direction in (1, -1):
                            def solve(caps, age, *, use_fleet=fleet, signed=False, trace_lp=None):
                                return solve_plan(use_fleet, k, caps, np.zeros(len(caps)), prefix,
                                                  histories[k], target=target, budget=age,
                                                  window=(k, fixed, direction, None), offline=True,
                                                  allow_signed=signed, diagnostics=trace_lp)

                            keys = dict(capacity=cap_name, group=group, scenario=name, start_hour=k,
                                        duration_h=duration, direction="Up" if direction==1 else "Down")
                            for paired in (False, True):
                                caps = grid[k:,3].copy()
                                if paired:
                                    caps[1:] = donor[k+1:,3]
                                limited_info, loose_info = {}, {}
                                limited = solve(caps, budget, trace_lp=limited_info)
                                loose = solve(caps, None, trace_lp=loose_info)
                                row = dict(**keys, pairing="Permuted future capacity" if paired else "Original",
                                           limited_kw=np.nan if limited is None else limited[1],
                                           loose_kw=np.nan if loose is None else loose[1],
                                           limited_status=limited_info["status"], loose_status=loose_info["status"],
                                           limited_cuts=limited_info["cuts"], loose_cuts=loose_info["cuts"],
                                           limited_iterations=limited_info["iterations"],
                                           loose_iterations=loose_info["iterations"],
                                           reason="", grid_relax_gain_kw=np.nan, charger_relax_gain_kw=np.nan,
                                           signed_best_kw=np.nan, loose_exceeds_budget_vehicles=np.nan,
                                           loose_peak_budget_excess=np.nan,
                                           first_window_cap_kw=caps[0], reference_min_kw=min(fixed))
                                if loose is None:
                                    signed = solve(caps, None, signed=True)
                                    if signed is not None:
                                        row["reason"] = "reference_direction_conflict"
                                        row["signed_best_kw"] = signed[1]
                                    elif solve(free_caps, None, signed=True) is not None:
                                        row["reason"] = "grid_blocks_fixed_service"
                                    else:
                                        row["reason"] = "fixed_service_infeasible_even_with_grid_relaxed"
                                elif limited is None:
                                    row["reason"] = ("aging_cut_limit" if limited_info["status"] == "cut_limit"
                                                     else "aging_arm_no_feasible_solution")
                                else:
                                    loss = replay_aging(fleet, prefix + loose[0])
                                    excess = loss-budget
                                    row["loose_exceeds_budget_vehicles"] = int((excess>TOL).sum())
                                    row["loose_peak_budget_excess"] = float(np.maximum(excess,0).max())
                                    relaxed_grid = solve(free_caps, None)
                                    relaxed_both = solve(free_caps, None, use_fleet=unlimited_charger)
                                    row["grid_relax_gain_kw"] = relaxed_grid[1]-loose[1]
                                    row["charger_relax_gain_kw"] = relaxed_both[1]-relaxed_grid[1]
                                    if abs(loose[1]-limited[1]) > DEVIATION_TOL:
                                        row["reason"] = "aging_changes_optimum"
                                    elif row["loose_exceeds_budget_vehicles"]:
                                        row["reason"] = "aging_changes_schedule_only"
                                    else:
                                        row["reason"] = "unbudgeted_optimum_respects_budget"
                                trace.append(row)
                            if floor == 0:
                                for cap_floor in (0., 5., 15., 30., 60.):
                                    caps = np.maximum(grid[k:,3], cap_floor)
                                    for allowance in (0., 1e-5, 1e-4, None):
                                        info = {}
                                        found = solve(caps, None if allowance is None else budget+allowance,
                                                      trace_lp=info)
                                        sweep.append(dict(**keys, floor_kw=cap_floor, allowance=(
                                            "No incremental limit" if allowance is None else str(allowance)),
                                            offline_kw=np.nan if found is None else found[1], status=info["status"]))
            print(cap_name, group, flush=True)
    return pd.DataFrame(trace), pd.DataFrame(sweep)


def controlled():
    # Fixed service: a 60 kWh EV at SOC .30 receives 14 kWh in four hours.
    # SOC stays below the high-SOC hinge. With fixed energy, the only variable
    # aging term is proportional to sum(p**2).
    fleet = Fleet(np.array([0]), np.array([4]), np.array([14.]), np.array([7.]),
                  np.array([60.]), np.array([.30]))
    ref = np.full((1,4),3.5)
    prefix = np.zeros_like(ref)
    budget = replay_aging(fleet, ref)
    q = AGING.k_power / AGING.p_ref_kw**2
    rows = []
    for cap in (3.5, 4., 5., 6., 7.):
        for direction in (1,-1):
            physical = cap-3.5 if direction==1 else min(3.5,3*(cap-3.5))
            crossover = (4/3)*q*physical**2
            for allowance in (0.,1e-6,5e-6,1e-5,2e-5,3e-5,None):
                analytic = physical if allowance is None else min(physical,np.sqrt(allowance/((4/3)*q)))
                info = {}
                solution = solve_plan(
                    fleet,0,np.full(4,cap),np.zeros(4),prefix,np.zeros(1),
                    target=fleet.demand,budget=None if allowance is None else budget+allowance,
                    window=(0,np.array([3.5]),direction,None),offline=True,diagnostics=info,
                    cut_limit=100,aging_tol=1e-10)
                numeric = np.nan if solution is None else solution[1]
                if not np.isfinite(numeric) or abs(numeric-analytic) > .01:
                    raise AssertionError((cap,direction,allowance,analytic,numeric,info))
                rows.append(dict(capacity_kw=cap,direction="Up" if direction==1 else "Down",
                                 allowance="No incremental limit" if allowance is None else str(allowance),
                                 physical_limit_kw=physical,crossover_budget_soh=crossover,
                                 analytic_kw=analytic,numeric_kw=numeric,aging_cuts=info["cuts"]))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"aging_capacity_diagnostics")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    synthetic=controlled()
    synthetic.to_csv(args.output/"controlled_headroom_sweep.csv",index=False)
    trace,sweep=audit()
    trace.to_csv(args.output/"archive_pair_diagnostics.csv",index=False)
    sweep.to_csv(args.output/"archive_capacity_budget_sweep.csv",index=False)
    print(trace.groupby(["reason","limited_status","loose_status"]).size().to_string(),flush=True)


if __name__=="__main__":
    main()
