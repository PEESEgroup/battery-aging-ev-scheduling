"""Replay locked feedback configurations on new archived-model evening-peak draws."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lock_recovery_holdout import digest, fingerprint, load_generators
from run_directional_pinn_validation import Aging, read_case
from run_gemini_feedback_pair import CHECKPOINTS, PROBES, load_cases, margins, metrics, replay
from run_llm_timing_pilot import dump


REGIME = "evening_peak_cluster"
FIRST_SEED = 923401
PARENTS = 40


def run(source, grid_data, prior, output):
    output.mkdir(parents=True,exist_ok=False)
    groups,_ = load_cases(source)
    _,forecast = groups[REGIME]
    excluded = {key for part,_ in groups.values() for cases in part.values() for key in cases}
    for name in ("recovery_departure_original_locked_holdout","joint_context_fresh_holdout",
                 "low_confidence_candidate_pool"):
        path = prior.parent/name/"locked_manifest.csv"
        if path.exists():
            excluded.update(pd.read_csv(path).parent_fingerprint)
    configs = {}
    for arm in ("bo","chatgpt"):
        lock = json.loads((prior/f"{arm}_configuration_lock.json").read_text(encoding="utf-8"))
        configs[arm] = {str(b):lock[REGIME][str(b)] for b in CHECKPOINTS}
    old_protocol = json.loads((prior/"protocol.json").read_text(encoding="utf-8"))
    aging = Aging(source)
    assert digest(aging.path)==old_protocol["pinn_sha256"]
    ev,grid,seed_row,hashes = load_generators(source,grid_data)
    from build_stress_scenarios import StressScenarioConfig, stress_ev_sessions, stress_grid_hourly
    from build_stress_scenarios import build_ev_hourly_from_sessions, build_joint_hourly
    from build_optimization_inputs import OptimizationInputConfig, build_ev_session_input, build_grid_input

    stress = StressScenarioConfig(stress_mode=REGIME)
    opts = OptimizationInputConfig()
    dump(output/"protocol.json",dict(regime=REGIME,planned_parents=PARENTS,first_seed=FIRST_SEED,
        generator_sha256=hashes,pinn_sha256=digest(aging.path),source_protocol_sha256=digest(prior/"protocol.json"),
        configuration_sha256={arm:digest(prior/f"{arm}_configuration_lock.json") for arm in configs},
        generation="archived CVAE-diffusion checkpoints, original evening-peak transform, no outcome screening",
        forecast="same seven original development donor paths; no holdout adaptation",
        feasibility="2 kW qualification at least BO; prior service and CVaR tolerances vs first shared probe",
        subset="both methods achieve qualified 2 kW delivery at the same checkpoint",
        capacity="same 1500 kW background floor as matched-feedback directional replay",
        configurations={a:{b:r["point"] for b,r in c.items()} for a,c in configs.items()},api_calls=0))
    start = pd.Timestamp(seed_row["timestamp"])
    cases,manifest = {},[]
    for seed in range(FIRST_SEED,FIRST_SEED+PARENTS*2):
        ev.rng = np.random.default_rng(seed)
        grid.rng = np.random.default_rng(seed+10000)
        torch.manual_seed(seed)
        sessions = ev.sample_sessions_for_horizon(str(start),hours=24,source="acn",tail_strength=.6,
                                                   use_tail_enhancement=True,include_pinn_context=False)
        if sessions.empty:
            continue
        torch.manual_seed(seed+10000)
        hourly = grid.sample_trajectory(seed_row,horizon=24,stochastic=True,tail_strength=.6)
        sessions = stress_ev_sessions(sessions,stress)
        hourly = stress_grid_hourly(hourly,stress)
        joint = build_joint_hourly(hourly,build_ev_hourly_from_sessions(sessions,hourly))
        grid_input = build_grid_input(joint,opts)
        ev_input = build_ev_session_input(sessions,grid_input,opts)
        if ev_input.empty:
            continue
        dest = output/"scenarios"/f"parent_{seed}"
        dest.mkdir(parents=True)
        ev_path,grid_path = dest/"optimizer_ev_sessions.csv",dest/"optimizer_grid.csv"
        ev_input.to_csv(ev_path,index=False)
        grid_input.to_csv(grid_path,index=False)
        fleet, signals, temp = read_case(dest)
        key = fingerprint(fleet)
        if key in excluded or key in cases:
            continue
        signals[:,3] = np.maximum(signals[:,3],1500.)
        cases[key] = fleet,signals,temp,None
        manifest.append(dict(seed=seed,parent=key,scenario=str(dest.relative_to(output)),
                             ev_sha256=digest(ev_path),grid_sha256=digest(grid_path)))
        if len(cases)%10 == 0:
            print(f"Generated {len(cases)}/{PARENTS} evening-peak parents.",flush=True)
        if len(cases)==PARENTS:
            break
    assert len(cases)==PARENTS
    pd.DataFrame(manifest).to_csv(output/"locked_manifest.csv",index=False)
    dump(output/"scenario_lock.json",dict(parents=len(cases),sha256=digest(output/"locked_manifest.csv")))
    print("Fresh parent trajectories locked before outcome evaluation.",flush=True)
    reference = metrics(replay(cases,forecast,aging,PROBES[0]))
    cache = {}
    for b in CHECKPOINTS:
        paired = {}
        for arm in configs:
            point=tuple(configs[arm][str(b)]["point"])
            if point not in cache:
                cache[point]=replay(cases,forecast,aging,point)
            paired[arm]=pd.DataFrame(cache[point])
        df = paired["chatgpt"].merge(paired["bo"],on="parent",suffixes=("_gpt","_bo"),validate="one_to_one")
        df["both_qualified"]=df.qualified_gpt & df.qualified_bo
        df["budget"]=b
        if b==CHECKPOINTS[0]:
            individual,summary=[],[]
        individual.append(df)
        overall={arm:metrics(rows) for arm,rows in paired.items()}
        for panel,part in (("all",df),("jointly_qualified",df[df.both_qualified])):
            for arm,suffix in (("chatgpt","gpt"),("bo","bo")):
                columns = ["qualified","served_ratio","unmet_kwh","mean_soh_loss"]
                m=metrics(part[[f"{c}_{suffix}" for c in columns]].set_axis(columns,axis=1)) if len(part) else {}
                guard=margins(overall[arm],reference)>=-1e-7
                summary.append(dict(budget=b,panel=panel,arm=arm,parents=len(part),**m,
                    delivery_vs_bo_ok=overall[arm]["qualified"]>=overall["bo"]["qualified"]-1e-7,
                    service_ok=bool(guard[0]),cvar_ok=bool(guard[1])))
    pd.concat(individual,ignore_index=True).to_csv(output/"paired_holdout.csv",index=False)
    pd.DataFrame(summary).to_csv(output/"summary.csv",index=False)
    dump(output/"fixed_reference.json",reference)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source","grid_data","prior","output"):
        p.add_argument(name,type=Path)
    torch.set_num_threads(1)
    run(*(path.resolve() for path in vars(p.parse_args()).values()))
