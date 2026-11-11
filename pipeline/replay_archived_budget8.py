"""Freeze eight-evaluation incumbents and replay them on fresh generator draws."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lock_recovery_holdout import digest, fingerprint, load_generators
from rerun_archived_gpt6_benchmark import GROUPS, load_benchmark
from run_directional_pinn_validation import read_case


def lock_configs(history_path, output):
    history = pd.read_csv(history_path)
    winners = []
    for group in GROUPS:
        frame = history[history.group==group]
        one = frame[frame.method=="LLM one-shot"]
        feedback = frame[frame.method=="LLM feedback"]
        seeds = feedback[feedback.candidate.str.startswith("seed_")]
        assert seeds.param_hash.tolist()==one.param_hash.tolist()
        assert np.allclose(seeds.actual_score, one.actual_score)
        gpt = pd.concat([one,feedback[~feedback.candidate.str.startswith("seed_")]],ignore_index=True).iloc[:8]
        bo = frame[frame.method=="Bayesian opt."].iloc[:8]
        assert len(gpt)==len(bo)==8
        for arm, rows in (("GPT-6 Sol",gpt),("BO",bo),("One-shot",one),
                          ("Base",frame[frame.method=="Fixed baseline"]),
                          ("Rule",frame[frame.method=="Rule-based prior"]),
                          ("LHS",frame[frame.method=="Random/LHS"])):
            pick = rows.loc[rows.actual_score.idxmin()].to_dict()
            pick.update(arm=arm,budget=len(rows),winning_step=rows.index.get_loc(rows.actual_score.idxmin())+1)
            winners.append(pick)
    config = output/"budget8_lock.csv"
    pd.DataFrame(winners).to_csv(config,index=False)
    return config


def generate(source, grid_data, output, group, exclude):
    from build_stress_scenarios import (StressScenarioConfig, stress_ev_sessions, stress_grid_hourly,
                                       build_ev_hourly_from_sessions, build_joint_hourly)
    from build_optimization_inputs import OptimizationInputConfig, build_ev_session_input, build_grid_input
    ev, grid, row, hashes = load_generators(source,grid_data)
    start = pd.Timestamp(row["timestamp"])
    stress, options = StressScenarioConfig(stress_mode=group), OptimizationInputConfig()
    first_seed = 951001+1000*GROUPS.index(group)
    inputs = output/"inputs"/group
    audit = []
    for seed in range(first_seed,first_seed+120):
        ev.rng = np.random.default_rng(seed)
        grid.rng = np.random.default_rng(seed+10000)
        torch.manual_seed(seed)
        sessions = ev.sample_sessions_for_horizon(str(start),hours=24,source="acn",tail_strength=.6,
                                                  use_tail_enhancement=True,include_pinn_context=False)
        if sessions.empty:
            continue
        torch.manual_seed(seed+10000)
        hourly = grid.sample_trajectory(row,horizon=24,stochastic=True,tail_strength=.6)
        sessions = stress_ev_sessions(sessions,stress)
        hourly = stress_grid_hourly(hourly,stress)
        joint = build_joint_hourly(hourly,build_ev_hourly_from_sessions(sessions,hourly))
        g = build_grid_input(joint,options)
        e = build_ev_session_input(sessions,g,options)
        if e.empty:
            continue
        if group=="capacity_shortage":
            hours = np.arange(24)
            arrival = e.arrival_step.to_numpy()[:,None]
            depart = e.departure_step.to_numpy()[:,None]
            power = (e.max_charging_power_kw.to_numpy()[:,None]*((arrival<=hours)&(depart>hours))).sum(axis=0)
            if np.count_nonzero((power>0)&(g.ev_capacity_limit_kw.to_numpy()<1.1*power))<4:
                continue
        folder = inputs/f"scenario_{len(audit):02}"
        folder.mkdir(parents=True,exist_ok=True)
        ep,gp = folder/"optimizer_ev_sessions.csv",folder/"optimizer_grid.csv"
        e.to_csv(ep,index=False)
        g.to_csv(gp,index=False)
        fleet,_,_ = read_case(folder)
        parent = fingerprint(fleet)
        if parent in exclude:
            continue
        exclude.add(parent)
        audit.append(dict(seed=seed,parent=parent,scenario=folder.name,
                          ev_sha256=digest(ep),grid_sha256=digest(gp)))
        if len(audit)==40:
            break
    assert len(audit)==40
    manifest=output/f"manifest_{group}.csv"
    pd.DataFrame(audit).to_csv(manifest,index=False)
    (output/f"lock_{group}.json").write_text(json.dumps(dict(manifest_sha256=digest(manifest),
         model_hashes=hashes, seed_input_sha256=digest(grid_data), parents=40, first_seed=first_seed),
         indent=2),encoding="utf-8")
    print(group,"40 scenarios locked",flush=True)
    return inputs


def group_run(source,grid_data,output,group,config,exclude):
    torch.set_num_threads(1)
    inputs = generate(source,grid_data,output,group,set(exclude))
    m = load_benchmark(source)
    optimizer,path = m.load_optimization_module()
    optimizer = m.patch_optimizer_for_top_config(optimizer)
    selected = pd.read_csv(config)
    selection = selected[(selected.group==group)&selected.arm.isin(["BO","GPT-6 Sol"])]
    for item in selection.to_dict("records"):
        arm = item["arm"]
        values = {k:item[f"param_{k}"] for k in m.PARAM_COLS}
        folder = output/"replay"/group/arm
        cfg = m.params_to_optimizer_config(optimizer,group,values,inputs,folder)
        summary = optimizer.run_all(cfg)
        assert len(summary)==40
        summary.to_csv(output/f"{group}_{arm}.csv",index=False)
    print(group,"paired replay complete",flush=True)


def run(source,grid_data,history,output):
    output.mkdir(parents=True,exist_ok=False)
    config=lock_configs(history,output)
    excluded=set()
    for f in output.parent.glob("*/locked_manifest.csv"):
        d=pd.read_csv(f)
        for k in ("parent","parent_fingerprint"):
            if k in d:
                excluded.update(d[k])
    excluded.update(pd.read_csv(output.parent/"bo_parity_feedback_pair/manifest.csv").parent)
    (output/"selection_protocol.json").write_text(json.dumps(dict(history_sha256=digest(history),
         selection_sha256=digest(config), main_budget=8, archived_seed_rows_counted_twice=False,
         new_parents_per_regime=40, capacity_filter="At least four active hours below 10% raw capacity margin in shortage group.",
         replay="Frozen winners, original optimizer and capacity floor, no additional proposals."),indent=2),encoding="utf-8")
    with ProcessPoolExecutor(max_workers=4) as pool:
        tasks=[pool.submit(group_run,source,grid_data,output,g,config,excluded) for g in GROUPS]
        for task in tasks:
            task.result()
    fingerprints=[x for g in GROUPS for x in pd.read_csv(output/f"manifest_{g}.csv").parent]
    assert len(set(fingerprints))==160


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source","grid_data","history","output"):
        p.add_argument(name,type=Path)
    run(*(v.resolve() for v in vars(p.parse_args()).values()))
