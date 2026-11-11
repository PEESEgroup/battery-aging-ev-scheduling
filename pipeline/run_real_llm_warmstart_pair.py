"""Compare real-LLM and rule starts with a shared scheduler and local search."""
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

BUDGET = 6
SEED = 42
GROUPS = ("joint", "evening_peak_cluster", "sustained_high_price", "capacity_shortage")


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def setup(source, output):

    sys.path.insert(0, str(source))
    core = importlib.import_module("llm_top_config_optimization_actual_eval_v7_ablation_layout")
    core.OUT_DIR = output
    core.OPTIMIZATION_SCRIPT_CANDIDATES = [source / p for p in core.OPTIMIZATION_SCRIPT_CANDIDATES]
    core.GROUP_INPUT_DIRS = {k: [source / p for p in v] for k,v in core.GROUP_INPUT_DIRS.items()}
    optimizer, script = core.load_optimization_module()
    core.patch_optimizer_for_top_config(optimizer)
    paths = {}
    for group in GROUPS:
        _, cases = core.list_scenario_dirs_for_group(group)
        assert [p.name for p in cases] == [f"scenario_{i:02d}" for i in range(10)]
        paths[group] = {k: core.materialize_eval_input_dir(group, s, k)
                        for k,s in (("dev",cases[:7]),("test",cases[7:]))}
    return core, optimizer, script, paths


def evaluate(core, optimizer, paths, group, split, params, baseline, arm, step):
    dev_dirs = optimizer.list_scenario_dirs(paths[group]["dev"], "scenario_")
    grids = optimizer.load_all_grid_scenarios(dev_dirs)
    robust = optimizer.build_robust_hourly_signals
    # Retain candidate-specific risk weights while using development-only forecast inputs.
    with patch.object(optimizer, "build_robust_hourly_signals", lambda _grids, config: robust(grids, config)):
        return core.evaluate_actual_candidate(optimizer, group, f"step_{step:02d}", params,
                                              paths[group][split], baseline, f"{split}_{arm}")


def state(core, group, metrics, normal):
    keys = {"cvar_unmet_demand":"cvar_unmet_energy_kwh",
            "cost_per_served_kwh":"weighted_price_per_kwh",
            "carbon_per_served_kwh":"weighted_emission_per_kwh",
            "degradation_per_served_kwh":"delta_soh_per_kwh",
            "peak_load":"mean_peak_ev_power_kw"}
    values = {k: float(metrics[m]) for k,m in keys.items()}
    relative = {f"{k}_rel": values[k]/max(float(normal[m]),1e-12) for k,m in keys.items()}
    return core.StateSummary(scenario_name=group, representative_lambda=None, stress_type=group,
                             objective_priority="service and composite score",
                             primary_bottleneck="not independently classified",
                             service_ratio=float(metrics["service_ratio"]), **values, **relative)


def run(source, output, stage):
    output.mkdir(parents=True, exist_ok=True)
    core, optimizer, script, paths = setup(source, output)
    if stage == "prepare":
        baseline = {}
        for group in GROUPS:
            row = evaluate(core, optimizer, paths, group, "dev", core.BASELINE_PARAMS, None, "baseline", 0)
            baseline[group] = {k: float(v) for k,v in row.items() if isinstance(v,(int,float)) and np.isfinite(v)}
            print(f"Development baseline complete: {group}", flush=True)
        states = {g: state(core,g,baseline[g],baseline["joint"]) for g in GROUPS}
        write(output/"development_baselines.json",baseline)
        write(output/"proposal_inputs.json",{
            "bounds":core.PARAM_BOUNDS, "default_weights":core.BASELINE_PARAMS,
            "states":{g:{k:round(v,7) if isinstance(v,float) else v for k,v in st.to_dict().items() if v is not None} for g,st in states.items()},
            "rule_seeds":{g:core.rule_based_prior(st) for g,st in states.items()}})
        write(output/"protocol.json",{"seed":SEED,"budget_per_arm":BUDGET,"development":"00–06",
                                       "test":"07–09","forecast":"development grid trajectories in both splits",
                                       "optimizer":str(script.resolve()),
                                       "optimizer_sha256":hashlib.sha256(script.read_bytes()).hexdigest(),
                                       "aging":"archived configuration optimizer uses its semi-empirical model",
                                       "outcome":"departure service and unmet energy"})
        return

    inputs=json.loads((output/"proposal_inputs.json").read_text(encoding="utf-8"))
    seeds=json.loads((output/"real_llm_seeds.json").read_text(encoding="utf-8-sig")) if stage == "evaluate" else {}
    baseline=json.loads((output/"development_baselines.json").read_text(encoding="utf-8"))
    assert stage == "rule" or set(seeds)==set(GROUPS)
    curve, held, per_scenario = [], [], []
    for group in GROUPS:
        test_base=evaluate(core,optimizer,paths,group,"test",core.BASELINE_PARAMS,None,"baseline",0)
        starts = [("rule",inputs["rule_seeds"][group])]
        if stage == "evaluate":
            starts.append(("llm",seeds[group]["parameters"]))
        for arm, start in starts:
            assert set(start)==set(core.PARAM_COLS)
            assert all(core.PARAM_BOUNDS[k][0]<=float(v)<=core.PARAM_BOUNDS[k][1] for k,v in start.items())
            rng=np.random.default_rng(SEED)
            records=[]
            for step in range(1,BUDGET+1):
                candidate=start if step==1 else core._suggest_llm_feedback_local_bo_candidate(records,start,rng)
                row=evaluate(core,optimizer,paths,group,"dev",candidate,baseline[group],arm,step)
                row["step"]=step
                records.append(row)
                curve.append(row)
                print(f"{group} {arm} development {step}/{BUDGET}: {row['actual_score']:.5f}", flush=True)
            best=min(records,key=lambda r:r["actual_score"])
            selected={k:best[f"param_{k}"] for k in core.PARAM_COLS}
            for tag,params in (("initial",start),("selected",selected)):
                evaluated=evaluate(core,optimizer,paths,group,"test",params,test_base,f"{arm}_{tag}",best["step"] if tag=="selected" else 1)
                evaluated["arm"]=arm
                evaluated["stage"]=tag
                evaluated["dev_selection_step"]=best["step"] if tag=="selected" else 1
                held.append(evaluated)
                path=output/"actual_runs"/group/evaluated["method"]/(evaluated["candidate"]+"_"+evaluated["param_hash"])/"stochastic_results_summary.csv"
                frame=pd.read_csv(path)
                frame["group"],frame["arm"],frame["stage"]=group,arm,tag
                per_scenario.append(frame)
            pd.DataFrame(curve).to_csv(output/"development_curve.csv",index=False)
            pd.DataFrame(held).to_csv(output/"heldout_metrics.csv",index=False)
            pd.concat(per_scenario,ignore_index=True).to_csv(output/"heldout_scenarios.csv",index=False)
    expected = 16 if stage == "evaluate" else 8
    assert len(held)==expected and sum(len(f) for f in per_scenario)==3*expected


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("stage",choices=("prepare","rule","evaluate"))
    args=parser.parse_args()
    run(args.source.resolve(),args.output.resolve(),args.stage)



