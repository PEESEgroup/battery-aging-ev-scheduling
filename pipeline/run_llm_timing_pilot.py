"""Paired timing-pilot warm starts with a hard-cap rolling replay."""
import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from nonanticipative_scheduler import Forecast, cvar
from run_directional_pinn_validation import Aging, TOL, read_case, plan, replay

START, RECOVERY, REQUEST = 17, 18, 2.
BUDGET, SEED = 6, 42
CHECKPOINTS = (1, 3, 6)
BOUNDS = {"lambda_risk": (0., .9), "w_price": (.4, 2.2),
          "unmet_penalty_per_kwh": (.5, 10.), "recovery_reserve_kw": (0., 6.)}
DEFAULTS = {"lambda_risk": .3, "w_price": 1., "unmet_penalty_per_kwh": 2., "recovery_reserve_kw": 0.}


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


class TimingForecast(Forecast):
    def scenario_paths(self, t, observed):
        # Only the observed prefix and development donors enter the forecast.
        history = self.signals(observed)
        distance = (((self.signals(self.donors[:, :t+1]) - history) / self.scale)**2).mean(axis=(1,2))
        paths = self.donors[np.argsort(distance, kind="stable")[:4],t:].copy()
        paths[:,0] = observed[-1]
        return paths

    def build_case(self, t, observed, params, announced):
        paths = self.scenario_paths(t, observed)
        q = np.clip((self.signals(paths)-self.center)/self.scale,0.,1.)
        risk = params["lambda_risk"]
        stress = q.mean(axis=0) + risk*q.std(axis=0) + risk*cvar(q,.95)
        weights = np.array([params["w_price"],30.,.2,1.])
        caps = paths[...,3].min(axis=0)
        for hour, cap in announced.items():
            if hour >= t:
                caps[hour-t] = min(caps[hour-t],cap)
        # The reserve limits the reference only. Activation may use the physical room.
        planning = caps.copy()
        if t <= RECOVERY:
            planning[RECOVERY-t] = max(0., planning[RECOVERY-t]-params["recovery_reserve_kw"])
        return caps, planning, stress @ weights / weights.sum()


def rolling_reference(case, forecast, params):
    fleet, grid, temp, cap_hour = case
    announced = {} if cap_hour is None else {cap_hour: float(grid[cap_hour,3])}
    ref = np.zeros((len(fleet.arrival),24))
    for t in range(24):
        caps, planning, score = forecast.build_case(t, grid[:t+1], params, announced)
        # The imported LP uses score - 2. An additive offset sets the requested service penalty.
        projected = plan(fleet,t,planning,score+2.-params["unmet_penalty_per_kwh"],ref)
        if t == START:
            proposal, activation_caps = projected, caps
        ref[:,t] = projected[:,t]
    assert np.all(ref.sum(axis=0)<=grid[:,3]+TOL)
    return ref, proposal, activation_caps


def simulate(case, forecast, aging, params, *, reference=None, debt_recovery=False):
    fleet, grid, temp, _ = case
    if reference is None:
        reference = rolling_reference(case, forecast, params)
    ref, proposal, activation_caps = reference
    baseline = aging.losses(fleet,ref,temp)
    if debt_recovery:
        from debt_aware_recovery import replay_with_debt
        action, actual, service, health, service_at_recovery = replay_with_debt(
            case, forecast, params, aging, ref, proposal, activation_caps, REQUEST, baseline)
        physical = True  # The debt replay audits every physical constraint before returning.
    else:
        action, actual, service, health, physical = replay(
            fleet,grid,ref,proposal,activation_caps,-1,REQUEST,aging,temp,baseline)
    assert np.all(action.sum(axis=0)<=grid[:,3]+TOL) and physical
    tracked = actual >= REQUEST-TOL
    qualifies = tracked and service and health and bool(np.all(baseline<=aging.limit+TOL))
    served = float(action.sum())
    total_loss = aging.losses(fleet,action,temp)
    hourly = action.sum(axis=0)
    metrics = dict(unmet_kwh=float(fleet.demand.sum()-served),
                price_per_kwh=float(hourly @ grid[:,0])/max(served,1e-9),
                carbon_per_kwh=float(hourly @ grid[:,1])/max(served,1e-9),
                loss_per_kwh=float(total_loss.sum())/max(served,1e-9),
                peak_kw=float(hourly.max()), qualified=qualifies, tracked=tracked, service_match=service, aging_pass=health,
                reference_floor_pass=bool(np.all(baseline<=aging.limit+TOL)),
                served_ratio=float(action.sum()/fleet.demand.sum()),
                reference_served_ratio=float(ref.sum()/fleet.demand.sum()),
                mean_soh_loss=float(total_loss.mean()), actual_kw=actual)
    if debt_recovery:
        metrics.update(service_at_recovery=service_at_recovery,
                       qualified_at_recovery=bool(qualifies and service_at_recovery))
    return metrics


def evaluate(cases, forecast, aging, params):
    return [{"parent":name,**simulate(case,forecast,aging,params)} for name,case in cases.items()]


def summarize(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!="parent"}


def score(metrics):
    # One additional qualified dev case dominates all service/health tie-breaks.
    return -10*metrics["qualified"] + (1-metrics["served_ratio"]) + metrics["mean_soh_loss"]


def bounded(params):
    assert set(params)==set(BOUNDS)
    return {k:float(np.clip(params[k],*BOUNDS[k])) for k in BOUNDS}


def timing_prior(original, state):
    activation = state["activation_capacity_to_connected_power"]
    recovery = state["recovery_capacity_to_connected_power"]
    late = state["due_by_recovery_energy_share"]
    high = state["recovery_minus_activation_price_per_kwh"]
    weights = dict(original)
    weights["recovery_reserve_kw"] = min(6.,2.+2.*late) if recovery < .5 else 0.
    if activation < .5 or recovery < .5:
        weights["unmet_penalty_per_kwh"] += 2.*late + .5
    if recovery < .5:
        weights["lambda_risk"] += .15
    if high > .025 and recovery < .5:
        weights["w_price"] *= .85
    return bounded(weights)


def make_state(group, cases, baseline):
    fleet_rows=[]
    for fleet,grid,temp,hour in cases.values():
        demand=fleet.demand.sum()
        connected=[fleet.power[(fleet.arrival<=t)&(fleet.departure>t)].sum() for t in (START,RECOVERY)]
        fleet_rows.append({
            "activation_capacity_to_connected_power":min(2.,grid[START,3]/max(connected[0],1e-6)),
            "recovery_capacity_to_connected_power":min(2.,grid[RECOVERY,3]/max(connected[1],1e-6)),
            "recovery_minus_activation_price_per_kwh":float(grid[RECOVERY,0]-grid[START,0]),
            "due_by_recovery_energy_share":float(fleet.demand[(fleet.departure<=RECOVERY+1)&(fleet.departure>START)].sum()/demand)})
    enhanced=summarize(fleet_rows)
    return {"regime":"normal" if group=="control" else "timing stress",
            "baseline_served_ratio":baseline["served_ratio"],"baseline_qualified_rate":baseline["qualified"],
            "baseline_soh_loss":baseline["mean_soh_loss"],"fixed_request_kw":REQUEST,
            "zero_incremental_aging_budget":True,
            **{k:round(v,6) for k,v in enhanced.items()}}


def run(source, scenarios, output, stage):
    output.mkdir(parents=True,exist_ok=True)
    aging=Aging(source)
    core=importlib.import_module("llm_top_config_optimization_actual_eval_v7_ablation_layout")
    manifest=pd.read_csv(scenarios/"manifest.csv")
    assert len(manifest)==50
    assert manifest.groupby("parent")["split"].nunique().eq(1).all()
    datasets={}
    for group, subset in manifest.groupby("cell",sort=True):
        cases={}
        for r in subset.itertuples():
            fleet,grid,temp=read_case(scenarios/group/r.parent)
            cases.setdefault(r.split,{})[r.parent]=(fleet,grid,temp,int(r.cap_hour) if pd.notna(r.cap_hour) else None)
        assert len(cases["development"])==7 and len(cases["pilot_test"])==3
        forecast=TimingForecast([case[1] for case in cases["development"].values()])
        datasets[group]=cases,forecast
    if stage=="prepare":
        prep={}
        aggregates={}
        for group,(cases,forecast) in datasets.items():
            rows=evaluate(cases["development"],forecast,aging,DEFAULTS)
            aggregates[group]=summarize(rows)
            aggregates[group]["tail_unmet_kwh"]=float(cvar(np.array([r["unmet_kwh"] for r in rows]),.95))
        normal=aggregates["control"]
        keys={"cvar_unmet_demand":"tail_unmet_kwh","cost_per_served_kwh":"price_per_kwh",
              "carbon_per_served_kwh":"carbon_per_kwh","degradation_per_served_kwh":"loss_per_kwh",
              "peak_load":"peak_kw"}
        for group,(cases,forecast) in datasets.items():
            baseline=aggregates[group]
            state_metrics={k:baseline[v] for k,v in keys.items()}
            state_metrics.update({k+"_rel":baseline[v]/max(normal[v],1e-12) for k,v in keys.items()})
            state=core.StateSummary(scenario_name="joint",representative_lambda=None,stress_type="normal",
                 objective_priority="service",primary_bottleneck="unclassified",service_ratio=baseline["served_ratio"],
                 **state_metrics)
            original_full=core.rule_based_prior(state)
            original=bounded({**{k:original_full[k] for k in BOUNDS if k!="recovery_reserve_kw"},
                               "recovery_reserve_kw":0.})
            enhanced=make_state(group,cases["development"],baseline)
            enhanced.update({k:round(v,8) for k,v in state_metrics.items()})
            prep[group]=dict(enhanced_state=enhanced, original_rule=original, timing_rule=timing_prior(original,enhanced))
            print(group,flush=True)
        dump(output/"states.json",prep)
        dump(output/"protocol.json",dict(budget=BUDGET, checkpoints=list(CHECKPOINTS),seed=SEED,parameters=BOUNDS,
             defaults=DEFAULTS, score="-10*qualified + (1-served_ratio) + mean_soh_loss",
             split="same 7 development and 3 pilot-test parents in every cell",
             request="2 kW downward at 17:00, recovery at 18:00", data="50-case timing pilot",
             proposal="one GPT-4.1 mini output per cell; only enhanced development summaries, defaults and bounds"))
        return

    prep=json.loads((output/"states.json").read_text(encoding="utf-8"))
    llm=json.loads((output/"llm_seeds.json").read_text(encoding="utf-8-sig"))
    assert set(llm)==set(datasets)
    core.PARAM_COLS=list(BOUNDS)
    core.PARAM_BOUNDS=BOUNDS
    core.BASELINE_PARAMS={**core.BASELINE_PARAMS, **DEFAULTS}
    dev_rows,test_rows=[],[]
    for group,(cases,forecast) in datasets.items():
        arms={"original_rule":prep[group]["original_rule"],"timing_rule":prep[group]["timing_rule"],
              "llm":llm[group]["parameters"]}
        for arm,start in arms.items():
            rng=np.random.default_rng(SEED)
            records=[]
            for step in range(1,BUDGET+1):
                candidate=bounded(start if step==1 else core._suggest_llm_feedback_local_bo_candidate(records,start,rng))
                metrics=summarize(evaluate(cases["development"],forecast,aging,candidate))
                row=dict(cell=group,arm=arm,step=step,actual_score=score(metrics),**metrics,
                         **{f"param_{k}":v for k,v in candidate.items()})
                records.append(row)
                dev_rows.append(row)
            assert len({tuple(r[f"param_{k}"] for k in BOUNDS) for r in records})==BUDGET
            for step in CHECKPOINTS:
                best=min(records[:step],key=lambda r:r["actual_score"])
                params={k:best[f"param_{k}"] for k in BOUNDS}
                for row in evaluate(cases["pilot_test"],forecast,aging,params):
                    test_rows.append(dict(cell=group,arm=arm,budget=step,selected_step=best["step"],**row))
            print(group,arm,"complete",flush=True)
        pd.DataFrame(dev_rows).to_csv(output/"development.csv",index=False)
        pd.DataFrame(test_rows).to_csv(output/"heldout.csv",index=False)
    df=pd.DataFrame(test_rows)
    assert len(df)==135 and not df.duplicated(["cell","arm","budget","parent"]).any()
    df["bottleneck"]=np.select([df.cell.str.startswith("cap_18"),df.cell.str.startswith("cap_17")],
                              ["recovery","activation"],default="control")
    for key in ("cell","bottleneck"):
        df.groupby([key,"arm","budget"],sort=True).agg(
            cases=("parent","size"),qualified_rate=("qualified","mean"),tracking_rate=("tracked","mean"),
            served_ratio=("served_ratio","mean"),mean_soh_loss=("mean_soh_loss","mean"),
            reference_floor_rate=("reference_floor_pass","mean")).reset_index().to_csv(output/f"summary_{key}.csv",index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("scenarios",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("stage",choices=("prepare","evaluate"))
    a=parser.parse_args()
    torch.set_num_threads(1)
    run(a.source.resolve(),a.scenarios.resolve(),a.output.resolve(),a.stage)


