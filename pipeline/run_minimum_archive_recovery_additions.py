"""Minimum common recovery capacity additions under visible-deadline dispatch."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_binding_archive_recovery import capacity, certified_shift
from run_paper_archive_recovery import EPS, START, flat_plan, targets
from run_section_iv_replay import ROOT, load, plt
from run_visible_deadline_archive import dispatch

REQUESTS = (4.1, 4.5, 5.5)
HOURS = tuple(range(18,24))
PRESSURES = ("00","01","10","11")


def prepare(reserve):
    pairs=[]
    for normal,peak in zip(load("joint",0)[7:],load("evening_peak_cluster",0)[7:]):
        name,nf,_=normal
        _,pf,_=peak
        caps=capacity((nf,pf),reserve)
        for ev,fleet in enumerate((nf,pf)):
            ref=flat_plan(fleet)
            for grid,cap in enumerate(caps):
                assert np.all(ref.sum(axis=0)<=cap+EPS)
                pairs.append((name,f"{ev}{grid}",fleet,ref,cap))
    return pairs


def energy_audit(pairs):
    rows=[]
    for name,pressure,fleet,ref,cap in pairs:
        if pressure!="11":
            continue
        for hour in HOURS:
            base=float(ref[:,hour].sum())
            rows.append(dict(scenario_id=name,hour=hour,baseline_kw=base,capacity_kw=cap[hour],
                             remaining_room_kwh=cap[hour]-base))
        required=float(targets(fleet).sum()-ref[:,:START+1].sum())
        np.testing.assert_allclose(required,ref[:,18:].sum(),rtol=0,atol=1e-8)
    return pd.DataFrame(rows)


def candidate_results(pairs,reserve,request,addition,name):
    rows,traces,vehicles=[],[],[]
    for case,pressure,fleet,ref,original in pairs:
        cap=original.copy()
        # Change only the capacity-stress arms. Keep the event action and
        # the no-response baseline fixed in every intervention.
        if pressure[1]=="1":
            cap[18:24]+=addition
        delta,possible=certified_shift(fleet,ref,4*reserve,request)
        result,trace,per_vehicle=dispatch(fleet,ref,cap,delta,request,possible,"visible_deadline")
        keys=dict(scenario_id=case,pressure=pressure,request_kw=request,allocation=name)
        rows.append(dict(**keys,added_kwh=float(addition.sum()) if pressure[1]=="1" else 0.,
                         **{f"added_{h}_kw":addition[j] if pressure[1]=="1" else 0.
                            for j,h in enumerate(HOURS)},**result))
        traces.extend(dict(**keys,**row) for row in trace)
        vehicles.extend(dict(**keys,**row) for row in per_vehicle)
    return rows,traces,vehicles


def summarize(rows,audit,out):
    joint=rows[rows.pressure=="11"]
    scores=joint.groupby(["request_kw","allocation"],as_index=False).agg(
        delivered_cases=("delivered","sum"),service_cases=("full_feasible_target","sum"),
        aging_cases=("aging_pass","sum"),added_kwh=("added_kwh","first"),
        max_extra_soh=("max_extra_soh","max"),worst_unmet_kwh=("unmet_feasible_kwh","max"))
    scores.to_csv(out/"joint_allocation_results.csv",index=False)
    all_counts=rows.groupby(["request_kw","allocation","pressure"],as_index=False)[
        ["full_feasible_target","full_requested","tracking","aging_pass","delivered"]
    ].sum()
    all_counts.to_csv(out/"four_group_counts.csv",index=False)

    lines=["# Frozen recovery-capacity sensitivity","",
           "The 17:00 action, flat benchmark, aging budget and visible-deadline recovery rule are unchanged.",
           "Each candidate adds capacity only after 17:00 and only to the capacity-stress arms.",
           "Additions are identical across the three archive IDs and are visible only when the relevant hour is dispatched.",
           "For an hour-long activation, q kW creates q kWh of recovery demand. With baseline charging protected, the binding event's cumulative recovery headroom gives an energy-conservation lower bound.",
           "", "## Joint stress, minimum-energy allocations", "",
           "| Request kW | Energy lower bound kWh | Single-hour placements delivering 3/3 | Uniform placement delivery |",
           "|---|---:|---|---:|"]
    for q in REQUESTS:
        part=scores[scores.request_kw.eq(q)]
        successful=part[part.allocation.str.startswith("hour_")&part.delivered_cases.eq(3)]
        uniform=part[part.allocation.eq("uniform")].iloc[0]
        placements=", ".join(f"{int(s.split('_')[1]):02d}:00" for s in successful.allocation) or "none"
        lines.append(f"| {q:.2f} | {uniform.added_kwh:.2f} | {placements} | {uniform.delivered_cases:g}/3 |")
    lines+=["","If a placement reaches the energy lower bound and delivers all three cases, it is minimum in total added energy under the unchanged 17:00 action.",
            "Hour labels denote the start of a one-hour slot. A reported added kWh in that slot equals the constant added kW.",
            "Service means the physically feasible per-vehicle departure target. Raw requested service remains in the event table.",
            "These are the same three previously inspected archive events. No LLM was called."]
    (out/"conclusions.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    labels=["baseline",*(f"hour_{hour}" for hour in HOURS),"uniform"]
    chart=scores.pivot(index="request_kw",columns="allocation",values="delivered_cases").reindex(columns=labels)
    fig,ax=plt.subplots(figsize=(9,3.4),constrained_layout=True)
    im=ax.imshow(chart,cmap="Blues",vmin=0,vmax=3,aspect="auto")
    ax.set(xticks=range(len(labels)),xticklabels=["Frozen cap",*(f"{h}:00" for h in HOURS),"Uniform"],
           yticks=range(len(chart)),yticklabels=[f"{q:.2f} kW" for q in chart.index],
           title="Joint-stress commitment deliveries out of three")
    for (i,j),value in np.ndenumerate(chart.to_numpy()):
        ax.text(j,i,f"{value:g}/3",ha="center",va="center",color="white" if value>=2 else "black")
    fig.colorbar(im,ax=ax,ticks=range(4))
    fig.savefig(out/"capacity_placement.png",dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"minimum_archive_recovery_additions_results")
    out=parser.parse_args().output
    out.mkdir(parents=True,exist_ok=True)
    prior=ROOT/"binding_archive_recovery_results"/"frozen_configuration.json"
    reserve=np.asarray(json.loads(prior.read_text())["reserve_kw_by_hour"])
    pairs=prepare(reserve)
    audit=energy_audit(pairs)
    audit.to_csv(out/"joint_recovery_energy_audit.csv",index=False)
    tight=float(audit.groupby("scenario_id").remaining_room_kwh.sum().min())
    rows,traces,vehicles=[],[],[]
    for q in REQUESTS:
        need=round(max(0.,q-tight),10)
        allocations={"baseline":np.zeros(6),"uniform":np.repeat(need/6,6)}
        for j,hour in enumerate(HOURS):
            allocation=np.zeros(6)
            allocation[j]=need
            allocations[f"hour_{hour}"]=allocation
        for name,addition in allocations.items():
            batch,hourly,individual=candidate_results(pairs,reserve,q,addition,name)
            rows.extend(batch)
            traces.extend(hourly)
            vehicles.extend(individual)
        print(f"{q:.2f} kW: energy lower bound {need:.2f} kWh",flush=True)
    rows=pd.DataFrame(rows)
    assert len(rows)==288
    assert (rows.delivered==(rows.full_feasible_target & rows.tracking & rows.aging_pass)).all()
    actions=rows.groupby(["scenario_id","pressure","request_kw"]).actual_reduction_kw.agg(["min","max"])
    np.testing.assert_allclose(actions["min"],actions["max"],rtol=0,atol=1e-8)
    baseline=rows[rows.allocation=="baseline"]
    previous=pd.read_csv(ROOT/"visible_deadline_archive_results"/"event_replays.csv",dtype={"pressure":str})
    previous=previous[(previous.split=="test")&(previous.policy=="visible_deadline")]
    compare=baseline.merge(previous,on=["scenario_id","pressure","request_kw"],validate="one_to_one")
    assert len(compare)==36 and (compare.delivered_x==compare.delivered_y).all()
    rows.to_csv(out/"event_replays.csv",index=False)
    pd.DataFrame(traces).to_csv(out/"hourly_recovery_trace.csv",index=False)
    pd.DataFrame(vehicles).to_csv(out/"vehicle_service_and_aging.csv",index=False)
    summarize(rows,audit,out)


if __name__=="__main__":
    main()
