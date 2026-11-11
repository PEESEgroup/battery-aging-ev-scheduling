"""Diagnose zero requests and replay a frozen, binding recovery capacity design."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from nonanticipative_scheduler import AGING, TOL, aging, replay_aging
from run_paper_archive_recovery import ALLOWANCE, EPS, START, controlled_cases, flat_plan, status, targets
from run_section_iv_replay import ROOT, load, plt

RECOVERY = slice(START+1,24)
POLICIES = ("reserved", "nominal_probe")


def audit(output, prior):
    before=pd.read_csv(prior/"calibration_replays.csv",dtype={"pressure":str})
    zero=before[before.request_kw.eq(0)].copy()
    zero["reference_tracking_failed"]=~zero.tracking
    zero["executed_aging_exceeded"]=~zero.aging_pass
    zero["offline_unresolved"]=zero.offline_health_status.ne("solved")
    zero["first_rolling_cut_limit"]=zero.solve_failures.fillna("").str.startswith("17:cut_limit")
    config=json.loads((prior/"frozen_configuration.json").read_text(encoding="utf-8"))
    archive=dict(controlled_cases(load("joint",0),load("evening_peak_cluster",0),
                                  config["recovery_headroom_kw"]))
    witnesses=[]
    for row in zero.itertuples():
        fleet,grid,ref=archive[row.scenario_id][row.pressure]
        loss=replay_aging(fleet,ref)
        witnesses.append(bool(np.allclose(ref.sum(axis=1),targets(fleet)) and
                              np.all(ref.sum(axis=0)<=grid[:,3]+EPS) and
                              np.all(loss<=1-AGING.min_soh+TOL)))
    zero["zero_reference_health_and_service_witness"]=witnesses
    assert len(zero)==24 and all(witnesses)
    zero.to_csv(output/"prior_zero_request_audit.csv",index=False)
    return zero


def physical_bound(fleet, ref, caps):
    ids,hours=np.where((fleet.arrival[:,None]<=np.arange(START,24)) &
                       (fleet.departure[:,None]>np.arange(START,24)))
    columns=np.arange(len(ids))
    aeq=np.zeros((len(fleet.arrival),len(ids)))
    aub=np.zeros((len(caps),len(ids)))
    aeq[ids,columns]=1
    aub[hours,columns]=1
    objective=(hours==0).astype(float)
    result=linprog(objective,A_eq=aeq,b_eq=targets(fleet)-ref[:,:START].sum(axis=1),
                   A_ub=aub,b_ub=caps,bounds=list(zip(np.zeros(len(ids)),fleet.power[ids])),
                   method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return float(ref[:,START].sum()-result.fun)


def capacity(fleets, reserve):
    profiles=[flat_plan(fleet).sum(axis=0) for fleet in fleets]
    base=np.full(24,1500.)
    tight=base.copy()
    tight[RECOVERY]=np.maximum(*profiles)[RECOVERY]+reserve
    return base,tight


def fit(pairs):
    planning=pairs[:4]
    calibration=pairs[4:7]
    reference_kw=np.median([flat_plan(fleet)[:,START].sum()
                            for _,fleets in planning for fleet in fleets])
    # Candidate budgets are total extra energy after 17:00, not per-hour caps.
    candidate_totals=(round(.10*reference_kw,1),round(.075*reference_kw,1),round(.05*reference_kw,1))
    checks=[]
    for total in candidate_totals:
        reserved=np.repeat(total/6,6)
        for split, cases in (("planning",planning),("calibration",calibration)):
            for name,fleets in cases:
                loose,tight=capacity(fleets,reserved)
                for ev,fleet in enumerate(fleets):
                    ref=flat_plan(fleet)
                    assert np.all(ref.sum(axis=0)<=tight+EPS)
                    a,b=[physical_bound(fleet,ref,cap[START:]) for cap in (loose,tight)]
                    checks.append(dict(total_reserve_kwh=total,split=split,scenario_id=name,
                                       ev_pressure=ev,loose_kw=a,tight_kw=b,binding=a>b+EPS))
        sample=[row for row in checks if row["total_reserve_kwh"]==total and row["split"]=="calibration" and row["ev_pressure"]==1]
        if all(row["binding"] for row in sample):
            # Use the least restrictive candidate that binds every late-peak calibration case.
            return reserved, tuple(sorted({0.,.5,1.,2.,round(total,1),round(2*total,1)})), pd.DataFrame(checks)
    raise RuntimeError("No candidate bound all late-peak calibration cases")


def certified_shift(fleet, ref, headroom, q):
    # At 17:00 only connected vehicles are eligible. Each shifted kWh has a
    # fixed even recovery pattern within that vehicle's remaining plug window.
    idx=np.flatnonzero((fleet.arrival<=START)&(fleet.departure>START+1))
    patterns=np.zeros((len(idx),24))
    upper=np.zeros(len(idx))
    base_loss=replay_aging(fleet,ref)
    for j,i in enumerate(idx):
        end=fleet.departure[i]
        count=end-START-1
        patterns[j,START]=-1
        patterns[j,START+1:end]=1/count
        hi=max(0.,min(ref[i,START],count*np.min(fleet.power[i]-ref[i,START+1:end])))
        lo=0.
        start=fleet.arrival[i]
        # Along the flat reference, loss is convex in the shift. Its sublevel
        # set containing zero yields an interval of permissible shifts.
        for _ in range(24):
            mid=(lo+hi)/2
            power=(ref[i]+mid*patterns[j])[start:end]
            loss=aging(power,fleet.soc[i],fleet.capacity[i])[0].sum()
            if loss<=min(base_loss[i]+ALLOWANCE,1-AGING.min_soh)+1e-12:
                lo=mid
            else:
                hi=mid
        upper[j]=lo
    if q<=EPS or not len(idx):
        return np.zeros_like(ref), 0.
    result=linprog(-np.ones(len(idx)),A_ub=patterns[:,RECOVERY].T,b_ub=headroom,
                   bounds=list(zip(np.zeros(len(idx)),upper)),method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    possible=float(result.x.sum())
    delta=np.zeros_like(ref)
    if possible>0 and possible>=q-EPS:
        delta[idx]=patterns*(result.x*q/possible)[:,None]
    return delta,possible


def replay(fleet, ref, caps, delta, q, possible):
    # Causal reserved schedule. During recovery, protect the observed flat
    # service first, then spend actual remaining capacity on make-up energy.
    executed=ref.copy()
    executed[:,START]+=delta[:,START]
    backlog=-delta[:,START].copy()
    for hour in range(START+1,24):
        room=max(0.,caps[hour]-ref[:,hour].sum())
        connected=(fleet.arrival<=hour)&(fleet.departure>hour)
        for i in np.argsort(fleet.departure,kind="stable"):
            if not connected[i] or backlog[i]<=EPS:
                continue
            # Try scheduled restoration first, with deferred energy pulled
            # forward only when it was previously curtailed.
            scheduled=delta[i,hour:24].sum()
            due=max(0.,backlog[i]-scheduled+delta[i,hour])
            take=max(0.,min(room,backlog[i],due,fleet.power[i]-ref[i,hour]))
            executed[i,hour]+=take
            backlog[i]-=take
            room-=take
    assert np.all(executed.sum(axis=0)<=caps+EPS)
    assert np.all(executed<=fleet.power[:,None]+EPS)
    result=status(fleet,executed,targets(fleet))
    extra=replay_aging(fleet,executed)-replay_aging(fleet,ref)
    actual=float(ref[:,START].sum()-executed[:,START].sum())
    tracking=actual>=q-EPS
    health=bool(np.all(extra<=ALLOWANCE+TOL))
    delivered=result["full_feasible_target"] and tracking and health
    cause=("delivered" if delivered else "recovery_service_shortfall" if not result["full_feasible_target"]
           else "not_admitted_by_17_plan" if not tracking else "executed_aging_excess")
    return dict(**result,actual_reduction_kw=actual,tracking=tracking,aging_pass=health,
                delivered=delivered,strict_delivery=delivered and result["full_requested"],
                cause=cause,max_extra_soh=float(extra.max()),plan_admission_bound_kw=possible)


def case_rows(pairs,reserve,requests,split):
    rows,bounds=[],[]
    for name,fleets in pairs:
        loose,tight=capacity(fleets,reserve)
        for ev,fleet in enumerate(fleets):
            ref=flat_plan(fleet)
            assert np.allclose(ref.sum(axis=1),targets(fleet))
            for stress,caps in enumerate((loose,tight)):
                assert np.all(ref.sum(axis=0)<=caps+EPS)
                bounds.append(dict(split=split,scenario_id=name,pressure=f"{ev}{stress}",
                                   offline_capacity_kw=physical_bound(fleet,ref,caps[START:]),
                                   baseline_feasible_service=True,baseline_requested_service=bool(np.all(targets(fleet)>=fleet.demand-EPS))))
            for policy,headroom in (("reserved",reserve),("nominal_probe",reserve*4)):
                for q in requests:
                    delta,possible=certified_shift(fleet,ref,headroom,q)
                    for stress,caps in enumerate((loose,tight)):
                        result=replay(fleet,ref,caps,delta,q,possible)
                        rows.append(dict(split=split,scenario_id=name,pressure=f"{ev}{stress}",policy=policy,
                                         request_kw=q,vehicles=len(fleet.arrival),**result))
        print(split,name,flush=True)
    return pd.DataFrame(rows),pd.DataFrame(bounds)


def report(zero,events,bounds,output,frozen):
    tests=events[events.split.eq("test")]
    counts=tests.groupby(["policy","pressure","request_kw"])[
        ["full_requested","full_feasible_target","tracking","aging_pass","delivered"]].sum().reset_index()
    counts.to_csv(output/"test_outcomes.csv",index=False)
    tests.groupby(["policy","pressure","request_kw","cause"]).size().rename("count").reset_index().to_csv(
        output/"test_failure_causes.csv",index=False)
    lines=["# Zero-request diagnosis and binding recovery replay","",
           "The original output is preserved. All tests use the archive rosters and a constructed capacity control.",
           "The prior zero request was a reference ceiling, not the no-activation reference itself.",
           "The new zero request executes that reference exactly.",
           "The stress cap is the pointwise maximum of both flat benchmark loads plus a frozen per-slot reserve from 18:00 to 24:00.",
           "The environment builds the common cap from the rosters. The controller receives only current capacity and the pre-announced reserve; future roster and capacity do not choose its action.",
           "The capacity of 1500 kW outside the constructed recovery stress is an experimental control.",
           "The nominal probe assumes four times the announced reserve and uses the same causal dispatcher.",
           "All service and commitment counts refer to each vehicle's physically feasible target; raw requests are reported separately.",
           "", "## Prior calibration at zero", "",
           "| Rule | Tracking failures | Aging exceedances | Both | Delivered | Offline unresolved |",
           "|---|---:|---:|---:|---:|---:|"]
    for rule,part in zero.groupby("method"):
        lines.append(f"| {rule} | {part.reference_tracking_failed.sum()} | {part.executed_aging_exceeded.sum()} | "
                     f"{(part.reference_tracking_failed & part.executed_aging_exceeded).sum()} | "
                     f"{part.adjusted_delivery.sum()} | {part.offline_unresolved.sum()} |")
    pair=bounds[bounds.split.eq("test")].pivot(index="scenario_id",columns="pressure",values="offline_capacity_kw")
    lines+=["",f"Frozen reserve: {sum(frozen['reserve_kw_by_hour']):.2f} kWh over six hours.",
            f"Held-out offline capacity-only loss, normal/peak respectively: "
            f"{(pair['00']-pair['01']).round(3).tolist()} / {(pair['10']-pair['11']).round(3).tolist()} kW.",
            "The capacity-only bounds do not prove an aging-constrained optimum.",
            "", "## Calibration-frozen requests and test results", "",
            "| Policy | Pressure | Frozen kW | Full original / 3 | Full feasible / 3 | Delivered / 3 |",
            "|---|---|---:|---:|---:|---:|"]
    for policy,q in frozen["selected_requests_kw"].items():
        for row in counts[(counts.policy==policy)&(counts.request_kw==q)].itertuples():
            lines.append(f"| {policy} | {row.pressure} | {q:g} | {row.full_requested}/3 | "
                         f"{row.full_feasible_target}/3 | {row.delivered}/3 |")
    lines+=["","The three held-out archive IDs were used in earlier experiments. No LLM was called."]
    (output/"conclusions.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for (policy,group),part in counts.groupby(["policy","pressure"]):
        if group not in ("10","11"):
            continue
        for ax,metric,title in zip(axes,("full_feasible_target","delivered"),
                                  ("Full feasible-target service","Commitment delivery")):
            ax.plot(part.request_kw,part[metric],"o-",label=f"{policy} {group}")
            ax.set(xlabel="Request (kW)",ylabel="Held-out events out of 3",yticks=range(4),title=title)
            ax.legend(fontsize=8)
    fig.savefig(output/"binding_recovery_delivery.png",dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"binding_archive_recovery_results")
    output=parser.parse_args().output
    output.mkdir(parents=True,exist_ok=True)
    zero=audit(output,ROOT/"paper_archive_recovery_results")
    pairs=[(a[0],(a[1],b[1])) for a,b in zip(load("joint",0),load("evening_peak_cluster",0))]
    reserve,requests,fit_rows=fit(pairs)
    fit_rows.to_csv(output/"planning_calibration_capacity_checks.csv",index=False)
    cal,cal_bounds=case_rows(pairs[4:7],reserve,requests,"calibration")
    selected=cal.groupby(["policy","request_kw"]).delivered.sum()
    selected=selected[selected==12].reset_index().groupby("policy").request_kw.max()
    frozen=dict(reserve_kw_by_hour=reserve.tolist(),requests_kw=requests,selected_requests_kw=selected.to_dict())
    (output/"frozen_configuration.json").write_text(json.dumps(frozen,indent=2)+"\n",encoding="utf-8")
    print("Frozen",frozen,flush=True)
    test,test_bounds=case_rows(pairs[7:],reserve,requests,"test")
    events=pd.concat([cal,test],ignore_index=True)
    bounds=pd.concat([cal_bounds,test_bounds],ignore_index=True)
    assert len(events)==6*4*len(POLICIES)*len(requests)
    first=events.pivot(index=["scenario_id","policy","request_kw"],columns="pressure",values="actual_reduction_kw")
    for a,b in (("00","01"),("10","11")):
        np.testing.assert_allclose(first[a],first[b],atol=1e-9,rtol=0)
    assert events[events.request_kw.eq(0)].delivered.all()
    events.to_csv(output/"event_replays.csv",index=False)
    bounds.to_csv(output/"offline_capacity_bounds.csv",index=False)
    report(zero,events,bounds,output,frozen)


if __name__=="__main__":
    main()
