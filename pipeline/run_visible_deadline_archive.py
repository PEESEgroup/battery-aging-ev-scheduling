"""Causal earliest-departure recovery allocation on the frozen archive caps."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from nonanticipative_scheduler import AGING, TOL, aging, replay_aging
from run_binding_archive_recovery import capacity, certified_shift, replay
from run_paper_archive_recovery import ALLOWANCE, EPS, START, flat_plan, status, targets
from run_section_iv_replay import ROOT, load, plt

REQUESTS = tuple(q/10 for q in range(39,56))
POLICIES = ("scheduled", "visible_deadline")


def health_room(fleet, i, hour, executed, ref, backlog, budget):
    start, end = fleet.arrival[i], fleet.departure[i]
    future = end-hour-1
    charger_room = fleet.power[i]-ref[i,hour]
    lower = max(0.,backlog-future*charger_room)
    upper = min(backlog,charger_room)
    if upper < lower-EPS:
        return 0.
    lower = min(lower, upper)
    point=ref[i,start:end].copy()
    slot=hour-start
    point[:slot]=executed[i,start:hour]

    def loss(amount):
        point[slot]=ref[i,hour]+amount
        if future:
            point[slot+1:]=ref[i,hour+1:end]+(backlog-amount)/future
        return aging(point,fleet.soc[i],fleet.capacity[i])[0].sum()

    if loss(upper)<=budget+1e-12:
        return upper
    if upper-lower<EPS:
        return 0.
    # The remaining-completion loss is convex in the present allocation.
    optimum=minimize_scalar(loss,bounds=(lower,upper),method="bounded",
                            options={"xatol":1e-10})
    safe=min((lower,optimum.x),key=loss)
    if loss(safe)>budget+1e-12:
        return 0.
    lo,hi=safe,upper
    for _ in range(26):
        mid=(lo+hi)/2
        if loss(mid)<=budget+1e-12:
            lo=mid
        else:
            hi=mid
    return lo


def dispatch(fleet,ref,caps,delta,q,possible,policy):
    executed=ref.copy()
    executed[:,START]+=delta[:,START]
    backlog=-delta[:,START].copy()
    ref_loss=replay_aging(fleet,ref)
    budget=np.minimum(ref_loss+ALLOWANCE,1-AGING.min_soh)
    trace=[]
    for hour in range(START+1,24):
        base=ref[:,hour].sum()
        room=max(0.,caps[hour]-base)
        before=backlog.sum()
        for i in np.argsort(fleet.departure,kind="stable"):
            if fleet.arrival[i]>hour or fleet.departure[i]<=hour or backlog[i]<=EPS:
                continue
            if policy=="scheduled":
                due=max(0.,backlog[i]-delta[i,hour:24].sum()+delta[i,hour])
            else:
                due=health_room(fleet,i,hour,executed,ref,backlog[i],budget[i])
            take=max(0.,min(room,backlog[i],due,fleet.power[i]-ref[i,hour]))
            executed[i,hour]+=take
            backlog[i]-=take
            room-=take
        trace.append(dict(hour=hour,observed_capacity_kw=caps[hour],baseline_kw=base,
                          available_recovery_kw=max(0.,caps[hour]-base),
                          scheduled_recovery_kwh=delta[:,hour].sum(),
                          actual_recovery_kwh=before-backlog.sum(),unused_recovery_kw=room,
                          remaining_kwh=backlog.sum()))
    assert np.all(executed.sum(axis=0)<=caps+EPS)
    assert np.all(executed<=fleet.power[:,None]+EPS)
    assert np.all(executed>=-EPS)
    assert np.all(executed.sum(axis=1)<=targets(fleet)+EPS)
    extra=replay_aging(fleet,executed)-ref_loss
    result=status(fleet,executed,targets(fleet))
    actual=float(ref[:,START].sum()-executed[:,START].sum())
    tracking=actual>=q-EPS
    health=bool(np.all(extra<=ALLOWANCE+TOL))
    delivered=result["full_feasible_target"] and tracking and health
    result.update(actual_reduction_kw=actual,tracking=tracking,aging_pass=health,
                  delivered=delivered,max_extra_soh=float(extra.max()),
                  plan_admission_bound_kw=possible,
                  cause="delivered" if delivered else "service_shortfall" if not result["full_feasible_target"]
                  else "not_admitted" if not tracking else "aging_excess")
    per_vehicle=[dict(vehicle_index=i,departure_hour=fleet.departure[i],
                      shifted_kwh=-delta[i,START],recovered_kwh=executed[i,START+1:].sum()-ref[i,START+1:].sum(),
                      feasible_target_kwh=targets(fleet)[i],delivered_kwh=executed[i].sum(),
                      incremental_soh=extra[i],**{f"power_{h}_kw":executed[i,h] for h in range(17,24)})
                 for i in range(len(fleet.arrival))]
    return result,trace,per_vehicle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"visible_deadline_archive_results")
    out=parser.parse_args().output
    out.mkdir(parents=True,exist_ok=True)
    prior=ROOT/"binding_archive_recovery_results"/"frozen_configuration.json"
    reserve=np.array(json.loads(prior.read_text())["reserve_kw_by_hour"])
    rows,traces,vehicles=[],[],[]
    for idx, (normal,peak) in enumerate(zip(load("joint",0),load("evening_peak_cluster",0))):
        if idx<4:
            continue
        name,nf,_=normal
        _,pf,_=peak
        caps=capacity((nf,pf),reserve)
        for ev,fleet in enumerate((nf,pf)):
            ref=flat_plan(fleet)
            for q in REQUESTS:
                delta,possible=certified_shift(fleet,ref,4*reserve,q)
                for stress,cap in enumerate(caps):
                    for policy in POLICIES:
                        keys=dict(split="calibration" if idx<7 else "test",scenario_id=name,
                                  pressure=f"{ev}{stress}",request_kw=q,policy=policy)
                        result,trace,per_vehicle=dispatch(fleet,ref,cap,delta,q,possible,policy)
                        if policy=="scheduled":
                            old=replay(fleet,ref,cap,delta,q,possible)
                            assert result["delivered"]==old["delivered"]
                            np.testing.assert_allclose(result["unmet_feasible_kwh"],old["unmet_feasible_kwh"],atol=1e-8)
                        rows.append(dict(**keys,**result))
                        if idx>=7:
                            traces.extend(dict(**keys,**row) for row in trace)
                            vehicles.extend(dict(**keys,**row) for row in per_vehicle)
        print(name,flush=True)
    events=pd.DataFrame(rows)
    assert len(events)==816
    assert (events.delivered==(events.tracking & events.aging_pass & events.full_feasible_target)).all()
    first=events.pivot(index=["scenario_id","request_kw"],columns=["policy","pressure"],values="actual_reduction_kw")
    for policy in POLICIES:
        for a,b in (("00","01"),("10","11")):
            np.testing.assert_allclose(first[policy,a],first[policy,b],atol=1e-9,rtol=0)
    for group in ("00","01","10","11"):
        np.testing.assert_allclose(first[POLICIES[0],group],first[POLICIES[1],group],atol=1e-9,rtol=0)
    events.to_csv(out/"event_replays.csv",index=False)
    pd.DataFrame(traces).to_csv(out/"hourly_recovery_trace.csv",index=False)
    pd.DataFrame(vehicles).to_csv(out/"vehicle_service_and_aging.csv",index=False)
    summarize(events,out)


def summarize(events,out):
    test=events[events.split=="test"]
    counts=test.groupby(["policy","pressure","request_kw"])[
        ["full_feasible_target","full_requested","tracking","aging_pass","delivered"]].sum().reset_index()
    counts.to_csv(out/"test_delivery_counts.csv",index=False)
    joint=test[test.pressure=="11"]
    maxima=joint[joint.delivered].groupby(["scenario_id","policy"]).request_kw.max().unstack()
    bounds=pd.read_csv(ROOT/"dense_binding_archive_results"/"certified_offline_bounds.csv",
                       dtype={"pressure":str})
    bounds=bounds[(bounds.split=="test")&(bounds.pressure=="11")].set_index("scenario_id")
    maxima=maxima.join(bounds[["aging_lower_kw","aging_upper_kw"]])
    maxima.to_csv(out/"joint_event_maxima.csv")
    common=counts[counts.delivered.eq(3)].groupby(["policy","pressure"]).request_kw.max()
    lines=["# Visible-deadline recovery reallocation","",
           "Frozen archive capacities, identical 17:00 action and aging budget in both policies.",
           "The new policy spends currently observed spare capacity in earliest-departure order.",
           "Each recovery addition is checked using the executed prefix and an even completion over that vehicle's known remaining parking window. The completion check does not use future capacity.",
           "Both policies protect flat baseline charging before allocating recovery.",
           "", "## Joint-stress test maxima", "",
           "| Event | Offline aging-constrained bound kW | Scheduled kW | Visible-deadline kW |",
           "|---|---:|---:|---:|"]
    for event,row in maxima.iterrows():
        lines.append(f"| {event} | {row.aging_lower_kw:.4f} | {row.scheduled:.2f} | {row.visible_deadline:.2f} |")
    lines+=["","## Highest request fully delivered in all three tests","",
            "| Pressure | Scheduled kW | Visible-deadline kW |","|---|---:|---:|"]
    for group in ("00","01","10","11"):
        lines.append(f"| {group} | {common.get(('scheduled',group),np.nan):g} | "
                     f"{common.get(('visible_deadline',group),np.nan):g} |")
    trace=pd.read_csv(out/"hourly_recovery_trace.csv",dtype={"pressure":str})
    first=trace[(trace.scenario_id=="scenario_07")&(trace.pressure=="11")&
                (trace.request_kw==5.5)&(trace.hour==18)].set_index("policy")
    probe=joint[(joint.scenario_id=="scenario_07")&(joint.request_kw==5.5)].set_index("policy")
    lines+=["","## Why scenario 07 missed the 5.50 kW request","",
            f"At 18:00 the scheduled policy left {first.loc['scheduled','unused_recovery_kw']:.4f} kWh "
            f"of the {first.loc['scheduled','available_recovery_kw']:.4f} kWh available in that one-hour slot unused. "
            "The later slots had only 0.65 kW each.",
            f"The deadline rule used {first.loc['visible_deadline','actual_recovery_kwh']:.4f} kWh at 18:00 "
            f"and reduced feasible-target unmet energy from {probe.loc['scheduled','unmet_feasible_kwh']:.4f} kWh to zero.",
            f"Maximum per-vehicle incremental SOH loss under the deadline rule was "
            f"{probe.loc['visible_deadline','max_extra_soh']:.6g}, below the {ALLOWANCE:g} budget."]
    lines+=["","The grid ends at 5.50 kW. Delivery requires tracking, each physically feasible departure target and the incremental aging budget.",
            "Previously used archive events support a paired mechanism comparison. No LLM was called."]
    (out/"conclusions.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,(metric,title) in zip(axes,(("full_feasible_target","Full service"),("delivered","Full commitment"))):
        for policy,part in counts[counts.pressure.eq("11")].groupby("policy"):
            ax.plot(part.request_kw,part[metric],"o-",markersize=3,label=policy)
        ax.set(xlabel="Request (kW)",ylabel="Joint-stress events out of 3",yticks=range(4),title=title)
        ax.legend()
    fig.savefig(out/"visible_deadline_comparison.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":
    main()
