"""Certified archive aging bounds and dense recovery replay on frozen caps."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, linprog, minimize
from scipy.sparse import coo_matrix, csr_matrix

from nonanticipative_scheduler import AGING, TOL, aging, replay_aging
from run_binding_archive_recovery import RECOVERY, capacity, certified_shift, replay
from run_paper_archive_recovery import ALLOWANCE, EPS, START, flat_plan, targets
from run_section_iv_replay import ROOT, load, plt

GAP_KW = 0.03
MAX_ITERATIONS = 100
REQUESTS = tuple(round(q/10, 1) for q in range(39,79))


def solve_bounds(fleet, ref, cap):
    """LP outer bound plus a feasible inner point on the line to the reference."""
    n=len(fleet.arrival)
    ids,hours=np.where((fleet.arrival[:,None]<=np.arange(START,24)) &
                       (fleet.departure[:,None]>np.arange(START,24)))
    slots=np.arange(len(ids))
    rhs=targets(fleet)-ref[:,:START].sum(axis=1)
    use=np.bincount(ids,minlength=n)
    offsets=np.r_[0,np.cumsum(use)]
    objective=(hours==0).astype(float)
    ref_flat=ref[ids,START+hours]
    ref_kw=float(ref_flat @ objective)
    aeq=csr_matrix((np.ones(len(ids)),(ids,slots)),shape=(n,len(ids)))
    aub=csr_matrix((np.ones(len(ids)),(hours,slots)),shape=(24-START,len(ids)))
    limits=list(zip(np.zeros(len(ids)),fleet.power[ids]))
    options={"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9}
    full_ref_loss=replay_aging(fleet,ref)
    active=np.flatnonzero(use)
    ref_suffix_loss=np.zeros(n)
    start_soc=fleet.soc+ref[:,:START].sum(axis=1)/fleet.capacity
    for i in active:
        sl=slice(offsets[i],offsets[i+1])
        ref_suffix_loss[i]=aging(ref_flat[sl],start_soc[i],fleet.capacity[i])[0].sum()
    allowed=np.minimum(full_ref_loss+ALLOWANCE,1-AGING.min_soh)-(full_ref_loss-ref_suffix_loss)
    assert np.all(ref_suffix_loss<=allowed+1e-10)

    def optimize(rows,cols,values,upper):
        matrix=coo_matrix((values,(rows,cols)),shape=(len(upper),len(ids))).tocsr()
        result=linprog(objective,A_eq=aeq,b_eq=rhs,A_ub=matrix,b_ub=upper,
                       bounds=limits,method="highs",options=options)
        if result.status==4:
            result=linprog(objective,A_eq=aeq,b_eq=rhs,A_ub=matrix,b_ub=upper,
                           bounds=limits,method="highs-ipm",options=options)
        if not result.success:
            raise RuntimeError(result.message)
        return result

    def validate(values, budget):
        plan=ref.copy()
        plan[:,START:]=0
        plan[ids,START+hours]=values
        np.testing.assert_allclose(plan.sum(axis=1),targets(fleet),rtol=0,atol=1e-7)
        assert np.all(plan.sum(axis=0)[START:]<=cap[START:]+1e-7)
        assert np.all(plan>=-1e-8) and np.all(plan<=fleet.power[:,None]+1e-7)
        loss=replay_aging(fleet,plan)
        assert np.all(loss<=budget+1e-10)
        return float((loss-full_ref_loss).max())

    rows,cols,values=hours.tolist(),slots.tolist(),[1.]*len(ids)
    upper=cap[START:].tolist()
    result=optimize(rows,cols,values,upper)
    physical=ref_kw-result.fun
    validate(result.x,np.full(n,1-AGING.min_soh))
    best,best_point=0.,ref_flat.copy()
    scale=1e5
    # Eliminate each vehicle's energy equality and all fixed profiles before
    # solving the smooth convex primal. Tangent LPs certify its accuracy.
    variable=[]
    rr,cc,vv=[],[],[]
    for i in active:
        left,right=offsets[i],offsets[i+1]
        energy=ref_flat[left:right].sum()
        if right-left<=1 or energy<=1e-9 or energy>=(right-left)*fleet.power[i]-1e-9:
            continue
        variable.append(i)
        for slot in range(left,right-1):
            col=len(cc)//2
            rr.extend((slot,right-1))
            cc.extend((col,col))
            vv.extend((1.,-1.))
    transform=csr_matrix((vv,(rr,cc)),shape=(len(ids),len(cc)//2))
    if transform.shape[1]:
        groups=[slice(offsets[i],offsets[i+1]) for i in variable]
        dense=transform.toarray()
        grid_matrix=(aub@transform).toarray()
        obj=np.asarray(transform.T@objective)

        def health(point):
            return np.array([(allowed[i]-aging(point[sl],start_soc[i],fleet.capacity[i])[0].sum())*scale
                             for i,sl in zip(variable,groups)])

        def health_jac(point):
            return np.stack([-scale*aging(point[sl],start_soc[i],fleet.capacity[i])[1]@dense[sl]
                             for i,sl in zip(variable,groups)])

        primal=minimize(lambda x: obj@x,np.zeros(transform.shape[1]),jac=lambda x:obj,
                        method="SLSQP",constraints=[
                            LinearConstraint(dense,-ref_flat,fleet.power[ids]-ref_flat),
                            LinearConstraint(grid_matrix,-np.inf,cap[START:]-aub@ref_flat),
                            {"type":"ineq","fun":lambda x:health(ref_flat+transform@x),
                             "jac":lambda x:health_jac(ref_flat+transform@x)},
                        ],options={"ftol":1e-10,"maxiter":350})
        point=ref_flat+transform@primal.x
        # Convex interpolation toward the checked reference removes small
        # nonlinear feasibility errors without changing service or grid use.
        alpha=1.
        for i,sl in zip(variable,groups):
            direction=point[sl]-ref_flat[sl]
            lo,hi=0.,1.
            for _ in range(30):
                mid=(lo+hi)/2
                loss=aging(ref_flat[sl]+mid*direction,start_soc[i],fleet.capacity[i])[0].sum()
                if loss<=allowed[i]:
                    lo=mid
                else:
                    hi=mid
            alpha=min(alpha,lo)
        candidate=ref_flat+alpha*(point-ref_flat)
        if (candidate.min()>=-1e-8 and np.all(candidate<=fleet.power[ids]+1e-7)
                and np.all(aub@candidate<=cap[START:]+1e-7)):
            best=max(0.,ref_kw-candidate@objective)
            if best>0:
                best_point=candidate
        for i,sl in zip(variable,groups):
            loss,grad=aging(point[sl],start_soc[i],fleet.capacity[i])
            rows.extend([len(upper)]*len(grad))
            cols.extend(range(sl.start,sl.stop))
            values.extend(grad*scale)
            upper.append((allowed[i]-loss.sum()+grad@point[sl])*scale)
        result=optimize(rows,cols,values,upper)
    for iteration in range(1,MAX_ITERATIONS+1):
        outer=max(best,ref_kw-result.fun)
        violated=[]
        alpha=1.
        for i in active:
            sl=slice(offsets[i],offsets[i+1])
            part=result.x[sl]
            loss=aging(part,start_soc[i],fleet.capacity[i])[0].sum()
            if loss<=allowed[i]+1e-12:
                continue
            direction=part-ref_flat[sl]
            lo,hi=0.,1.
            for _ in range(29):
                mid=(lo+hi)/2
                loss=aging(ref_flat[sl]+mid*direction,start_soc[i],fleet.capacity[i])[0].sum()
                if loss<=allowed[i]:
                    lo=mid
                else:
                    hi=mid
            alpha=min(alpha,lo)
            boundary=ref_flat[sl]+lo*direction
            loss,grad=aging(boundary,start_soc[i],fleet.capacity[i])
            violated.append((sl,grad,(allowed[i]-loss.sum()+grad@boundary)*scale))
        candidate=ref_flat+alpha*(result.x-ref_flat)
        value=ref_kw-candidate@objective
        if value>best:
            best,best_point=value,candidate
        if outer-best<=GAP_KW or not violated or iteration==MAX_ITERATIONS:
            extra=validate(best_point,full_ref_loss+ALLOWANCE)
            return dict(no_incremental_budget_kw=physical,aging_lower_kw=best,
                        aging_upper_kw=outer,certificate_gap_kw=outer-best,
                        iterations=iteration,cuts=len(upper)-(24-START),
                        aging_witness_max_extra_soh=extra,
                        status="certified" if outer-best<=GAP_KW else "gap_limit")
        for sl,grad,intercept in violated:
            rows.extend([len(upper)]*len(grad))
            cols.extend(range(sl.start,sl.stop))
            values.extend(grad*scale)
            upper.append(intercept)
        result=optimize(rows,cols,values,upper)


def run_cases(pairs,reserve,split,run_offline=True):
    rows,bounds=[],[]
    for name,fleets in pairs:
        caps=capacity(fleets,reserve)
        for ev,fleet in enumerate(fleets):
            ref=flat_plan(fleet)
            for stress,cap in enumerate(caps):
                if run_offline:
                    result=solve_bounds(fleet,ref,cap)
                    bounds.append(dict(split=split,scenario_id=name,pressure=f"{ev}{stress}",**result))
                    print(f"{split} {name} {ev}{stress}: {result['status']} "
                          f"gap={result['certificate_gap_kw']:.4f} iterations={result['iterations']}",flush=True)
            for policy,headroom in (("reserved",reserve),("nominal_probe",4*reserve)):
                for q in REQUESTS:
                    delta,possible=certified_shift(fleet,ref,headroom,q)
                    for stress,cap in enumerate(caps):
                        result=replay(fleet,ref,cap,delta,q,possible)
                        rows.append(dict(split=split,scenario_id=name,pressure=f"{ev}{stress}",policy=policy,
                                         request_kw=q,**result))
    return pd.DataFrame(rows),pd.DataFrame(bounds)


def interval(lo,hi):
    return f"{lo:.2f}–{hi:.2f}"


def report(bounds,events,chosen,output):
    test=events[events.split=="test"]
    off=bounds[bounds.split=="test"]
    levels=["split","scenario_id","pressure"]
    capacities=off.pivot(index="scenario_id",columns="pressure",values="no_incremental_budget_kw")
    aginglo=off.pivot(index="scenario_id",columns="pressure",values="aging_lower_kw")
    aginghi=off.pivot(index="scenario_id",columns="pressure",values="aging_upper_kw")
    online=test.assign(q=test.request_kw.where(test.delivered)).groupby(levels+["policy"],as_index=False).q.max()
    online=online.rename(columns={"q":"max_tested_delivery_kw"})
    effects=online.merge(off,on=levels,validate="many_to_one")
    effects["online_gap_lower_kw"]=(effects.aging_lower_kw-effects.max_tested_delivery_kw).clip(lower=0)
    effects["online_gap_upper_kw"]=(effects.aging_upper_kw-effects.max_tested_delivery_kw).clip(lower=0)
    effects["aging_loss_lower_kw"]=effects.no_incremental_budget_kw-effects.aging_upper_kw
    effects["aging_loss_upper_kw"]=effects.no_incremental_budget_kw-effects.aging_lower_kw
    effects.to_csv(output/"event_decomposition.csv",index=False)
    paired=[]
    for case in capacities.index:
        for group in ("0","1"):
            control,stress=group+"0",group+"1"
            paired.append(dict(scenario_id=case,ev_type=group,
                               capacity_loss_without_aging_kw=capacities.loc[case,control]-capacities.loc[case,stress],
                               capacity_loss_with_aging_lower_kw=aginglo.loc[case,control]-aginghi.loc[case,stress],
                               capacity_loss_with_aging_upper_kw=aginghi.loc[case,control]-aginglo.loc[case,stress]))
    pd.DataFrame(paired).to_csv(output/"paired_capacity_effects.csv",index=False)
    counts=test.groupby(["policy","pressure","request_kw"])[
        ["full_feasible_target","full_requested","tracking","aging_pass","delivered"]].sum().reset_index()
    counts.to_csv(output/"test_delivery_counts.csv",index=False)
    lines=[
        "# Certified offline aging and dense online replay","",
        "The recovery caps and 0.65 kW hourly reserve are unchanged. Archive planning, calibration and held-out IDs are unchanged.",
        f"Each offline result is bounded by a verified feasible schedule and a tangent-LP upper bound. Target gap {GAP_KW:.2f} kW.",
        "The no-incremental-budget LP witness is independently checked against the absolute SOH constraint.",
        "The online grid uses the two unchanged dispatch policies and 0.1 kW steps, including both endpoints.",
        "A positive delivery requires requested power, feasible-target departure service, and the incremental aging budget.",
        "The online gap includes policy restrictions and finite grid resolution. It is not extrapolated beyond 7.8 kW.",
        "",f"Certified offline instances: {bounds.status.eq('certified').sum()}/{len(bounds)}. "
        f"Largest gap: {bounds.certificate_gap_kw.max():.5f} kW.",
        "", "## Held-out mean offline bounds",
        "", "| Pressure | No incremental budget kW | With budget kW | Aging loss kW |",
        "|---|---:|---:|---:|",
    ]
    for group,part in off.groupby("pressure"):
        cap,lo,hi=part[["no_incremental_budget_kw","aging_lower_kw","aging_upper_kw"]].mean()
        lines.append(f"| {group} | {cap:.2f} | {interval(lo,hi)} | {interval(cap-hi,cap-lo)} |")
    lines+=["", "## Held-out online grid", "",
            "| Policy | Pressure | Largest grid request delivered 3/3 (kW) | Full service at that request |",
            "|---|---|---:|---:|"]
    for (policy,group),part in counts.groupby(["policy","pressure"]):
        best=part[part.delivered==3].tail(1)
        if best.empty:
            lines.append(f"| {policy} | {group} | — | — |")
        else:
            row=best.iloc[0]
            lines.append(f"| {policy} | {group} | {row.request_kw:g} | {row.full_feasible_target:g}/3 |")
    lines+=["", "## Commitments frozen before replay of test events", "",
            "| Policy | Request kW | Feasible service / 12 | Delivery / 12 |",
            "|---|---:|---:|---:|"]
    for policy,q in chosen.items():
        part=test[(test.policy==policy)&(test.request_kw==q)]
        lines.append(f"| {policy} | {q:g} | {part.full_feasible_target.sum()}/12 | {part.delivered.sum()}/12 |")
    lines+=["","The held-out IDs have been used previously. Results are descriptive. No LLM was called."]
    (output/"conclusions.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    means=off.groupby("pressure")[["no_incremental_budget_kw","aging_lower_kw"]].mean()
    means.plot.bar(ax=axes[0],rot=0)
    axes[0].set(xlabel="EV / grid pressure",ylabel="kW",title="Offline physical and aging bounds")
    for policy,part in counts[counts.pressure=="11"].groupby("policy"):
        axes[1].plot(part.request_kw,part.delivered,"o-",markersize=2,label=policy)
    axes[1].set(xlabel="Request (kW)",ylabel="Events delivered out of 3",
                yticks=range(4),title="Joint-stress dense replay")
    axes[1].legend()
    fig.savefig(output/"dense_archive_comparison.png",dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"dense_binding_archive_results")
    output=parser.parse_args().output
    output.mkdir(parents=True,exist_ok=True)
    prior=ROOT/"binding_archive_recovery_results"/"frozen_configuration.json"
    reserve=np.asarray(json.loads(prior.read_text())["reserve_kw_by_hour"])
    pairs=[(a[0],(a[1],b[1])) for a,b in zip(load("joint",0),load("evening_peak_cluster",0))]
    cal,cb=run_cases(pairs[4:7],reserve,"calibration")
    eligible=cal.groupby(["policy","request_kw"]).delivered.sum()
    chosen=eligible[eligible==12].reset_index().groupby("policy").request_kw.max().to_dict()
    (output/"frozen_dense_configuration.json").write_text(json.dumps(dict(
        reserve_kw_by_hour=reserve.tolist(),requests_kw=REQUESTS,selected_requests_kw=chosen
    ),indent=2)+"\n",encoding="utf-8")
    print("Frozen online requests",chosen,flush=True)
    test,tb=run_cases(pairs[7:],reserve,"test")
    events=pd.concat([cal,test],ignore_index=True)
    bounds=pd.concat([cb,tb],ignore_index=True)
    assert len(events)==6*4*2*len(REQUESTS) and len(bounds)==24
    assert (events.delivered==(events.tracking & events.aging_pass & events.full_feasible_target)).all()
    first=events.pivot(index=["scenario_id","policy","request_kw"],columns="pressure",values="actual_reduction_kw")
    for a,b in (("00","01"),("10","11")):
        np.testing.assert_allclose(first[a],first[b],atol=1e-9,rtol=0)
    events.to_csv(output/"dense_event_replays.csv",index=False)
    bounds.to_csv(output/"certified_offline_bounds.csv",index=False)
    report(bounds,events,chosen,output)


if __name__=="__main__":
    main()
