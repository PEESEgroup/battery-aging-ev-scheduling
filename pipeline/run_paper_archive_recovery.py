"""Run raw and controlled paper-scale archive replays without using an LLM."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from nonanticipative_scheduler import Forecast, TOL, reference_replay, replay_aging, solve_plan, update
from run_full_service_recovery_pilot import retry_linprog
from run_section_iv_replay import GROUPS, ROOT, load, plt

START = 17
RECOVERY = slice(18, 21)
STRESSES = ("00", "10", "01", "11")
METHODS = ("original", "improved")
ALLOWANCE = 1e-5
EPS = 1e-6


def targets(fleet):
    return np.minimum(fleet.target, fleet.power*(fleet.departure-fleet.arrival))


def flat_plan(fleet):
    hours = np.arange(24)
    connected = (fleet.arrival[:, None] <= hours) & (hours < fleet.departure[:, None])
    return connected * (targets(fleet)/(fleet.departure-fleet.arrival))[:, None]


def status(fleet, plan, benchmark):
    served = plan.sum(axis=1)
    return dict(full_requested=bool(np.all(served >= fleet.demand-EPS)),
                full_feasible_target=bool(np.all(served >= targets(fleet)-EPS)),
                baseline_service_matched=bool(np.all(served >= benchmark-EPS)),
                unmet_requested_kwh=float(np.maximum(fleet.demand-served, 0).sum()),
                unmet_feasible_kwh=float(np.maximum(targets(fleet)-served, 0).sum()),
                vehicles_short_requested=int(np.count_nonzero(served < fleet.demand-EPS)),
                intrinsic_unmet_kwh=float((fleet.demand-targets(fleet)).sum()))


def raw_replay(job):
    label, case, fleet, grid, forecast = job
    with patch("nonanticipative_scheduler.linprog", retry_linprog):
        plan, _, _, _ = reference_replay(fleet, grid, forecast, 4)
    result = status(fleet, plan, targets(fleet))
    return dict(group=label, scenario_id=case, vehicles=len(fleet.arrival),
                zero_capacity_hours=int(np.count_nonzero(grid[:,3] == 0)),
                reference_17_kw=float(plan[:,START].sum()),
                baseline_service_limited=not result["full_requested"],
                capacity_or_dispatch_limited=not result["full_feasible_target"], **result)


def controlled_cases(normal, peak, headroom):
    cases = []
    for (case, nf, archive_grid), (other, pf, _) in zip(normal, peak):
        assert case == other
        refs = [flat_plan(fleet) for fleet in (nf, pf)]
        grid = archive_grid.copy()
        grid[:,3] = np.maximum(grid[:,3], 1500)
        stressed = grid.copy()
        shared_load = np.maximum(*(ref.sum(axis=0) for ref in refs))
        stressed[RECOVERY,3] = np.minimum(grid[RECOVERY,3], shared_load[RECOVERY]+headroom)
        arms = {}
        for ev, (fleet, ref) in enumerate(zip((nf,pf),refs)):
            for stress, actual in enumerate((grid,stressed)):
                assert np.all(ref.sum(axis=0) <= actual[:,3]+EPS)
                assert np.allclose(ref.sum(axis=1), targets(fleet))
                arms[f"{ev}{stress}"] = (fleet, actual, ref)
        cases.append((case, arms))
    return cases


def visible_load(fleet, ref):
    return ref[fleet.arrival <= START, RECOVERY].sum(axis=0)


def fit(cases):
    # Each fit sample is one constructed stressed capacity path. The improved
    # model regresses capacity on the currently visible flat reference with
    # a fixed unit slope and a per-hour residual estimated on planning data.
    planning, calibration = cases[:4], cases[4:7]
    absolute = np.stack([arms["01"][1][RECOVERY,3] for _, arms in planning]).min(axis=0)
    residuals = [
        grid[RECOVERY,3] - visible_load(fleet, ref)
        for _, arms in planning for group in ("01","11")
        for fleet, grid, ref in [arms[group]]
    ]
    residual = np.stack(residuals).min(axis=0)
    fixed_shortfalls = []
    residual_shortfalls = []
    for _, arms in calibration:
        for group in ("01","11"):
            fleet, grid, ref = arms[group]
            actual = grid[RECOVERY,3]
            fixed_shortfalls.extend(absolute-actual)
            residual_shortfalls.extend(visible_load(fleet,ref)+residual-actual)
    reserve_fixed = max(0., max(fixed_shortfalls))
    reserve_improved = max(0., max(residual_shortfalls))
    return absolute, residual, reserve_fixed, reserve_improved


def replan(fleet, grid, ref, forecast, floor, q):
    benchmark = targets(fleet)
    ref_loss = replay_aging(fleet, ref)
    power_ref = ref[:,START:START+1].sum(axis=0)
    window = START, power_ref, -1, q
    executed = ref.copy()
    executed[:,START:] = 0
    # replay_aging on a padded prefix includes calendar loss after the prefix;
    # use the existing per-hour updater to initialize only elapsed losses.
    prefix = np.zeros_like(ref)
    losses = np.zeros(len(fleet.arrival))
    for hour in range(START):
        update(fleet, prefix, losses, hour, ref[:,hour])
    np.testing.assert_allclose(prefix, executed)
    budget = ref_loss+ALLOWANCE
    fallback = []
    failures = []
    solver_errors = 0

    def solve(hour, caps, score, **kwargs):
        nonlocal solver_errors
        detail = {}
        try:
            ans = solve_plan(fleet, hour, caps, score, executed, losses,
                             diagnostics=detail, **kwargs)
        except RuntimeError as error:
            if "HiGHS" not in str(error):
                raise
            solver_errors += 1
            ans = None
            detail["status"] = "solver_error"
        return ans, detail["status"]

    initial_power = np.nan
    for hour in range(START, len(grid)):
        caps, score = forecast.build(hour, grid[:hour+1], 4)
        if hour == START:
            caps[1:4] = floor
        elif 18 <= hour <= 20:
            # The capacity just observed informs only the next recovery slots.
            caps[1:21-hour] = grid[hour,3]
        # In the paired design, the 1500 kW floor outside the recovery block
        # is a known engineering assumption shared by all treatments.
        later = max(21-hour,1)
        caps[later:] = np.maximum(caps[later:],1500)
        planned, reason = solve(hour,caps,score,target=benchmark,budget=budget,window=window)
        if planned is None:
            failures.append(f"{hour}:{reason}")
            for label, aging in (("service_health",budget),("service_absolute",None)):
                planned, _ = solve(hour,caps,np.zeros_like(score),target=benchmark,budget=aging)
                if planned is not None:
                    fallback.append(f"{hour}:{label}")
                    break
            if planned is None:
                planned, _ = solve(hour,caps,np.zeros_like(score))
                fallback.append(f"{hour}:best_effort")
        action = np.zeros(len(fleet.arrival)) if planned is None else planned[0][:,hour]
        if action.sum() > grid[hour,3]+EPS:
            raise AssertionError("Actual capacity exceeded")
        if hour == START:
            initial_power = action.sum()
        update(fleet, executed, losses, hour, action)
    results = status(fleet, executed, benchmark)
    actual_reduction = power_ref[0]-initial_power
    track = actual_reduction >= q-EPS
    health = bool(np.all(losses <= budget+TOL))
    delivery = results["full_feasible_target"] and track and health
    reason = ("delivered" if delivery else "service_shortfall" if not results["full_feasible_target"]
              else "tracking_shortfall" if not track else "aging_budget")
    return dict(**results, actual_reduction_kw=float(actual_reduction), tracking=bool(track),
                aging_pass=health, adjusted_delivery=bool(delivery),
                strict_delivery=bool(delivery and results["full_requested"]),
                cause=reason, max_extra_soh=float((losses-ref_loss).max()),
                fallback=";".join(fallback), solve_failures=";".join(failures),
                solver_errors=solver_errors)


def run_case(job):
    split, case, group, fleet, grid, ref, forecast, model, requests = job
    absolute,residual,old_margin,new_margin = model
    floors = (absolute-old_margin, visible_load(fleet,ref)+residual-new_margin)
    rows = []
    with patch("nonanticipative_scheduler.linprog", retry_linprog):
        prefix=ref.copy()
        prefix[:,START:]=0
        losses=np.zeros(len(fleet.arrival))
        previous=np.zeros_like(ref)
        for hour in range(START):
            update(fleet,previous,losses,hour,ref[:,hour])
        bounds=[]
        for budget in (None,replay_aging(fleet,ref)+ALLOWANCE):
            info={}
            try:
                answer=solve_plan(
                    fleet,START,grid[START:,3],np.zeros(len(grid)-START),
                    prefix,losses,target=targets(fleet),budget=budget,
                    window=(START,ref[:,START:START+1].sum(axis=0),-1,None),
                    offline=True,allow_signed=True,diagnostics=info,
                )
            except RuntimeError as error:
                if "HiGHS" not in str(error):
                    raise
                answer=None
                info["status"]="solver_error"
            bounds.append((float(answer[1]) if answer else np.nan,info["status"]))
        for (method,floor), q in product(zip(METHODS,floors),requests):
            row = replan(fleet,grid,ref,forecast,np.maximum(floor,0),q)
            if not row["adjusted_delivery"]:
                if bounds[0][1]=="solved" and q>bounds[0][0]+EPS:
                    row["cause"]="offline_capacity_or_deadline"
                elif bounds[1][1]=="solved" and q>bounds[1][0]+EPS:
                    row["cause"]="offline_aging_budget"
                else:
                    row["cause"]="online_"+row["cause"]
            rows.append(dict(split=split,scenario_id=case,pressure=group,method=method,
                             vehicles=len(fleet.arrival),request_kw=q,reference_kw=ref[:,START].sum(),
                             baseline_service_limited=bool(np.any(targets(fleet)<fleet.demand-EPS)),
                             recovery_estimate_kwh=float(np.maximum(floor,0).sum()),
                             actual_recovery_capacity_kwh=float(grid[RECOVERY,3].sum()),
                             offline_capacity_kw=bounds[0][0],offline_capacity_status=bounds[0][1],
                             offline_health_kw=bounds[1][0],offline_health_status=bounds[1][1],**row))
    print(f"{split} {case} {group}",flush=True)
    return rows


def run_jobs(fn,jobs,workers):
    if workers == 1:
        return list(map(fn,jobs))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn,jobs))


def select_requests(cal):
    counts=cal.groupby(["method","request_kw"]).adjusted_delivery.sum()
    return counts[counts==12].reset_index().groupby("method").request_kw.max()


def summarize(raw, events, output, model, headroom, requests):
    test = events[events.split=="test"]
    metrics = ["full_requested","full_feasible_target","tracking","aging_pass",
               "strict_delivery","adjusted_delivery"]
    counts = test.groupby(["method","pressure","request_kw"])[metrics].sum().reset_index()
    counts.to_csv(output/"test_outcomes.csv",index=False)
    cal = events[events.split=="calibration"]
    selected = select_requests(cal)
    frozen = counts.merge(selected.rename("frozen_kw"),on="method")
    frozen = frozen[frozen.request_kw==frozen.frozen_kw]
    frozen.to_csv(output/"frozen_outcomes.csv",index=False)
    paired = test.pivot(index=["scenario_id","pressure","request_kw"],columns="method",
                        values=metrics+["actual_reduction_kw"])
    paired.columns = [f"{metric}_{method}" for metric,method in paired.columns]
    paired.to_csv(output/"paired_outcomes.csv")
    test.groupby(["method","pressure","request_kw","cause"]).size().rename("count").reset_index().to_csv(
        output/"failure_causes.csv",index=False)
    absolute,residual,fixed_margin,improved_margin=model
    lines=[
        "# Paper-scale archive validation","",
        f"Raw replay used all {len(raw)} archive trajectories. Controlled comparison used the normal and evening sessions with their paired normal grid path.",
        "All original requests exceed individual charging or SOC limits in at least one session per event. Raw requested service and feasible-target service are reported separately.",
        "The controlled benchmark clips each request only to that session's charger, parking and SOC limits. The flat benchmark exactly meets those targets.",
        "The four-group grid uses a constructed 1500 kW floor outside and before recovery. Recovery stress caps 18:00–21:00 using the common flat load envelope plus planning-only headroom.",
        "The common cap can use both full rosters to construct the test environment. Neither policy can observe that path ahead of execution.",
        f"Planning-only recovery headroom: {headroom:g} kW. Planning-derived requests: {', '.join(map(str,requests))} kW.",
        f"Original absolute planning floors before margin: {absolute.round(3).tolist()} kW. Calibration margin: {fixed_margin:.3f} kW.",
        f"Improved estimate adds the visible connected-fleet flat profile to planning residuals {residual.round(3).tolist()} kW. Calibration margin: {improved_margin:.3f} kW.",
        "Planning IDs 00–03, calibration IDs 04–06, held-out replay IDs 07–09. These archive cases were previously inspected in other experiments.",
        "", "## Raw reference service", "",
        "| Group | Baseline limited | Full feasible-target service | Median unmet original kWh |",
        "|---|---:|---:|---:|",
    ]
    for label, part in raw.groupby("group"):
        lines.append(f"| {label} | {part.baseline_service_limited.sum()}/{len(part)} | "
                     f"{part.full_feasible_target.sum()}/{len(part)} | {part.unmet_requested_kwh.median():.2f} |")
    lines += ["", "## Frozen requests on held-out events", "",
              "| Method | Pressure | Request kW | Full original / 3 | Full feasible target / 3 | Feasible-target delivery / 3 |",
              "|---|---|---:|---:|---:|---:|"]
    for row in frozen.itertuples():
        lines.append(f"| {row.method} | {row.pressure} | {row.request_kw:g} | {row.full_requested}/3 | "
                     f"{row.full_feasible_target}/3 | {row.adjusted_delivery}/3 |")
    for method in METHODS:
        if method not in selected:
            lines.append(f"| {method} | all | no eligible request, including zero | — | — | — |")
    service_loss=(paired.full_feasible_target_original.astype(bool)&~paired.full_feasible_target_improved.astype(bool)).sum()
    gain=(~paired.adjusted_delivery_original.astype(bool)&paired.adjusted_delivery_improved.astype(bool)).sum()
    loss=(paired.adjusted_delivery_original.astype(bool)&~paired.adjusted_delivery_improved.astype(bool)).sum()
    lines += ["", f"Matched test comparisons: {len(paired)}. Feasible-target service regressions: {service_loss}. "
              f"Delivery gains: {gain}. Delivery losses: {loss}.",
              f"Unresolved numerical errors after retries: {events.solver_errors.sum()}.",
              "Failure causes and rolling solve statuses are recorded separately. No LLM was called."]
    lines += ["", "## Descriptive sweep on the held-out IDs", "",
              "| Method | Full requested | Full feasible target | Feasible-target delivery | Tracking failures | Final aging failures after tracking |",
              "|---|---:|---:|---:|---:|---:|"]
    for method, part in test.groupby("method"):
        total = len(part)
        lines.append(f"| {method} | {part.full_requested.sum()}/{total} | "
                     f"{part.full_feasible_target.sum()}/{total} | {part.adjusted_delivery.sum()}/{total} | "
                     f"{part.cause.eq('online_tracking_shortfall').sum()} | "
                     f"{part.cause.eq('online_aging_budget').sum()} |")
    offline=test.drop_duplicates(["scenario_id","pressure"])
    capacity=offline.pivot(index="scenario_id",columns="pressure",values="offline_capacity_kw")
    unchanged=sum(np.isclose(capacity[a],capacity[b],atol=EPS,rtol=0).sum()
                  for a,b in (("00","01"),("10","11")))
    lines += ["", f"The constructed cap leaves {unchanged}/{2*len(capacity)} paired offline capacity-only bounds unchanged. "
              "The current stress does not isolate a capacity bottleneck at the offline optimum.",
              f"Health-constrained offline solves reaching the cut limit: "
              f"{offline.offline_health_status.eq('cut_limit').sum()}/{len(offline)}. "
              "A missing offline health bound is not evidence of aging infeasibility."]
    (output/"conclusions.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,metric,title in zip(axes,("full_feasible_target","adjusted_delivery"),("Full feasible-target service","Feasible-target commitment delivery")):
        for (method,pressure), part in counts.groupby(["method","pressure"]):
            ax.plot(part.request_kw,part[metric],"o-" if method=="original" else "s--",
                    markersize=3,label=f"{method} {pressure}")
        ax.set(xlabel="Request (kW)",ylabel="Held-out events out of 3",yticks=range(4),title=title)
    axes[1].legend(fontsize=7,ncol=2)
    fig.savefig(output/"archive_delivery.png",dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"paper_archive_recovery_results")
    parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    raw_jobs=[]
    for label,group in GROUPS.items():
        cases=load(group,0)
        forecast=Forecast([row[2] for row in cases[:4]])
        raw_jobs.extend((label,*row,forecast) for row in cases)
    raw=pd.DataFrame(run_jobs(raw_replay,raw_jobs,args.workers))
    raw.to_csv(args.output/"raw_reference_service.csv",index=False)
    print(f"Raw reference replays complete: {len(raw)}",flush=True)
    normal,peak=load("joint",0),load("evening_peak_cluster",0)
    p17=np.median([flat_plan(row[1])[:,START].sum() for row in normal[:4]+peak[:4]])
    headroom=max(1.,round(0.05*p17,1))
    requests=tuple(sorted({0.,*(max(.5,round(p17*frac*2)/2) for frac in (.005,.01,.025,.05,.10,.20))}))
    cases=controlled_cases(normal,peak,headroom)
    model=fit(cases)
    forecast=Forecast([arms[group][1] for _,arms in cases[:4] for group in ("00","01")])
    jobs=[("calibration" if idx<7 else "test",case,group,*arms[group],forecast,model,requests)
          for idx,(case,arms) in enumerate(cases) if idx>=4 for group in STRESSES]
    cal=pd.DataFrame([row for part in run_jobs(run_case,jobs[:12],args.workers) for row in part])
    cal.to_csv(args.output/"calibration_replays.csv",index=False)
    absolute,residual,original_margin,improved_margin=model
    frozen={
        "planning_ids":["00","01","02","03"],
        "calibration_ids":["04","05","06"],
        "recovery_headroom_kw":headroom,
        "original_hourly_kw":absolute.tolist(),
        "improved_hourly_residual_kw":residual.tolist(),
        "original_margin_kw":original_margin,
        "improved_margin_kw":improved_margin,
        "request_grid_kw":requests,
        "selected_requests_kw":select_requests(cal).to_dict(),
    }
    (args.output/"frozen_configuration.json").write_text(json.dumps(frozen,indent=2)+"\n",encoding="utf-8")
    print(f"Frozen commitments: {frozen['selected_requests_kw']}",flush=True)
    test=pd.DataFrame([row for part in run_jobs(run_case,jobs[12:],args.workers) for row in part])
    events=pd.concat([cal,test],ignore_index=True)
    assert len(events)==len(jobs)*len(METHODS)*len(requests)
    first=events.pivot(index=["scenario_id","method","request_kw"],columns="pressure",values="actual_reduction_kw")
    for a,b in (("00","01"),("10","11")):
        np.testing.assert_allclose(first[a],first[b],atol=1e-8,rtol=0)
    events.to_csv(args.output/"event_replays.csv",index=False)
    summarize(raw,events,args.output,model,headroom,requests)


if __name__=="__main__":
    main()
