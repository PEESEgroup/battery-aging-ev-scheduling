"""Matched Gemini and constrained BO candidate proposals on the four manuscript groups."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from time import sleep
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import warnings

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import torch

from lock_recovery_holdout import digest, fingerprint
from run_directional_pinn_validation import Aging, GROUPS, read_case
from run_llm_timing_pilot import DEFAULTS, TimingForecast, cvar, dump, simulate


KEYS = ("lambda_risk", "w_price", "unmet_penalty_per_kwh")
BOUNDS = np.array([[0.,.9],[.4,2.2],[.5,10.]])
PROBES = ((.3,1.,2.),(.75,1.2,2.))
SERVICE_MARGIN, CVAR_MARGIN = .001, .5
MODEL = "gemini-2.5-flash"
CHATGPT_MODEL = "gpt-6-sol"
CHECKPOINTS = (3,6)


def metrics(rows):
    frame = pd.DataFrame(rows)
    return dict(qualified=float(frame.qualified.mean()), service=float(frame.served_ratio.mean()),
        cvar_kwh=float(cvar(frame.unmet_kwh.to_numpy(), .95)), soh=float(frame.mean_soh_loss.mean()))


def replay(cases, forecast, aging, point):
    params = dict(DEFAULTS, **dict(zip(KEYS, point)))
    return [dict(parent=key, **simulate(case, forecast, aging, params)) for key, case in cases.items()]


def load_cases(source):
    data, all_split, provenance = {}, {}, []
    for group in GROUPS:
        partition = {"development":{}, "holdout":{}}
        paths = sorted((source/f"optimization_inputs_{group}").glob("scenario_*"))
        assert len(paths) == 10
        for i, path in enumerate(paths):
            fleet, grid, temp = read_case(path)
            split = "development" if i < 7 else "holdout"
            key = fingerprint(fleet)
            assert all_split.setdefault(key, split) == split, "Same fleet crossed the split"
            assert key not in partition[split]
            grid[:,3] = np.maximum(grid[:,3], 1500.)
            if group == "capacity_shortage":
                grid[17:21,3] = np.minimum(grid[17:21,3],15.)
            partition[split][key] = fleet,grid,temp,None
            provenance.append(dict(group=group, split=split, parent=key, source=str(path),
                ev_sha256=digest(path/"optimizer_ev_sessions.csv"), grid_sha256=digest(path/"optimizer_grid.csv")))
        data[group] = partition, TimingForecast([case[1] for case in partition["development"].values()])
    return data, provenance


def margins(m, reference, enforce_delivery=False):
    values = [m["service"]-reference["service"]+SERVICE_MARGIN,
              reference["cvar_kwh"]+CVAR_MARGIN-m["cvar_kwh"]]
    if enforce_delivery:
        values.append(m["qualified"]-reference["qualified"])
    return np.array(values)


def rank(row, reference, enforce_delivery=False):
    m = row["metrics"]
    scale = [SERVICE_MARGIN, CVAR_MARGIN] + ([1/7] if enforce_delivery else [])
    violation = np.maximum(0., -margins(m, reference, enforce_delivery)/scale).sum()
    return (float(violation), m["soh"], -m["qualified"], -m["service"])


def peak_exploration_ok(point):
    point = np.asarray(point)
    widths = BOUNDS[:, 1] - BOUNDS[:, 0]
    return np.any(np.abs(point[..., 1:] - np.asarray(PROBES[1])[1:]) >= .1 * widths[1:] - 1e-12, axis=-1)


def propose_bo(history, reference, rng, enforce_delivery=False, explore=False):
    low, high = BOUNDS.T
    x = (np.array([r["point"] for r in history])-low)/(high-low)
    y = np.array([[r["metrics"]["soh"], *margins(r["metrics"],reference,enforce_delivery)] for r in history])
    pool = rng.random((512,len(KEYS)))
    if explore:
        pool = pool[peak_exploration_ok(low + pool * (high - low))]
    predictions = []
    for target in y.T:
        gp = GaussianProcessRegressor(kernel=Matern(length_scale=np.ones(len(KEYS)),nu=2.5)+WhiteKernel(1e-5),
              normalize_y=True,random_state=int(rng.integers(1,2**31-1)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",ConvergenceWarning)
            gp.fit(x,target)
        mu, sd = gp.predict(pool,return_std=True)
        predictions.append((mu,np.maximum(sd,1e-10)))
    feasible = (y[:,1:]>=0).all(axis=1)
    best = y[feasible,0].min() if feasible.any() else y[:,0].min()
    mu, sd = predictions[0]
    z = (best-mu)/sd
    ei = np.maximum(0., (best-mu)*ndtr(z)+sd*np.exp(-.5*z*z)/np.sqrt(2*np.pi))
    feasibility = np.prod([ndtr(m/s) for m,s in predictions[1:]],axis=0)
    i = np.argmax(feasibility*(ei+1e-12))
    return tuple(float(v) for v in low+pool[i]*(high-low))


def propose_gemini(history, reference, key, retry=False):
    # The only data-bearing fields are development aggregates and configuration metadata.
    contents = dict(feedback=[dict(point=dict(zip(KEYS,r["point"])),metrics=r["metrics"]) for r in history],
        bounds=dict(zip(KEYS,BOUNDS.tolist())), defaults=dict(zip(KEYS,PROBES[0])),
        service_floor=reference["service"]-SERVICE_MARGIN,cvar_ceiling_kwh=reference["cvar_kwh"]+CVAR_MARGIN)
    instructions = ("Propose a distinct configuration inside bounds. Minimize predicted mean SOH loss, "
        "subject to departure service at or above the floor and unmet-departure-energy CVaR95 at or below the ceiling. "
        "Qualified delivery is calibrated-PINN qualified 2 kW down at 17:00 with recovery at 18:00. "
        "Service, SOH loss, and qualification are fractions. Return only a JSON object with the three parameter keys.")
    if retry:
        rng = np.random.default_rng(739+len(history))
        center = np.array(min(history,key=lambda r:rank(r,reference))["point"])
        low, high = BOUNDS.T
        choices = np.clip(center+rng.uniform(-.2,.2,(12,len(KEYS)))*(high-low),low,high)
        choices = [dict(zip(KEYS,p.tolist())) for p in choices
                   if not any(np.allclose(p,r["point"],atol=1e-8,rtol=0) for r in history)]
        contents["unevaluated_configuration_options"] = choices
        instructions += (" The previous response repeated an evaluated configuration. "
            "Copy exactly one point from unevaluated_configuration_options.")
    payload = dict(contents=[dict(parts=[dict(text=instructions+"\n"+json.dumps(contents,allow_nan=False))])],
        generationConfig=dict(temperature=0,responseMimeType="application/json",maxOutputTokens=1024,
                              thinkingConfig=dict(thinkingBudget=0)))
    req = Request(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        data=json.dumps(payload).encode(), headers={"Content-Type":"application/json","x-goog-api-key":key})
    for attempt in range(4):
        try:
            with urlopen(req,timeout=75) as response:
                result = json.load(response)
            break
        except HTTPError as exc:
            try:
                error = json.load(exc)["error"]
            except (ValueError, KeyError, TypeError):
                error = {}
            details = error.get("details",[])
            quota = [v.get("quotaId","") for d in details for v in d.get("violations",[])]
            delay = next((d.get("retryDelay","") for d in details if "retryDelay" in d),"")
            parsed = re.fullmatch(r"([0-9.]+)s",delay)
            if exc.code == 429 and attempt < 3 and not any("perday" in q.lower() for q in quota):
                seconds = min(75.,max(2.,float(parsed[1])+2.)) if parsed else 65.
                print(f"Gemini rate limit; waiting {seconds:.0f}s.",flush=True)
                sleep(seconds)
                continue
            raise RuntimeError(f"Gemini API HTTP {exc.code}; quota identifiers: {quota}") from None
    candidate = result["candidates"][0]["content"]["parts"]
    values = json.loads("".join(p["text"] for p in candidate if "text" in p and not p.get("thought")))
    assert set(values)==set(KEYS)
    point = tuple(float(values[k]) for k in KEYS)
    assert np.all(np.isfinite(point)) and np.all(np.array(point)>=BOUNDS[:,0]) and np.all(np.array(point)<=BOUNDS[:,1])
    if any(np.allclose(point,r["point"],rtol=0,atol=1e-8) for r in history):
        if not retry:
            return propose_gemini(history, reference, key, retry=True)
        raise RuntimeError("Gemini repeated a point after retry")
    return point, dict(model=result.get("modelVersion",MODEL), response_id=result.get("responseId"),
                       repeated_point_retry=retry, sent=contents)


def propose_chatgpt(history, reference, enforce_delivery=False, bo_delivery_reference=False, explore=False):
    exe = os.environ["MATCHED_FEEDBACK_CODEX_EXE"]
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    contents = dict(feedback=[dict(point=dict(zip(KEYS,r["point"])),metrics=r["metrics"]) for r in history],
        bounds=dict(zip(KEYS,BOUNDS.tolist())), defaults=dict(zip(KEYS,PROBES[0])),
        service_floor=reference["service"]-SERVICE_MARGIN,cvar_ceiling_kwh=reference["cvar_kwh"]+CVAR_MARGIN)
    instructions = ("Propose one distinct in-bounds configuration from only the supplied feedback. "
        "Minimize predicted mean SOH loss subject to departure service at or above the floor and CVaR95 "
        "of unmet departure energy at or below the ceiling. Qualified delivery is calibrated-PINN "
        "qualified 2 kW down at 17:00 with recovery at 18:00. Service, SOH and qualification are fractions. "
        "Do not use tools. Return the three numeric parameter keys and no commentary.")
    if enforce_delivery:
        contents["qualified_delivery_floor"] = reference["qualified"]
        instructions += " Qualified 2 kW delivery must be at or above qualified_delivery_floor."
    if bo_delivery_reference:
        instructions += " This floor is the frozen, same-budget constrained BO qualification rate on development cases."
    if explore:
        instructions += (" For this proposal require at least one of: absolute difference from 1.2 in "
                         "w_price >= 0.18, or absolute difference from 2.0 in unmet_penalty_per_kwh >= 0.95.")
    schema = dict(type="object", properties={k:{"type":"number"} for k in KEYS},
                  required=KEYS, additionalProperties=False)
    with TemporaryDirectory(prefix="feedback-subscription-") as path:
        path = Path(path)
        (path/"schema.json").write_text(json.dumps(schema),encoding="utf-8")
        for attempt in range(2):
            if attempt:
                rng = np.random.default_rng(739+len(history))
                center = np.array(min(history,key=lambda r:rank(r,reference,enforce_delivery))["point"])
                low, high = BOUNDS.T
                options = np.clip(center+rng.uniform(-.2,.2,(12,len(KEYS)))*(high-low),low,high)
                contents["unevaluated_configuration_options"] = [dict(zip(KEYS,p.tolist())) for p in options
                    if (not explore or peak_exploration_ok(p))
                    and not any(np.allclose(p,r["point"],atol=1e-8,rtol=0) for r in history)]
                instructions += " Copy exactly one new point from unevaluated_configuration_options."
            cmd = [exe,"exec","--ephemeral","--ignore-user-config","--disable","shell_tool","--disable","plugins",
                   "-c",'web_search="disabled"',"-c",'approval_policy="never"', "--sandbox","read-only",
                   "--skip-git-repo-check","-C",str(path),"-m",CHATGPT_MODEL,
                   "--output-schema",str(path/"schema.json"),"-o",str(path/"result.json"),"-"]
            response = subprocess.run(cmd,input=instructions+"\n"+json.dumps(contents,allow_nan=False),
                text=True,encoding="utf-8",capture_output=True,timeout=120,env=env,cwd=path)
            if response.returncode:
                errors = [line for line in response.stderr.splitlines() if line.startswith(("ERROR:", "error:"))]
                raise RuntimeError(f"Subscription Codex proposal failed with exit code {response.returncode}: "
                                   + " ".join(errors[-2:]))
            values=json.loads((path/"result.json").read_text(encoding="utf-8"))
            point=tuple(float(values[k]) for k in KEYS)
            if (np.all(np.isfinite(point)) and np.all(np.array(point)>=BOUNDS[:,0])
                and np.all(np.array(point)<=BOUNDS[:,1])
                and (not explore or peak_exploration_ok(point))
                and not any(np.allclose(point,r["point"],rtol=0,atol=1e-8) for r in history)):
                sid=re.search(r"session id:\s*([0-9a-f-]+)",response.stderr)
                return point,dict(model=CHATGPT_MODEL,subscription=True,session_id=sid[1] if sid else None,
                                  repeated_or_invalid_point_retry=bool(attempt),sent=contents)
    raise RuntimeError("ChatGPT returned no distinct valid candidate after retry")


def run(source, output, arm, enforce_delivery=False, bo_delivery_reference=None):
    output.mkdir(parents=True,exist_ok=True)
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if arm=="gemini" and not key:
        raise RuntimeError("Gemini credential unavailable")
    data, provenance = load_cases(source)
    aging = Aging(source)
    pin = digest(aging.path)
    manifest = output/"manifest.csv"
    pd.DataFrame(provenance).to_csv(manifest,index=False)
    protocol = dict(probes=PROBES, parameters=dict(zip(KEYS,BOUNDS.tolist())), checkpoints=CHECKPOINTS,
        service_margin=SERVICE_MARGIN,cvar_margin_kwh=CVAR_MARGIN,pinn_sha256=pin,
        enforce_delivery=enforce_delivery, guardrail_reference="first shared probe within each split and regime",
        manifest_sha256=digest(manifest), model=CHATGPT_MODEL if arm=="chatgpt" else MODEL,
        recovery="2 kW down at 17:00; same-car recovery at 18:00; calibrated PINN and actual hard capacity",
        capacity="manuscript directional replay background 1500 kW and capacity-shortage 15 kW at 17:00–20:00",
        selection="feasible first; lowest SOH then qualified delivery; infeasible: smallest normalized violation",
        scope="three active weights in the hard-cap directional replay; independent parent split per regime")
    dump(output/"protocol.json",protocol)
    probes_path=output/"shared_probes.json"
    if probes_path.exists():
        saved=json.loads(probes_path.read_text(encoding="utf-8"))
        assert saved["manifest_sha256"]==protocol["manifest_sha256"] and saved["pinn_sha256"]==pin
        probes=saved["groups"]
    else:
        probes={}
        for group,(cases,forecast) in data.items():
            probes[group]=[dict(step=i+1,point=p,metrics=metrics(replay(cases["development"],forecast,aging,p)))
                           for i,p in enumerate(PROBES)]
        dump(probes_path,dict(manifest_sha256=protocol["manifest_sha256"],pinn_sha256=pin,groups=probes))
    refs = {g:dict(rows[0]["metrics"]) for g,rows in probes.items()}
    if bo_delivery_reference is not None:
        bo_lock = json.loads(bo_delivery_reference.read_text(encoding="utf-8"))
        assert set(bo_lock) == set(refs)
        for group, ref in refs.items():
            floors = {bo_lock[group][str(b)]["metrics"]["qualified"] for b in CHECKPOINTS}
            if len(floors) != 1:
                raise ValueError("Different checkpoint floors require a budget-specific search protocol.")
            ref["qualified"] = floors.pop()
        protocol.update(bo_delivery_reference_sha256=digest(bo_delivery_reference),
                        development_delivery_floors={g:r["qualified"] for g,r in refs.items()},
                        guardrail_reference="same-budget frozen BO development delivery; first shared probe for service and CVaR")
        dump(output/"protocol.json",protocol)
    histories={}
    if (output/f"{arm}_development.json").exists():
        histories=json.loads((output/f"{arm}_development.json").read_text(encoding="utf-8"))
    for group,(cases,forecast) in data.items():
        history=histories.get(group,list(probes[group]))
        rng=np.random.default_rng(204)
        while len(history)<6:
            reference=refs[group]
            if arm=="bo":
                point=propose_bo(history,reference,rng,enforce_delivery)
                details={}
            elif arm=="gemini":
                point,details=propose_gemini(history,reference,key)
            else:
                point,details=propose_chatgpt(history,reference,enforce_delivery,bo_delivery_reference is not None)
            history.append(dict(step=len(history)+1,point=point,
                 metrics=metrics(replay(cases["development"],forecast,aging,point)),**details))
            histories[group]=history
            dump(output/f"{arm}_development.json",histories)
        print(f"{arm} {group}: six development evaluations complete.",flush=True)
    lock={group:{str(b):min(hist[:b],key=lambda r:rank(r,refs[group],enforce_delivery)) for b in CHECKPOINTS}
          for group,hist in histories.items()}
    dump(output/f"{arm}_configuration_lock.json",lock)
    test, holdout_reference=[], {}
    for group,(cases,forecast) in data.items():
        baseline_rows = replay(cases["holdout"],forecast,aging,PROBES[0])
        holdout_reference[group] = metrics(baseline_rows)
        cache = {tuple(PROBES[0]): baseline_rows}
        for b in CHECKPOINTS:
            chosen=lock[group][str(b)]
            point = tuple(chosen["point"])
            if point not in cache:
                cache[point] = replay(cases["holdout"],forecast,aging,point)
            for row in cache[point]:
                test.append(dict(group=group,budget=b,selected_step=chosen["step"],**row))
    df=pd.DataFrame(test)
    assert len(df)==24
    df.to_csv(output/f"{arm}_holdout.csv",index=False)
    dump(output/"holdout_fixed_reference.json",holdout_reference)
    summary = []
    for (group,budget),rows in df.groupby(["group","budget"]):
        result = metrics(rows)
        checks = margins(result,holdout_reference[group],enforce_delivery)>=-1e-7
        summary.append(dict(group=group,budget=budget,**result,service_ok=bool(checks[0]),cvar_ok=bool(checks[1]),
            delivery_ok=bool(result["qualified"] >= holdout_reference[group]["qualified"]-1e-7),
            within_holdout_guardrail=bool(checks.all())))
    for b,rows in df.groupby("budget"):
        summary.append(dict(group="pooled",budget=b,**metrics(rows),within_holdout_guardrail=all(
            r["within_holdout_guardrail"] for r in summary if r["budget"]==b)))
    pd.DataFrame(summary).to_csv(output/f"{arm}_summary.csv",index=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("arm",choices=("bo","gemini","chatgpt"))
    p.add_argument("--enforce-delivery",action="store_true")
    p.add_argument("--bo-delivery-reference",type=Path)
    args=p.parse_args()
    torch.set_num_threads(1)
    if args.enforce_delivery and args.arm == "gemini":
        p.error("Delivery guardrail is implemented for BO and subscription GPT only.")
    if args.bo_delivery_reference and not args.enforce_delivery:
        p.error("The BO reference requires --enforce-delivery.")
    run(args.source.resolve(),args.output.resolve(),args.arm,args.enforce_delivery,args.bo_delivery_reference)
