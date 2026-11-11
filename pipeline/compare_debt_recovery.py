"""Paired rolling debt recovery vs same-hour overlay on the frozen low-confidence panel."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from check_low_confidence_risk_boundary import reconstruct
from diagnose_low_confidence_reversals import find_reversals
from lock_recovery_holdout import digest
from run_joint_context_warm_start import LIMIT, split_metrics
from run_llm_timing_pilot import DEFAULTS, dump, rolling_reference, simulate
from screen_adaptive_recovery_reserve import construct
from screen_aging_derating_overlap import summarize
from screen_forecast_adaptive_reserve import BASELINE, KEYS


POINTS = ((.6, 8.), BASELINE, (1.1, 8.), (1.2, 8.), (1.25, 8.),
          (1.1, 0.), (1.1, 1.), (1.1, 2.))
MATCH = ["parent", "bucket", "branch", *KEYS]


def replay(entries, aging, split):
    rows = []
    for i, (row, case, forecast) in enumerate(entries, 1):
        for risk, reserve in POINTS:
            params = dict(DEFAULTS, lambda_risk=risk, recovery_reserve_kw=reserve)
            ref = rolling_reference(case, forecast, params)
            original = simulate(case, forecast, aging, params, reference=ref)
            original.update(qualified_at_recovery=original["qualified"], service_at_recovery=original["service_match"])
            revised = simulate(case, forecast, aging, params, reference=ref, debt_recovery=True)
            assert abs(revised["actual_kw"]-original["actual_kw"]) <= 1e-7
            for policy, metrics in (("overlay", original), ("debt_aware", revised)):
                rows.append({k: row[k] for k in ("parent", "bucket", "branch", "event", "probability")} |
                            dict(policy=policy, risk=risk, reserve_kw=reserve, **metrics))
        if i % 24 == 0 or i == len(entries):
            print(f"{split}: {i}/{len(entries)} branches.", flush=True)
    df = pd.DataFrame(rows)
    assert not df.duplicated(["policy", *MATCH]).any()
    assert len(df) == 2*len(entries)*len(POINTS)
    return df


def state_grid(df, policy, strict=False):
    part = df[df.policy == policy].copy()
    if strict:
        part["qualified"] = part.qualified_at_recovery
    grid = pd.DataFrame([dict(bucket=b, risk=r, reserve_kw=k, **summarize(rows).to_dict())
        for (b,r,k), rows in part.groupby(["bucket", *KEYS])])
    baseline = grid[(grid.risk == BASELINE[0]) & (grid.reserve_kw == BASELINE[1])].set_index("bucket").cvar_unmet_kwh
    grid["cvar_ok"] = grid.cvar_unmet_kwh <= grid.bucket.map(baseline)+LIMIT+1e-7
    return grid


def check_reversals(dev, held):
    result = []
    for policy in ("overlay", "debt_aware"):
        for strict in (True, False):
            candidates = find_reversals(state_grid(dev, policy, strict))
            h = state_grid(held, policy, strict).set_index([*KEYS, "bucket"])
            for row in candidates.itertuples(index=False):
                a, b = (row.risk_a, row.reserve_a), (row.risk_b, row.reserve_b)
                d, e = (h.loc[(*a,k),"qualified_rate"]-h.loc[(*b,k),"qualified_rate"]
                        for k in (row.state_a, row.state_b))
                guard = all(h.loc[(*p,k), "cvar_ok"] for p in (a,b) for k in (row.state_a,row.state_b))
                replicated = guard and d*row.dev_gap_state_a > 0 and e*row.dev_gap_state_b > 0 and min(abs(d),abs(e)) >= .01
                result.append(row._asdict() | dict(policy=policy, window="18:00" if strict else "departure",
                    holdout_gap_state_a=d, holdout_gap_state_b=e, holdout_cvar_ok=guard, replicated=replicated))
    return pd.DataFrame(result)


def summaries(dev, held):
    result = []
    for split, frame in (("development", dev), ("holdout", held)):
        for (policy, risk, reserve), rows in frame.groupby(["policy", *KEYS]):
            for panel, metrics in split_metrics(rows).items():
                part = rows if panel == "overall" else rows[rows.bucket.isin([1,3])]
                strict = part.assign(qualified=part.qualified_at_recovery)
                result.append(dict(split=split, policy=policy, risk=risk, reserve_kw=reserve, panel=panel,
                    **metrics, qualified_18=float(summarize(strict).qualified_rate),
                    service_pass=float((part.service_match*part.probability).sum()/part.probability.sum()),
                    service_18_pass=float((part.service_at_recovery*part.probability).sum()/part.probability.sum())))
    return pd.DataFrame(result)


def run(source, factorial, fresh, prior_stress, panel, output):
    output.mkdir(parents=True, exist_ok=False)
    lock = json.loads((panel/"scenario_lock.json").read_text(encoding="utf-8"))
    manifest_file = panel/"locked_manifest.csv"
    assert digest(manifest_file) == lock["manifest_sha256"]
    manifest = pd.read_csv(manifest_file)
    assert manifest.groupby("split").parent.nunique().to_dict() == {"development":8, "holdout":14}
    assert manifest.groupby("parent").split.nunique().eq(1).all()
    dump(output/"protocol.json", dict(
        points=POINTS, reserve_control_risk=1.1, same_hour_metric="restore all curtailed energy by end of 18:00",
        departure_metric="restore all same-vehicle energy by original departure; no schedule extension",
        activation="same forecast and PINN screened 17:00 swap for both policies",
        change="18:00 onward residual-capacity LP carries each EV's unrecovered debt; forecast PINN and realized PINN both checked",
        reversal="same +/-1 pp criterion and +0.5 kWh per-state CVaR guard as previous audit",
        panel_sha256=lock["manifest_sha256"], api_calls=0))
    aging, _, source_cases = construct(source, factorial, fresh, prior_stress, output/"source_design", audit_physical=False)
    cases = reconstruct(source_cases, manifest, 2.25)
    dev = replay([x for x in cases if x[0]["split"] == "development"], aging, "Development")
    dev.to_csv(output/"development_replays.csv", index=False)
    development_candidates = {}
    for policy in ("overlay", "debt_aware"):
        for strict in (True, False):
            key = f"{policy}_{'18' if strict else 'departure'}"
            development_candidates[key] = find_reversals(state_grid(dev, policy, strict)).to_dict("records")
    dump(output/"configuration_lock.json", dict(points=POINTS,
        development_sha256=digest(output/"development_replays.csv"), candidate_reversals=development_candidates,
        pinn_sha256=digest(aging.path)))
    print("Development choices locked before holdout.", flush=True)
    held = replay([x for x in cases if x[0]["split"] == "holdout"], aging, "Holdout")
    held.to_csv(output/"holdout_replays.csv", index=False)
    results = summaries(dev, held)
    results.to_csv(output/"summary.csv", index=False)
    results[(results.risk == 1.1) & results.reserve_kw.isin([0.,1.,2.])].to_csv(
        output/"unsaturated_reserve_summary.csv", index=False)
    reversals = check_reversals(dev, held)
    reversals.to_csv(output/"reversal_validation.csv", index=False)
    dump(output/"completion.json", dict(dev_replays=len(dev), holdout_replays=len(held),
        development_reversals=len(reversals), replicated_reversals=int(reversals.replicated.sum()) if not reversals.empty else 0,
        config_sha256=digest(output/"configuration_lock.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "factorial", "fresh", "prior_stress", "panel", "output"):
        parser.add_argument(name, type=Path)
    torch.set_num_threads(1)
    run(*(p.resolve() for p in vars(parser.parse_args()).values()))
