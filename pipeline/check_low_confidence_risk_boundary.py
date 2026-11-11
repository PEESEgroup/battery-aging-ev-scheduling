"""Extend the risk grid on the frozen 2.25 kW low-confidence panel."""
import argparse
from dataclasses import replace
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lock_recovery_holdout import digest
from run_joint_context_warm_start import CHECKPOINTS, LIMIT, chosen_rows, local_search, split_metrics
from run_llm_timing_pilot import DEFAULTS, RECOVERY, dump, rolling_reference, simulate
from screen_adaptive_recovery_reserve import construct
from screen_aging_derating_overlap import summarize
from screen_forecast_adaptive_reserve import BASELINE, KEYS, POINTS
from screen_low_confidence_recovery import LowConfidenceForecast, source_entries_and_forecast


RISK_EXTENSION = (1.1, 1.15, 1.25, 1.35, 1.5, 1.8)
RESERVES = (6., 8., 10.)
OLD_WINNER = (1.2, 8.)
ROW_ID = ["parent", "bucket", "branch"]


def reconstruct(entries, manifest, cap):
    sources, base = source_entries_and_forecast(entries, cap)
    by_parent = {row["parent"]: case for row, case, _ in sources}
    cases = []
    for (parent, bucket), group in manifest.groupby(["parent", "bucket"], sort=False):
        fleet, grid, temp, _ = by_parent[parent]
        variants = group.set_index("branch")
        caps = tuple(variants.loc[b, "recovery_hard_cap_kw"] for b in ("nominal", "moderate", "severe"))
        assert caps[-1] == cap
        forecast = LowConfidenceForecast(base, (*caps, grid[RECOVERY+1, 3]), 1+bucket//2)
        assert group.soc_values.nunique() == 1
        soc = fleet.soc.copy()
        for i, value in json.loads(group.soc_values.iloc[0]):
            soc[i] = value
        adjusted = replace(fleet, soc=soc)
        assert np.allclose(adjusted.target, fleet.target, atol=1e-7, rtol=0.)
        for row in group.to_dict("records"):
            actual = grid.copy()
            actual[RECOVERY, 3] = row["recovery_hard_cap_kw"]
            cases.append((row, (adjusted, actual, temp, None), forecast))
    return cases


def replay(entries, aging, choose, stage):
    records = []
    references = {}
    parent = None
    for i, (row, case, forecast) in enumerate(entries, 1):
        if row["parent"] != parent:
            references.clear()
            parent = row["parent"]
        for risk, reserve in choose(row["bucket"]):
            params = dict(DEFAULTS, lambda_risk=risk, recovery_reserve_kw=reserve)
            key = (row["bucket"]//2, row["branch"], risk, reserve)
            if key not in references:
                references[key] = rolling_reference(case, forecast, params)
            result = simulate(case, forecast, aging, params, reference=references[key])
            records.append({k: row[k] for k in (*ROW_ID, "event", "probability")} |
                           dict(risk=risk, reserve_kw=reserve, **result))
        if i % 24 == 0 or i == len(entries):
            print(f"{stage}: {i}/{len(entries)} branches", flush=True)
    df = pd.DataFrame(records)
    assert not df.duplicated(ROW_ID+KEYS).any()
    return df


def stats(rows):
    return {k: float(v) for k, v in summarize(rows).items()}


def evaluate(dataset, policy, baseline):
    panels = split_metrics(chosen_rows(dataset, policy))
    return {name: m | dict(cvar_delta_kwh=m["cvar_unmet_kwh"]-baseline[name]["cvar_unmet_kwh"],
        cvar_ok=m["cvar_unmet_kwh"] <= baseline[name]["cvar_unmet_kwh"]+LIMIT+1e-7)
        for name, m in panels.items()}


def run(source, factorial, fresh, prior_stress, prior, output):
    output.mkdir(parents=True, exist_ok=False)
    source_lock = json.loads((prior/"scenario_lock.json").read_text(encoding="utf-8"))
    assert digest(prior/"locked_manifest.csv") == source_lock["manifest_sha256"]
    protocol = json.loads((prior/"protocol.json").read_text(encoding="utf-8"))
    assert protocol["severe_absolute_cap_kw"] == 2.25 and protocol["cvar95_margin_kwh"] == LIMIT
    manifest = pd.read_csv(prior/"locked_manifest.csv")
    assert manifest.groupby("split").parent.nunique().to_dict() == {"development": 8, "holdout": 14}
    assert manifest.groupby("parent").split.nunique().eq(1).all()
    aging, _, source_cases = construct(source, factorial, fresh, prior_stress, output/"source_design", audit_physical=False)
    cases = reconstruct(source_cases, manifest, 2.25)
    dev_cases = [x for x in cases if x[0]["split"] == "development"]
    held_cases = [x for x in cases if x[0]["split"] == "holdout"]
    old = pd.read_csv(prior/"development_replays.csv")
    assert len(old) == len(dev_cases)*len(POINTS)
    new_points = tuple(product(RISK_EXTENSION, RESERVES))
    all_points = tuple(sorted((*POINTS, *new_points)))
    dump(output/"protocol.json", dict(
        panel_lock_sha256=digest(prior/"scenario_lock.json"), pinn_sha256=digest(aging.path),
        existing_dev_replays_sha256=digest(prior/"development_replays.csv"),
        risk_grid=sorted({r for r, _ in all_points}), reserve_grid_kw=RESERVES,
        baseline=BASELINE, prior_winner=OLD_WINNER, tolerance_kwh=LIMIT,
        guardrail="CVaR95 of unmet departure energy vs fixed reference, overall and high exposure separately",
        fixed_selection="maximize qualification subject to both CVaR limits, then CVaR, demand satisfaction, SOH loss",
        state_selection="same selection with one CVaR limit in each cell, then use common six-evaluation local search with both pooled guardrails",
        holdout="fixed and adaptive policies and all four source-state winners replayed in every target state",
        api_calls=0))
    # Verify one reused point across the four states before extending the grid.
    spot = [next(x for x in dev_cases if x[0]["bucket"] == b) for b in range(4)]
    check = replay(spot, aging, lambda _: [BASELINE], "Reconstruction audit")
    joined = check.merge(old, on=ROW_ID+KEYS, suffixes=("_new", "_old"))
    assert len(joined) == len(spot)
    for key in ("qualified", "served_ratio", "mean_soh_loss", "unmet_kwh", "actual_kw"):
        assert np.allclose(joined[key+"_new"], joined[key+"_old"], atol=1e-7, rtol=0.)
    dev = pd.concat([old, replay(dev_cases, aging, lambda _: new_points, "Development")], ignore_index=True)
    assert len(dev) == len(dev_cases)*len(all_points)
    dev.to_csv(output/"development_replays.csv", index=False)
    base = split_metrics(chosen_rows(dev, [BASELINE]*4))
    fixed, per_state = [], []
    for point in all_points:
        panels = evaluate(dev, [point]*4, base)
        if all(m["cvar_ok"] for m in panels.values()):
            m = panels["overall"]
            fixed.append(((-m["qualified_rate"], m["cvar_unmet_kwh"], -m["served_ratio"],
                           m["mean_soh_loss"]), point))
        for b, part in dev[(dev.risk == point[0]) & (dev.reserve_kw == point[1])].groupby("bucket"):
            per_state.append(dict(bucket=int(b), risk=point[0], reserve_kw=point[1], **stats(part)))
    fixed = min(fixed)[1]
    grid = pd.DataFrame(per_state)
    cell_base = grid[(grid.risk == BASELINE[0]) & (grid.reserve_kw == BASELINE[1])].set_index("bucket").cvar_unmet_kwh
    grid["cvar_ok"] = grid.cvar_unmet_kwh <= grid.bucket.map(cell_base)+LIMIT+1e-7
    grid.to_csv(output/"development_grid.csv", index=False)
    winners = grid[grid.cvar_ok].sort_values(
        ["qualified_rate", "cvar_unmet_kwh", "served_ratio", "mean_soh_loss", *KEYS],
        ascending=[False, True, False, True, True, True]).drop_duplicates("bucket").sort_values("bucket")
    winners.to_csv(output/"development_state_winners.csv", index=False)
    initial = tuple(winners[KEYS].itertuples(index=False, name=None))
    search = local_search(dev, initial, base, all_points)
    selected = {b: min(search[:b], key=lambda r: r["rank"]) for b in CHECKPOINTS}
    dump(output/"configuration_lock.json", dict(
        fixed_best=fixed, state_winners=initial, selected=selected, development_search=search,
        development_replay_sha256=digest(output/"development_replays.csv"),
        panel_sha256=source_lock["manifest_sha256"]))
    print(f"Locked fixed {fixed}, state winners {initial}.", flush=True)

    common = {BASELINE, OLD_WINNER, fixed, *initial}
    needed = {b: sorted(common | {r["policy"][b] for r in selected.values()}) for b in range(4)}
    held = replay(held_cases, aging, lambda b: needed[b], "Holdout")
    held.to_csv(output/"holdout_replays.csv", index=False)
    summaries, cross = [], []
    for split, dataset in (("development", dev), ("holdout", held)):
        reference = split_metrics(chosen_rows(dataset, [BASELINE]*4))
        policies = [("fixed_reference", 0, [BASELINE]*4), ("prior_1.2_8", 0, [OLD_WINNER]*4),
                    ("fixed_best", 0, [fixed]*4)] + [
                    ("adaptive", b, selected[b]["policy"]) for b in CHECKPOINTS]
        for arm, budget, p in policies:
            assert len(chosen_rows(dataset, p)) == (len(dev_cases) if split == "development" else len(held_cases))
            for panel, metrics in evaluate(dataset, p, reference).items():
                summaries.append(dict(split=split, arm=arm, budget=budget, panel=panel, **metrics))
        for source_bucket, (risk, reserve) in enumerate(initial):
            for target_bucket, part in dataset[(dataset.risk == risk) & (dataset.reserve_kw == reserve)].groupby("bucket"):
                cross.append(dict(split=split, source_bucket=source_bucket, target_bucket=target_bucket,
                                  risk=risk, reserve_kw=reserve, **stats(part)))
    pd.DataFrame(summaries).to_csv(output/"summary.csv", index=False)
    pd.DataFrame(cross).to_csv(output/"cross_replay.csv", index=False)
    dump(output/"completion.json", dict(new_development_replays=len(dev)-len(old), holdout_replays=len(held),
        lock_sha256=digest(output/"configuration_lock.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "factorial", "fresh", "prior_stress", "prior", "output"):
        parser.add_argument(name, type=Path)
    torch.set_num_threads(1)
    run(*(p.resolve() for p in vars(parser.parse_args()).values()))
