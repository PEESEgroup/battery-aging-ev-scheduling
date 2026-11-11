"""Paired offline aging and capacity-timing ablation at fixed reference service."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from nonanticipative_scheduler import Forecast, reference_replay, solve_plan
from run_section_iv_replay import DURATIONS, GROUPS, ROOT, STARTS, load

def run():
    rows = []
    for cap_name, floor in (("Raw", 0), ("1500 kW floor", 1500)):
        for label, group in GROUPS.items():
            cases = load(group, floor)
            forecast = Forecast([c[2] for c in cases[:4]])
            test = cases[7:]
            for i, (name, fleet, grid) in enumerate(test):
                donor_name, _, donor_grid = test[(i+1) % len(test)]
                ref, losses, plans, histories = reference_replay(fleet, grid, forecast, 4)
                target = ref.sum(axis=1)
                for k in STARTS:
                    prefix = ref.copy()
                    prefix[:, k:] = 0
                    for duration in DURATIONS:
                        fixed = plans[k].sum(axis=0)[k:k+duration]
                        for direction in (1, -1):
                            for pairing in ("Original", "Permuted future capacity"):
                                caps = grid[k:, 3].copy()
                                if pairing != "Original":
                                    caps[1:] = donor_grid[k+1:, 3]
                                for aging in ("Zero additional", "No incremental limit"):
                                    result = solve_plan(
                                        fleet, k, caps, np.zeros(len(caps)), prefix, histories[k],
                                        target=target, budget=losses if aging == "Zero additional" else None,
                                        window=(k, fixed, direction, None), offline=True)
                                    rows.append(dict(capacity=cap_name, group=label, scenario=name,
                                                     donor=donor_name, start_hour=k, duration_h=duration,
                                                     direction="Up" if direction==1 else "Down",
                                                     pairing=pairing, aging_budget=aging,
                                                     reference_served_kwh=target.sum(),
                                                     offline_kw=np.nan if result is None else max(0., result[1])))
            print(cap_name, label, flush=True)
    return pd.DataFrame(rows)

def summarize(frame):
    labels = ["capacity","group","scenario","start_hour","duration_h","direction"]
    specs = [
        ("Remove aging budget", "aging_budget", "No incremental limit", "Zero additional", "pairing"),
        ("Re-pair future capacity", "pairing", "Permuted future capacity", "Original", "aging_budget"),
    ]
    contrasts = []
    for name, arm, changed, base, fixed in specs:
        x = frame.pivot(index=labels+[fixed], columns=arm, values="offline_kw").reset_index()
        x["contrast"] = name
        x["fixed_arm"] = x[fixed]
        x["baseline_kw"] = x[base]
        x["changed_kw"] = x[changed]
        x["delta_kw"] = x[changed]-x[base]
        contrasts.append(x[labels+["contrast","fixed_arm","baseline_kw","changed_kw","delta_kw"]])
    paired = pd.concat(contrasts, ignore_index=True)
    paired["positive"] = paired.delta_kw.gt(1e-6)
    paired["negative"] = paired.delta_kw.lt(-1e-6)
    stats = paired.groupby(["capacity","contrast","fixed_arm","direction"]).delta_kw.agg(
        total="size", resolved="count", minimum="min", median="median", maximum="max").reset_index()
    stats = stats.merge(paired.groupby(["capacity","contrast","fixed_arm","direction"])[["positive","negative"]]
                        .sum().reset_index())
    return paired, stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"fixed_service_ablation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = run()
    paired, stats = summarize(frame)
    frame.to_csv(args.output/"cases.csv", index=False)
    paired.to_csv(args.output/"paired_contrasts.csv", index=False)
    stats.to_csv(args.output/"summary.csv", index=False)
