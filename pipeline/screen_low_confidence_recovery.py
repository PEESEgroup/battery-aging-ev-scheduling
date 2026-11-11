"""Physics-lock a fresh low-confidence panel and run a local equal-budget baseline."""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lock_recovery_holdout import digest
from run_joint_context_warm_start import CHECKPOINTS, LIMIT, local_search, split_metrics, chosen_rows
from run_llm_timing_pilot import DEFAULTS, RECOVERY, REQUEST, START, TimingForecast, dump, rolling_reference, simulate
from screen_adaptive_recovery_reserve import construct, make_fleet
from screen_aging_derating_overlap import HOURS, summarize
from screen_forecast_adaptive_reserve import BASELINE, KEYS, POINTS
from screen_recovery_departure import adjacent_bound


QUOTA = 14
SEVERE = 1.5
MIN_CONFIDENCE, MAX_CONFIDENCE = .25, .75
HIGH_MIN, LOW_MAX = .60, .45
SEVERITIES = (1, 2)


class LowConfidenceForecast(TimingForecast):
    def __init__(self, base, caps, severe_paths):
        self.donors, self.center, self.scale = base.donors, base.center, base.scale
        self.caps, self.severe_paths = caps, severe_paths

    def scenario_paths(self, t, observed):
        paths = super().scenario_paths(t, observed)
        if RECOVERY > t:
            paths[:, RECOVERY-t, 3] = self.caps[0]
            paths[-self.severe_paths-1, RECOVERY-t, 3] = self.caps[1]
            paths[-self.severe_paths:, RECOVERY-t, 3] = self.caps[2]
        if RECOVERY+1 > t:
            paths[:, RECOVERY+1-t, 3] = self.caps[3]
        return paths


def context(case, forecast, params, threshold):
    fleet, grid, _, _ = case
    reference = rolling_reference(case, forecast, params)
    proposal = reference[1]
    at_recovery = proposal[:, RECOVERY]
    room = forecast.scenario_paths(START, grid[:START+1])[:, RECOVERY-START, 3]-at_recovery.sum()
    eligible = (fleet.arrival <= START) & (fleet.departure > RECOVERY)
    movable = np.maximum(0., np.minimum(proposal[:, START], fleet.power-at_recovery))*eligible
    soc = fleet.soc+proposal[:, :RECOVERY].sum(axis=1)/fleet.capacity
    rebound = movable*min(1., REQUEST/max(movable.sum(), 1e-9))
    return reference, float(np.mean(np.minimum(room, movable.sum()) >= REQUEST-1e-7)), float(
        rebound[soc >= threshold].sum()/max(rebound.sum(), 1e-9)), float(room.min())


def source_entries_and_forecast(entries, severe_cap):
    # Reuse the archived vehicle/grid draws and donor bank. Restrict to one
    # background variant before constructing the three explicitly weighted tails.
    entries = [e for e in entries if not e[0]["event"] and e[0]["event_probability"] == .25
               and e[0]["target_remaining_room_kw"] == 4.25 and e[0]["target_high_soc_share"] == .25]
    original = entries[0][2]
    donors = list(original.donors)
    for row, case, _ in entries:
        if row["split"] == "development":
            donor = case[1].copy()
            donor[RECOVERY, 3] = severe_cap
            donors.append(donor)
    norm = TimingForecast(donors)
    base = TimingForecast(original.donors)
    base.center, base.scale = norm.center, norm.scale
    return entries, base


def build(entries, threshold, output, excluded, severe_cap):
    entries, base = source_entries_and_forecast(entries, severe_cap)
    params = dict(DEFAULTS, lambda_risk=BASELINE[0], recovery_reserve_kw=0.)
    dump(output/"protocol.json", dict(
        screening="first 14 fresh parents passing all four forecast/physics states; no PINN outcomes used",
        confidence_range=[MIN_CONFIDENCE, MAX_CONFIDENCE], high_soc_rebound_min=HIGH_MIN,
        control_soc_rebound_max=LOW_MAX, severe_absolute_cap_kw=severe_cap,
        branches="1 or 2 of 4 severe paths, one moderate, balance nominal; both non-severe branches have hard caps >=2 kW",
        moderate_cap="max(2.25 kW, 55% of nominal), never above nominal",
        screen_physical="at least one non-severe branch supports 2 kW per state; retain all three branches",
        main="pooled probabilities; high exposure comprises buckets 1 and 3",
        cvar95_margin_kwh=LIMIT, fixed_reference=BASELINE, common_points=POINTS, common_budgets=CHECKPOINTS,
        shared_prior="identical state summaries plus identical nine-point development replay tables for future LLM and rule",
        api_calls=0))
    audit, locked, cases = [], [], []
    n = 0
    entries.sort(key=lambda x: (x[0]["split"], int(Path(x[0]["source"]).name.split("_")[-1])))
    for row, case, _ in entries:
        split, parent = row["split"], row["parent"]
        assert parent not in excluded or split == "development"
        fleet, original_grid, temp, _ = case
        cap = original_grid[RECOVERY, 3]
        if cap < 2.25:
            continue
        caps = (cap, max(2.25, .55*cap), severe_cap, original_grid[RECOVERY+1, 3])
        parent_rows, parent_cases, eligibility = [], [], []
        for s, count in enumerate(SEVERITIES):
            forecast = LowConfidenceForecast(base, caps, count)
            proposal = rolling_reference(case, forecast, params)[1]
            for high in (False, True):
                stress, _, _, _ = make_fleet(fleet, proposal, .9 if high else .25,
                    np.random.default_rng(int(parent[:12], 16)), threshold)
                point_case = stress, original_grid, temp, None
                _, confidence, rebound, minimum_room = context(point_case, forecast, params, threshold)
                qualifies = MIN_CONFIDENCE <= confidence <= MAX_CONFIDENCE and (
                    rebound >= HIGH_MIN if high else rebound <= LOW_MAX)
                specs = (("nominal", caps[0], (3-count)/4), ("moderate", caps[1], .25),
                         ("severe", caps[2], count/4))
                feasible = False
                for branch, hard_cap, prob in specs:
                    grid = original_grid.copy()
                    grid[RECOVERY, 3] = hard_cap
                    test = stress, grid, temp, None
                    ref = rolling_reference(test, forecast, params)
                    bound = adjacent_bound(stress, grid, ref[0])
                    feasible |= branch != "severe" and bound >= REQUEST-1e-7
                    changes = np.flatnonzero(np.abs(stress.soc-fleet.soc) > 1e-7)
                    saved = dict(split=split, parent=parent, source=row["source"], bucket=2*s+int(high),
                        event=branch != "nominal", branch=branch, probability=prob, high_exposure=high,
                        severe_probability=count/4, forecast_confidence=confidence, high_soc_rebound_share=rebound,
                        forecast_min_room_kw=minimum_room, recovery_hard_cap_kw=hard_cap,
                        physical_kw=bound, physical=bound >= REQUEST-1e-7,
                        soc_values=json.dumps([[int(i), float(stress.soc[i])] for i in changes]))
                    parent_rows.append(saved)
                    parent_cases.append((saved, test, forecast))
                eligibility.append(bool(qualifies and feasible))
        passes = all(eligibility)
        take = passes and (split == "development" or n < QUOTA)
        if take:
            locked.extend(parent_rows)
            cases.extend(parent_cases)
            n += int(split == "holdout")
        audit.append(dict(split=split, parent=parent, states_passing=sum(eligibility), passes=passes, locked=take))
    pd.DataFrame(audit).to_csv(output/"screen_audit.csv", index=False)
    manifest = pd.DataFrame(locked)
    manifest.to_csv(output/"locked_manifest.csv", index=False)
    counts = manifest.groupby("split").parent.nunique().to_dict() if not manifest.empty else {}
    dump(output/"scenario_lock.json", dict(parents=counts, quota=QUOTA,
        manifest_sha256=digest(output/"locked_manifest.csv"), audit_sha256=digest(output/"screen_audit.csv")))
    print(f"Physics-locked parents: {counts}.", flush=True)
    return manifest, cases


def replay(entries, aging, choose, label):
    rows = []
    for i, (row, case, forecast) in enumerate(entries, 1):
        for risk, reserve in choose(row["bucket"]):
            result = simulate(case, forecast, aging, dict(DEFAULTS, lambda_risk=risk, recovery_reserve_kw=reserve))
            rows.append({k: row[k] for k in ("parent", "bucket", "event", "branch", "probability")} |
                        dict(risk=risk, reserve_kw=reserve, **result))
        if i % 60 == 0 or i == len(entries):
            print(f"{label}: {i}/{len(entries)} branches", flush=True)
    return pd.DataFrame(rows)


def choose_initial(dev, output):
    grid = []
    for (bucket, risk, reserve), rows in dev.groupby(["bucket", *KEYS]):
        grid.append(dict(bucket=int(bucket), risk=risk, reserve_kw=reserve,
                         **{k: float(v) for k, v in summarize(rows).items() if k != "parents"}))
    grid = pd.DataFrame(grid)
    base = grid[(grid.risk == BASELINE[0]) & (grid.reserve_kw == BASELINE[1])].set_index("bucket").cvar_unmet_kwh
    grid["within_cell_cvar"] = grid.cvar_unmet_kwh <= grid.bucket.map(base)+LIMIT+1e-7
    grid.to_csv(output/"common_development_grid.csv", index=False)
    winners = grid[grid.within_cell_cvar].sort_values(
        ["qualified_rate", "cvar_unmet_kwh", "served_ratio", "mean_soh_loss"],
        ascending=[False, True, False, True]).drop_duplicates("bucket").set_index("bucket").sort_index()
    initial = tuple(tuple(p) for p in winners[KEYS].itertuples(index=False, name=None))
    source = [(int(b), *p) for b, p in enumerate(initial)]
    cross = pd.DataFrame(source, columns=["source_bucket", *KEYS]).merge(grid, on=KEYS)
    cross.to_csv(output/"development_cross_replay.csv", index=False)
    return initial


def run(source, factorial, fresh, prior_stress, old, recent, output, severe_cap=SEVERE):
    output.mkdir(parents=True, exist_ok=False)
    excluded = set().union(*(set(pd.read_csv(p/"locked_manifest.csv").parent_fingerprint) for p in (old, recent)))
    aging, _, entries = construct(source, factorial, fresh, prior_stress, output/"source_design", audit_physical=False)
    manifest, cases = build(entries, aging.model.config.soc_threshold, output, excluded, severe_cap)
    counts = manifest.groupby("split").parent.nunique().to_dict() if not manifest.empty else {}
    if counts.get("development", 0) < 5 or counts.get("holdout", 0) < 8:
        print("Fewer than five development or eight holdout parents met the frozen physics criteria.", flush=True)
        return
    dev = replay([e for e in cases if e[0]["split"] == "development"], aging, lambda _: POINTS, "Development")
    dev.to_csv(output/"development_replays.csv", index=False)
    init = choose_initial(dev, output)
    base = split_metrics(chosen_rows(dev, [BASELINE]*4))
    fixed = []
    for p in POINTS:
        m = split_metrics(chosen_rows(dev, [p]*4))
        if all(m[k]["cvar_unmet_kwh"] <= base[k]["cvar_unmet_kwh"]+LIMIT+1e-7 for k in m):
            fixed.append((-m["overall"]["qualified_rate"], m["overall"]["cvar_unmet_kwh"],
                          m["overall"]["mean_soh_loss"], p))
    best_fixed = min(fixed)[-1]
    records = local_search(dev, init, base)
    selected = {k: min(records[:k], key=lambda r: r["rank"]) for k in CHECKPOINTS}
    summary = manifest[manifest.split == "development"].drop_duplicates(["parent", "bucket"]).groupby("bucket")[
        ["forecast_confidence", "high_soc_rebound_share", "forecast_min_room_kw", "severe_probability"]].mean()
    table = pd.read_csv(output/"common_development_grid.csv")
    dump(output/"shared_offline_prior.json", dict(default=BASELINE, bounds=POINTS, cvar_margin_kwh=LIMIT,
         states={str(b): dict(state=summary.loc[b].to_dict(),
             offline_prior=table[table.bucket.eq(b)].drop(columns="bucket").to_dict("records")) for b in range(4)}))
    dump(output/"configuration_lock.json", dict(initial=init, selected=selected, fixed=best_fixed,
        development_search=records, scenario_lock_sha256=digest(output/"scenario_lock.json"),
        shared_prior_sha256=digest(output/"shared_offline_prior.json")))
    print(f"Locked strong rule {init}, fixed {best_fixed}; evaluating holdout.", flush=True)
    need = {b: sorted({BASELINE, best_fixed, *(r["policy"][b] for r in selected.values())}) for b in range(4)}
    held = replay([e for e in cases if e[0]["split"] == "holdout"], aging, lambda b: need[b], "Holdout")
    held.to_csv(output/"holdout_replays.csv", index=False)
    results = []
    for split, dataset in (("development", dev), ("holdout", held)):
        reference = split_metrics(chosen_rows(dataset, [BASELINE]*4))
        for arm, budget, policy in [("fixed_reference", 0, [BASELINE]*4),
                                    ("fixed_best", 0, [best_fixed]*4)]+[
                ("strong_rule", b, selected[b]["policy"]) for b in CHECKPOINTS]:
            for panel, m in split_metrics(chosen_rows(dataset, policy)).items():
                delta = m["cvar_unmet_kwh"]-reference[panel]["cvar_unmet_kwh"]
                results.append(dict(split=split, arm=arm, budget=budget, panel=panel, **m,
                    cvar_delta_kwh=delta, cvar_ok=delta<=LIMIT+1e-7))
    pd.DataFrame(results).to_csv(output/"summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "factorial", "fresh", "prior_stress", "old", "recent", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--severe-cap", type=float, default=SEVERE)
    torch.set_num_threads(1)
    args = parser.parse_args()
    run(*(v.resolve() if isinstance(v, Path) else v for v in vars(args).values()))
