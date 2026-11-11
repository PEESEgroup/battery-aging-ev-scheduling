"""Paired genuine API warm starts on a newly generated, locked recovery panel."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lock_recovery_holdout import digest
from run_llm_timing_pilot import DEFAULTS, RECOVERY, REQUEST, START, dump, rolling_reference
from screen_adaptive_recovery_reserve import STATE, construct
from screen_aging_derating_overlap import summarize
from screen_forecast_adaptive_reserve import BASELINE, ID, KEYS, POINTS, replay, state_bin


CHECKPOINTS = (1, 3, 6)
LIMIT = .5
ARMS = ("llm_original", "llm_extended", "adaptive_rule")
SEED = 182


def forecast_features(case, forecast, threshold):
    fleet, grid, _, _ = case
    _, proposal, _ = rolling_reference(case, forecast, dict(DEFAULTS, lambda_risk=.9))
    paths = forecast.scenario_paths(START, grid[:START+1])
    cap = paths[:, RECOVERY-START, 3]
    recovery = proposal[:, RECOVERY]
    room = cap-recovery.sum()
    eligible = (fleet.arrival <= START) & (fleet.departure > RECOVERY)
    movable = np.maximum(0., np.minimum(proposal[:, START], fleet.power-recovery))*eligible
    soc = fleet.soc+proposal[:, :RECOVERY].sum(axis=1)/fleet.capacity
    high = soc >= threshold
    amount = movable.sum()
    confidence = float(np.mean(np.minimum(room, amount) >= REQUEST-1e-7))
    rebound = movable*min(1., REQUEST/max(amount, 1e-9))
    total_recovery = recovery+rebound
    return dict(predicted_room_kw=float(max(0., room.min())),
                predicted_high_soc_share=float(movable[high].sum()/max(amount, 1e-9)),
                forecast_recovery_confidence=confidence,
                forecast_expected_uncovered_kw=float(np.maximum(0., REQUEST-np.minimum(room, amount)).mean()),
                high_soc_recovery_energy_share=float(total_recovery[high].sum()/max(total_recovery.sum(), 1e-9)))


def state_frame(entries, threshold, cuts):
    rows = []
    for row, case, forecast in entries:
        if row["event"]:
            continue
        features = forecast_features(case, forecast, threshold)
        bucket = state_bin(features["predicted_room_kw"], features["predicted_high_soc_share"], cuts)
        rows.append({k: row[k] for k in ["parent", *STATE]} | features | dict(bucket=bucket))
    df = pd.DataFrame(rows)
    assert not df.duplicated(["parent", *STATE]).any()
    return df


def split_metrics(rows):
    high = rows[rows.bucket.isin([1, 3])]
    assert not high.empty
    return {panel: {k: float(v) for k, v in summarize(part).items()}
            for panel, part in (("overall", rows), ("high_exposure", high))}


def prepare(dev, states, output, previous, provenance):
    anchor = dev[(dev.risk == BASELINE[0]) & (dev.reserve_kw == BASELINE[1])]
    summaries, rule = {}, {}
    historical = json.loads((previous/"configuration_lock.json").read_text(encoding="utf-8"))["adaptive"]
    for bucket, state in states.groupby("bucket", sort=True):
        rows = anchor[anchor.bucket == bucket]
        base = {k: float(v) for k, v in summarize(rows).items() if k != "parents"}
        extra = {k: float(state[k].mean()) for k in (
            "predicted_room_kw", "predicted_high_soc_share", "forecast_recovery_confidence",
            "forecast_expected_uncovered_kw", "high_soc_recovery_energy_share")}
        key = f"context_{int(bucket)+1}"
        summaries[key] = dict(original=base, extended=base | extra)
        point = tuple(historical[str(bucket)])
        # Freeze a conservative guard on the prior validated forecast-state rule.
        if bucket >= 2 and (extra["forecast_recovery_confidence"] < .75
                            or extra["high_soc_recovery_energy_share"] > .65):
            point = (.9, min(8., point[1]))
        rule[str(bucket)] = point
    assert len(summaries) == 4
    dump(output/"api_states.json", summaries)
    dump(output/"protocol.json", dict(**provenance, rule=rule, budget=CHECKPOINTS, default=BASELINE,
        points=POINTS, cvar_increment_limit_kwh=LIMIT, seed=SEED,
        high_exposure="17:00 predicted transferable-power high-SOC share above frozen development cutoff",
        state_assignment="same four forecast-state bins for all arms; one point per bin",
        confidence="fraction of four forecast paths with enough grid and same-vehicle headroom for 2 kW",
        risk_metric="CVaR95 of unmet departure kWh, overall and high-exposure separately, relative to fixed 0.9/6 kW",
        selection="feasibility under both CVaR caps first; then overall qualification, demand served, SOH; if none feasible, smallest violation",
        search="identical seeded discrete coordinate local search; only six policy scores exposed per arm",
        rule_basis="previous locked forecast-state policy with confidence below .75 or high-SOC recovery energy above .65 triggering conservative 0.9 risk and at most 8 kW in high-room bins",
        api_data="development aggregates, default weights and bounds only"))


def chosen_rows(table, policy):
    rule = pd.DataFrame([(b, *p) for b, p in enumerate(policy)], columns=["bucket", *KEYS])
    return table.merge(rule, on=["bucket", *KEYS])


def local_search(dev, initial, baseline, points=POINTS):
    rng = np.random.default_rng(SEED)
    records, seen = [], set()
    initial = tuple(tuple(p) for p in initial)
    for step in range(1, 7):
        if step == 1:
            policy = initial
        else:
            incumbent = min(records, key=lambda r: r["rank"])["policy"]
            pool = sorted({incumbent[:b]+(p,)+incumbent[b+1:] for b in range(4) for p in points}-seen)
            assert pool
            policy = pool[int(rng.integers(len(pool)))]
        assert policy not in seen and all(p in points for p in policy)
        seen.add(policy)
        m = split_metrics(chosen_rows(dev, policy))
        violation = max(max(0., m[k]["cvar_unmet_kwh"]-baseline[k]["cvar_unmet_kwh"]-LIMIT)
                        for k in m)
        overall = m["overall"]
        rank = (round(violation, 9), -round(overall["qualified_rate"], 10),
                -round(overall["served_ratio"], 10), round(overall["mean_soh_loss"], 10))
        records.append(dict(step=step, policy=policy, rank=rank, metrics=m))
    return records


def run(source, factorial, fresh, prior_stress, previous, old_holdout, output, stage):
    output.mkdir(parents=True, exist_ok=True)
    old_parents = set(pd.read_csv(old_holdout/"locked_manifest.csv").parent_fingerprint)
    fresh_parents = set(pd.read_csv(fresh/"locked_manifest.csv").parent_fingerprint)
    assert len(fresh_parents) == 20 and not (old_parents & fresh_parents)
    scenario_output = output/"scenario_design"
    if stage == "evaluate":
        scenario_output = output/"reconstructed_design"
    aging, design, entries = construct(source, factorial, fresh, prior_stress, scenario_output, audit_physical=False)
    dev_entries = [e for e in entries if e[0]["split"] == "development"]
    hold_entries = [e for e in entries if e[0]["split"] == "holdout"]
    assert len(dev_entries) == 240 and len(hold_entries) == 320
    previous_lock = json.loads((previous/"configuration_lock.json").read_text(encoding="utf-8"))
    cuts = previous_lock["forecast_room_cut_kw"], previous_lock["forecast_high_soc_share_cut"]
    dev_cache = pd.read_csv(previous/"development_replays.csv")
    assert digest(previous/"development_replays.csv") == previous_lock["development_replay_sha256"]
    prior_design = pd.read_csv(previous/"manifest.csv")
    old_dev = prior_design[prior_design.split == "development"]
    new_dev = design[design.split == "development"]
    pd.testing.assert_frame_equal(old_dev.reset_index(drop=True), new_dev.reset_index(drop=True),
                                  check_exact=False, atol=1e-8, rtol=0.)
    provenance = dict(fresh_lock_sha256=digest(fresh/"lock.json"), pinn_sha256=digest(aging.path),
                      development_cache_sha256=digest(previous/"development_replays.csv"),
                      forecast_cuts=cuts)
    if stage == "prepare":
        states = state_frame(dev_entries, aging.model.config.soc_threshold, cuts)
        states.to_csv(output/"development_states.csv", index=False)
        audit = dev_cache.drop_duplicates(["parent", *STATE]).merge(states, on=["parent", *STATE], suffixes=("_cached", ""))
        assert len(audit) == 120 and (audit.bucket_cached == audit.bucket).all()
        prepare(dev_cache, states, output, previous, provenance)
        print("Frozen protocol and prepared only development aggregates.", flush=True)
        return

    protocol = json.loads((output/"protocol.json").read_text(encoding="utf-8"))
    assert all(protocol[k] == v for k, v in provenance.items() if k != "forecast_cuts")
    llm = json.loads((output/"api_seeds.json").read_text(encoding="utf-8"))
    starts = {arm: [llm[arm][f"context_{b+1}"]["point"] for b in range(4)] for arm in ARMS[:2]}
    starts["adaptive_rule"] = [protocol["rule"][str(b)] for b in range(4)]
    baseline = split_metrics(chosen_rows(dev_cache, [BASELINE]*4))
    chosen, searches = {}, []
    for arm, start in starts.items():
        records = local_search(dev_cache, start, baseline)
        searches.extend(dict(arm=arm, **r) for r in records)
        for budget in CHECKPOINTS:
            chosen[arm, budget] = min(records[:budget], key=lambda r: r["rank"])
    dump(output/"development_search.json", searches)
    dump(output/"configuration_lock.json", dict(
        selected=[dict(arm=arm, budget=budget, **r) for (arm, budget), r in chosen.items()],
        fresh_lock_sha256=provenance["fresh_lock_sha256"], dev_baseline=baseline))
    print("Locked all checkpoints before scoring the fresh holdout.", flush=True)

    states = state_frame(hold_entries, aging.model.config.soc_threshold, cuts)
    states.to_csv(output/"holdout_states.csv", index=False)
    state_map = {(r.parent, *(getattr(r, k) for k in STATE)):
                 (r.predicted_room_kw, r.predicted_high_soc_share, r.bucket) for r in states.itertuples()}
    need = {b: sorted({BASELINE, *(r["policy"][b] for r in chosen.values())}) for b in range(4)}
    held = replay(hold_entries, state_map, aging, lambda b: need[b], "Fresh holdout")
    held.to_csv(output/"holdout_candidate_replays.csv", index=False)
    held_base = split_metrics(chosen_rows(held, [BASELINE]*4))
    report = []
    for split, table, reference in (("development", dev_cache, baseline), ("holdout", held, held_base)):
        for panel, m in reference.items():
            report.append(dict(split=split, arm="fixed_reference", budget=0, selected_step=0,
                               panel=panel, **m, cvar_delta_kwh=0., cvar_ok=True))
        for (arm, budget), r in chosen.items():
            rows = chosen_rows(table, r["policy"])
            assert len(rows) == (240 if split == "development" else 320)
            for panel, m in split_metrics(rows).items():
                delta = m["cvar_unmet_kwh"]-reference[panel]["cvar_unmet_kwh"]
                report.append(dict(split=split, arm=arm, budget=budget, selected_step=r["step"], panel=panel,
                                   **m, cvar_delta_kwh=delta, cvar_ok=delta <= LIMIT+1e-7))
    pd.DataFrame(report).to_csv(output/"summary.csv", index=False)
    dump(output/"completion.json", dict(holdout_parents=len(fresh_parents), replays=len(held),
        lock_sha256=digest(output/"configuration_lock.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "factorial", "fresh", "prior_stress", "previous", "old_holdout", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("stage", choices=("prepare", "evaluate"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(*(v.resolve() if isinstance(v, Path) else v for v in vars(args).values()))
