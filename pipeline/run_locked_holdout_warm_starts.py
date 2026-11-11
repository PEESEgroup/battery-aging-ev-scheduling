"""Compare three warm starts on the frozen recovery/departure holdout."""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

from lock_recovery_holdout import digest
from run_directional_pinn_validation import Aging, TOL, read_case
from run_llm_timing_pilot import (
    BOUNDS, BUDGET, CHECKPOINTS, DEFAULTS, RECOVERY, REQUEST, SEED, TimingForecast,
    bounded, cvar, dump, evaluate, make_state, score, summarize, timing_prior,
)
from screen_recovery_departure import parent_cases


AGGREGATES = {
    "cvar_unmet_demand": "tail_unmet_kwh", "cost_per_served_kwh": "price_per_kwh",
    "carbon_per_served_kwh": "carbon_per_kwh", "degradation_per_served_kwh": "loss_per_kwh",
    "peak_load": "peak_kw",
}


def development_inputs(source, factorial, floor):
    original = {key: (case, split) for _, _, key, _, case, split in parent_cases(source)}
    design = pd.read_csv(factorial / "manifest.csv")
    dev = design[design.split == "development"]
    assert dev.parent_sha256.nunique() == 21 and len(dev) == 42
    donors = []
    for key in dev.parent_sha256.unique():
        case, split = original[key]
        assert split == "development"
        donor = case[1].copy()
        donor[:, 3] = np.maximum(donor[:, 3], floor)
        donors.append(donor)
    datasets = {}
    for row in dev.itertuples():
        (fleet, grid, temp), _ = original[row.parent_sha256]
        grid = grid.copy()
        grid[:, 3] = np.maximum(grid[:, 3], floor)
        grid[RECOVERY, 3] = row.cap_kw
        datasets.setdefault(row.cell, {})[row.parent_sha256] = fleet, grid, temp, RECOVERY
    return dev, datasets, TimingForecast(donors)


def prepare(output, dev, datasets, forecast, aging, core, provenance):
    aggregates = {}
    for cell, cases in datasets.items():
        rows = evaluate(cases, forecast, aging, DEFAULTS)
        aggregates[cell] = summarize(rows)
        aggregates[cell]["tail_unmet_kwh"] = float(cvar(np.array([r["unmet_kwh"] for r in rows]), .95))
    normal = aggregates["ample_low"]
    prep = {}
    for cell, cases in datasets.items():
        baseline = aggregates[cell]
        state_metrics = {k: baseline[v] for k, v in AGGREGATES.items()}
        state_metrics.update({k+"_rel": baseline[v]/max(normal[v], 1e-12) for k, v in AGGREGATES.items()})
        aggregate = core.StateSummary(scenario_name="joint", representative_lambda=None, stress_type="normal",
                  objective_priority="service", primary_bottleneck="unclassified", service_ratio=baseline["served_ratio"],
                  **state_metrics)
        original_full = core.rule_based_prior(aggregate)
        original = bounded({k: original_full[k] if k != "recovery_reserve_kw" else 0. for k in BOUNDS})
        enhanced = make_state(cell, cases, baseline)
        enhanced.update({k: round(v, 8) for k, v in state_metrics.items()})
        cell_dev = dev[dev.cell == cell]
        enhanced.update({
            "mean_recovery_grid_headroom_kw": float(cell_dev.default_grid_room_kw.mean()),
            "mean_urgent_remaining_energy_share": float(cell_dev.urgent_share.mean()),
            "mean_urgent_overlap_remaining_energy_share": float(cell_dev.urgent_overlap_share.mean()),
            "mean_default_two_slot_physical_bound_kw": float(cell_dev.default_physical_kw.mean()),
        })
        prep[cell] = dict(enhanced_state=enhanced, original_rule=original,
                          timing_rule=timing_prior(original, enhanced))
    dump(output / "states.json", prep)
    dump(output / "protocol.json", dict(
        **provenance, defaults=DEFAULTS, parameters=BOUNDS, budget=BUDGET, checkpoints=CHECKPOINTS, seed=SEED,
        main="critical_high and realized default recovery room within tolerance of 2.25 kW; 31 parents",
        sensitivity="all 112 scenarios; two capacity variants per parent; equal case weighting",
        score="-10*qualified + (1-served_ratio) + mean_soh_loss",
        reference_for_aggregate_ratios="ample_low development group",
        timing_rule="frozen timing pilot rule; same enhanced state object supplied to GPT",
        search="same archived local BO, seed and budget; duplicate proposals are redrawn without evaluation",
        api="only enhanced development aggregates, default weights and parameter bounds",
    ))
    print("Prepared four development-only states.", flush=True)


def search(start, cases, forecast, aging, core):
    rng = np.random.default_rng(SEED)
    records, seen = [], set()
    for step in range(1, BUDGET+1):
        for _ in range(100):
            candidate = bounded(start if step == 1 else
                                core._suggest_llm_feedback_local_bo_candidate(records, start, rng))
            key = tuple(candidate.values())
            if key not in seen:
                break
        else:
            raise RuntimeError("Local search did not propose a distinct point")
        seen.add(key)
        metrics = summarize(evaluate(cases, forecast, aging, candidate))
        records.append(dict(step=step, actual_score=score(metrics), **metrics,
                            **{f"param_{k}": v for k, v in candidate.items()}))
    return records


def summarize_pairs(heldout, output):
    panels = {"primary": heldout[heldout.primary], "full": heldout}
    panels.update({cell: rows for cell, rows in heldout.groupby("cell")})
    summaries, contrasts = [], []
    for panel, rows in panels.items():
        for (arm, budget), part in rows.groupby(["arm", "budget"]):
            summaries.append(dict(panel=panel, arm=arm, budget=budget, parents=part.parent.nunique(),
                cases=len(part), qualified=int(part.qualified.sum()), qualified_rate=part.qualified.mean(),
                served_ratio=part.served_ratio.mean(), mean_soh_loss=part.mean_soh_loss.mean()))
        for budget, part in rows.groupby("budget"):
            key = ["cell", "parent"]
            by_arm = {arm: p.set_index(key).sort_index() for arm, p in part.groupby("arm")}
            llm = by_arm["llm"]
            for baseline in ("original_rule", "timing_rule"):
                other = by_arm[baseline]
                assert llm.index.equals(other.index)
                wins = int((llm.qualified & ~other.qualified).sum())
                losses = int((~llm.qualified & other.qualified).sum())
                p = (binomtest(wins, wins+losses).pvalue if wins+losses else 1.) if panel == "primary" else None
                contrasts.append(dict(panel=panel, budget=budget, baseline=baseline, llm_only=wins,
                    baseline_only=losses, qualified_diff=(wins-losses)/len(llm),
                    served_diff=(llm.served_ratio-other.served_ratio).mean(),
                    soh_loss_diff=(llm.mean_soh_loss-other.mean_soh_loss).mean(), primary_exact_p=p))
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(output / "paired_contrasts.csv", index=False)


def run(source, factorial, locked, output, stage):
    output.mkdir(parents=True, exist_ok=True)
    lock = json.loads((locked / "lock.json").read_text(encoding="utf-8"))
    assert lock["quotas_met"] and lock["manifest_sha256"] == digest(locked / "locked_manifest.csv")
    prior = json.loads((factorial / "protocol.json").read_text(encoding="utf-8"))
    dev, datasets, forecast = development_inputs(source, factorial, prior["capacity_background_floor_kw"])
    aging = Aging(source)
    core = importlib.import_module("llm_top_config_optimization_actual_eval_v7_ablation_layout")
    provenance = dict(holdout_manifest_sha256=lock["manifest_sha256"],
                      development_manifest_sha256=digest(factorial / "manifest.csv"), pinn_sha256=digest(aging.path))
    if stage == "prepare":
        prepare(output, dev, datasets, forecast, aging, core, provenance)
        return
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert all(protocol[k] == v for k, v in provenance.items())
    prep = json.loads((output / "states.json").read_text(encoding="utf-8"))
    llm = json.loads((output / "llm_seeds.json").read_text(encoding="utf-8"))
    assert set(llm) == set(datasets) == set(prep)
    manifest = pd.read_csv(locked / "locked_manifest.csv")
    manifest["primary"] = manifest.cell.eq("critical_high") & (
        manifest.grid_room_kw - prior["target_room_kw"]["critical"]).abs().le(TOL)
    assert len(manifest) == 112 and manifest.parent_fingerprint.nunique() == 56 and manifest.primary.sum() == 31
    assert not manifest.duplicated(["cell", "parent_fingerprint"]).any()
    holdout = {}
    primary = {}
    for row in manifest.itertuples():
        path = locked / row.path
        assert digest(path / "optimizer_ev_sessions.csv") == row.ev_sha256
        assert digest(path / "optimizer_grid.csv") == row.grid_sha256
        fleet, grid, temp = read_case(path)
        assert abs(grid[RECOVERY, 3] - row.cap_kw) <= TOL
        holdout.setdefault(row.cell, {})[row.parent_fingerprint] = fleet, grid, temp, RECOVERY
        primary[row.cell, row.parent_fingerprint] = row.primary

    core.PARAM_COLS = list(BOUNDS)
    core.PARAM_BOUNDS = BOUNDS
    core.BASELINE_PARAMS = {**core.BASELINE_PARAMS, **DEFAULTS}
    dev_rows, test_rows = [], []
    for cell in sorted(datasets):
        starts = {k: prep[cell][k] for k in ("original_rule", "timing_rule")}
        starts["llm"] = llm[cell]["parameters"]
        for arm, start in starts.items():
            records = search(start, datasets[cell], forecast, aging, core)
            dev_rows.extend(dict(cell=cell, arm=arm, **r) for r in records)
            cached = {}
            for budget in CHECKPOINTS:
                best = min(records[:budget], key=lambda r: r["actual_score"])
                step = best["step"]
                if step not in cached:
                    params = {k: best[f"param_{k}"] for k in BOUNDS}
                    cached[step] = evaluate(holdout[cell], forecast, aging, params)
                for row in cached[step]:
                    test_rows.append(dict(cell=cell, arm=arm, budget=budget, selected_step=step,
                                          primary=primary[cell, row["parent"]], **row))
            print(f"{cell} {arm} complete", flush=True)
        pd.DataFrame(dev_rows).to_csv(output / "development.csv", index=False)
        pd.DataFrame(test_rows).to_csv(output / "heldout.csv", index=False)
    heldout = pd.DataFrame(test_rows)
    assert len(heldout) == len(manifest)*3*len(CHECKPOINTS)
    assert not heldout.duplicated(["cell", "parent", "arm", "budget"]).any()
    summarize_pairs(heldout, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "factorial", "locked", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("stage", choices=("prepare", "evaluate"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.source.resolve(), args.factorial.resolve(), args.locked.resolve(), args.output.resolve(), args.stage)
