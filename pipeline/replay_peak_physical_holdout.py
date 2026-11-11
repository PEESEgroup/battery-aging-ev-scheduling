"""Replay the frozen evening-peak BO and GPT-6 Sol selections without retuning."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_gemini_feedback_pair import (
    Aging, CHECKPOINTS, PROBES, TimingForecast, digest, dump, fingerprint,
    margins, metrics, read_case, replay,
)


def read_verified_case(path, row):
    assert digest(path / "optimizer_ev_sessions.csv") == row.ev_sha256
    assert digest(path / "optimizer_grid.csv") == row.grid_sha256
    fleet, grid, temp = read_case(path)
    assert fingerprint(fleet) == row.parent
    grid[:, 3] = np.maximum(grid[:, 3], 1500.)
    return fleet, grid, temp, None


def paired_soh(frame):
    d = frame.gpt_minus_bo_soh_pp.to_numpy()
    if not len(d):
        return dict(n=0, mean_pp=None)
    rng = np.random.default_rng(943501)
    bootstrap = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    return dict(n=len(d), mean_pp=float(d.mean()),
                bootstrap_95_pp=np.quantile(bootstrap, [.025, .975]).tolist(),
                lower=int((d < -1e-8).sum()), higher=int((d > 1e-8).sum()),
                tied=int((np.abs(d) <= 1e-8).sum()))


def run(source, prior, development, selection, holdout):
    lock = json.loads((holdout / "scenario_lock.json").read_text())
    screen = holdout / "screen_protocol.json"
    manifest = holdout / "locked_manifest.csv"
    assert digest(screen) == lock["protocol_sha256"] and digest(manifest) == lock["sha256"]
    screen_protocol = json.loads(screen.read_text())
    assert screen_protocol["split"] == "holdout"
    assert digest(selection) == screen_protocol["frozen_selection_sha256"]
    selected = {r["budget"]: r["selected"] for r in json.loads(selection.read_text())}
    assert set(selected) == set(CHECKPOINTS)
    protocol = json.loads((prior / "protocol.json").read_text())
    assert digest(prior / "manifest.csv") == protocol["manifest_sha256"]
    archive = pd.read_csv(prior / "manifest.csv")
    dev_manifest = development / "locked_manifest.csv"
    dev_lock = json.loads((development / "scenario_lock.json").read_text())
    assert digest(dev_manifest) == dev_lock["sha256"]
    excluded = set(archive.parent) | set(pd.read_csv(dev_manifest).parent)
    donors = [read_verified_case(source / "optimization_inputs_evening_peak_cluster" / Path(row.source).name, row)[1]
              for row in archive.itertuples() if row.group == "evening_peak_cluster" and row.split == "development"]
    assert len(donors) == 7
    forecast = TimingForecast(donors)
    cases = {}
    for row in pd.read_csv(manifest).itertuples():
        assert row.parent not in excluded and row.parent not in cases
        cases[row.parent] = read_verified_case(holdout / row.scenario, row)
    assert len(cases) == lock["parents"]
    aging = Aging(source)
    assert digest(aging.path) == protocol["pinn_sha256"]
    cache = {}
    def evaluate(point):
        point = tuple(point)
        if point not in cache:
            cache[point] = replay(cases, forecast, aging, point)
        return cache[point]

    reference = metrics(evaluate(PROBES[0]))
    results, frames = [], []
    for budget in CHECKPOINTS:
        arm = selected[budget]
        bo, gpt = (pd.DataFrame(evaluate(arm[name]["point"])) for name in ("bo", "chatgpt"))
        a, b = metrics(bo), metrics(gpt)
        compare = b["qualified"] >= a["qualified"]
        within = {key: (margins(m, reference) >= -1e-7).tolist()
                  for key, m in (("bo", a), ("chatgpt", b))}
        df = bo.merge(gpt, on="parent", suffixes=("_bo", "_gpt"), validate="one_to_one")
        df["budget"] = budget
        df["gpt_minus_bo_soh_pp"] = 100 * (df.mean_soh_loss_gpt - df.mean_soh_loss_bo)
        both = df.qualified_bo & df.qualified_gpt
        outcomes = dict(both=int(both.sum()), only_bo=int((df.qualified_bo & ~df.qualified_gpt).sum()),
                        only_gpt=int((~df.qualified_bo & df.qualified_gpt).sum()),
                        neither=int((~df.qualified_bo & ~df.qualified_gpt).sum()))
        result = dict(budget=budget, configurations={name: arm[name]["point"] for name in arm},
                      bo=a, chatgpt=b, same_budget_delivery_ok=compare,
                      service_cvar_within_prior_tolerances=within, outcomes=outcomes,
                      all_soh=paired_soh(df), joint_soh=paired_soh(df[both]))
        results.append(result)
        frames.append(df)
        print(json.dumps(result), flush=True)
    pd.concat(frames, ignore_index=True).to_csv(holdout / "paired_holdout.csv", index=False)
    dump(holdout / "replay_summary.json", dict(reference_first_shared_probe=reference,
         source_protocol_sha256=digest(prior/"protocol.json"), pinned_selection_sha256=digest(selection),
         holdout_manifest_sha256=digest(manifest), results=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "prior", "development", "selection", "holdout"):
        parser.add_argument(name, type=Path)
    torch.set_num_threads(1)
    run(**vars(parser.parse_args()))
