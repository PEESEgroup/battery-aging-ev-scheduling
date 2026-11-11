"""Fixed-configuration sensitivity to an announced 18:00 charging cap."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from replay_peak_physical_holdout import read_verified_case
from run_directional_pinn_validation import RECOVERY
from run_gemini_feedback_pair import Aging, TimingForecast, digest, replay


def run(source, root, derating):
    prior = root / "bo_parity_feedback_pair"
    held = root / "peak_physical_confirmatory_holdout"
    summary = json.loads((held / "replay_summary.json").read_text())
    lock = json.loads((held / "scenario_lock.json").read_text())
    assert digest(held / "locked_manifest.csv") == lock["sha256"]
    assert digest(held / "screen_protocol.json") == lock["protocol_sha256"]
    protocol = json.loads((prior / "protocol.json").read_text())
    assert digest(prior / "manifest.csv") == protocol["manifest_sha256"]
    archive = pd.read_csv(prior / "manifest.csv")
    donors = [read_verified_case(source / "optimization_inputs_evening_peak_cluster" / Path(row.source).name, row)[1]
              for row in archive.itertuples() if row.group == "evening_peak_cluster" and row.split == "development"]
    forecast = TimingForecast(donors)
    aging = Aging(source)
    assert digest(aging.path) == protocol["pinn_sha256"]
    cases = {}
    for row in pd.read_csv(held / "locked_manifest.csv").itertuples():
        fleet, grid, temp, _ = read_verified_case(held / row.scenario, row)
        active = (fleet.arrival <= RECOVERY) & (fleet.departure > RECOVERY)
        grid[RECOVERY, 3] = min(grid[RECOVERY, 3], (1-derating/100)*fleet.power[active].sum())
        cases[row.parent] = fleet, grid, temp, RECOVERY
    assert len(cases) == 120
    cache, frames = {}, []
    for item in summary["results"]:
        out = {}
        for arm, values in item["configurations"].items():
            point = tuple(values)
            if point not in cache:
                cache[point] = pd.DataFrame(replay(cases, forecast, aging, point))
            out[arm] = cache[point]
        paired = out["bo"].merge(out["chatgpt"], on="parent", suffixes=("_bo", "_gpt"), validate="one_to_one")
        paired["budget"] = item["budget"]
        paired["derating"] = derating
        paired["gpt_minus_bo_soh_pp"] = 100*(paired.mean_soh_loss_gpt-paired.mean_soh_loss_bo)
        frames.append(paired)
    dest = root / "peak_recovery_curve"
    dest.mkdir(exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(dest / f"derating_{derating}.csv", index=False)
    print(f"{derating}%: complete", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("root", type=Path)
    p.add_argument("derating", type=int, choices=[20, 40, 60, 80])
    torch.set_num_threads(1)
    run(**vars(p.parse_args()))
