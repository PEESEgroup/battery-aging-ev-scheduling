"""Matched proposal rerun using only the seven archived evening-peak development cases."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_gemini_feedback_pair import (
    Aging, CHECKPOINTS, PROBES, TimingForecast, digest, dump, fingerprint, margins,
    metrics, peak_exploration_ok, propose_bo, propose_chatgpt, rank, read_case, replay,
)


def run(source, prior, output, locked_development=None):
    output.mkdir(parents=True, exist_ok=True)
    group = "evening_peak_cluster"
    protocol = json.loads((prior / "protocol.json").read_text())
    assert digest(prior / "manifest.csv") == protocol["manifest_sha256"]
    cases = {}
    for row in pd.read_csv(prior / "manifest.csv").itertuples():
        if row.group != group or row.split != "development":
            continue
        path = source / f"optimization_inputs_{group}" / Path(row.source).name
        assert digest(path / "optimizer_ev_sessions.csv") == row.ev_sha256
        assert digest(path / "optimizer_grid.csv") == row.grid_sha256
        fleet, grid, temp = read_case(path)
        assert fingerprint(fleet) == row.parent and row.parent not in cases
        grid[:, 3] = np.maximum(grid[:, 3], 1500.)
        cases[row.parent] = fleet, grid, temp, None
    assert len(cases) == 7
    forecast = TimingForecast([case[1] for case in cases.values()])
    lock_sha256 = None
    if locked_development is not None:
        lock = json.loads((locked_development / "scenario_lock.json").read_text())
        manifest = locked_development / "locked_manifest.csv"
        lock_sha256 = digest(manifest)
        assert lock_sha256 == lock["sha256"]
        assert digest(locked_development / "screen_protocol.json") == lock["protocol_sha256"]
        original = set(pd.read_csv(prior / "manifest.csv").parent)
        cases = {}
        for row in pd.read_csv(manifest).itertuples():
            path = locked_development / row.scenario
            assert digest(path / "optimizer_ev_sessions.csv") == row.ev_sha256
            assert digest(path / "optimizer_grid.csv") == row.grid_sha256
            fleet, grid, temp = read_case(path)
            assert fingerprint(fleet) == row.parent and row.parent not in cases and row.parent not in original
            grid[:, 3] = np.maximum(grid[:, 3], 1500.)
            cases[row.parent] = fleet, grid, temp, None
        assert len(cases) == lock["parents"]
    aging = Aging(source)
    assert digest(aging.path) == protocol["pinn_sha256"]
    archived = json.loads((prior / "shared_probes.json").read_text())["groups"][group]
    rows = {}
    shared = []
    for i, point in enumerate(PROBES):
        rows[point] = replay(cases, forecast, aging, point)
        observed = metrics(rows[point])
        if locked_development is None:
            assert all(np.isclose(v, archived[i]["metrics"][k], atol=1e-7, rtol=0)
                       for k, v in observed.items()), (observed, archived[i]["metrics"])
            observed = archived[i]["metrics"]
        shared.append(dict(step=i + 1, point=point, metrics=observed))
    reference = dict(shared[0]["metrics"], qualified=protocol["development_delivery_floors"][group])
    dump(output / "protocol.json", dict(prior=protocol, group=group, parents=len(cases),
         step_3_minimum_change=dict(w_price=.18, unmet_penalty_per_kwh=.95),
         locked_development_sha256=lock_sha256, reference=reference))
    histories = {}
    for arm in ("bo", "chatgpt"):
        history = list(shared)
        rng = np.random.default_rng(204)
        while len(history) < 6:
            explore = len(history) == 2
            if arm == "bo":
                point = propose_bo(history, reference, rng, enforce_delivery=True, explore=explore)
                details = {}
            else:
                point, details = propose_chatgpt(history, reference, enforce_delivery=True,
                                                 bo_delivery_reference=True, explore=explore)
            assert not explore or peak_exploration_ok(point)
            rows[point] = replay(cases, forecast, aging, point)
            history.append(dict(step=len(history) + 1, point=point, metrics=metrics(rows[point]), **details))
            dump(output / f"{arm}_development.json", history)
            print(arm, history[-1]["step"], point, history[-1]["metrics"], flush=True)
        histories[arm] = history
    paired, summary = [], []
    for budget in CHECKPOINTS:
        selected = {arm: min(history[:budget], key=lambda r: rank(r, reference, True))
                    for arm, history in histories.items()}
        differences, joint = [], []
        for bo, gpt in zip(*(rows[tuple(selected[arm]["point"])] for arm in ("bo", "chatgpt")), strict=True):
            assert bo["parent"] == gpt["parent"]
            delta = 100 * (gpt["mean_soh_loss"] - bo["mean_soh_loss"])
            differences.append(delta)
            if bo["qualified"] and gpt["qualified"]:
                joint.append(delta)
            paired.append(dict(budget=budget, parent=bo["parent"], bo_qualified=bo["qualified"],
                               gpt_qualified=gpt["qualified"], bo_soh_loss=bo["mean_soh_loss"],
                               gpt_soh_loss=gpt["mean_soh_loss"], gpt_minus_bo_soh_pp=delta,
                               bo_service=bo["served_ratio"], gpt_service=gpt["served_ratio"],
                               bo_unmet_kwh=bo["unmet_kwh"], gpt_unmet_kwh=gpt["unmet_kwh"]))
        summary.append(dict(budget=budget, selected=selected,
            guardrails={arm: margins(r["metrics"], reference, True).tolist() for arm, r in selected.items()},
            same_budget_delivery_ok=selected["chatgpt"]["metrics"]["qualified"] >= selected["bo"]["metrics"]["qualified"],
            mean_paired_soh_pp=float(np.mean(differences)), joint_count=len(joint),
            joint_mean_paired_soh_pp=float(np.mean(joint)) if joint else None))
    pd.DataFrame(paired).to_csv(output / "paired_development.csv", index=False)
    dump(output / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "prior", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--locked-development", type=Path)
    run(**vars(parser.parse_args()))
