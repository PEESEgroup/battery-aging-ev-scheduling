"""Retrain the local generators and evaluate chronological holdouts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
import torch

from train_ev_generator_f import EVGenerator, EVGeneratorConfig
from train_grid_generator_f import GridGenerator, GridGeneratorConfig
from world_model_common import sample_with_diffusion, set_seed

ROOT = Path(__file__).resolve().parent.parent
MODES = ("CVAE", "CVAE + diffusion")
FIELDS = ["system_load_mw", "weighted_lbmp_mwh", "weighted_imer_co2_ton_per_mwh",
          "congestion_intensity"]


def draw(model, c, r, mode, strength):
    c = torch.as_tensor(model.c_norm.transform(c), dtype=torch.float32)
    r = torch.as_tensor(model.r_norm.transform(r), dtype=torch.float32)
    with torch.no_grad():
        x = model.cvae.sample(c)
        if mode == MODES[1]:
            x = sample_with_diffusion(model.denoiser, model.scheduler, x, c, r, strength)
    return model.x_norm.inverse_transform(x.numpy())


def marginal(actual, generated):
    a, g = np.asarray(actual), np.asarray(generated)
    spread = np.quantile(a, .75) - np.quantile(a, .25)
    q95 = np.quantile(a, .95)
    return dict(n_real=len(a), n_generated=len(g), wasserstein=wasserstein_distance(a, g),
                wasserstein_over_real_iqr=wasserstein_distance(a, g) / spread if spread else np.nan,
                ks=ks_2samp(a, g).statistic, real_q95=q95, generated_q95=np.quantile(g, .95))


def split_index(n):
    return int(.8*n), int(.9*n)


def prepare_grid():
    # Each archived isolf file includes forecasts for more than its named date.
    # Retain the date-matched NYISO series and rebuild lags on unique clock hours.
    loads = []
    for path in sorted((ROOT / "Load").rglob("2025*isolf.csv")):
        day = pd.to_datetime(path.name[:8], format="%Y%m%d")
        part = pd.read_csv(path, usecols=["Time Stamp", "NYISO"])
        part["timestamp"] = pd.to_datetime(part["Time Stamp"], errors="coerce")
        loads.append(part.loc[part.timestamp.dt.normalize().eq(day), ["timestamp", "NYISO"]])
    load = pd.concat(loads).drop_duplicates()
    load = load.loc[~load.timestamp.duplicated(keep=False)].rename(columns={"NYISO": FIELDS[0]})
    market = pd.read_csv(ROOT / "processed_grid/grid_generator_ready.csv")
    keys = ["timestamp", *FIELDS[1:], "weighted_loss_mwh", "weighted_congestion_mwh"]
    market = market[keys].drop_duplicates()
    assert market.timestamp.is_unique
    market["timestamp"] = pd.to_datetime(market.timestamp)
    frame = market.merge(load, on="timestamp", validate="one_to_one").sort_values("timestamp")
    frame = frame.set_index("timestamp").asfreq("h")
    for key in FIELDS:
        for lag in (1, 2, 24):
            frame[f"{key}_lag{lag}"] = frame[key].shift(lag)
        frame[f"{key}_next1"] = frame[key].shift(-1)
    frame["hour"], frame["day_of_week"], frame["month"] = (
        frame.index.hour, frame.index.dayofweek, frame.index.month)
    frame["is_weekend"] = (frame.index.dayofweek >= 5).astype(int)
    for prefix, key, period in (("hour","hour",24),("month","month",12),("dow","day_of_week",7)):
        frame[prefix+"_sin"] = np.sin(2*np.pi*frame[key]/period)
        frame[prefix+"_cos"] = np.cos(2*np.pi*frame[key]/period)
    return frame.reset_index()


def ev_check(seed, out, stratified=False):
    set_seed(seed)
    model = EVGenerator(EVGeneratorConfig(
        sessions_path=ROOT / "EV/processed/ev_sessions_unified.csv",
        save_path=out / f"ev_retrained_{seed}.pkl", random_seed=seed))
    data = model.load_data()
    days = sorted(data.date.unique())
    a, b = split_index(len(days))
    boundaries = {}
    if stratified:
        parts = []
        for source, part in data.groupby("source"):
            source_days = sorted(part.date.unique())
            left, right = split_index(len(source_days))
            parts.append((part[part.date < source_days[left]], part[part.date >= source_days[right]]))
            boundaries[source] = {"first_validation_day": source_days[left],
                                  "first_test_day": source_days[right]}
        train = pd.concat([pair[0] for pair in parts])
        test = pd.concat([pair[1] for pair in parts])
    else:
        train = data[data.date < days[a]].copy()
        test = data[data.date >= days[b]].copy()
    test = test[test.source.isin(train.source.unique())]
    model.fit_count_model(train)
    model.fit_world_model(train)
    model.save()
    stats = []
    trajectories = []
    for mode in MODES:
        set_seed(seed + 100)
        model.rng = np.random.default_rng(seed + 100)
        for source, part in test.groupby("source"):
            x, c, r = model._build_training_arrays(part)
            predicted = draw(model, c, r, mode, .6)
            for i, feature in enumerate(model.feature_cols):
                stats.append(dict(seed=seed, mode=mode, scope="session_given_observed_context",
                                  source=source, variable=feature, **marginal(x[:, i], predicted[:, i])))
            dates = sorted(part.date.unique())
            idx = np.linspace(0, len(dates)-1, min(12, len(dates)), dtype=int)
            for day in [dates[i] for i in idx]:
                actual = part[part.date == day]
                simulated = model.sample_sessions_for_horizon(
                    day, hours=24, source=source, use_tail_enhancement=(mode == MODES[1]))
                def features(frame):
                    if frame.empty:
                        return 0, 0.0, np.nan
                    e = frame.energy_delivered_kwh
                    eve = frame.arrival_hour.between(17, 21, inclusive="left")
                    return len(e), float(e.sum()), float(e[eve].sum()/e.sum())
                for key, a0, g0 in zip(("session_count", "total_energy_kwh", "evening_energy_share"),
                                       features(actual), features(simulated)):
                    trajectories.append(dict(seed=seed, mode=mode, source=source, date=day,
                                             variable=key, observed=a0, generated=g0))
    meta = dict(rows_total=len(data), rows_train=len(train), rows_test=len(test),
                first_day=str(days[0]), last_day=str(days[-1]), sources=sorted(test.source.unique()), stratified=stratified,
                test_by_source=test.groupby("source").date.agg(["size", "min", "max"]).to_dict("index"))
    meta["boundaries"] = boundaries if stratified else {
        "first_validation_day": days[a], "first_test_day": days[b]}
    return stats, trajectories, meta


def grid_check(seed, out):
    set_seed(seed)
    prepared = out / f"grid_date_matched_{seed}.csv"
    observed = prepare_grid()
    observed.to_csv(prepared, index=False)
    model = GridGenerator(GridGeneratorConfig(
        data_path=prepared, save_path=out / f"grid_retrained_{seed}.pkl",
        metrics_path=out / f"grid_training_validation_{seed}.csv",
        importance_path=out / f"grid_placeholder_importance_{seed}.csv",
        val_pred_path=out / f"grid_training_predictions_{seed}.csv", random_seed=seed, val_ratio=1/9))
    data = model.load_data()
    a, b = split_index(len(data))
    # A row's next-hour target must remain in its own partition.
    stamp = data.timestamp
    adjacent = stamp.shift(-1) - stamp == pd.Timedelta(hours=1)
    dev = data.iloc[:b-1].copy()
    cut = a - 1
    model.load_data = lambda: dev
    model.time_split = lambda frame: (frame.iloc[:cut].loc[adjacent], frame.iloc[a:].loc[adjacent])
    model.fit()
    model.save()
    (out / f"grid_placeholder_importance_{seed}.csv").unlink()
    test = data.iloc[b:].loc[adjacent].copy()
    X = test[model.feature_cols].to_numpy(np.float32)
    Y = test[list(model.config.target_cols)].to_numpy(np.float32)
    R = model._build_risk(test)
    actual_history = observed.dropna(subset=FIELDS).set_index("timestamp")[FIELDS]
    window_candidates = []
    for pos in range(b, len(data)-24, 24):
        t0 = stamp.iloc[pos]
        required = pd.date_range(t0-pd.Timedelta(hours=24), periods=49, freq="h")
        if required.isin(actual_history.index).all():
            window_candidates.append(pos)
    if not window_candidates:
        raise ValueError("No contiguous held-out 24-hour grid windows")
    selected = [window_candidates[i] for i in np.linspace(0, len(window_candidates)-1,
                min(16, len(window_candidates)), dtype=int)]
    stats, rollouts = [], []
    for mode in MODES:
        set_seed(seed + 100)
        pred = np.maximum(draw(model, X, R, mode, .6), 0)
        for i, key in enumerate(FIELDS):
            current = test[key].to_numpy()
            stats.append(dict(seed=seed, mode=mode, scope="one_step", variable=key,
                mae=float(np.mean(abs(Y[:, i]-pred[:, i]))),
                persistence_mae=float(np.mean(abs(Y[:, i]-current))),
                **marginal(Y[:, i], pred[:, i])))
        state = data.iloc[selected].copy().reset_index(drop=True)
        for horizon in range(1, 25):
            prediction = np.maximum(draw(model, state[model.feature_cols].to_numpy(np.float32),
                                        model._build_risk(state), mode, .6), 0)
            ts = state.timestamp + pd.Timedelta(hours=1)
            actual = actual_history.loc[list(ts)].to_numpy()
            for i, key in enumerate(FIELDS):
                for j in range(len(state)):
                    rollouts.append(dict(seed=seed, mode=mode, window=j, lead_h=horizon,
                                         variable=key, observed=actual[j, i], generated=prediction[j, i]))
                state[key+"_lag2"] = state[key+"_lag1"]
                state[key+"_lag1"] = state[key]
                state[key+"_lag24"] = actual_history.loc[list(ts-pd.Timedelta(hours=24)), key].to_numpy()
                state[key] = prediction[:, i]
            state["timestamp"] = ts
            for k, v in {"hour":ts.dt.hour, "day_of_week":ts.dt.dayofweek, "month":ts.dt.month,
                         "is_weekend":(ts.dt.dayofweek>=5).astype(int)}.items():
                state[k] = v
            for prefix, key, period in (("hour","hour",24),("month","month",12),("dow","day_of_week",7)):
                state[prefix+"_sin"] = np.sin(2*np.pi*state[key]/period)
                state[prefix+"_cos"] = np.cos(2*np.pi*state[key]/period)
    meta = dict(rows=len(data), rows_test=len(test), test_first=str(test.timestamp.iloc[0]),
                test_last=str(test.timestamp.iloc[-1]), windows=len(selected),
                origins=[str(stamp.iloc[i]) for i in selected])
    return stats, rollouts, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--component", choices=["all", "ev", "grid"], default="all")
    parser.add_argument("--ev-source-stratified", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, check in (("ev", ev_check), ("grid", grid_check)):
        if args.component not in ("all", name):
            continue
        stats, paths, meta = (check(args.seed, args.output, args.ev_source_stratified)
                              if name == "ev" else check(args.seed, args.output))
        pd.DataFrame(stats).to_csv(args.output / f"{name}_metrics_{args.seed}.csv", index=False)
        pd.DataFrame(paths).to_csv(args.output / f"{name}_paths_{args.seed}.csv", index=False)
        (args.output / f"{name}_split_{args.seed}.json").write_text(json.dumps(meta, indent=2))
        print(name, meta, flush=True)


if __name__ == "__main__":
    main()
