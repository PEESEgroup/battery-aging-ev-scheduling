#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_grid_generator.py

Train a lightweight probabilistic grid generator from:
    processed_grid/grid_generator_ready.csv

Targets:
    - system_load_mw_next1
    - weighted_lbmp_mwh_next1
    - weighted_imer_co2_ton_per_mwh_next1
    - congestion_intensity_next1

Model:
    - one RandomForestRegressor per target
    - Gaussian residual sampling for trajectory generation

Outputs:
    processed_grid/
        - grid_generator.pkl
        - grid_generator_metrics.csv
        - grid_generator_feature_importance.csv
        - grid_generator_val_predictions.csv
        - grid_generator_sample.csv
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

@dataclass
class GridGeneratorConfig:
    data_path: Path = Path("processed_grid/grid_generator_ready.csv")
    save_path: Path = Path("processed_grid/grid_generator.pkl")
    metrics_path: Path = Path("processed_grid/grid_generator_metrics.csv")
    importance_path: Path = Path("processed_grid/grid_generator_feature_importance.csv")
    val_pred_path: Path = Path("processed_grid/grid_generator_val_predictions.csv")
    sample_path: Path = Path("processed_grid/grid_generator_sample.csv")

    random_seed: int = 42
    n_estimators: int = 300
    max_depth: int | None = 12
    min_samples_leaf: int = 5
    val_ratio: float = 0.2

    target_cols: tuple[str, ...] = (
        "system_load_mw_next1",
        "weighted_lbmp_mwh_next1",
        "weighted_imer_co2_ton_per_mwh_next1",
        "congestion_intensity_next1",
    )


class GridGenerator:
    """
    One-step grid generator.
    Train one RF model per target, then generate trajectories by recursive rollout.
    """

    def __init__(self, config: GridGeneratorConfig):
        self.config = config
        self.models: Dict[str, RandomForestRegressor] = {}
        self.feature_cols: List[str] = []
        self.residual_std: Dict[str, float] = {}
        self.rng = np.random.default_rng(config.random_seed)

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.data_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

        # numeric coercion
        for c in df.columns:
            if c != "timestamp":
                df[c] = pd.to_numeric(df[c], errors="ignore")

        # feature set: exclude timestamp and future targets
        exclude = {"timestamp", *self.config.target_cols}
        self.feature_cols = [c for c in df.columns if c not in exclude]

        keep_cols = ["timestamp"] + self.feature_cols + list(self.config.target_cols)
        df = df[keep_cols].copy()
        df = df.dropna().reset_index(drop=True)
        return df

    def time_split(self, df: pd.DataFrame):
        n = len(df)
        split_idx = int((1 - self.config.val_ratio) * n)
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()
        return train_df, val_df

    def fit(self):
        df = self.load_data()
        train_df, val_df = self.time_split(df)

        X_train = train_df[self.feature_cols]
        X_val = val_df[self.feature_cols]

        metrics_rows = []
        pred_df = val_df[["timestamp"]].copy()
        importance_rows = []

        for target in self.config.target_cols:
            y_train = train_df[target]
            y_val = val_df[target]

            model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=self.config.random_seed,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            self.models[target] = model

            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            # residual std from train residuals
            resid = y_train.values - train_pred
            resid_std = float(np.std(resid))
            if resid_std <= 1e-8:
                resid_std = 1e-8
            self.residual_std[target] = resid_std

            rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
            mae = float(mean_absolute_error(y_val, val_pred))
            r2 = float(r2_score(y_val, val_pred))

            metrics_rows.append({
                "target": target,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "train_residual_std": resid_std,
            })

            pred_df[f"{target}_true"] = y_val.values
            pred_df[f"{target}_pred"] = val_pred

            for feat, imp in zip(self.feature_cols, model.feature_importances_):
                importance_rows.append({
                    "target": target,
                    "feature": feat,
                    "importance": float(imp),
                })

        metrics_df = pd.DataFrame(metrics_rows)
        importance_df = pd.DataFrame(importance_rows).sort_values(
            ["target", "importance"], ascending=[True, False]
        )
        pred_df.to_csv(self.config.val_pred_path, index=False)
        metrics_df.to_csv(self.config.metrics_path, index=False)
        importance_df.to_csv(self.config.importance_path, index=False)

        return metrics_df, importance_df, pred_df

    def save(self):
        payload = {
            "config": vars(self.config),
            "models": self.models,
            "feature_cols": self.feature_cols,
            "residual_std": self.residual_std,
        }
        self.config.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.save_path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "GridGenerator":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        config_dict = payload["config"]
        config = GridGeneratorConfig(**config_dict)

        obj = cls(config)
        obj.models = payload["models"]
        obj.feature_cols = payload["feature_cols"]
        obj.residual_std = payload["residual_std"]
        return obj

    def _sample_one_step(self, row: pd.Series, stochastic: bool = True) -> Dict[str, float]:
        x = row[self.feature_cols].to_frame().T
        out = {}
        for target, model in self.models.items():
            mu = float(model.predict(x)[0])
            if stochastic:
                sigma = self.residual_std.get(target, 0.0)
                val = mu + self.rng.normal(0.0, sigma)
            else:
                val = mu
            out[target] = val
        return out

    def sample_trajectory(
        self,
        seed_row: pd.Series,
        horizon: int = 24,
        stochastic: bool = True,
    ) -> pd.DataFrame:
        """
        Recursive rollout from the last observed row.
        Assumes lag1/lag2/lag24 are present in feature space.
        """
        current = seed_row.copy()
        rows = []

        for _ in range(horizon):
            ts = pd.Timestamp(current["timestamp"]) + pd.Timedelta(hours=1)

            preds = self._sample_one_step(current, stochastic=stochastic)

            next_row = current.copy()
            next_row["timestamp"] = ts

            # write realized current-step state (which is previous next1)
            next_row["system_load_mw"] = preds["system_load_mw_next1"]
            next_row["weighted_lbmp_mwh"] = preds["weighted_lbmp_mwh_next1"]
            next_row["weighted_imer_co2_ton_per_mwh"] = preds["weighted_imer_co2_ton_per_mwh_next1"]
            next_row["congestion_intensity"] = preds["congestion_intensity_next1"]

            # update calendar features
            next_row["hour"] = ts.hour
            next_row["day_of_week"] = ts.dayofweek
            next_row["month"] = ts.month
            next_row["is_weekend"] = ts.dayofweek >= 5

            next_row["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24.0)
            next_row["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24.0)
            next_row["month_sin"] = np.sin(2 * np.pi * ts.month / 12.0)
            next_row["month_cos"] = np.cos(2 * np.pi * ts.month / 12.0)
            next_row["dow_sin"] = np.sin(2 * np.pi * ts.dayofweek / 7.0)
            next_row["dow_cos"] = np.cos(2 * np.pi * ts.dayofweek / 7.0)

            # update lag features
            for base in [
                "system_load_mw",
                "weighted_lbmp_mwh",
                "weighted_imer_co2_ton_per_mwh",
                "congestion_intensity",
            ]:
                lag1 = current.get(base, np.nan)
                lag2 = current.get(f"{base}_lag1", np.nan)
                lag24 = current.get(f"{base}_lag23", current.get(f"{base}_lag24", np.nan))

                next_row[f"{base}_lag1"] = lag1
                next_row[f"{base}_lag2"] = lag2
                next_row[f"{base}_lag24"] = lag24

            rows.append({
                "timestamp": ts,
                "system_load_mw": next_row["system_load_mw"],
                "weighted_lbmp_mwh": next_row["weighted_lbmp_mwh"],
                "weighted_imer_co2_ton_per_mwh": next_row["weighted_imer_co2_ton_per_mwh"],
                "congestion_intensity": next_row["congestion_intensity"],
            })

            # keep a rolling memory for lag24 approximation
            for base in [
                "system_load_mw",
                "weighted_lbmp_mwh",
                "weighted_imer_co2_ton_per_mwh",
                "congestion_intensity",
            ]:
                # shift lag memory manually if intermediate lag columns exist
                for k in range(23, 1, -1):
                    prev = f"{base}_lag{k-1}"
                    cur = f"{base}_lag{k}"
                    if prev in current.index:
                        next_row[cur] = current[prev]
                if f"{base}_lag1" not in next_row.index:
                    next_row[f"{base}_lag1"] = current.get(base, np.nan)

            current = next_row

        return pd.DataFrame(rows)

def plot_generated_grid_samples(
    real_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    save_path: Path,
    history_hours: int = 24,
) -> None:
    """
    Plot real recent trajectory vs generated future trajectory.
    """
    real_df = real_df.copy()
    sampled_df = sampled_df.copy()

    real_df["timestamp"] = pd.to_datetime(real_df["timestamp"], errors="coerce")
    sampled_df["timestamp"] = pd.to_datetime(sampled_df["timestamp"], errors="coerce")

    # 取最近 history_hours 小时真实数据
    real_plot = real_df.sort_values("timestamp").tail(history_hours).copy()
    sampled_plot = sampled_df.sort_values("timestamp").copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    variables = [
        ("system_load_mw", "System Load (MW)"),
        ("weighted_lbmp_mwh", "Weighted Price ($/MWh)"),
        ("weighted_imer_co2_ton_per_mwh", "CO2 Emission Rate (ton/MWh)"),
        ("congestion_intensity", "Congestion Intensity"),
    ]

    for ax, (col, title) in zip(axes.flatten(), variables):
        ax.plot(real_plot["timestamp"], real_plot[col], label="Recent Real", linewidth=2)
        ax.plot(sampled_plot["timestamp"], sampled_plot[col], label="Generated", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=30)
        ax.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_generated_grid_uncertainty(
    real_df: pd.DataFrame,
    sampled_list: list[pd.DataFrame],
    save_path: Path,
    history_hours: int = 24,
) -> None:
    """
    Plot recent real trajectory + multiple generated trajectories + mean uncertainty band.
    """
    real_df = real_df.copy()
    real_df["timestamp"] = pd.to_datetime(real_df["timestamp"], errors="coerce")
    real_plot = real_df.sort_values("timestamp").tail(history_hours).copy()

    variables = [
        ("system_load_mw", "System Load (MW)"),
        ("weighted_lbmp_mwh", "Weighted Price ($/MWh)"),
        ("weighted_imer_co2_ton_per_mwh", "CO2 Emission Rate (ton/MWh)"),
        ("congestion_intensity", "Congestion Intensity"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for ax, (col, title) in zip(axes.flatten(), variables):
        # plot recent real history
        ax.plot(
            real_plot["timestamp"],
            real_plot[col],
            label="Recent Real",
            linewidth=2
        )

        # align generated trajectories
        gen_times = sampled_list[0]["timestamp"]
        gen_matrix = np.vstack([pd.to_numeric(df[col], errors="coerce").values for df in sampled_list])

        gen_mean = np.nanmean(gen_matrix, axis=0)
        gen_std = np.nanstd(gen_matrix, axis=0)
        gen_p10 = np.nanpercentile(gen_matrix, 10, axis=0)
        gen_p90 = np.nanpercentile(gen_matrix, 90, axis=0)

        # plot individual samples
        for i, df in enumerate(sampled_list):
            ax.plot(
                df["timestamp"],
                df[col],
                alpha=0.25,
                linewidth=1
            )

        # plot uncertainty band
        ax.fill_between(
            gen_times,
            gen_p10,
            gen_p90,
            alpha=0.25,
            label="P10-P90 band"
        )

        # plot mean
        ax.plot(
            gen_times,
            gen_mean,
            linewidth=2.5,
            label="Generated Mean"
        )

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=30)
        ax.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def main() -> None:
    config = GridGeneratorConfig()
    gen = GridGenerator(config)

    metrics_df, importance_df, pred_df = gen.fit()
    gen.save()

    print(f"[ok] grid generator saved   -> {config.save_path}")
    print(f"[ok] metrics saved          -> {config.metrics_path}")
    print(f"[ok] importance saved       -> {config.importance_path}")
    print(f"[ok] val predictions saved  -> {config.val_pred_path}")

    # quick sample rollout from the last available row
    data = gen.load_data()
    seed_row = data.iloc[-1].copy()

    n_samples = 10
    sampled_list = []
    for i in range(n_samples):
        sample_df = gen.sample_trajectory(seed_row=seed_row, horizon=24, stochastic=True)
        sample_df["sample_id"] = i
        sampled_list.append(sample_df)

    all_samples = pd.concat(sampled_list, ignore_index=True)
    all_samples.to_csv(config.sample_path, index=False)
    print(f"[ok] sample rollouts saved  -> {config.sample_path}")

    # plot uncertainty
    fig_path = Path("processed_grid/grid_generator_uncertainty_plot.png")
    plot_generated_grid_uncertainty(
        real_df=data,
        sampled_list=sampled_list,
        save_path=fig_path,
        history_hours=24,
    )
    print(f"[ok] uncertainty plot saved -> {fig_path}")

    # plot generated samples
    fig_path = Path("processed_grid/grid_generator_samples_plot.png")
    plot_generated_grid_samples(
        real_df=data,
        sampled_df=sample_df,
        save_path=fig_path,
        history_hours=24,
    )
    print(f"[ok] sample plot saved      -> {fig_path}")

    print("\nValidation metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
