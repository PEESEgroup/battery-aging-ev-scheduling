#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

@dataclass
class EVGeneratorConfig:
    sessions_path: Path = Path("processed_ev/ev_sessions_unified.csv")
    save_path: Path = Path("processed_ev/ev_generator.pkl")
    random_seed: int = 42
    gmm_components: int = 5
    min_samples_per_group: int = 50


class EVGenerator:
    """
    EV generator v1:
    1) Hourly count generator using empirical Poisson rates
    2) Session attribute generator using GMM
    """

    def __init__(self, config: EVGeneratorConfig):
        self.config = config
        self.count_table: pd.DataFrame | None = None
        self.gmm_models: Dict[Tuple[str, bool], GaussianMixture] = {}
        self.feature_cols = ["arrival_hour", "parking_duration_hours", "energy_delivered_kwh"]
        self.rng = np.random.default_rng(config.random_seed)

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.sessions_path)

        # basic cleaning
        df = df.copy()
        for col in self.feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["month"] = pd.to_numeric(df["month"], errors="coerce")
        df["is_weekend"] = df["is_weekend"].astype(bool)

        df = df.dropna(subset=["source", "month", "is_weekend", "arrival_hour",
                               "parking_duration_hours", "energy_delivered_kwh"])

        # mild filtering
        df = df[(df["arrival_hour"] >= 0) & (df["arrival_hour"] <= 24)]
        df = df[(df["parking_duration_hours"] > 0) & (df["parking_duration_hours"] <= 72)]
        df = df[(df["energy_delivered_kwh"] > 0) & (df["energy_delivered_kwh"] <= 200)]

        return df.reset_index(drop=True)

    def fit_count_model(self, df: pd.DataFrame) -> None:
        """
        Estimate hourly Poisson rates:
        lambda(source, is_weekend, month, hour) = average session count
        """
        tmp = df.copy()
        tmp["hour_bin"] = np.floor(tmp["arrival_hour"]).clip(0, 23).astype(int)

        # count sessions per date per group
        daily_hour_counts = (
            tmp.groupby(["source", "is_weekend", "month", "date", "hour_bin"])
            .size()
            .reset_index(name="n_sessions")
        )

        count_table = (
            daily_hour_counts.groupby(["source", "is_weekend", "month", "hour_bin"])["n_sessions"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "lambda_hour"})
        )

        # fill possible missing combinations
        all_sources = sorted(tmp["source"].dropna().unique())
        all_weekend = [False, True]
        all_months = sorted(tmp["month"].dropna().astype(int).unique())
        all_hours = list(range(24))

        full_idx = pd.MultiIndex.from_product(
            [all_sources, all_weekend, all_months, all_hours],
            names=["source", "is_weekend", "month", "hour_bin"]
        )

        count_table = count_table.set_index(["source", "is_weekend", "month", "hour_bin"]).reindex(full_idx).reset_index()

        # backfill missing lambda using coarser averages
        global_hour_mean = (
            daily_hour_counts.groupby(["source", "is_weekend", "hour_bin"])["n_sessions"]
            .mean()
            .reset_index()
            .rename(columns={"n_sessions": "fallback_lambda"})
        )

        count_table = count_table.merge(
            global_hour_mean,
            on=["source", "is_weekend", "hour_bin"],
            how="left"
        )

        count_table["lambda_hour"] = count_table["lambda_hour"].fillna(count_table["fallback_lambda"]).fillna(0.0)
        self.count_table = count_table[["source", "is_weekend", "month", "hour_bin", "lambda_hour"]]

    def fit_gmm_models(self, df: pd.DataFrame) -> None:
        """
        Fit one GMM per (source, is_weekend).
        """
        for (source, is_weekend), g in df.groupby(["source", "is_weekend"]):
            x = g[self.feature_cols].dropna().values

            if len(x) < self.config.min_samples_per_group:
                continue

            n_components = min(self.config.gmm_components, max(1, len(x) // 20))
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                random_state=self.config.random_seed,
                reg_covar=1e-4,
            )
            gmm.fit(x)
            self.gmm_models[(source, bool(is_weekend))] = gmm

    def fit(self) -> None:
        df = self.load_data()
        self.fit_count_model(df)
        self.fit_gmm_models(df)

    def save(self) -> None:
        payload = {
            "config": vars(self.config),
            "count_table": self.count_table,
            "gmm_models": self.gmm_models,
            "feature_cols": self.feature_cols,
        }
        self.config.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.save_path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "EVGenerator":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        config_dict = payload["config"]
        config = EVGeneratorConfig(**config_dict)

        obj = cls(config)
        obj.count_table = payload["count_table"]
        obj.gmm_models = payload["gmm_models"]
        obj.feature_cols = payload["feature_cols"]
        return obj

    def _get_lambda(self, source: str, is_weekend: bool, month: int, hour_bin: int) -> float:
        if self.count_table is None:
            raise RuntimeError("Count model has not been fitted.")

        row = self.count_table[
            (self.count_table["source"] == source) &
            (self.count_table["is_weekend"] == is_weekend) &
            (self.count_table["month"] == month) &
            (self.count_table["hour_bin"] == hour_bin)
        ]
        if len(row) == 0:
            return 0.0
        return float(row.iloc[0]["lambda_hour"])

    def sample_sessions_for_horizon(
        self,
        start_time: str,
        hours: int = 24,
        source: str = "acn",
    ) -> pd.DataFrame:
        """
        Generate synthetic EV sessions over a horizon.
        """
        start_ts = pd.Timestamp(start_time)
        rows: List[Dict[str, Any]] = []

        for h in range(hours):
            current_ts = start_ts + pd.Timedelta(hours=h)
            month = int(current_ts.month)
            hour_bin = int(current_ts.hour)
            is_weekend = current_ts.dayofweek >= 5

            lam = self._get_lambda(source, is_weekend, month, hour_bin)
            n_sessions = self.rng.poisson(lam)

            gmm = self.gmm_models.get((source, is_weekend), None)
            if gmm is None or n_sessions == 0:
                continue

            samples, _ = gmm.sample(n_sessions)

            for i, s in enumerate(samples):
                arrival_hour, parking_h, energy_kwh = s.tolist()

                # keep generated values in a reasonable range
                arrival_hour = float(np.clip(arrival_hour, hour_bin, min(hour_bin + 1, 24)))
                parking_h = float(np.clip(parking_h, 0.25, 72.0))
                energy_kwh = float(np.clip(energy_kwh, 0.5, 200.0))

                minute_part = (arrival_hour - int(arrival_hour)) * 60.0
                arrival_time = current_ts.floor("h") + pd.Timedelta(minutes=float(minute_part))
                departure_time = arrival_time + pd.Timedelta(hours=parking_h)

                rows.append(
                    {
                        "source": source,
                        "timestamp_hour": current_ts.floor("h"),
                        "session_id": f"{source}_{current_ts:%Y%m%d%H}_{i}",
                        "arrival_time": arrival_time,
                        "departure_time": departure_time,
                        "arrival_hour": arrival_hour,
                        "parking_duration_hours": parking_h,
                        "energy_delivered_kwh": energy_kwh,
                        "month": month,
                        "is_weekend": is_weekend,
                    }
                )

        return pd.DataFrame(rows)

def plot_generated_samples(real_df: pd.DataFrame, sampled_df: pd.DataFrame, save_path: Path) -> None:
    """
    Plot real vs generated sample distributions for:
    1) arrival_hour
    2) parking_duration_hours
    3) energy_delivered_kwh
    """
    # 只保留有效数据
    real_df = real_df.copy()
    sampled_df = sampled_df.copy()

    cols = ["arrival_hour", "parking_duration_hours", "energy_delivered_kwh"]
    for c in cols:
        real_df[c] = pd.to_numeric(real_df[c], errors="coerce")
        sampled_df[c] = pd.to_numeric(sampled_df[c], errors="coerce")

    real_df = real_df.dropna(subset=cols)
    sampled_df = sampled_df.dropna(subset=cols)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1) arrival hour
    axes[0].hist(real_df["arrival_hour"], bins=24, alpha=0.6, density=True, label="Real")
    axes[0].hist(sampled_df["arrival_hour"], bins=24, alpha=0.6, density=True, label="Generated")
    axes[0].set_title("Arrival Hour")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # 2) parking duration
    max_parking = min(
        max(real_df["parking_duration_hours"].quantile(0.99), 1),
        72
    )
    axes[1].hist(
        real_df.loc[real_df["parking_duration_hours"] <= max_parking, "parking_duration_hours"],
        bins=30, alpha=0.6, density=True, label="Real"
    )
    axes[1].hist(
        sampled_df.loc[sampled_df["parking_duration_hours"] <= max_parking, "parking_duration_hours"],
        bins=30, alpha=0.6, density=True, label="Generated"
    )
    axes[1].set_title("Parking Duration")
    axes[1].set_xlabel("Hours")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    # 3) delivered energy
    max_energy = min(
        max(real_df["energy_delivered_kwh"].quantile(0.99), 1),
        100
    )
    axes[2].hist(
        real_df.loc[real_df["energy_delivered_kwh"] <= max_energy, "energy_delivered_kwh"],
        bins=30, alpha=0.6, density=True, label="Real"
    )
    axes[2].hist(
        sampled_df.loc[sampled_df["energy_delivered_kwh"] <= max_energy, "energy_delivered_kwh"],
        bins=30, alpha=0.6, density=True, label="Generated"
    )
    axes[2].set_title("Delivered Energy")
    axes[2].set_xlabel("kWh")
    axes[2].set_ylabel("Density")
    axes[2].legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    config = EVGeneratorConfig()
    gen = EVGenerator(config)
    gen.fit()
    gen.save()

    print(f"[ok] EV generator saved to: {config.save_path}")

    # load real data for comparison
    real_df = gen.load_data()
    real_df = real_df[real_df["source"] == "acn"].copy()

    # quick sanity sample
    sampled = gen.sample_sessions_for_horizon(
        start_time="2022-10-01 00:00:00",
        hours=24 * 30,   # 生成30天，这样分布更稳定
        source="acn"
    )

    sample_path = Path("processed_ev/ev_generator_sample.csv")
    sampled.to_csv(sample_path, index=False)
    print(f"[ok] Sampled sessions saved to: {sample_path}")
    print(f"[ok] Number of sampled sessions: {len(sampled)}")

    # draw comparison figure
    fig_path = Path("processed_ev/ev_generator_samples_plot.png")
    plot_generated_samples(real_df, sampled, fig_path)
    print(f"[ok] Comparison plot saved to: {fig_path}")


if __name__ == "__main__":
    main()