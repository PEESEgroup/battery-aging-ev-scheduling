#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


@dataclass
class EVGeneratorConfig:
    sessions_path: Path = Path("processed_ev/ev_sessions_unified.csv")
    save_path: Path = Path("processed_ev/ev_generator_v2.pkl")
    random_seed: int = 42

    min_samples_per_count_group: int = 20
    n_user_clusters: int = 4
    min_samples_per_cluster: int = 60
    gmm_components: int = 4
    min_samples_per_gmm_group: int = 50

    max_parking_hours: float = 72.0
    max_energy_kwh: float = 200.0
    clip_arrival_within_hour: bool = True


class EVGeneratorV2:
    """
    EV generator v2:
    1) Context-aware hourly arrival model with Poisson / Negative-Binomial behavior
    2) User-type clustering
    3) Conditional session generator using per-cluster GMMs
    """

    def __init__(self, config: EVGeneratorConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

        self.feature_cols = ["arrival_hour", "parking_duration_hours", "energy_delivered_kwh"]
        self.cluster_feature_cols = ["arrival_hour", "parking_duration_hours", "energy_delivered_kwh"]

        self.count_table: Optional[pd.DataFrame] = None
        self.cluster_scaler: Optional[StandardScaler] = None
        self.cluster_model: Optional[KMeans] = None
        self.cluster_priors: Optional[pd.DataFrame] = None

        self.gmm_models: Dict[Tuple[str, bool, int], GaussianMixture] = {}
        self.gmm_fallback: Dict[Tuple[str, bool], GaussianMixture] = {}

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.sessions_path, low_memory=False)

        df = df.copy()
        for col in self.feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["month"] = pd.to_numeric(df.get("month"), errors="coerce")
        df["is_weekend"] = df["is_weekend"].astype(bool)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        elif "arrival_time" in df.columns:
            arr = pd.to_datetime(df["arrival_time"], errors="coerce")
            df["date"] = arr.dt.date
        else:
            raise ValueError("Input sessions must contain either 'date' or 'arrival_time'.")

        df = df.dropna(subset=[
            "source", "month", "is_weekend", "date",
            "arrival_hour", "parking_duration_hours", "energy_delivered_kwh"
        ])

        df["month"] = df["month"].astype(int)
        df = df[(df["arrival_hour"] >= 0) & (df["arrival_hour"] <= 24)]
        df = df[(df["parking_duration_hours"] > 0) & (df["parking_duration_hours"] <= self.config.max_parking_hours)]
        df = df[(df["energy_delivered_kwh"] > 0) & (df["energy_delivered_kwh"] <= self.config.max_energy_kwh)]

        df["hour_bin"] = np.floor(df["arrival_hour"]).clip(0, 23).astype(int)
        df["is_weekend"] = df["is_weekend"].astype(bool)

        return df.reset_index(drop=True)

    def fit_count_model(self, df: pd.DataFrame) -> None:
        tmp = df.copy()

        daily_hour_counts = (
            tmp.groupby(["source", "is_weekend", "month", "date", "hour_bin"])
            .size()
            .reset_index(name="n_sessions")
        )

        records = []
        for (source, is_weekend, month), g in tmp.groupby(["source", "is_weekend", "month"]):
            dates = sorted(pd.Series(g["date"]).dropna().unique())
            if len(dates) == 0:
                continue
            full_idx = pd.MultiIndex.from_product(
                [[source], [bool(is_weekend)], [int(month)], dates, list(range(24))],
                names=["source", "is_weekend", "month", "date", "hour_bin"]
            )
            g_counts = (
                daily_hour_counts[
                    (daily_hour_counts["source"] == source) &
                    (daily_hour_counts["is_weekend"] == bool(is_weekend)) &
                    (daily_hour_counts["month"] == int(month))
                ]
                .set_index(["source", "is_weekend", "month", "date", "hour_bin"])
                .reindex(full_idx)
                .reset_index()
            )
            g_counts["n_sessions"] = g_counts["n_sessions"].fillna(0.0)
            records.append(g_counts)

        if len(records) == 0:
            raise RuntimeError("No valid count records were constructed.")

        daily_hour_counts_full = pd.concat(records, ignore_index=True)

        count_table = (
            daily_hour_counts_full.groupby(["source", "is_weekend", "month", "hour_bin"])["n_sessions"]
            .agg(["mean", "var", "count"])
            .reset_index()
            .rename(columns={"mean": "lambda_hour", "var": "var_hour", "count": "n_days"})
        )

        fallback_table = (
            daily_hour_counts_full.groupby(["source", "is_weekend", "hour_bin"])["n_sessions"]
            .agg(["mean", "var", "count"])
            .reset_index()
            .rename(columns={"mean": "fallback_lambda", "var": "fallback_var", "count": "fallback_n_days"})
        )

        count_table = count_table.merge(
            fallback_table,
            on=["source", "is_weekend", "hour_bin"],
            how="left"
        )

        count_table["lambda_hour"] = count_table["lambda_hour"].fillna(count_table["fallback_lambda"]).fillna(0.0)
        count_table["var_hour"] = count_table["var_hour"].fillna(count_table["fallback_var"]).fillna(count_table["lambda_hour"])
        count_table["n_days"] = count_table["n_days"].fillna(count_table["fallback_n_days"]).fillna(0)

        self.count_table = count_table[[
            "source", "is_weekend", "month", "hour_bin",
            "lambda_hour", "var_hour", "n_days"
        ]].copy()

    def _get_count_stats(self, source: str, is_weekend: bool, month: int, hour_bin: int) -> Tuple[float, float, int]:
        if self.count_table is None:
            raise RuntimeError("Count model has not been fitted.")

        row = self.count_table[
            (self.count_table["source"] == source) &
            (self.count_table["is_weekend"] == bool(is_weekend)) &
            (self.count_table["month"] == int(month)) &
            (self.count_table["hour_bin"] == int(hour_bin))
        ]
        if len(row) == 0:
            return 0.0, 0.0, 0
        r = row.iloc[0]
        return float(r["lambda_hour"]), float(r["var_hour"]), int(r["n_days"])

    def _sample_hourly_count(self, lam: float, var: float, n_days: int) -> int:
        lam = max(float(lam), 0.0)
        var = max(float(var), lam)
        if lam <= 1e-12:
            return 0
        if n_days < self.config.min_samples_per_count_group or var <= lam + 1e-9:
            return int(self.rng.poisson(lam))

        k = lam * lam / max(var - lam, 1e-9)
        theta = lam / max(k, 1e-9)
        gamma_rate = self.rng.gamma(shape=k, scale=theta)
        return int(self.rng.poisson(gamma_rate))

    def fit_user_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.cluster_feature_cols].values
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        n_clusters = min(self.config.n_user_clusters, max(2, len(df) // self.config.min_samples_per_cluster))
        n_clusters = max(2, n_clusters)

        cluster_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.config.random_seed,
            n_init=20,
        )
        cluster_id = cluster_model.fit_predict(x_scaled)

        out = df.copy()
        out["cluster_id"] = cluster_id.astype(int)

        self.cluster_scaler = scaler
        self.cluster_model = cluster_model

        cluster_priors = (
            out.groupby(["source", "is_weekend", "month", "hour_bin", "cluster_id"])
            .size()
            .reset_index(name="n_cluster")
        )
        cluster_totals = (
            cluster_priors.groupby(["source", "is_weekend", "month", "hour_bin"])["n_cluster"]
            .sum()
            .reset_index(name="n_total")
        )
        cluster_priors = cluster_priors.merge(
            cluster_totals,
            on=["source", "is_weekend", "month", "hour_bin"],
            how="left"
        )
        cluster_priors["cluster_prob"] = cluster_priors["n_cluster"] / cluster_priors["n_total"]

        self.cluster_priors = cluster_priors[[
            "source", "is_weekend", "month", "hour_bin", "cluster_id", "cluster_prob"
        ]].copy()

        return out

    def _sample_cluster_ids(self, source: str, is_weekend: bool, month: int, hour_bin: int, n_sessions: int) -> np.ndarray:
        if self.cluster_priors is None:
            raise RuntimeError("User clusters have not been fitted.")

        sub = self.cluster_priors[
            (self.cluster_priors["source"] == source) &
            (self.cluster_priors["is_weekend"] == bool(is_weekend)) &
            (self.cluster_priors["month"] == int(month)) &
            (self.cluster_priors["hour_bin"] == int(hour_bin))
        ]

        if len(sub) == 0:
            all_clusters = sorted(self.cluster_priors["cluster_id"].unique())
            probs = np.ones(len(all_clusters), dtype=float) / max(len(all_clusters), 1)
            return self.rng.choice(all_clusters, size=n_sessions, p=probs)

        clusters = sub["cluster_id"].values.astype(int)
        probs = sub["cluster_prob"].values.astype(float)
        probs = probs / probs.sum()
        return self.rng.choice(clusters, size=n_sessions, p=probs)

    def fit_session_generators(self, df_clustered: pd.DataFrame) -> None:
        for (source, is_weekend), g in df_clustered.groupby(["source", "is_weekend"]):
            x = g[self.feature_cols].dropna().values
            if len(x) >= self.config.min_samples_per_gmm_group:
                n_components = min(self.config.gmm_components, max(1, len(x) // 25))
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type="full",
                    random_state=self.config.random_seed,
                    reg_covar=1e-4,
                )
                gmm.fit(x)
                self.gmm_fallback[(source, bool(is_weekend))] = gmm

        for (source, is_weekend, cluster_id), g in df_clustered.groupby(["source", "is_weekend", "cluster_id"]):
            x = g[self.feature_cols].dropna().values
            if len(x) < self.config.min_samples_per_gmm_group:
                continue

            n_components = min(self.config.gmm_components, max(1, len(x) // 20))
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                random_state=self.config.random_seed,
                reg_covar=1e-4,
            )
            gmm.fit(x)
            self.gmm_models[(source, bool(is_weekend), int(cluster_id))] = gmm

    def _conditional_sample_from_gmm(self, gmm: GaussianMixture, hour_bin: int, n_samples: int) -> np.ndarray:
        samples, _ = gmm.sample(n_samples)
        out = samples.copy()

        if self.config.clip_arrival_within_hour:
            lo = float(hour_bin)
            hi = min(float(hour_bin + 1), 24.0)
            out[:, 0] = np.clip(out[:, 0], lo, hi)
        else:
            out[:, 0] = np.clip(out[:, 0], 0.0, 24.0)

        out[:, 1] = np.clip(out[:, 1], 0.25, self.config.max_parking_hours)
        out[:, 2] = np.clip(out[:, 2], 0.5, self.config.max_energy_kwh)
        return out

    def fit(self) -> None:
        df = self.load_data()
        self.fit_count_model(df)
        df_clustered = self.fit_user_clusters(df)
        self.fit_session_generators(df_clustered)

    def save(self) -> None:
        payload = {
            "config": vars(self.config),
            "count_table": self.count_table,
            "cluster_scaler": self.cluster_scaler,
            "cluster_model": self.cluster_model,
            "cluster_priors": self.cluster_priors,
            "gmm_models": self.gmm_models,
            "gmm_fallback": self.gmm_fallback,
            "feature_cols": self.feature_cols,
            "cluster_feature_cols": self.cluster_feature_cols,
        }
        self.config.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.save_path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "EVGeneratorV2":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        config = EVGeneratorConfig(**payload["config"])
        obj = cls(config)
        obj.count_table = payload["count_table"]
        obj.cluster_scaler = payload["cluster_scaler"]
        obj.cluster_model = payload["cluster_model"]
        obj.cluster_priors = payload["cluster_priors"]
        obj.gmm_models = payload["gmm_models"]
        obj.gmm_fallback = payload["gmm_fallback"]
        obj.feature_cols = payload["feature_cols"]
        obj.cluster_feature_cols = payload["cluster_feature_cols"]
        return obj

    def sample_sessions_for_horizon(self, start_time: str, hours: int = 24, source: str = "acn") -> pd.DataFrame:
        start_ts = pd.Timestamp(start_time)
        rows: List[Dict[str, Any]] = []

        for h in range(hours):
            current_ts = start_ts + pd.Timedelta(hours=h)
            month = int(current_ts.month)
            hour_bin = int(current_ts.hour)
            is_weekend = current_ts.dayofweek >= 5

            lam, var, n_days = self._get_count_stats(source, is_weekend, month, hour_bin)
            n_sessions = self._sample_hourly_count(lam, var, n_days)
            if n_sessions <= 0:
                continue

            cluster_ids = self._sample_cluster_ids(source, is_weekend, month, hour_bin, n_sessions)

            for i, cluster_id in enumerate(cluster_ids):
                gmm = self.gmm_models.get((source, bool(is_weekend), int(cluster_id)), None)
                if gmm is None:
                    gmm = self.gmm_fallback.get((source, bool(is_weekend)), None)
                if gmm is None:
                    continue

                s = self._conditional_sample_from_gmm(gmm, hour_bin, 1)[0]
                arrival_hour, parking_h, energy_kwh = s.tolist()

                minute_part = (arrival_hour - int(arrival_hour)) * 60.0
                arrival_time = current_ts.floor("h") + pd.Timedelta(minutes=float(minute_part))
                departure_time = arrival_time + pd.Timedelta(hours=float(parking_h))

                rows.append({
                    "source": source,
                    "timestamp_hour": current_ts.floor("h"),
                    "session_id": f"{source}_{current_ts:%Y%m%d%H}_{i}",
                    "arrival_time": arrival_time,
                    "departure_time": departure_time,
                    "arrival_hour": float(arrival_hour),
                    "parking_duration_hours": float(parking_h),
                    "energy_delivered_kwh": float(energy_kwh),
                    "month": month,
                    "is_weekend": bool(is_weekend),
                    "cluster_id": int(cluster_id),
                })

        return pd.DataFrame(rows)


def plot_generated_samples(real_df: pd.DataFrame, sampled_df: pd.DataFrame, save_path: Path) -> None:
    real_df = real_df.copy()
    sampled_df = sampled_df.copy()

    cols = ["arrival_hour", "parking_duration_hours", "energy_delivered_kwh"]
    for c in cols:
        real_df[c] = pd.to_numeric(real_df[c], errors="coerce")
        sampled_df[c] = pd.to_numeric(sampled_df[c], errors="coerce")

    real_df = real_df.dropna(subset=cols)
    sampled_df = sampled_df.dropna(subset=cols)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(real_df["arrival_hour"], bins=24, alpha=0.6, density=True, label="Real")
    axes[0].hist(sampled_df["arrival_hour"], bins=24, alpha=0.6, density=True, label="Generated")
    axes[0].set_title("Arrival Hour")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    max_parking = min(max(real_df["parking_duration_hours"].quantile(0.99), 1), 72)
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

    max_energy = min(max(real_df["energy_delivered_kwh"].quantile(0.99), 1), 100)
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
    gen = EVGeneratorV2(config)
    gen.fit()
    gen.save()

    print(f"[ok] EV generator v2 saved to: {config.save_path}")

    real_df = gen.load_data()
    real_df = real_df[real_df["source"] == "acn"].copy()

    sampled = gen.sample_sessions_for_horizon(
        start_time="2022-10-01 00:00:00",
        hours=24 * 30,
        source="acn",
    )

    sample_path = Path("processed_ev/ev_generator_v2_sample.csv")
    sampled.to_csv(sample_path, index=False)
    print(f"[ok] Sampled sessions saved to: {sample_path}")
    print(f"[ok] Number of sampled sessions: {len(sampled)}")

    fig_path = Path("processed_ev/ev_generator_v2_samples_plot.png")
    plot_generated_samples(real_df, sampled, fig_path)
    print(f"[ok] Comparison plot saved to: {fig_path}")


if __name__ == "__main__":
    main()
