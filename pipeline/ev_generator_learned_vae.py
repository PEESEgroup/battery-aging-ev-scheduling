#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class EVLearnedGeneratorConfig:
    sessions_path: Path = Path("processed_ev/ev_sessions_unified.csv")
    model_path: Path = Path("processed_ev/ev_generator_learned.pt")
    artifact_path: Path = Path("processed_ev/ev_generator_learned.pkl")

    random_seed: int = 42

    # arrival-count model
    min_samples_per_count_group: int = 20

    # cVAE
    batch_size: int = 256
    epochs: int = 120
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    latent_dim: int = 8
    hidden_dim: int = 128
    beta_kl: float = 0.05

    # clustering
    use_cluster_id: bool = True
    n_user_clusters: int = 4
    min_cluster_samples: int = 60

    # cleaning / clipping
    max_parking_hours: float = 72.0
    max_energy_kwh: float = 200.0
    clip_arrival_within_hour: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cvar_style_plot(real_df: pd.DataFrame, sampled_df: pd.DataFrame, save_path: Path) -> None:
    real_df = real_df.copy()
    sampled_df = sampled_df.copy()

    cols = ["arrival_hour", "parking_duration_hours", "energy_delivered_kwh"]
    for c in cols:
        real_df[c] = pd.to_numeric(real_df[c], errors="coerce")
        sampled_df[c] = pd.to_numeric(sampled_df[c], errors="coerce")

    real_df = real_df.dropna(subset=cols)
    sampled_df = sampled_df.dropna(subset=cols)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(real_df["arrival_hour"], bins=24, alpha=0.6, density=True, label="Real")
    axes[0].hist(sampled_df["arrival_hour"], bins=24, alpha=0.6, density=True, label="Generated")
    axes[0].set_title("Arrival Hour")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)

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
    axes[1].legend(frameon=False)

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
    axes[2].legend(frameon=False)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------
class EVLearnedPreprocessor:
    def __init__(self, config: EVLearnedGeneratorConfig):
        self.config = config
        self.x_cols = ["arrival_hour", "parking_duration_hours", "energy_delivered_kwh"]
        self.cat_cols = ["source", "month", "is_weekend", "hour_bin"]
        self.cond_cols: List[str] = []

        self.x_scaler: Optional[StandardScaler] = None
        self.cluster_scaler: Optional[StandardScaler] = None
        self.cluster_model: Optional[KMeans] = None

        self.cond_levels: Dict[str, List[Any]] = {}
        self.cond_dim: int = 0
        self.x_dim: int = len(self.x_cols)

    def load_and_clean(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.sessions_path, low_memory=False).copy()

        for col in self.x_cols:
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

        df = df.dropna(subset=["source", "month", "is_weekend", "date"] + self.x_cols)
        df["month"] = df["month"].astype(int)

        df = df[(df["arrival_hour"] >= 0) & (df["arrival_hour"] <= 24)]
        df = df[(df["parking_duration_hours"] > 0) & (df["parking_duration_hours"] <= self.config.max_parking_hours)]
        df = df[(df["energy_delivered_kwh"] > 0) & (df["energy_delivered_kwh"] <= self.config.max_energy_kwh)]

        df["hour_bin"] = np.floor(df["arrival_hour"]).clip(0, 23).astype(int)
        df["is_weekend"] = df["is_weekend"].astype(bool)
        return df.reset_index(drop=True)

    def fit_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.use_cluster_id:
            out = df.copy()
            out["cluster_id"] = 0
            return out

        x = df[self.x_cols].values
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        n_clusters = min(self.config.n_user_clusters, max(2, len(df) // self.config.min_cluster_samples))
        n_clusters = max(2, n_clusters)

        km = KMeans(
            n_clusters=n_clusters,
            random_state=self.config.random_seed,
            n_init=20,
        )
        cluster_id = km.fit_predict(x_scaled)

        out = df.copy()
        out["cluster_id"] = cluster_id.astype(int)
        self.cluster_scaler = scaler
        self.cluster_model = km
        return out

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.fit_clusters(df)

        self.cond_cols = self.cat_cols + (["cluster_id"] if self.config.use_cluster_id else [])
        self.cond_levels = {}
        for col in self.cond_cols:
            vals = sorted(pd.Series(df[col]).dropna().unique().tolist())
            self.cond_levels[col] = vals

        self.cond_dim = int(sum(len(v) for v in self.cond_levels.values()))
        self.x_scaler = StandardScaler()
        self.x_scaler.fit(df[self.x_cols].values)
        return df

    def transform_x(self, df: pd.DataFrame) -> np.ndarray:
        if self.x_scaler is None:
            raise RuntimeError("x_scaler has not been fitted.")
        return self.x_scaler.transform(df[self.x_cols].values).astype(np.float32)

    def inverse_transform_x(self, x_scaled: np.ndarray) -> np.ndarray:
        if self.x_scaler is None:
            raise RuntimeError("x_scaler has not been fitted.")
        return self.x_scaler.inverse_transform(x_scaled)

    def encode_conditions(self, df: pd.DataFrame) -> np.ndarray:
        if not self.cond_levels:
            raise RuntimeError("Condition encoder has not been fitted.")

        n = len(df)
        cond = np.zeros((n, self.cond_dim), dtype=np.float32)

        offset = 0
        for col in self.cond_cols:
            levels = self.cond_levels[col]
            level_to_idx = {v: i for i, v in enumerate(levels)}
            for i, v in enumerate(df[col].tolist()):
                if v in level_to_idx:
                    cond[i, offset + level_to_idx[v]] = 1.0
            offset += len(levels)
        return cond

    def encode_condition_dict(self, cond_dict: Dict[str, Any]) -> np.ndarray:
        out = np.zeros((1, self.cond_dim), dtype=np.float32)
        offset = 0
        for col in self.cond_cols:
            levels = self.cond_levels[col]
            level_to_idx = {v: i for i, v in enumerate(levels)}
            val = cond_dict.get(col, None)
            if val in level_to_idx:
                out[0, offset + level_to_idx[val]] = 1.0
            offset += len(levels)
        return out

    def save(self, path: Path) -> None:
        payload = {
            "config": asdict(self.config),
            "x_cols": self.x_cols,
            "cat_cols": self.cat_cols,
            "cond_cols": self.cond_cols,
            "cond_levels": self.cond_levels,
            "cond_dim": self.cond_dim,
            "x_scaler": self.x_scaler,
            "cluster_scaler": self.cluster_scaler,
            "cluster_model": self.cluster_model,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "EVLearnedPreprocessor":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        cfg = EVLearnedGeneratorConfig(**payload["config"])
        obj = cls(cfg)
        obj.x_cols = payload["x_cols"]
        obj.cat_cols = payload["cat_cols"]
        obj.cond_cols = payload["cond_cols"]
        obj.cond_levels = payload["cond_levels"]
        obj.cond_dim = payload["cond_dim"]
        obj.x_scaler = payload["x_scaler"]
        obj.cluster_scaler = payload["cluster_scaler"]
        obj.cluster_model = payload["cluster_model"]
        return obj


# ---------------------------------------------------------------------
# Count model
# ---------------------------------------------------------------------
class ArrivalCountModel:
    """
    Context-aware hourly arrival model.
    Uses Poisson when variance is close to mean, and Gamma-Poisson
    (Negative-Binomial-like) sampling when arrivals are over-dispersed.
    """
    def __init__(self, config: EVLearnedGeneratorConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.count_table: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame) -> None:
        daily_hour_counts = (
            df.groupby(["source", "is_weekend", "month", "date", "hour_bin"])
            .size()
            .reset_index(name="n_sessions")
        )

        records = []
        for (source, is_weekend, month), g in df.groupby(["source", "is_weekend", "month"]):
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

    def get_count_stats(self, source: str, is_weekend: bool, month: int, hour_bin: int) -> Tuple[float, float, int]:
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

    def sample_hourly_count(self, lam: float, var: float, n_days: int) -> int:
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


# ---------------------------------------------------------------------
# Dataset and cVAE
# ---------------------------------------------------------------------
class EVDataset(Dataset):
    def __init__(self, x: np.ndarray, c: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.c = torch.tensor(c, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.c[idx]


class ConditionalVAE(nn.Module):
    def __init__(self, x_dim: int, cond_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        enc_in = x_dim + cond_dim
        dec_in = latent_dim + cond_dim

        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        h = self.encoder(torch.cat([x, c], dim=-1))
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor):
        return self.decoder(torch.cat([z, c], dim=-1))

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, c)
        return x_hat, mu, logvar


def vae_loss_fn(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta_kl: float):
    recon = torch.mean((x - x_hat) ** 2)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + beta_kl * kl
    return total, recon, kl


# ---------------------------------------------------------------------
# Unified learned generator
# ---------------------------------------------------------------------
class EVGeneratorLearned:
    """
    Complete learned EV generator:
    1) Context-aware count model
    2) Conditional VAE for session attributes
    """
    def __init__(self, config: EVLearnedGeneratorConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

        self.preprocessor = EVLearnedPreprocessor(config)
        self.count_model = ArrivalCountModel(config, self.rng)
        self.model: Optional[ConditionalVAE] = None

    def fit(self) -> Dict[str, List[float]]:
        set_seed(self.config.random_seed)

        df = self.preprocessor.load_and_clean()
        df = self.preprocessor.fit(df)

        # fit count model on cleaned raw data
        self.count_model.fit(df)

        # fit cVAE on session attributes conditioned on context
        x = self.preprocessor.transform_x(df)
        c = self.preprocessor.encode_conditions(df)

        dataset = EVDataset(x, c)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        model = ConditionalVAE(
            x_dim=self.preprocessor.x_dim,
            cond_dim=self.preprocessor.cond_dim,
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
        ).to(self.config.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        history = {"loss": [], "recon": [], "kl": []}
        model.train()

        for epoch in range(self.config.epochs):
            loss_meter = 0.0
            recon_meter = 0.0
            kl_meter = 0.0
            n_batches = 0

            for xb, cb in loader:
                xb = xb.to(self.config.device)
                cb = cb.to(self.config.device)

                optimizer.zero_grad()
                x_hat, mu, logvar = model(xb, cb)
                loss, recon, kl = vae_loss_fn(xb, x_hat, mu, logvar, self.config.beta_kl)
                loss.backward()
                optimizer.step()

                loss_meter += loss.item()
                recon_meter += recon.item()
                kl_meter += kl.item()
                n_batches += 1

            history["loss"].append(loss_meter / max(n_batches, 1))
            history["recon"].append(recon_meter / max(n_batches, 1))
            history["kl"].append(kl_meter / max(n_batches, 1))

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(
                    f"[epoch {epoch+1:03d}/{self.config.epochs}] "
                    f"loss={history['loss'][-1]:.5f}, "
                    f"recon={history['recon'][-1]:.5f}, "
                    f"kl={history['kl'][-1]:.5f}"
                )

        self.model = model
        return history

    def save(self) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")

        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": asdict(self.config),
            "model_state_dict": self.model.state_dict(),
            "x_dim": self.preprocessor.x_dim,
            "cond_dim": self.preprocessor.cond_dim,
            "hidden_dim": self.config.hidden_dim,
            "latent_dim": self.config.latent_dim,
        }, self.config.model_path)

        payload = {
            "config": asdict(self.config),
            "count_table": self.count_model.count_table,
        }
        with open(self.config.artifact_path, "wb") as f:
            pickle.dump(payload, f)

        self.preprocessor.save(self.config.artifact_path.with_suffix(".preproc.pkl"))

    @classmethod
    def load(cls, model_path: Path, artifact_path: Path) -> "EVGeneratorLearned":
        checkpoint = torch.load(model_path, map_location="cpu")
        with open(artifact_path, "rb") as f:
            payload = pickle.load(f)

        cfg = EVLearnedGeneratorConfig(**checkpoint["config"])
        obj = cls(cfg)

        obj.preprocessor = EVLearnedPreprocessor.load(artifact_path.with_suffix(".preproc.pkl"))
        obj.count_model.count_table = payload["count_table"]

        model = ConditionalVAE(
            x_dim=checkpoint["x_dim"],
            cond_dim=checkpoint["cond_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            latent_dim=checkpoint["latent_dim"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        obj.model = model
        return obj

    def sample_sessions_for_horizon(
        self,
        start_time: str,
        hours: int = 24,
        source: str = "acn",
    ) -> pd.DataFrame:
        """
        Full learned generator:
        1) sample counts by context
        2) sample session attributes from cVAE conditioned on context
        """
        if self.model is None:
            raise RuntimeError("Model has not been loaded or fitted.")

        self.model.eval()
        start_ts = pd.Timestamp(start_time)
        rows: List[Dict[str, Any]] = []

        with torch.no_grad():
            for h in range(hours):
                current_ts = start_ts + pd.Timedelta(hours=h)
                month = int(current_ts.month)
                hour_bin = int(current_ts.hour)
                is_weekend = current_ts.dayofweek >= 5

                lam, var, n_days = self.count_model.get_count_stats(source, is_weekend, month, hour_bin)
                n_sessions = self.count_model.sample_hourly_count(lam, var, n_days)
                if n_sessions <= 0:
                    continue

                cluster_ids = [0]
                if self.config.use_cluster_id:
                    cluster_ids = self.preprocessor.cond_levels.get("cluster_id", [0])

                chosen_clusters = np.random.choice(cluster_ids, size=n_sessions, replace=True)

                for i, cluster_id in enumerate(chosen_clusters):
                    cond_dict = {
                        "source": source,
                        "month": month,
                        "is_weekend": bool(is_weekend),
                        "hour_bin": hour_bin,
                    }
                    if self.config.use_cluster_id:
                        cond_dict["cluster_id"] = int(cluster_id)

                    c_np = self.preprocessor.encode_condition_dict(cond_dict)
                    c = torch.tensor(c_np, dtype=torch.float32)

                    z = torch.randn((1, self.config.latent_dim), dtype=torch.float32)
                    x_hat_scaled = self.model.decode(z, c).cpu().numpy()
                    x_hat = self.preprocessor.inverse_transform_x(x_hat_scaled)[0]

                    arrival_hour = float(x_hat[0])
                    parking_h = float(x_hat[1])
                    energy_kwh = float(x_hat[2])

                    if self.config.clip_arrival_within_hour:
                        arrival_hour = float(np.clip(arrival_hour, hour_bin, min(hour_bin + 1, 24.0)))
                    else:
                        arrival_hour = float(np.clip(arrival_hour, 0.0, 24.0))

                    parking_h = float(np.clip(parking_h, 0.25, self.config.max_parking_hours))
                    energy_kwh = float(np.clip(energy_kwh, 0.5, self.config.max_energy_kwh))

                    minute_part = (arrival_hour - int(arrival_hour)) * 60.0
                    arrival_time = current_ts.floor("h") + pd.Timedelta(minutes=float(minute_part))
                    departure_time = arrival_time + pd.Timedelta(hours=float(parking_h))

                    rows.append({
                        "source": source,
                        "timestamp_hour": current_ts.floor("h"),
                        "session_id": f"{source}_{current_ts:%Y%m%d%H}_{i}",
                        "arrival_time": arrival_time,
                        "departure_time": departure_time,
                        "arrival_hour": arrival_hour,
                        "parking_duration_hours": parking_h,
                        "energy_delivered_kwh": energy_kwh,
                        "month": month,
                        "is_weekend": bool(is_weekend),
                        "hour_bin": hour_bin,
                        "cluster_id": int(cluster_id) if self.config.use_cluster_id else 0,
                    })

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    cfg = EVLearnedGeneratorConfig()
    gen = EVGeneratorLearned(cfg)
    history = gen.fit()
    gen.save()

    print(f"[ok] learned EV generator model saved to: {cfg.model_path}")
    print(f"[ok] learned EV generator artifact saved to: {cfg.artifact_path}")

    real_df = gen.preprocessor.load_and_clean()
    real_df = gen.preprocessor.fit_clusters(real_df)
    real_df = real_df[real_df["source"] == "acn"].copy()

    sampled = gen.sample_sessions_for_horizon(
        start_time="2022-10-01 00:00:00",
        hours=24 * 30,
        source="acn",
    )

    sample_path = Path("processed_ev/ev_generator_learned_sample.csv")
    sampled.to_csv(sample_path, index=False)
    print(f"[ok] Sampled sessions saved to: {sample_path}")
    print(f"[ok] Number of sampled sessions: {len(sampled)}")

    fig_path = Path("processed_ev/ev_generator_learned_samples_plot.png")
    cvar_style_plot(real_df, sampled, fig_path)
    print(f"[ok] Comparison plot saved to: {fig_path}")

    hist_path = Path("processed_ev/ev_generator_learned_train_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[ok] Training history saved to: {hist_path}")


if __name__ == "__main__":
    main()
