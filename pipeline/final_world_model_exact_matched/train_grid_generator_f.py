
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from world_model_common import (
    ConditionalCVAE, ConditionalDenoiser, DiffusionNoiseScheduler,
    StandardNormalizer, TrainingConfig, fit_cvae_model,
    fit_diffusion_model, sample_with_diffusion,
)


@dataclass
class GridGeneratorConfig:
    data_path: Path = Path("processed_grid/grid_generator_ready.csv")
    save_path: Path = Path("processed_grid/grid_generator.pkl")
    metrics_path: Path = Path("processed_grid/grid_generator_metrics.csv")
    importance_path: Path = Path("processed_grid/grid_generator_feature_importance.csv")
    val_pred_path: Path = Path("processed_grid/grid_generator_val_predictions.csv")
    sample_path: Path = Path("processed_grid/grid_generator_sample.csv")

    random_seed: int = 42
    val_ratio: float = 0.2
    target_cols: tuple[str, ...] = (
        "system_load_mw_next1",
        "weighted_lbmp_mwh_next1",
        "weighted_imer_co2_ton_per_mwh_next1",
        "congestion_intensity_next1",
    )

    z_dim: int = 16
    hidden_dims: tuple[int, ...] = (128, 128)
    diffusion_hidden_dims: tuple[int, ...] = (128, 128)
    diffusion_steps: int = 30
    cvae_epochs: int = 50
    diffusion_epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    device: str = "cpu"

    # proxy temperature for PINN context if weather data are unavailable
    temperature_mean_c: float = 15.0
    temperature_season_amp_c: float = 12.0
    temperature_diurnal_amp_c: float = 4.0


class GridGenerator:
    """
    Interface-compatible grid generator.

    Public API preserved:
      - fit()
      - save()
      - load()
      - load_data()
      - sample_trajectory(...)

    Added for PINN dataset construction:
      - add_pinn_context_to_trajectory(...)
      - build_pinn_hourly_context(...)
      - temperature proxy and grid stress columns
    """
    def __init__(self, config: GridGeneratorConfig):
        self.config = config
        self.feature_cols: List[str] = []
        self.residual_std: Dict[str, float] = {}
        self.rng = np.random.default_rng(config.random_seed)

        self.x_norm = StandardNormalizer()
        self.c_norm = StandardNormalizer()
        self.r_norm = StandardNormalizer()

        self.cvae: Optional[ConditionalCVAE] = None
        self.denoiser: Optional[ConditionalDenoiser] = None
        self.scheduler = DiffusionNoiseScheduler(timesteps=config.diffusion_steps)

    def _init_models(self) -> None:
        if not self.feature_cols:
            raise RuntimeError("feature_cols must be initialized before model creation.")
        self.cvae = ConditionalCVAE(
            len(self.config.target_cols),
            len(self.feature_cols),
            z_dim=self.config.z_dim,
            hidden_dims=list(self.config.hidden_dims),
        )
        self.denoiser = ConditionalDenoiser(
            len(self.config.target_cols),
            len(self.feature_cols),
            3,
            hidden_dims=list(self.config.diffusion_hidden_dims),
        )

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.data_path, low_memory=False)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

        for c in df.columns:
            if c != "timestamp":
                df[c] = pd.to_numeric(df[c], errors="coerce")

        required = [
            "timestamp",
            "system_load_mw", "weighted_lbmp_mwh", "weighted_imer_co2_ton_per_mwh", "congestion_intensity",
            "hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
            "system_load_mw_lag1", "weighted_lbmp_mwh_lag1", "weighted_imer_co2_ton_per_mwh_lag1", "congestion_intensity_lag1",
            "system_load_mw_lag2", "weighted_lbmp_mwh_lag2", "weighted_imer_co2_ton_per_mwh_lag2", "congestion_intensity_lag2",
            "system_load_mw_lag24", "weighted_lbmp_mwh_lag24", "weighted_imer_co2_ton_per_mwh_lag24", "congestion_intensity_lag24",
            *self.config.target_cols,
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required grid columns: {missing}")

        exclude = {"timestamp", *self.config.target_cols}
        self.feature_cols = [c for c in df.columns if c not in exclude]
        keep_cols = ["timestamp"] + self.feature_cols + list(self.config.target_cols)
        df = df[keep_cols].dropna().reset_index(drop=True)

        # Important: do not overwrite loaded weights when the object has already been restored.
        if self.cvae is None or self.denoiser is None:
            self._init_models()
        return df

    def time_split(self, df: pd.DataFrame):
        split_idx = int((1 - self.config.val_ratio) * len(df))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    def _build_risk(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            df["congestion_intensity"].values.astype(np.float32),
            df["weighted_lbmp_mwh"].values.astype(np.float32),
            df["system_load_mw"].values.astype(np.float32),
        ]).astype(np.float32)

    def fit(self):
        df = self.load_data()
        train_df, val_df = self.time_split(df)

        X_train = train_df[self.feature_cols].values.astype(np.float32)
        Y_train = train_df[list(self.config.target_cols)].values.astype(np.float32)
        R_train = self._build_risk(train_df)

        X_val = val_df[self.feature_cols].values.astype(np.float32)
        Y_val = val_df[list(self.config.target_cols)].values.astype(np.float32)

        self.c_norm.fit(X_train)
        self.x_norm.fit(Y_train)
        self.r_norm.fit(R_train)

        c_train = self.c_norm.transform(X_train)
        x_train = self.x_norm.transform(Y_train)
        r_train = self.r_norm.transform(R_train)

        cfg = TrainingConfig(
            batch_size=self.config.batch_size,
            lr=self.config.lr,
            epochs=self.config.cvae_epochs,
            diffusion_epochs=self.config.diffusion_epochs,
            device=self.config.device,
            seed=self.config.random_seed,
        )
        fit_cvae_model(self.cvae, x_train, c_train, cfg)

        with torch.no_grad():
            base_x = self.cvae.sample(torch.tensor(c_train, dtype=torch.float32)).cpu().numpy()

        fit_diffusion_model(
            self.denoiser, self.scheduler, x_train, base_x, c_train, r_train, cfg
        )

        # compatibility outputs
        pred_df = val_df[["timestamp"]].copy()
        c_val = self.c_norm.transform(X_val)
        with torch.no_grad():
            y_pred = self.x_norm.inverse_transform(
                self.cvae.sample(torch.tensor(c_val, dtype=torch.float32)).cpu().numpy()
            )

        metrics_rows = []
        base_x_un = self.x_norm.inverse_transform(base_x)
        for i, target in enumerate(self.config.target_cols):
            y_true = Y_val[:, i]
            yp = y_pred[:, i]
            resid = Y_train[:, i] - base_x_un[:, i]
            self.residual_std[target] = max(float(np.std(resid)), 1e-8)

            metrics_rows.append({
                "target": target,
                "rmse": float(np.sqrt(mean_squared_error(y_true, yp))),
                "mae": float(mean_absolute_error(y_true, yp)),
                "r2": float(r2_score(y_true, yp)),
                "train_residual_std": self.residual_std[target],
            })
            pred_df[f"{target}_true"] = y_true
            pred_df[f"{target}_pred"] = yp

        metrics_df = pd.DataFrame(metrics_rows)
        importance_df = pd.DataFrame({
            "target": np.repeat(list(self.config.target_cols), len(self.feature_cols)),
            "feature": self.feature_cols * len(self.config.target_cols),
            "importance": 1.0 / max(1, len(self.feature_cols)),
        })

        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(self.config.val_pred_path, index=False)
        metrics_df.to_csv(self.config.metrics_path, index=False)
        importance_df.to_csv(self.config.importance_path, index=False)
        return metrics_df, importance_df, pred_df

    def save(self):
        payload = {
            "config": vars(self.config),
            "feature_cols": self.feature_cols,
            "residual_std": self.residual_std,
            "x_norm": self.x_norm.state_dict(),
            "c_norm": self.c_norm.state_dict(),
            "r_norm": self.r_norm.state_dict(),
            "cvae_state": self.cvae.state_dict(),
            "denoiser_state": self.denoiser.state_dict(),
        }
        self.config.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.save_path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "GridGenerator":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        config = GridGeneratorConfig(**payload["config"])
        obj = cls(config)
        obj.feature_cols = payload["feature_cols"]
        obj.residual_std = payload["residual_std"]
        obj._init_models()
        obj.x_norm.load_state_dict(payload["x_norm"])
        obj.c_norm.load_state_dict(payload["c_norm"])
        obj.r_norm.load_state_dict(payload["r_norm"])
        obj.cvae.load_state_dict(payload["cvae_state"])
        obj.denoiser.load_state_dict(payload["denoiser_state"])
        return obj

    def _predict_one_step_vector(
        self,
        row: pd.Series,
        stochastic: bool = True,
        tail_strength: float = 0.5,
    ) -> Dict[str, float]:
        x = row[self.feature_cols].to_frame().T.values.astype(np.float32)
        c = self.c_norm.transform(x)
        r = self.r_norm.transform(self._build_risk(pd.DataFrame([row])))

        self.cvae.to(self.config.device)
        self.denoiser.to(self.config.device)
        self.scheduler.to(torch.device(self.config.device))

        with torch.no_grad():
            c_tensor = torch.tensor(c, dtype=torch.float32, device=self.config.device)
            base_x = self.cvae.sample(c_tensor)
            if stochastic:
                r_tensor = torch.tensor(r, dtype=torch.float32, device=self.config.device)
                x_next_n = sample_with_diffusion(
                    self.denoiser, self.scheduler, base_x, c_tensor, r_tensor,
                    tail_strength=tail_strength
                ).cpu().numpy()
            else:
                x_next_n = base_x.cpu().numpy()

        x_next = self.x_norm.inverse_transform(x_next_n)[0]
        out = {target: float(x_next[i]) for i, target in enumerate(self.config.target_cols)}
        out["system_load_mw_next1"] = max(out["system_load_mw_next1"], 0.0)
        out["weighted_lbmp_mwh_next1"] = max(out["weighted_lbmp_mwh_next1"], 0.0)
        out["weighted_imer_co2_ton_per_mwh_next1"] = max(out["weighted_imer_co2_ton_per_mwh_next1"], 0.0)
        out["congestion_intensity_next1"] = max(out["congestion_intensity_next1"], 0.0)
        return out

    def _estimate_temperature_c(self, timestamps: pd.Series) -> np.ndarray:
        ts = pd.to_datetime(timestamps, errors="coerce")
        month = ts.dt.month.astype(float).values
        hour = ts.dt.hour.astype(float).values

        seasonal = self.config.temperature_mean_c + self.config.temperature_season_amp_c * np.cos(
            2 * np.pi * (month - 7.0) / 12.0
        )
        diurnal = self.config.temperature_diurnal_amp_c * np.cos(
            2 * np.pi * (hour - 15.0) / 24.0
        )
        return (seasonal + diurnal).astype(np.float32)

    @staticmethod
    def _minmax01(x: pd.Series) -> pd.Series:
        xmin = float(x.min())
        xmax = float(x.max())
        if abs(xmax - xmin) < 1e-8:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - xmin) / (xmax - xmin)

    def add_pinn_context_to_trajectory(self, grid_hourly: pd.DataFrame) -> pd.DataFrame:
        """
        Add grid-side context fields that are useful for PINN training.
        """
        out = grid_hourly.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

        if "hour" not in out.columns:
            out["hour"] = out["timestamp"].dt.hour
        if "day_of_week" not in out.columns:
            out["day_of_week"] = out["timestamp"].dt.dayofweek
        if "month" not in out.columns:
            out["month"] = out["timestamp"].dt.month
        if "is_weekend" not in out.columns:
            out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

        if "hour_sin" not in out.columns:
            out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
        if "hour_cos" not in out.columns:
            out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
        if "month_sin" not in out.columns:
            out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
        if "month_cos" not in out.columns:
            out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
        if "dow_sin" not in out.columns:
            out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
        if "dow_cos" not in out.columns:
            out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)

        if "temperature_c" not in out.columns:
            out["temperature_c"] = self._estimate_temperature_c(out["timestamp"])
            out["temperature_is_proxy"] = 1
        else:
            out["temperature_is_proxy"] = 0

        out["is_peak_hour"] = out["hour"].isin([17, 18, 19, 20, 21]).astype(int)
        out["price_emission_score"] = out["weighted_lbmp_mwh"] * out["weighted_imer_co2_ton_per_mwh"]

        out["high_price_flag"] = (
            out["weighted_lbmp_mwh"] >= out["weighted_lbmp_mwh"].quantile(0.75)
        ).astype(int)
        out["high_congestion_flag"] = (
            out["congestion_intensity"] >= out["congestion_intensity"].quantile(0.75)
        ).astype(int)
        out["high_load_flag"] = (
            out["system_load_mw"] >= out["system_load_mw"].quantile(0.75)
        ).astype(int)

        load_norm = self._minmax01(out["system_load_mw"])
        price_norm = self._minmax01(out["weighted_lbmp_mwh"])
        congestion_norm = self._minmax01(out["congestion_intensity"])
        out["grid_risk_score"] = ((load_norm + price_norm + congestion_norm) / 3.0).astype(float)
        return out

    def build_pinn_hourly_context(
        self,
        grid_hourly: pd.DataFrame,
        scenario_id: Optional[int] = None,
        default_battery_capacity_kwh: float = 60.0,
    ) -> pd.DataFrame:
        out = self.add_pinn_context_to_trajectory(grid_hourly)
        if scenario_id is not None:
            out["scenario_id"] = int(scenario_id)
        out["battery_capacity_kwh"] = float(default_battery_capacity_kwh)

        # placeholders to be filled after stochastic optimization and teacher labeling
        out["charge_power_kw"] = np.nan
        out["soc"] = np.nan
        out["soh"] = np.nan
        out["cumulative_throughput_kwh"] = np.nan
        out["delta_soh_true"] = np.nan
        out["soh_next"] = np.nan
        out["sample_type"] = "joint_hourly_context"
        return out

    def sample_trajectory(
        self,
        seed_row: pd.Series,
        horizon: int = 24,
        stochastic: bool = True,
        tail_strength: float = 0.6,
    ) -> pd.DataFrame:
        current = seed_row.copy()
        rows = []

        for _ in range(horizon):
            ts = pd.Timestamp(current["timestamp"]) + pd.Timedelta(hours=1)
            preds = self._predict_one_step_vector(
                current, stochastic=stochastic, tail_strength=tail_strength
            )

            next_row = current.copy()
            next_row["timestamp"] = ts
            next_row["system_load_mw"] = preds["system_load_mw_next1"]
            next_row["weighted_lbmp_mwh"] = preds["weighted_lbmp_mwh_next1"]
            next_row["weighted_imer_co2_ton_per_mwh"] = preds["weighted_imer_co2_ton_per_mwh_next1"]
            next_row["congestion_intensity"] = preds["congestion_intensity_next1"]

            next_row["hour"] = ts.hour
            next_row["day_of_week"] = ts.dayofweek
            next_row["month"] = ts.month
            next_row["is_weekend"] = int(ts.dayofweek >= 5)
            next_row["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24.0)
            next_row["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24.0)
            next_row["month_sin"] = np.sin(2 * np.pi * ts.month / 12.0)
            next_row["month_cos"] = np.cos(2 * np.pi * ts.month / 12.0)
            next_row["dow_sin"] = np.sin(2 * np.pi * ts.dayofweek / 7.0)
            next_row["dow_cos"] = np.cos(2 * np.pi * ts.dayofweek / 7.0)

            for base in [
                "system_load_mw", "weighted_lbmp_mwh",
                "weighted_imer_co2_ton_per_mwh", "congestion_intensity"
            ]:
                next_row[f"{base}_lag1"] = current.get(base, np.nan)
                next_row[f"{base}_lag2"] = current.get(f"{base}_lag1", np.nan)
                next_row[f"{base}_lag24"] = current.get(f"{base}_lag23", current.get(f"{base}_lag24", np.nan))

            rows.append({
                "timestamp": ts,
                "system_load_mw": next_row["system_load_mw"],
                "weighted_lbmp_mwh": next_row["weighted_lbmp_mwh"],
                "weighted_imer_co2_ton_per_mwh": next_row["weighted_imer_co2_ton_per_mwh"],
                "congestion_intensity": next_row["congestion_intensity"],
                "hour": next_row["hour"],
                "day_of_week": next_row["day_of_week"],
                "month": next_row["month"],
                "is_weekend": next_row["is_weekend"],
                "hour_sin": next_row["hour_sin"],
                "hour_cos": next_row["hour_cos"],
                "month_sin": next_row["month_sin"],
                "month_cos": next_row["month_cos"],
                "dow_sin": next_row["dow_sin"],
                "dow_cos": next_row["dow_cos"],
            })

            for base in [
                "system_load_mw", "weighted_lbmp_mwh",
                "weighted_imer_co2_ton_per_mwh", "congestion_intensity"
            ]:
                for k in range(23, 1, -1):
                    prev = f"{base}_lag{k-1}"
                    cur = f"{base}_lag{k}"
                    if prev in current.index:
                        next_row[cur] = current[prev]

            current = next_row

        out = pd.DataFrame(rows)
        return self.add_pinn_context_to_trajectory(out)


def plot_generated_grid_samples(
    real_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    save_path: Path,
    history_hours: int = 24,
) -> None:
    real_df = real_df.copy()
    sampled_df = sampled_df.copy()
    real_df["timestamp"] = pd.to_datetime(real_df["timestamp"], errors="coerce")
    sampled_df["timestamp"] = pd.to_datetime(sampled_df["timestamp"], errors="coerce")
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
        ax.plot(real_plot["timestamp"], real_plot[col], label="Recent Real", linewidth=2)
        gen_times = sampled_list[0]["timestamp"]
        gen_matrix = np.vstack([pd.to_numeric(df[col], errors="coerce").values for df in sampled_list])
        gen_mean = np.nanmean(gen_matrix, axis=0)
        gen_p10 = np.nanpercentile(gen_matrix, 10, axis=0)
        gen_p90 = np.nanpercentile(gen_matrix, 90, axis=0)
        for df in sampled_list:
            ax.plot(df["timestamp"], df[col], alpha=0.25, linewidth=1)
        ax.fill_between(gen_times, gen_p10, gen_p90, alpha=0.25, label="P10-P90 band")
        ax.plot(gen_times, gen_mean, linewidth=2.5, label="Generated Mean")
        ax.set_title(title)
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
    data = gen.load_data()
    seed_row = data.iloc[-1].copy()

    sampled_list = []
    for i in range(10):
        sample_df = gen.sample_trajectory(seed_row=seed_row, horizon=24, stochastic=True)
        sample_df["sample_id"] = i
        sampled_list.append(sample_df)

    all_samples = pd.concat(sampled_list, ignore_index=True)
    config.sample_path.parent.mkdir(parents=True, exist_ok=True)
    all_samples.to_csv(config.sample_path, index=False)

    plot_generated_grid_uncertainty(
        data, sampled_list, Path("processed_grid/grid_generator_uncertainty_plot.png")
    )
    plot_generated_grid_samples(
        data, sampled_list[-1], Path("processed_grid/grid_generator_samples_plot.png")
    )

    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
