
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from world_model_common import (
    ConditionalCVAE, ConditionalDenoiser, DiffusionNoiseScheduler,
    StandardNormalizer, TrainingConfig, fit_cvae_model,
    fit_diffusion_model, sample_with_diffusion,
)


@dataclass
class EVGeneratorConfig:
    sessions_path: Path = Path("processed_ev/ev_sessions_unified.csv")
    save_path: Path = Path("processed_ev/ev_generator.pkl")
    random_seed: int = 42
    min_samples_per_group: int = 50

    z_dim: int = 16
    hidden_dims: tuple[int, ...] = (128, 128)
    diffusion_hidden_dims: tuple[int, ...] = (128, 128)
    diffusion_steps: int = 30
    cvae_epochs: int = 50
    diffusion_epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    device: str = "cpu"

    # default battery context used for PINN data scaffolding
    default_battery_capacity_kwh: float = 60.0
    default_max_charge_power_kw: float = 11.0
    default_target_soc: float = 0.80
    min_init_soc: float = 0.05
    max_init_soc: float = 0.90
    default_temperature_c: float = 25.0


class EVGenerator:
    """
    Interface-compatible EV generator.

    Public API preserved:
      - fit()
      - save()
      - load()
      - load_data()
      - sample_sessions_for_horizon(...)

    Added for PINN dataset construction:
      - build_pinn_session_context(...)
      - session-level battery and risk context columns
    """
    def __init__(self, config: EVGeneratorConfig):
        self.config = config
        self.count_table: pd.DataFrame | None = None

        # generated EV session attributes
        self.feature_cols = ["arrival_hour", "parking_duration_hours", "energy_delivered_kwh"]

        # CVAE condition / diffusion risk columns
        self.cond_cols = [
            "source_code", "is_weekend_num", "month", "hour_bin",
            "hour_sin", "hour_cos", "month_sin", "month_cos"
        ]
        self.risk_cols = ["risk_lambda", "risk_peak_hour", "risk_month_end"]

        self.source_to_code: Dict[str, int] = {}
        self.code_to_source: Dict[int, str] = {}

        self.x_norm = StandardNormalizer()
        self.c_norm = StandardNormalizer()
        self.r_norm = StandardNormalizer()

        self.cvae = ConditionalCVAE(
            len(self.feature_cols),
            len(self.cond_cols),
            z_dim=config.z_dim,
            hidden_dims=list(config.hidden_dims),
        )
        self.denoiser = ConditionalDenoiser(
            len(self.feature_cols),
            len(self.cond_cols),
            len(self.risk_cols),
            hidden_dims=list(config.diffusion_hidden_dims),
        )
        self.scheduler = DiffusionNoiseScheduler(timesteps=config.diffusion_steps)
        self.rng = np.random.default_rng(config.random_seed)

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.sessions_path, low_memory=False).copy()

        required = [
            "source", "date", "month", "day_of_week", "is_weekend",
            "arrival_hour", "parking_duration_hours", "energy_delivered_kwh",
            "arrival_time", "departure_time"
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required EV columns: {missing}")

        for col in self.feature_cols + ["month", "day_of_week"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # robust weekend parsing
        if df["is_weekend"].dtype == bool:
            pass
        else:
            mapped = (
                df["is_weekend"].astype(str).str.strip().str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
            )
            df["is_weekend"] = mapped.fillna(False).astype(bool)

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
        df["arrival_time"] = pd.to_datetime(df["arrival_time"], errors="coerce")
        df["departure_time"] = pd.to_datetime(df["departure_time"], errors="coerce")

        df = df.dropna(
            subset=["source", "date", "month", "is_weekend", "arrival_hour", "parking_duration_hours", "energy_delivered_kwh"]
        )
        df = df[(df["arrival_hour"] >= 0) & (df["arrival_hour"] <= 24)]
        df = df[(df["parking_duration_hours"] > 0) & (df["parking_duration_hours"] <= 72)]
        df = df[(df["energy_delivered_kwh"] > 0) & (df["energy_delivered_kwh"] <= 200)]
        return df.reset_index(drop=True)

    def fit_count_model(self, df: pd.DataFrame) -> None:
        tmp = df.copy()
        tmp["hour_bin"] = np.floor(tmp["arrival_hour"]).clip(0, 23).astype(int)

        daily_hour_counts = (
            tmp.groupby(["source", "is_weekend", "month", "date", "hour_bin"])
            .size().reset_index(name="n_sessions")
        )
        count_table = (
            daily_hour_counts.groupby(["source", "is_weekend", "month", "hour_bin"])["n_sessions"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "lambda_hour"})
        )

        full_idx = pd.MultiIndex.from_product(
            [
                sorted(tmp["source"].dropna().astype(str).unique()),
                [False, True],
                sorted(tmp["month"].dropna().astype(int).unique()),
                list(range(24)),
            ],
            names=["source", "is_weekend", "month", "hour_bin"],
        )
        count_table = (
            count_table.set_index(["source", "is_weekend", "month", "hour_bin"])
            .reindex(full_idx)
            .reset_index()
        )

        fallback = (
            daily_hour_counts.groupby(["source", "is_weekend", "hour_bin"])["n_sessions"]
            .mean().reset_index().rename(columns={"n_sessions": "fallback_lambda"})
        )
        count_table = count_table.merge(
            fallback, on=["source", "is_weekend", "hour_bin"], how="left"
        )
        count_table["lambda_hour"] = (
            count_table["lambda_hour"].fillna(count_table["fallback_lambda"]).fillna(0.0)
        )
        self.count_table = count_table[["source", "is_weekend", "month", "hour_bin", "lambda_hour"]]

    def _fit_source_mapping(self, df: pd.DataFrame) -> None:
        sources = sorted(df["source"].astype(str).dropna().unique().tolist())
        self.source_to_code = {s: i for i, s in enumerate(sources)}
        self.code_to_source = {i: s for s, i in self.source_to_code.items()}

    def _build_training_arrays(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        work = df.copy()
        work["hour_bin"] = np.floor(work["arrival_hour"]).clip(0, 23).astype(int)
        work["source_code"] = work["source"].map(self.source_to_code).astype(float)
        work["is_weekend_num"] = work["is_weekend"].astype(int).astype(float)
        work["hour_sin"] = np.sin(2 * np.pi * work["hour_bin"] / 24.0)
        work["hour_cos"] = np.cos(2 * np.pi * work["hour_bin"] / 24.0)
        work["month_sin"] = np.sin(2 * np.pi * work["month"] / 12.0)
        work["month_cos"] = np.cos(2 * np.pi * work["month"] / 12.0)

        lam = self.count_table.rename(columns={"lambda_hour": "risk_lambda"})
        work = work.merge(
            lam, on=["source", "is_weekend", "month", "hour_bin"], how="left"
        )
        work["risk_lambda"] = work["risk_lambda"].fillna(0.0)
        work["risk_peak_hour"] = work["hour_bin"].isin([17, 18, 19, 20, 21]).astype(float)
        work["risk_month_end"] = work["month"].isin([7, 8, 12]).astype(float)

        x = work[self.feature_cols].values.astype(np.float32)
        c = work[self.cond_cols].values.astype(np.float32)
        r = work[self.risk_cols].values.astype(np.float32)
        return x, c, r

    def fit_world_model(self, df: pd.DataFrame) -> None:
        self._fit_source_mapping(df)
        x, c, r = self._build_training_arrays(df)

        self.x_norm.fit(x)
        self.c_norm.fit(c)
        self.r_norm.fit(r)

        x_n = self.x_norm.transform(x)
        c_n = self.c_norm.transform(c)
        r_n = self.r_norm.transform(r)

        cfg = TrainingConfig(
            batch_size=self.config.batch_size,
            lr=self.config.lr,
            epochs=self.config.cvae_epochs,
            diffusion_epochs=self.config.diffusion_epochs,
            device=self.config.device,
            seed=self.config.random_seed,
        )
        fit_cvae_model(self.cvae, x_n, c_n, cfg)

        with torch.no_grad():
            base_x = self.cvae.sample(torch.tensor(c_n, dtype=torch.float32)).cpu().numpy()

        fit_diffusion_model(
            self.denoiser, self.scheduler, x_n, base_x, c_n, r_n, cfg
        )

    def fit(self) -> None:
        df = self.load_data()
        self.fit_count_model(df)
        self.fit_world_model(df)

    def save(self) -> None:
        payload = {
            "config": vars(self.config),
            "count_table": self.count_table,
            "feature_cols": self.feature_cols,
            "cond_cols": self.cond_cols,
            "risk_cols": self.risk_cols,
            "source_to_code": self.source_to_code,
            "code_to_source": self.code_to_source,
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
    def load(cls, path: Path) -> "EVGenerator":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        config = EVGeneratorConfig(**payload["config"])
        obj = cls(config)
        obj.count_table = payload["count_table"]
        obj.feature_cols = payload["feature_cols"]
        obj.cond_cols = payload["cond_cols"]
        obj.risk_cols = payload["risk_cols"]
        obj.source_to_code = payload["source_to_code"]
        obj.code_to_source = payload["code_to_source"]
        obj.x_norm.load_state_dict(payload["x_norm"])
        obj.c_norm.load_state_dict(payload["c_norm"])
        obj.r_norm.load_state_dict(payload["r_norm"])
        obj.cvae.load_state_dict(payload["cvae_state"])
        obj.denoiser.load_state_dict(payload["denoiser_state"])
        return obj

    def _get_lambda(self, source: str, is_weekend: bool, month: int, hour_bin: int) -> float:
        row = self.count_table[
            (self.count_table["source"] == source)
            & (self.count_table["is_weekend"] == is_weekend)
            & (self.count_table["month"] == month)
            & (self.count_table["hour_bin"] == hour_bin)
        ]
        return 0.0 if len(row) == 0 else float(row.iloc[0]["lambda_hour"])

    def _build_sample_condition(self, source: str, is_weekend: bool, month: int, hour_bin: int):
        source_code = float(self.source_to_code.get(source, 0))
        c = np.array([[
            source_code,
            float(is_weekend),
            float(month),
            float(hour_bin),
            np.sin(2 * np.pi * hour_bin / 24.0),
            np.cos(2 * np.pi * hour_bin / 24.0),
            np.sin(2 * np.pi * month / 12.0),
            np.cos(2 * np.pi * month / 12.0),
        ]], dtype=np.float32)

        lam = self._get_lambda(source, is_weekend, month, hour_bin)
        r = np.array([[
            lam,
            float(hour_bin in [17, 18, 19, 20, 21]),
            float(month in [7, 8, 12]),
        ]], dtype=np.float32)
        return c, r

    def _make_temperature_map(
        self,
        temperature_by_hour: Optional[pd.DataFrame | pd.Series | Dict[Any, float]]
    ) -> Optional[pd.DataFrame]:
        if temperature_by_hour is None:
            return None
        if isinstance(temperature_by_hour, pd.DataFrame):
            temp_df = temperature_by_hour.copy()
            if "timestamp_hour" not in temp_df.columns:
                if "timestamp" in temp_df.columns:
                    temp_df = temp_df.rename(columns={"timestamp": "timestamp_hour"})
                else:
                    raise KeyError("temperature_by_hour DataFrame must contain 'timestamp_hour' or 'timestamp'.")
            if "temperature_c" not in temp_df.columns:
                raise KeyError("temperature_by_hour DataFrame must contain 'temperature_c'.")
            temp_df["timestamp_hour"] = pd.to_datetime(temp_df["timestamp_hour"], errors="coerce").dt.floor("h")
            return temp_df[["timestamp_hour", "temperature_c"]].drop_duplicates("timestamp_hour")
        if isinstance(temperature_by_hour, pd.Series):
            temp_df = temperature_by_hour.rename("temperature_c").reset_index()
            temp_df.columns = ["timestamp_hour", "temperature_c"]
            temp_df["timestamp_hour"] = pd.to_datetime(temp_df["timestamp_hour"], errors="coerce").dt.floor("h")
            return temp_df
        if isinstance(temperature_by_hour, dict):
            temp_df = pd.DataFrame({
                "timestamp_hour": list(temperature_by_hour.keys()),
                "temperature_c": list(temperature_by_hour.values()),
            })
            temp_df["timestamp_hour"] = pd.to_datetime(temp_df["timestamp_hour"], errors="coerce").dt.floor("h")
            return temp_df
        raise TypeError("Unsupported temperature_by_hour type.")

    def build_pinn_session_context(
        self,
        ev_sessions: pd.DataFrame,
        temperature_by_hour: Optional[pd.DataFrame | pd.Series | Dict[Any, float]] = None,
        battery_capacity_kwh: Optional[float] = None,
        max_charge_power_kw: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Add battery- and PINN-related context columns to EV sessions.

        This method does not create labels. It creates the context rows
        that can later be joined with stochastic optimization outputs and
        teacher-model degradation labels.
        """
        base_columns = [
            "source", "timestamp_hour", "session_id", "arrival_time", "departure_time",
            "arrival_hour", "parking_duration_hours", "energy_delivered_kwh",
            "month", "is_weekend",
        ]
        if ev_sessions.empty:
            empty = pd.DataFrame(columns=base_columns)
            for col in [
                "source_code", "is_weekend_num", "hour_bin", "hour_sin", "hour_cos",
                "month_sin", "month_cos", "risk_lambda", "risk_peak_hour", "risk_month_end",
                "battery_capacity_kwh", "max_charge_power_kw", "estimated_init_soc",
                "estimated_target_soc", "requested_avg_charge_power_kw", "requested_c_rate",
                "delta_soc_requested", "required_energy_remaining_kwh", "feasible_charge_hours",
                "charging_slack_hours", "temperature_c", "charge_power_kw", "soc", "soh",
                "cumulative_throughput_kwh", "delta_soh_true", "soh_next", "sample_type"
            ]:
                empty[col] = pd.Series(dtype="float64" if col not in {"source", "session_id", "sample_type"} else "object")
            return empty

        df = ev_sessions.copy()
        df["timestamp_hour"] = pd.to_datetime(df["timestamp_hour"], errors="coerce").dt.floor("h")

        df["hour_bin"] = np.floor(df["arrival_hour"]).clip(0, 23).astype(int)
        df["source_code"] = df["source"].map(self.source_to_code).fillna(0).astype(float)
        df["is_weekend_num"] = df["is_weekend"].astype(int).astype(float)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_bin"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_bin"] / 24.0)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

        lam = self.count_table.rename(columns={"lambda_hour": "risk_lambda"})
        df = df.merge(
            lam, on=["source", "is_weekend", "month", "hour_bin"], how="left"
        )
        df["risk_lambda"] = df["risk_lambda"].fillna(0.0)
        df["risk_peak_hour"] = df["hour_bin"].isin([17, 18, 19, 20, 21]).astype(float)
        df["risk_month_end"] = df["month"].isin([7, 8, 12]).astype(float)

        cap = float(self.config.default_battery_capacity_kwh if battery_capacity_kwh is None else battery_capacity_kwh)
        pmax = float(self.config.default_max_charge_power_kw if max_charge_power_kw is None else max_charge_power_kw)

        energy = df["energy_delivered_kwh"].astype(float).values
        parking = np.maximum(df["parking_duration_hours"].astype(float).values, 1e-6)
        delta_soc_requested = np.clip(energy / max(cap, 1e-6), 0.0, 1.0)
        estimated_init_soc = np.clip(
            self.config.default_target_soc - delta_soc_requested,
            self.config.min_init_soc,
            self.config.max_init_soc,
        )
        estimated_target_soc = np.clip(
            estimated_init_soc + delta_soc_requested,
            estimated_init_soc,
            0.98,
        )
        requested_avg_power = energy / parking
        requested_c_rate = requested_avg_power / max(cap, 1e-6)
        feasible_charge_hours = energy / max(pmax, 1e-6)
        charging_slack_hours = np.maximum(parking - feasible_charge_hours, 0.0)

        df["battery_capacity_kwh"] = cap
        df["max_charge_power_kw"] = pmax
        df["estimated_init_soc"] = estimated_init_soc
        df["estimated_target_soc"] = estimated_target_soc
        df["requested_avg_charge_power_kw"] = requested_avg_power
        df["requested_c_rate"] = requested_c_rate
        df["delta_soc_requested"] = delta_soc_requested
        df["required_energy_remaining_kwh"] = energy
        df["feasible_charge_hours"] = feasible_charge_hours
        df["charging_slack_hours"] = charging_slack_hours

        temp_df = self._make_temperature_map(temperature_by_hour)
        if temp_df is not None:
            df = df.merge(temp_df, on="timestamp_hour", how="left")
            df["temperature_c"] = df["temperature_c"].fillna(self.config.default_temperature_c)
        else:
            df["temperature_c"] = float(self.config.default_temperature_c)

        # placeholders to be filled by stochastic optimization and teacher labeling
        df["charge_power_kw"] = np.nan
        df["soc"] = np.nan
        df["soh"] = np.nan
        df["cumulative_throughput_kwh"] = np.nan
        df["delta_soh_true"] = np.nan
        df["soh_next"] = np.nan
        df["sample_type"] = "joint_session_context"
        return df

    def sample_sessions_for_horizon(
        self,
        start_time: str,
        hours: int = 24,
        source: str = "acn",
        tail_strength: float = 0.6,
        use_tail_enhancement: bool = True,
        include_pinn_context: bool = False,
        temperature_by_hour: Optional[pd.DataFrame | pd.Series | Dict[Any, float]] = None,
        battery_capacity_kwh: Optional[float] = None,
        max_charge_power_kw: Optional[float] = None,
    ) -> pd.DataFrame:
        start_ts = pd.Timestamp(start_time)
        rows: List[Dict[str, Any]] = []

        self.cvae.to(self.config.device)
        self.denoiser.to(self.config.device)
        self.scheduler.to(torch.device(self.config.device))

        for h in range(hours):
            current_ts = start_ts + pd.Timedelta(hours=h)
            month = int(current_ts.month)
            hour_bin = int(current_ts.hour)
            is_weekend = current_ts.dayofweek >= 5

            lam = self._get_lambda(source, is_weekend, month, hour_bin)
            n_sessions = self.rng.poisson(lam)
            if n_sessions == 0:
                continue

            c_raw, r_raw = self._build_sample_condition(source, is_weekend, month, hour_bin)
            c_n = self.c_norm.transform(np.repeat(c_raw, n_sessions, axis=0))
            r_n = self.r_norm.transform(np.repeat(r_raw, n_sessions, axis=0))

            with torch.no_grad():
                c_tensor = torch.tensor(c_n, dtype=torch.float32, device=self.config.device)
                base_x_n = self.cvae.sample(c_tensor)
                if use_tail_enhancement:
                    r_tensor = torch.tensor(r_n, dtype=torch.float32, device=self.config.device)
                    x_n = sample_with_diffusion(
                        self.denoiser, self.scheduler, base_x_n, c_tensor, r_tensor,
                        tail_strength=tail_strength
                    ).cpu().numpy()
                else:
                    x_n = base_x_n.cpu().numpy()

            samples = self.x_norm.inverse_transform(x_n)

            for i, s in enumerate(samples):
                arrival_hour, parking_h, energy_kwh = s.tolist()
                arrival_hour = float(np.clip(arrival_hour, hour_bin, min(hour_bin + 1, 24)))
                parking_h = float(np.clip(parking_h, 0.25, 72.0))
                energy_kwh = float(np.clip(energy_kwh, 0.5, 200.0))
                minute_part = max(0.0, min(59.0, (arrival_hour - int(arrival_hour)) * 60.0))
                arrival_time = current_ts.floor("h") + pd.Timedelta(minutes=minute_part)
                departure_time = arrival_time + pd.Timedelta(hours=parking_h)

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
                    "is_weekend": is_weekend,
                })

        out = pd.DataFrame(rows)
        if include_pinn_context:
            out = self.build_pinn_session_context(
                out,
                temperature_by_hour=temperature_by_hour,
                battery_capacity_kwh=battery_capacity_kwh,
                max_charge_power_kw=max_charge_power_kw,
            )
        return out


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
    real_df = gen.load_data()
    real_df = real_df[real_df["source"] == "acn"].copy()

    sampled = gen.sample_sessions_for_horizon(
        start_time="2025-10-01 00:00:00",
        hours=24 * 7,
        source="acn",
    )
    sample_path = Path("processed_ev/ev_generator_sample.csv")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(sample_path, index=False)

    fig_path = Path("processed_ev/ev_generator_samples_plot.png")
    plot_generated_samples(real_df, sampled, fig_path)

    print(f"[ok] Sampled sessions saved to: {sample_path}")
    print(f"[ok] Comparison plot saved to: {fig_path}")


if __name__ == "__main__":
    main()
