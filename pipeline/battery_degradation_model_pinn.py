#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
battery_degradation_model_pinn.py

Physics-informed battery degradation model for EV charging optimization.

This version upgrades the previous semi-empirical surrogate into a PINN-style model.
The model combines:
1) A mechanistic degradation backbone
2) A neural residual term
3) Physics-informed constraints in the training loss

Main state equations
--------------------
SOC_{t+1} = clamp(SOC_t + P_t * dt / C, 0, 1)
SOH_{t+1} = max(SOH_t - ΔSOH_t, min_soh)

where:
ΔSOH_t = ΔSOH_phys_t + ΔSOH_nn_t

The physics term keeps the model interpretable and stable.
The neural term learns residual patterns not captured by the semi-empirical structure.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StandardScalerNP:
    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "StandardScalerNP":
        self.mean_ = np.nanmean(x, axis=0).astype(np.float32)
        self.std_ = np.nanstd(x, axis=0).astype(np.float32)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler not fitted.")
        return ((x - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler not fitted.")
        return (x * self.std_ + self.mean_).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "std": None if self.std_ is None else self.std_.tolist(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.mean_ = None if state["mean"] is None else np.asarray(state["mean"], dtype=np.float32)
        self.std_ = None if state["std"] is None else np.asarray(state["std"], dtype=np.float32)


@dataclass
class DegradationConfig:
    dt_hours: float = 1.0
    p_ref_kw: float = 7.0
    soc_threshold: float = 0.70
    min_soh: float = 0.50

    k_throughput: float = 1.0e-4
    k_power: float = 8.0e-5
    k_soc_swing: float = 6.0e-5
    k_high_soc: float = 1.2e-4
    k_calendar: float = 2.0e-5
    k_high_soc_dwell: float = 5.0e-5
    k_temp: float = 1.0e-5

    power_exponent: float = 2.0
    high_soc_exponent: float = 2.0
    temp_ref_c: float = 25.0
    temp_scale_c: float = 10.0

    lambda_data: float = 1.0
    lambda_phy: float = 1.0
    lambda_nonneg: float = 0.5
    lambda_mono_power: float = 0.1
    lambda_mono_soc: float = 0.1
    lambda_smooth: float = 1.0e-4

    hidden_dims: Tuple[int, ...] = (64, 64)
    dropout: float = 0.0
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 200
    val_ratio: float = 0.2
    random_seed: int = 42
    device: str = "cpu"


@dataclass
class BatteryState:
    soc: float
    soh: float
    cumulative_throughput_kwh: float = 0.0


class BatteryDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y_delta: Optional[np.ndarray],
        y_soh_next: Optional[np.ndarray],
        aux: Dict[str, np.ndarray],
    ) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_delta = None if y_delta is None else torch.tensor(y_delta, dtype=torch.float32).view(-1, 1)
        self.y_soh_next = None if y_soh_next is None else torch.tensor(y_soh_next, dtype=torch.float32).view(-1, 1)
        self.aux = {k: torch.tensor(v, dtype=torch.float32).view(-1, 1) for k, v in aux.items()}

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        out = {"x": self.x[idx]}
        if self.y_delta is not None:
            out["y_delta"] = self.y_delta[idx]
        if self.y_soh_next is not None:
            out["y_soh_next"] = self.y_soh_next[idx]
        for k, v in self.aux.items():
            out[k] = v[idx]
        return out


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BatteryDegradationPINN:
    def __init__(self, config: DegradationConfig):
        self.config = config
        self.feature_cols = [
            "soc",
            "charge_power_kw",
            "battery_capacity_kwh",
            "temperature_c",
            "avg_soc",
            "delta_soc",
            "power_ratio",
            "high_soc_excess",
        ]
        self.scaler = StandardScalerNP()
        self.net = ResidualMLP(
            in_dim=len(self.feature_cols),
            hidden_dims=list(config.hidden_dims),
            dropout=config.dropout,
        )
        self.trained_ = False
        self.history_: List[Dict[str, float]] = []

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _compute_features_np(
        self,
        soc: np.ndarray,
        charge_power_kw: np.ndarray,
        battery_capacity_kwh: np.ndarray,
        temperature_c: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        cfg = self.config
        soc = np.clip(soc.astype(np.float32), 0.0, 1.0)
        charge_power_kw = np.maximum(charge_power_kw.astype(np.float32), 0.0)
        battery_capacity_kwh = np.maximum(battery_capacity_kwh.astype(np.float32), 1e-6)
        temperature_c = temperature_c.astype(np.float32)

        charge_energy_kwh = charge_power_kw * cfg.dt_hours
        delta_soc = charge_energy_kwh / battery_capacity_kwh
        soc_next = np.clip(soc + delta_soc, 0.0, 1.0)
        avg_soc = 0.5 * (soc + soc_next)
        high_soc_excess = np.maximum(avg_soc - cfg.soc_threshold, 0.0)
        power_ratio = charge_power_kw / max(cfg.p_ref_kw, 1e-6)
        temp_stress = np.maximum((temperature_c - cfg.temp_ref_c) / max(cfg.temp_scale_c, 1e-6), 0.0)

        return {
            "soc": soc,
            "soc_next": soc_next,
            "avg_soc": avg_soc,
            "charge_power_kw": charge_power_kw,
            "charge_energy_kwh": charge_energy_kwh,
            "delta_soc": delta_soc,
            "power_ratio": power_ratio,
            "high_soc_excess": high_soc_excess,
            "temperature_c": temperature_c,
            "temp_stress": temp_stress,
        }

    def _compute_features_torch(
        self,
        soc: torch.Tensor,
        charge_power_kw: torch.Tensor,
        battery_capacity_kwh: torch.Tensor,
        temperature_c: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.config
        soc = torch.clamp(soc, 0.0, 1.0)
        charge_power_kw = torch.clamp(charge_power_kw, min=0.0)
        battery_capacity_kwh = torch.clamp(battery_capacity_kwh, min=1e-6)

        charge_energy_kwh = charge_power_kw * cfg.dt_hours
        delta_soc = charge_energy_kwh / battery_capacity_kwh
        soc_next = torch.clamp(soc + delta_soc, 0.0, 1.0)
        avg_soc = 0.5 * (soc + soc_next)
        high_soc_excess = torch.clamp(avg_soc - cfg.soc_threshold, min=0.0)
        power_ratio = charge_power_kw / max(cfg.p_ref_kw, 1e-6)
        temp_stress = torch.clamp((temperature_c - cfg.temp_ref_c) / max(cfg.temp_scale_c, 1e-6), min=0.0)

        return {
            "soc": soc,
            "soc_next": soc_next,
            "avg_soc": avg_soc,
            "charge_power_kw": charge_power_kw,
            "charge_energy_kwh": charge_energy_kwh,
            "delta_soc": delta_soc,
            "power_ratio": power_ratio,
            "high_soc_excess": high_soc_excess,
            "temperature_c": temperature_c,
            "temp_stress": temp_stress,
        }

    def _physics_delta_torch(self, feat: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cfg = self.config
        throughput_term = cfg.k_throughput * feat["charge_energy_kwh"]
        power_term = cfg.k_power * (feat["power_ratio"] ** cfg.power_exponent)
        soc_swing_term = cfg.k_soc_swing * torch.abs(feat["delta_soc"])
        high_soc_term = cfg.k_high_soc * (feat["high_soc_excess"] ** cfg.high_soc_exponent)

        delta_soh_cycling = throughput_term + power_term + soc_swing_term + high_soc_term
        calendar_term = cfg.k_calendar * cfg.dt_hours
        high_soc_dwell_term = cfg.k_high_soc_dwell * feat["high_soc_excess"] * cfg.dt_hours
        temp_term = cfg.k_temp * feat["temp_stress"] * cfg.dt_hours
        delta_soh_calendar = calendar_term + high_soc_dwell_term + temp_term
        delta_soh_phys = delta_soh_cycling + delta_soh_calendar

        return {
            "delta_soh_phys": delta_soh_phys,
            "delta_soh_cycling": delta_soh_cycling,
            "delta_soh_calendar": delta_soh_calendar,
            "throughput_term": throughput_term,
            "power_term": power_term,
            "soc_swing_term": soc_swing_term,
            "high_soc_term": high_soc_term,
            "calendar_term": torch.full_like(delta_soh_phys, float(calendar_term)),
            "high_soc_dwell_term": high_soc_dwell_term,
            "temp_term": temp_term,
        }

    def _prepare_dataframe(self, df: pd.DataFrame):
        work = df.copy()
        required = ["soc", "charge_power_kw", "battery_capacity_kwh"]
        for col in required:
            if col not in work.columns:
                raise ValueError(f"Missing required column: {col}")

        if "temperature_c" not in work.columns:
            work["temperature_c"] = self.config.temp_ref_c

        for col in ["soc", "charge_power_kw", "battery_capacity_kwh", "temperature_c"]:
            work[col] = pd.to_numeric(work[col], errors="coerce")

        feat = self._compute_features_np(
            soc=work["soc"].values,
            charge_power_kw=work["charge_power_kw"].values,
            battery_capacity_kwh=work["battery_capacity_kwh"].values,
            temperature_c=work["temperature_c"].values,
        )

        battery_capacity_kwh = np.maximum(work["battery_capacity_kwh"].values.astype(np.float32), 1e-6)
        x = np.column_stack([
            feat["soc"],
            feat["charge_power_kw"],
            battery_capacity_kwh,
            feat["temperature_c"],
            feat["avg_soc"],
            feat["delta_soc"],
            feat["power_ratio"],
            feat["high_soc_excess"],
        ]).astype(np.float32)

        y_delta = None
        if "delta_soh_true" in work.columns:
            work["delta_soh_true"] = pd.to_numeric(work["delta_soh_true"], errors="coerce")
            y_delta = work["delta_soh_true"].values.astype(np.float32)

        y_soh_next = None
        if "soh_next" in work.columns:
            work["soh_next"] = pd.to_numeric(work["soh_next"], errors="coerce")
            y_soh_next = work["soh_next"].values.astype(np.float32)

        if y_delta is None and y_soh_next is None:
            raise ValueError("Need at least one target column: 'delta_soh_true' or 'soh_next'.")

        aux = {
            "soc": feat["soc"],
            "charge_power_kw": feat["charge_power_kw"],
            "battery_capacity_kwh": battery_capacity_kwh,
            "temperature_c": feat["temperature_c"],
            "soc_next_physics": feat["soc_next"],
        }

        valid_mask = np.isfinite(x).all(axis=1)
        if y_delta is not None:
            valid_mask &= np.isfinite(y_delta)
        if y_soh_next is not None:
            valid_mask &= np.isfinite(y_soh_next)

        x = x[valid_mask]
        y_delta = None if y_delta is None else y_delta[valid_mask]
        y_soh_next = None if y_soh_next is None else y_soh_next[valid_mask]
        aux = {k: v[valid_mask] for k, v in aux.items()}
        return x, y_delta, y_soh_next, aux

    def _split_train_val(self, x, y_delta, y_soh_next, aux):
        n = len(x)
        n_val = max(1, int(n * self.config.val_ratio))
        idx = np.arange(n)
        rng = np.random.default_rng(self.config.random_seed)
        rng.shuffle(idx)

        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        def take(arr, ids):
            return None if arr is None else arr[ids]

        train = (x[train_idx], take(y_delta, train_idx), take(y_soh_next, train_idx), {k: v[train_idx] for k, v in aux.items()})
        val = (x[val_idx], take(y_delta, val_idx), take(y_soh_next, val_idx), {k: v[val_idx] for k, v in aux.items()})
        return train, val

    def _build_input_tensor(self, feat: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([
            feat["soc"],
            feat["charge_power_kw"],
            feat["battery_capacity_kwh"],
            feat["temperature_c"],
            feat["avg_soc"],
            feat["delta_soc"],
            feat["power_ratio"],
            feat["high_soc_excess"],
        ], dim=1)

    def _forward_components(self, soc, charge_power_kw, battery_capacity_kwh, temperature_c):
        feat = self._compute_features_torch(soc, charge_power_kw, battery_capacity_kwh, temperature_c)
        feat["battery_capacity_kwh"] = battery_capacity_kwh
        physics = self._physics_delta_torch(feat)

        x_in = self._build_input_tensor(feat)
        x_in_scaled = torch.tensor(
            self.scaler.transform(x_in.detach().cpu().numpy()),
            dtype=torch.float32,
            device=x_in.device,
        )
        delta_nn = torch.nn.functional.softplus(self.net(x_in_scaled))
        delta_total = physics["delta_soh_phys"] + delta_nn
        return {**feat, **physics, "delta_soh_nn": delta_nn, "delta_soh_total": delta_total}

    def fit(self, df: pd.DataFrame) -> List[Dict[str, float]]:
        set_seed(self.config.random_seed)
        device = torch.device(self.config.device)
        self.net.to(device)

        x, y_delta, y_soh_next, aux = self._prepare_dataframe(df)
        self.scaler.fit(x)
        x_scaled = self.scaler.transform(x)

        train, val = self._split_train_val(x_scaled, y_delta, y_soh_next, aux)
        (x_tr, y_delta_tr, y_soh_tr, aux_tr), (x_va, y_delta_va, y_soh_va, aux_va) = train, val

        train_ds = BatteryDataset(x_tr, y_delta_tr, y_soh_tr, aux_tr)
        val_ds = BatteryDataset(x_va, y_delta_va, y_soh_va, aux_va)
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False, drop_last=False)

        opt = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)
        self.history_ = []

        for epoch in range(self.config.epochs):
            self.net.train()
            train_logs = []
            for batch in train_loader:
                soc = batch["soc"].to(device)
                p = batch["charge_power_kw"].to(device)
                cap = batch["battery_capacity_kwh"].to(device)
                temp = batch["temperature_c"].to(device)

                out = self._forward_components(soc, p, cap, temp)
                loss_data = torch.tensor(0.0, device=device)
                n_terms = 0

                if "y_delta" in batch:
                    y_delta_true = batch["y_delta"].to(device)
                    loss_data = loss_data + torch.mean((out["delta_soh_total"] - y_delta_true) ** 2)
                    n_terms += 1

                if "y_soh_next" in batch:
                    y_soh_true = batch["y_soh_next"].to(device)
                    soh_pred = torch.clamp(1.0 - out["delta_soh_total"], min=self.config.min_soh, max=1.0)
                    loss_data = loss_data + torch.mean((soh_pred - y_soh_true) ** 2)
                    n_terms += 1

                if n_terms > 0:
                    loss_data = loss_data / n_terms

                loss_phy = torch.mean(torch.relu(out["delta_soh_phys"] - out["delta_soh_total"]) ** 2)
                loss_nonneg = torch.mean(torch.relu(-out["delta_soh_total"]) ** 2)

                p_plus = p + 0.5
                soc_plus = torch.clamp(soc + 0.02, 0.0, 1.0)
                out_p_plus = self._forward_components(soc, p_plus, cap, temp)
                out_soc_plus = self._forward_components(soc_plus, p, cap, temp)
                loss_mono_power = torch.mean(torch.relu(out["delta_soh_total"] - out_p_plus["delta_soh_total"]) ** 2)
                loss_mono_soc = torch.mean(torch.relu(out["delta_soh_total"] - out_soc_plus["delta_soh_total"]) ** 2)
                loss_smooth = sum((param ** 2).mean() for param in self.net.parameters())

                loss = (
                    self.config.lambda_data * loss_data
                    + self.config.lambda_phy * loss_phy
                    + self.config.lambda_nonneg * loss_nonneg
                    + self.config.lambda_mono_power * loss_mono_power
                    + self.config.lambda_mono_soc * loss_mono_soc
                    + self.config.lambda_smooth * loss_smooth
                )

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_logs.append({
                    "loss": float(loss.detach().cpu()),
                    "loss_data": float(loss_data.detach().cpu()),
                    "loss_phy": float(loss_phy.detach().cpu()),
                    "loss_nonneg": float(loss_nonneg.detach().cpu()),
                    "loss_mono_power": float(loss_mono_power.detach().cpu()),
                    "loss_mono_soc": float(loss_mono_soc.detach().cpu()),
                })

            self.net.eval()
            with torch.no_grad():
                val_logs = []
                for batch in val_loader:
                    soc = batch["soc"].to(device)
                    p = batch["charge_power_kw"].to(device)
                    cap = batch["battery_capacity_kwh"].to(device)
                    temp = batch["temperature_c"].to(device)
                    out = self._forward_components(soc, p, cap, temp)
                    loss_data = torch.tensor(0.0, device=device)
                    n_terms = 0

                    if "y_delta" in batch:
                        y_delta_true = batch["y_delta"].to(device)
                        loss_data = loss_data + torch.mean((out["delta_soh_total"] - y_delta_true) ** 2)
                        n_terms += 1
                    if "y_soh_next" in batch:
                        y_soh_true = batch["y_soh_next"].to(device)
                        soh_pred = torch.clamp(1.0 - out["delta_soh_total"], min=self.config.min_soh, max=1.0)
                        loss_data = loss_data + torch.mean((soh_pred - y_soh_true) ** 2)
                        n_terms += 1
                    if n_terms > 0:
                        loss_data = loss_data / n_terms
                    val_logs.append({"val_loss_data": float(loss_data.detach().cpu())})

            epoch_log = {
                "epoch": epoch + 1,
                **{k: float(np.mean([d[k] for d in train_logs])) for k in train_logs[0]},
                **{k: float(np.mean([d[k] for d in val_logs])) for k in val_logs[0]},
            }
            self.history_.append(epoch_log)

        self.trained_ = True
        return self.history_

    def predict_increment(self, soc: float, charge_power_kw: float, battery_capacity_kwh: float, temperature_c: Optional[float] = None) -> Dict[str, float]:
        device = torch.device(self.config.device)
        self.net.to(device)
        self.net.eval()
        temperature_c = self.config.temp_ref_c if temperature_c is None else float(temperature_c)

        soc_t = torch.tensor([[float(soc)]], dtype=torch.float32, device=device)
        p_t = torch.tensor([[float(charge_power_kw)]], dtype=torch.float32, device=device)
        cap_t = torch.tensor([[float(battery_capacity_kwh)]], dtype=torch.float32, device=device)
        temp_t = torch.tensor([[float(temperature_c)]], dtype=torch.float32, device=device)

        with torch.no_grad():
            out = self._forward_components(soc_t, p_t, cap_t, temp_t)

        return {
            "delta_soh_total": float(out["delta_soh_total"].cpu().item()),
            "delta_soh_phys": float(out["delta_soh_phys"].cpu().item()),
            "delta_soh_nn": float(out["delta_soh_nn"].cpu().item()),
            "delta_soh_cycling": float(out["delta_soh_cycling"].cpu().item()),
            "delta_soh_calendar": float(out["delta_soh_calendar"].cpu().item()),
            "throughput_term": float(out["throughput_term"].cpu().item()),
            "power_term": float(out["power_term"].cpu().item()),
            "soc_swing_term": float(out["soc_swing_term"].cpu().item()),
            "high_soc_term": float(out["high_soc_term"].cpu().item()),
            "calendar_term": float(out["calendar_term"].cpu().item()),
            "high_soc_dwell_term": float(out["high_soc_dwell_term"].cpu().item()),
            "temp_term": float(out["temp_term"].cpu().item()),
            "soc_before": float(out["soc"].cpu().item()),
            "soc_after": float(out["soc_next"].cpu().item()),
            "avg_soc": float(out["avg_soc"].cpu().item()),
            "delta_soc": float(out["delta_soc"].cpu().item()),
            "power_ratio": float(out["power_ratio"].cpu().item()),
            "high_soc_excess": float(out["high_soc_excess"].cpu().item()),
            "charge_energy_kwh": float(out["charge_energy_kwh"].cpu().item()),
            "temperature_c": float(out["temperature_c"].cpu().item()),
        }

    def update_state(self, state: BatteryState, charge_power_kw: float, battery_capacity_kwh: float, temperature_c: Optional[float] = None) -> BatteryState:
        info = self.predict_increment(state.soc, charge_power_kw, battery_capacity_kwh, temperature_c)
        new_soc = self._clamp(info["soc_after"], 0.0, 1.0)
        new_soh = max(float(state.soh) - info["delta_soh_total"], self.config.min_soh)
        new_throughput = float(state.cumulative_throughput_kwh) + info["charge_energy_kwh"]
        return BatteryState(soc=float(new_soc), soh=float(new_soh), cumulative_throughput_kwh=float(new_throughput))

    def simulate_profile(self, initial_state: BatteryState, charge_power_profile_kw: List[float], battery_capacity_kwh: float, temperature_profile_c: Optional[List[float]] = None) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        temps = temperature_profile_c or [self.config.temp_ref_c] * len(charge_power_profile_kw)
        state = BatteryState(float(initial_state.soc), float(initial_state.soh), float(initial_state.cumulative_throughput_kwh))

        for step, (p_kw, temp_c) in enumerate(zip(charge_power_profile_kw, temps)):
            info = self.predict_increment(state.soc, float(p_kw), float(battery_capacity_kwh), float(temp_c))
            new_state = self.update_state(state, float(p_kw), float(battery_capacity_kwh), float(temp_c))
            rows.append({
                "step": int(step),
                "charge_power_kw": float(p_kw),
                "temperature_c": float(temp_c),
                "soc_before": float(state.soc),
                "soh_before": float(state.soh),
                "cumulative_throughput_before_kwh": float(state.cumulative_throughput_kwh),
                **info,
                "soc_after_update": float(new_state.soc),
                "soh_after_update": float(new_state.soh),
                "cumulative_throughput_after_kwh": float(new_state.cumulative_throughput_kwh),
            })
            state = new_state
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "feature_cols": self.feature_cols,
            "scaler": self.scaler.state_dict(),
            "net_state": self.net.state_dict(),
            "trained": self.trained_,
            "history": self.history_,
        }
        with open(path, "wb") as f:
            torch.save(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "BatteryDegradationPINN":
        payload = torch.load(path, map_location="cpu")
        config = DegradationConfig(**payload["config"])
        obj = cls(config)
        obj.feature_cols = payload["feature_cols"]
        obj.scaler.load_state_dict(payload["scaler"])
        obj.net.load_state_dict(payload["net_state"])
        obj.trained_ = payload["trained"]
        obj.history_ = payload.get("history", [])
        return obj

    def describe(self) -> Dict[str, Any]:
        return asdict(self.config)


if __name__ == "__main__":
    cfg = DegradationConfig(epochs=20, batch_size=64)
    model = BatteryDegradationPINN(cfg)

    rng = np.random.default_rng(42)
    n = 800
    df = pd.DataFrame({
        "soc": rng.uniform(0.1, 0.9, size=n),
        "charge_power_kw": rng.uniform(0.0, 11.0, size=n),
        "battery_capacity_kwh": rng.uniform(45.0, 80.0, size=n),
        "temperature_c": rng.uniform(15.0, 40.0, size=n),
    })

    charge_energy = df["charge_power_kw"].values * cfg.dt_hours
    delta_soc = charge_energy / df["battery_capacity_kwh"].values
    avg_soc = np.clip(df["soc"].values + 0.5 * delta_soc, 0.0, 1.0)
    high_soc = np.maximum(avg_soc - cfg.soc_threshold, 0.0)
    power_ratio = df["charge_power_kw"].values / cfg.p_ref_kw
    temp_stress = np.maximum((df["temperature_c"].values - cfg.temp_ref_c) / cfg.temp_scale_c, 0.0)
    df["delta_soh_true"] = (
        1.1e-4 * charge_energy
        + 8.0e-5 * (power_ratio ** 2)
        + 5.0e-5 * np.abs(delta_soc)
        + 1.4e-4 * (high_soc ** 2)
        + 2.0e-5
        + 4.0e-5 * high_soc
        + 1.0e-5 * temp_stress
        + rng.normal(0.0, 2e-5, size=n)
    ).clip(min=0.0)

    hist = model.fit(df)
    print("Last training log:", hist[-1])

    init_state = BatteryState(soc=0.35, soh=1.0)
    profile_kw = [7.0, 7.0, 3.5, 0.0, 0.0, 11.0]
    temp_profile = [25, 26, 28, 30, 29, 32]
    history = model.simulate_profile(init_state, profile_kw, 60.0, temp_profile)
    print(history.to_string(index=False))
