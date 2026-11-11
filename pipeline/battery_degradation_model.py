#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
battery_degradation_model.py

A first-version semi-empirical battery degradation surrogate model for
EV charging optimization.

Main idea
---------
This model upgrades a simple health penalty into a state-based degradation model:

    SOH_{t+1} = SOH_t - delta_soh_t
    SOC_{t+1} = SOC_t + charge_energy / battery_capacity

The degradation increment is decomposed into:
1) Cycling aging
2) Calendar aging

Cycling aging includes:
- throughput term
- power term
- SOC swing term
- high-SOC stress term

Calendar aging includes:
- base time-aging term
- high-SOC dwell term

This design is lightweight enough for optimization loops, while still providing
more interpretable and mechanistic behavior than a single proxy penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Any

import math
import pandas as pd


@dataclass
class DegradationConfig:
    """Configuration for the semi-empirical degradation model."""

    # time resolution
    dt_hours: float = 1.0

    # reference charging power for normalization
    p_ref_kw: float = 7.0

    # high-SOC threshold
    soc_threshold: float = 0.70

    # SOH lower bound to prevent negative drift
    min_soh: float = 0.50

    # degradation coefficients
    # These are surrogate-scale coefficients. They should be calibrated later
    # if you want quantitative realism.
    k_throughput: float = 1.0e-4
    k_power: float = 8.0e-5
    k_soc_swing: float = 6.0e-5
    k_high_soc: float = 1.2e-4
    k_calendar: float = 2.0e-5
    k_high_soc_dwell: float = 5.0e-5

    # optional nonlinear exponent for power penalty
    power_exponent: float = 2.0

    # optional exponent for high-SOC stress
    high_soc_exponent: float = 2.0


@dataclass
class BatteryState:
    """Battery state tracked during optimization or simulation."""

    soc: float
    soh: float
    cumulative_throughput_kwh: float = 0.0


class BatteryDegradationModel:
    """
    Semi-empirical degradation surrogate.

    Methods
    -------
    compute_increment(...)
        Compute degradation increment and component breakdown for one step.

    update_state(...)
        Update battery state after one charging step.

    simulate_profile(...)
        Roll the state forward over an entire charging power profile.
    """

    def __init__(self, config: DegradationConfig):
        self.config = config

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _compute_features(
        self,
        soc: float,
        charge_power_kw: float,
        battery_capacity_kwh: float,
    ) -> Dict[str, float]:
        """
        Compute degradation-related features for one charging step.
        """
        cfg = self.config

        soc = self._clamp(float(soc), 0.0, 1.0)
        charge_power_kw = max(float(charge_power_kw), 0.0)
        battery_capacity_kwh = max(float(battery_capacity_kwh), 1e-6)

        charge_energy_kwh = charge_power_kw * cfg.dt_hours
        delta_soc = charge_energy_kwh / battery_capacity_kwh
        soc_next = self._clamp(soc + delta_soc, 0.0, 1.0)

        avg_soc = 0.5 * (soc + soc_next)
        high_soc_excess = max(avg_soc - cfg.soc_threshold, 0.0)

        power_ratio = charge_power_kw / max(cfg.p_ref_kw, 1e-6)

        return {
            "soc": soc,
            "soc_next": soc_next,
            "avg_soc": avg_soc,
            "charge_power_kw": charge_power_kw,
            "charge_energy_kwh": charge_energy_kwh,
            "delta_soc": delta_soc,
            "power_ratio": power_ratio,
            "high_soc_excess": high_soc_excess,
        }

    def compute_increment(
        self,
        soc: float,
        charge_power_kw: float,
        battery_capacity_kwh: float,
    ) -> Dict[str, float]:
        """
        Compute degradation increment for one time step.

        Returns a detailed dictionary containing:
        - total degradation
        - cycling degradation
        - calendar degradation
        - each component term
        - basic state transition features
        """
        cfg = self.config
        feat = self._compute_features(
            soc=soc,
            charge_power_kw=charge_power_kw,
            battery_capacity_kwh=battery_capacity_kwh,
        )

        # Cycling aging terms
        throughput_term = cfg.k_throughput * feat["charge_energy_kwh"]
        power_term = cfg.k_power * (feat["power_ratio"] ** cfg.power_exponent)
        soc_swing_term = cfg.k_soc_swing * abs(feat["delta_soc"])
        high_soc_term = cfg.k_high_soc * (
            feat["high_soc_excess"] ** cfg.high_soc_exponent
        )

        delta_soh_cycling = (
            throughput_term
            + power_term
            + soc_swing_term
            + high_soc_term
        )

        # Calendar aging terms
        calendar_term = cfg.k_calendar * cfg.dt_hours
        high_soc_dwell_term = (
            cfg.k_high_soc_dwell * feat["high_soc_excess"] * cfg.dt_hours
        )

        delta_soh_calendar = calendar_term + high_soc_dwell_term

        delta_soh_total = delta_soh_cycling + delta_soh_calendar

        return {
            "delta_soh_total": float(delta_soh_total),
            "delta_soh_cycling": float(delta_soh_cycling),
            "delta_soh_calendar": float(delta_soh_calendar),
            "throughput_term": float(throughput_term),
            "power_term": float(power_term),
            "soc_swing_term": float(soc_swing_term),
            "high_soc_term": float(high_soc_term),
            "calendar_term": float(calendar_term),
            "high_soc_dwell_term": float(high_soc_dwell_term),
            "charge_energy_kwh": float(feat["charge_energy_kwh"]),
            "delta_soc": float(feat["delta_soc"]),
            "soc_before": float(feat["soc"]),
            "soc_after": float(feat["soc_next"]),
            "avg_soc": float(feat["avg_soc"]),
            "high_soc_excess": float(feat["high_soc_excess"]),
            "power_ratio": float(feat["power_ratio"]),
        }

    def update_state(
        self,
        state: BatteryState,
        charge_power_kw: float,
        battery_capacity_kwh: float,
    ) -> BatteryState:
        """
        Update battery state by one step using the degradation model.
        """
        info = self.compute_increment(
            soc=state.soc,
            charge_power_kw=charge_power_kw,
            battery_capacity_kwh=battery_capacity_kwh,
        )

        new_soc = self._clamp(info["soc_after"], 0.0, 1.0)
        new_soh = max(float(state.soh) - info["delta_soh_total"], self.config.min_soh)
        new_throughput = float(state.cumulative_throughput_kwh) + info["charge_energy_kwh"]

        return BatteryState(
            soc=float(new_soc),
            soh=float(new_soh),
            cumulative_throughput_kwh=float(new_throughput),
        )

    def simulate_profile(
        self,
        initial_state: BatteryState,
        charge_power_profile_kw: List[float],
        battery_capacity_kwh: float,
    ) -> pd.DataFrame:
        """
        Simulate battery degradation over a charging profile.

        Parameters
        ----------
        initial_state : BatteryState
            Initial battery state.
        charge_power_profile_kw : list of float
            Charging power at each step.
        battery_capacity_kwh : float
            Battery capacity.

        Returns
        -------
        pd.DataFrame
            Step-by-step simulation history.
        """
        rows: List[Dict[str, Any]] = []
        state = BatteryState(
            soc=float(initial_state.soc),
            soh=float(initial_state.soh),
            cumulative_throughput_kwh=float(initial_state.cumulative_throughput_kwh),
        )

        for step, p_kw in enumerate(charge_power_profile_kw):
            info = self.compute_increment(
                soc=state.soc,
                charge_power_kw=float(p_kw),
                battery_capacity_kwh=float(battery_capacity_kwh),
            )
            new_state = self.update_state(
                state=state,
                charge_power_kw=float(p_kw),
                battery_capacity_kwh=float(battery_capacity_kwh),
            )

            row = {
                "step": int(step),
                "charge_power_kw": float(p_kw),
                "soc_before": float(state.soc),
                "soh_before": float(state.soh),
                "cumulative_throughput_before_kwh": float(state.cumulative_throughput_kwh),
                **info,
                "soc_after_update": float(new_state.soc),
                "soh_after_update": float(new_state.soh),
                "cumulative_throughput_after_kwh": float(new_state.cumulative_throughput_kwh),
            }
            rows.append(row)
            state = new_state

        return pd.DataFrame(rows)

    def describe(self) -> Dict[str, Any]:
        """Return configuration as a dictionary."""
        return asdict(self.config)


if __name__ == "__main__":
    # Minimal example
    cfg = DegradationConfig()
    model = BatteryDegradationModel(cfg)

    init_state = BatteryState(soc=0.35, soh=1.0)
    profile_kw = [7.0, 7.0, 3.5, 0.0, 0.0, 11.0]

    history = model.simulate_profile(
        initial_state=init_state,
        charge_power_profile_kw=profile_kw,
        battery_capacity_kwh=60.0,
    )

    print("Degradation model config:")
    print(model.describe())
    print("\nSimulation history:")
    print(history.to_string(index=False))