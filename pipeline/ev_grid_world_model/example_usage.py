from pathlib import Path

import numpy as np
import pandas as pd

from ev_generator_world_model import EVWorldModel
from grid_generator_world_model import GridWorldModel
from joint_scenarios_world_model import JointScenarioConfig, JointScenarioWorldPipeline
from world_model_common import TrainingConfig


def build_synthetic_data(n: int = 500):
    rng = np.random.default_rng(123)

    cond = pd.DataFrame({
        "region_id": rng.integers(0, 4, size=n),
        "weekday": rng.integers(0, 2, size=n),
        "temperature": rng.normal(20, 8, size=n),
        "base_load_level": rng.normal(0.0, 1.0, size=n),
    })

    risk = pd.DataFrame({
        "risk_level": rng.uniform(0, 1, size=n),
    })

    ev = pd.DataFrame({
        "arrival": rng.uniform(16, 21, size=n),
        "departure": rng.uniform(6, 9, size=n) + 24.0,
        "init_soc": rng.uniform(0.2, 0.7, size=n),
        "target_soc": rng.uniform(0.7, 1.0, size=n),
        "energy": rng.uniform(8, 60, size=n),
        "max_power": rng.uniform(3.3, 11.0, size=n),
        "peak_overlap_proxy": rng.uniform(0, 1, size=n),
    })

    grid = pd.DataFrame({
        "load": 50 + 10 * cond["base_load_level"].values + 0.4 * ev["energy"].values + rng.normal(0, 3, size=n),
        "price": 30 + 4 * cond["base_load_level"].values + 8 * risk["risk_level"].values + rng.normal(0, 2, size=n),
        "carbon_intensity": 0.3 + 0.05 * rng.normal(size=n),
        "congestion_index": np.clip(0.3 * ev["peak_overlap_proxy"].values + 0.6 * risk["risk_level"].values, 0, None),
    })

    return ev, grid, cond, risk


def main():
    ev_df, grid_df, cond_df, risk_df = build_synthetic_data()

    ev_features = ["arrival", "departure", "init_soc", "target_soc", "energy", "max_power", "peak_overlap_proxy"]
    grid_features = ["load", "price", "carbon_intensity", "congestion_index"]

    ev_conditions = ["region_id", "weekday", "temperature", "base_load_level"]
    grid_conditions = ["region_id", "weekday", "temperature", "base_load_level", "ev_energy_proxy", "ev_peak_overlap_proxy"]

    # EV model
    ev_model = EVWorldModel(
        feature_names=ev_features,
        condition_names=ev_conditions,
        risk_names=["risk_level"],
        device="cpu",
    )
    ev_cfg = TrainingConfig(epochs=3, diffusion_epochs=2, batch_size=64, lr=1e-3, phy_weight=0.1, device="cpu")
    ev_model.fit_cvae(ev_df, cond_df, ev_cfg)
    ev_model.fit_diffusion(ev_df, cond_df, risk_df, ev_cfg)

    # Grid conditions for training: append EV proxies
    grid_cond_df = cond_df.copy()
    grid_cond_df["ev_energy_proxy"] = ev_df["energy"].values
    grid_cond_df["ev_peak_overlap_proxy"] = ev_df["peak_overlap_proxy"].values

    grid_model = GridWorldModel(
        feature_names=grid_features,
        condition_names=grid_conditions,
        risk_names=["risk_level"],
        device="cpu",
    )
    grid_cfg = TrainingConfig(epochs=3, diffusion_epochs=2, batch_size=64, lr=1e-3, phy_weight=0.1, device="cpu")
    grid_model.fit_cvae(grid_df, grid_cond_df, grid_cfg)
    grid_model.fit_diffusion(grid_df, grid_cond_df, risk_df, grid_cfg)

    pipeline = JointScenarioWorldPipeline(
        ev_model=ev_model,
        grid_model=grid_model,
        ev_to_grid_feature_map={
            "energy": "ev_energy_proxy",
            "peak_overlap_proxy": "ev_peak_overlap_proxy",
        },
    )

    future_cond = cond_df.iloc[:5].reset_index(drop=True)
    future_risk = risk_df.iloc[:5].reset_index(drop=True)

    sampled = pipeline.sample_joint_scenarios(
        conditions_df=future_cond,
        risk_df=future_risk,
        config=JointScenarioConfig(
            n_samples_per_condition=3,
            tail_strength=0.7,
            grid_condition_on_ev=True,
            sample_tail_for_ev=True,
            sample_tail_for_grid=True,
        ),
    )

    out_dir = Path("artifacts_example")
    out_dir.mkdir(exist_ok=True, parents=True)
    sampled["joint"].to_csv(out_dir / "joint_scenarios.csv", index=False)
    sampled["ev"].to_csv(out_dir / "ev_samples.csv", index=False)
    sampled["grid"].to_csv(out_dir / "grid_samples.csv", index=False)

    ev_model.save(out_dir / "ev_world")
    grid_model.save(out_dir / "grid_world")

    print(sampled["joint"].head())
    print(f"Saved example outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
