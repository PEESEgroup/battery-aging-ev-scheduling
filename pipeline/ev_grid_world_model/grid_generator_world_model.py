from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from world_model_common import (
    ConditionalCVAE,
    ConditionalDenoiser,
    DiffusionNoiseScheduler,
    StandardNormalizer,
    TrainingConfig,
    dataframe_or_none,
    ensure_2d_array,
    fit_cvae_model,
    fit_diffusion_model,
    load_json,
    sample_with_diffusion,
    save_json,
)


class GridWorldModel:
    """
    Grid generator:
      CVAE main generator + conditional diffusion tail enhancement

    Supports optional EV-coupled generation by allowing EV baseline features
    to be concatenated into the conditions at sampling time.
    """
    def __init__(
        self,
        feature_names: List[str],
        condition_names: List[str],
        risk_names: Optional[List[str]] = None,
        z_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        diffusion_hidden_dims: Optional[List[int]] = None,
        diffusion_steps: int = 50,
        device: str = "cpu",
    ) -> None:
        self.feature_names = feature_names
        self.condition_names = condition_names
        self.risk_names = risk_names or ["risk_level"]
        self.device = device

        x_dim = len(feature_names)
        c_dim = len(condition_names)
        r_dim = len(self.risk_names)

        self.x_norm = StandardNormalizer()
        self.c_norm = StandardNormalizer()
        self.r_norm = StandardNormalizer()

        self.cvae = ConditionalCVAE(
            x_dim=x_dim,
            c_dim=c_dim,
            z_dim=z_dim,
            hidden_dims=hidden_dims or [128, 128],
        )
        self.denoiser = ConditionalDenoiser(
            x_dim=x_dim,
            c_dim=c_dim,
            r_dim=r_dim,
            hidden_dims=diffusion_hidden_dims or [128, 128],
        )
        self.scheduler = DiffusionNoiseScheduler(timesteps=diffusion_steps)

        self.cvae_history_: Optional[List[Dict]] = None
        self.diffusion_history_: Optional[List[Dict]] = None

    def _phy_penalty(self, xb: torch.Tensor) -> torch.Tensor:
        penalties = torch.zeros(xb.shape[0], device=xb.device)
        name_to_idx = {n: i for i, n in enumerate(self.feature_names)}

        def has(*names): return all(n in name_to_idx for n in names)

        if has("price"):
            price = xb[:, name_to_idx["price"]]
            penalties = penalties + torch.relu(-price)

        if has("carbon_intensity"):
            ci = xb[:, name_to_idx["carbon_intensity"]]
            penalties = penalties + torch.relu(-ci)

        if has("load"):
            load = xb[:, name_to_idx["load"]]
            penalties = penalties + torch.relu(-load)

        if has("congestion_index"):
            g = xb[:, name_to_idx["congestion_index"]]
            penalties = penalties + torch.relu(-g)

        return penalties

    def fit_cvae(
        self,
        grid_df: pd.DataFrame,
        cond_df: pd.DataFrame,
        train_cfg: Optional[TrainingConfig] = None,
    ) -> List[Dict]:
        train_cfg = train_cfg or TrainingConfig(device=self.device)
        x = ensure_2d_array(grid_df[self.feature_names])
        c = ensure_2d_array(cond_df[self.condition_names])

        self.x_norm.fit(x)
        self.c_norm.fit(c)
        x_n = self.x_norm.transform(x)
        c_n = self.c_norm.transform(c)

        self.cvae_history_ = fit_cvae_model(
            self.cvae,
            x_n,
            c_n,
            train_cfg,
            phy_penalty_fn=self._phy_penalty,
        )
        return self.cvae_history_

    def fit_diffusion(
        self,
        grid_df: pd.DataFrame,
        cond_df: pd.DataFrame,
        risk_df: pd.DataFrame,
        train_cfg: Optional[TrainingConfig] = None,
    ) -> List[Dict]:
        train_cfg = train_cfg or TrainingConfig(device=self.device)

        x = ensure_2d_array(grid_df[self.feature_names])
        c = ensure_2d_array(cond_df[self.condition_names])
        r = ensure_2d_array(risk_df[self.risk_names])

        if self.x_norm.mean_ is None:
            raise RuntimeError("Fit CVAE before diffusion training.")

        x_n = self.x_norm.transform(x)
        c_n = self.c_norm.transform(c)
        self.r_norm.fit(r)
        r_n = self.r_norm.transform(r)

        with torch.no_grad():
            c_tensor = torch.tensor(c_n, dtype=torch.float32)
            base_x = self.cvae.sample(c_tensor).cpu().numpy()

        self.diffusion_history_ = fit_diffusion_model(
            self.denoiser,
            self.scheduler,
            x_true=x_n,
            base_x=base_x,
            c=c_n,
            r=r_n,
            train_cfg=train_cfg,
        )
        return self.diffusion_history_

    def sample_base(
        self,
        conditions_df: pd.DataFrame,
        n_samples_per_condition: int = 1,
    ) -> pd.DataFrame:
        c = ensure_2d_array(conditions_df[self.condition_names])
        c_n = self.c_norm.transform(c)

        if n_samples_per_condition > 1:
            c_n = np.repeat(c_n, n_samples_per_condition, axis=0)

        c_tensor = torch.tensor(c_n, dtype=torch.float32, device=self.device)
        self.cvae.to(self.device)
        with torch.no_grad():
            x_n = self.cvae.sample(c_tensor).cpu().numpy()
        x = self.x_norm.inverse_transform(x_n)
        return dataframe_or_none(x, self.feature_names)

    def sample_tail_enhanced(
        self,
        conditions_df: pd.DataFrame,
        risk_df: pd.DataFrame,
        n_samples_per_condition: int = 1,
        tail_strength: float = 0.5,
    ) -> pd.DataFrame:
        c = ensure_2d_array(conditions_df[self.condition_names])
        r = ensure_2d_array(risk_df[self.risk_names])
        c_n = self.c_norm.transform(c)
        r_n = self.r_norm.transform(r)

        if n_samples_per_condition > 1:
            c_n = np.repeat(c_n, n_samples_per_condition, axis=0)
            r_n = np.repeat(r_n, n_samples_per_condition, axis=0)

        self.cvae.to(self.device)
        self.denoiser.to(self.device)
        self.scheduler.to(torch.device(self.device))

        with torch.no_grad():
            c_tensor = torch.tensor(c_n, dtype=torch.float32, device=self.device)
            r_tensor = torch.tensor(r_n, dtype=torch.float32, device=self.device)
            base_x = self.cvae.sample(c_tensor)
            refined_x = sample_with_diffusion(
                self.denoiser,
                self.scheduler,
                base_x=base_x,
                c=c_tensor,
                r=r_tensor,
                tail_strength=tail_strength,
            )
            out = refined_x.cpu().numpy()
        x = self.x_norm.inverse_transform(out)
        return dataframe_or_none(x, self.feature_names)

    def save(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.cvae.state_dict(), save_dir / "cvae.pt")
        torch.save(self.denoiser.state_dict(), save_dir / "denoiser.pt")

        meta = {
            "feature_names": self.feature_names,
            "condition_names": self.condition_names,
            "risk_names": self.risk_names,
            "x_norm": self.x_norm.state_dict(),
            "c_norm": self.c_norm.state_dict(),
            "r_norm": self.r_norm.state_dict(),
            "device": self.device,
            "cvae_history": self.cvae_history_,
            "diffusion_history": self.diffusion_history_,
        }
        save_json(meta, save_dir / "meta.json")

    @classmethod
    def load(cls, save_dir: str | Path) -> "GridWorldModel":
        save_dir = Path(save_dir)
        meta = load_json(save_dir / "meta.json")
        obj = cls(
            feature_names=meta["feature_names"],
            condition_names=meta["condition_names"],
            risk_names=meta["risk_names"],
            device=meta.get("device", "cpu"),
        )
        obj.x_norm.load_state_dict(meta["x_norm"])
        obj.c_norm.load_state_dict(meta["c_norm"])
        obj.r_norm.load_state_dict(meta["r_norm"])
        obj.cvae.load_state_dict(torch.load(save_dir / "cvae.pt", map_location="cpu"))
        obj.denoiser.load_state_dict(torch.load(save_dir / "denoiser.pt", map_location="cpu"))
        obj.cvae_history_ = meta.get("cvae_history")
        obj.diffusion_history_ = meta.get("diffusion_history")
        return obj
