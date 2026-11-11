from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ev_generator_world_model import EVWorldModel
from grid_generator_world_model import GridWorldModel


@dataclass
class JointScenarioConfig:
    n_samples_per_condition: int = 10
    tail_strength: float = 0.5
    grid_condition_on_ev: bool = True
    sample_tail_for_ev: bool = True
    sample_tail_for_grid: bool = True


class JointScenarioWorldPipeline:
    """
    Adapter layer for a joint_scenarios pipeline.

    Design:
      1) Sample EV trajectories from EV world model.
      2) Optionally append selected EV outputs into grid conditions.
      3) Sample grid trajectories from Grid world model.
      4) Return joint scenario table plus separated EV / grid tables.

    This is intentionally flexible so it can plug into an existing pipeline.
    """
    def __init__(
        self,
        ev_model: EVWorldModel,
        grid_model: GridWorldModel,
        ev_to_grid_feature_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.ev_model = ev_model
        self.grid_model = grid_model
        self.ev_to_grid_feature_map = ev_to_grid_feature_map or {}

    def _repeat_rows(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df.loc[df.index.repeat(n)].reset_index(drop=True)

    def _augment_grid_conditions_with_ev(
        self,
        base_cond_df: pd.DataFrame,
        ev_samples_df: pd.DataFrame,
    ) -> pd.DataFrame:
        out = base_cond_df.copy()
        for ev_col, grid_cond_col in self.ev_to_grid_feature_map.items():
            if ev_col not in ev_samples_df.columns:
                raise KeyError(f"EV feature '{ev_col}' not found in EV samples.")
            out[grid_cond_col] = ev_samples_df[ev_col].values
        return out

    def sample_joint_scenarios(
        self,
        conditions_df: pd.DataFrame,
        risk_df: pd.DataFrame,
        config: Optional[JointScenarioConfig] = None,
    ) -> Dict[str, pd.DataFrame]:
        cfg = config or JointScenarioConfig()
        cond_rep = self._repeat_rows(conditions_df, cfg.n_samples_per_condition)
        risk_rep = self._repeat_rows(risk_df, cfg.n_samples_per_condition)

        # 1) EV samples
        if cfg.sample_tail_for_ev:
            ev_df = self.ev_model.sample_tail_enhanced(
                conditions_df=conditions_df,
                risk_df=risk_df,
                n_samples_per_condition=cfg.n_samples_per_condition,
                tail_strength=cfg.tail_strength,
            )
        else:
            ev_df = self.ev_model.sample_base(
                conditions_df=conditions_df,
                n_samples_per_condition=cfg.n_samples_per_condition,
            )

        # 2) Grid conditions
        if cfg.grid_condition_on_ev and self.ev_to_grid_feature_map:
            grid_cond_df = self._augment_grid_conditions_with_ev(cond_rep, ev_df)
        else:
            grid_cond_df = cond_rep.copy()

        # Keep only model-required grid condition columns
        missing = [c for c in self.grid_model.condition_names if c not in grid_cond_df.columns]
        if missing:
            raise KeyError(
                f"Missing grid condition columns after EV augmentation: {missing}. "
                f"Available columns: {list(grid_cond_df.columns)}"
            )
        grid_cond_df = grid_cond_df[self.grid_model.condition_names]

        # 3) Grid samples
        if cfg.sample_tail_for_grid:
            grid_df = self.grid_model.sample_tail_enhanced(
                conditions_df=grid_cond_df,
                risk_df=risk_rep[self.grid_model.risk_names],
                n_samples_per_condition=1,
                tail_strength=cfg.tail_strength,
            )
        else:
            grid_df = self.grid_model.sample_base(
                conditions_df=grid_cond_df,
                n_samples_per_condition=1,
            )

        # 4) Joint table
        joint = pd.concat(
            [
                cond_rep.reset_index(drop=True).add_prefix("cond_"),
                risk_rep.reset_index(drop=True).add_prefix("risk_"),
                ev_df.reset_index(drop=True).add_prefix("ev_"),
                grid_df.reset_index(drop=True).add_prefix("grid_"),
            ],
            axis=1,
        )

        joint["scenario_id"] = np.arange(len(joint))
        return {
            "joint": joint,
            "ev": ev_df.reset_index(drop=True),
            "grid": grid_df.reset_index(drop=True),
            "conditions": cond_rep.reset_index(drop=True),
            "risk": risk_rep.reset_index(drop=True),
        }

    def to_joint_scenarios_format(
        self,
        sampled: Dict[str, pd.DataFrame],
        ev_profile_cols: Optional[List[str]] = None,
        grid_profile_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Helper for converting to a flat `joint_scenarios` table.
        This can be adjusted to match your existing downstream optimizer exactly.
        """
        joint = sampled["joint"].copy()
        if ev_profile_cols is not None:
            missing = [c for c in ev_profile_cols if f"ev_{c}" not in joint.columns]
            if missing:
                raise KeyError(f"Missing EV profile columns in joint table: {missing}")
        if grid_profile_cols is not None:
            missing = [c for c in grid_profile_cols if f"grid_{c}" not in joint.columns]
            if missing:
                raise KeyError(f"Missing grid profile columns in joint table: {missing}")
        return joint
