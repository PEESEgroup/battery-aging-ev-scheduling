# EV–Grid World Model Code Package

This package implements:

- `EVWorldModel`: EV generator with **CVAE main generator + conditional diffusion tail enhancement**
- `GridWorldModel`: grid generator with **CVAE main generator + conditional diffusion tail enhancement**
- `JointScenarioWorldPipeline`: adapter that connects both generators to a `joint_scenarios` style pipeline

## File overview

- `world_model_common.py`  
  Shared modules, utilities, dataset wrappers, normalizers, CVAE and diffusion components.

- `ev_generator_world_model.py`  
  EV world model.

- `grid_generator_world_model.py`  
  Grid world model.

- `joint_scenarios_world_model.py`  
  Joint pipeline wrapper and scenario builder.

- `example_usage.py`  
  Example showing training and sampling with synthetic data.

## Expected integration pattern

The main API is:

```python
from ev_generator_world_model import EVWorldModel
from grid_generator_world_model import GridWorldModel
from joint_scenarios_world_model import JointScenarioWorldPipeline
```

### Train
```python
ev_model = EVWorldModel(ev_feature_names, ev_condition_names)
grid_model = GridWorldModel(grid_feature_names, grid_condition_names)

ev_model.fit_cvae(ev_df, cond_df)
ev_model.fit_diffusion(ev_df, cond_df, risk_df)

grid_model.fit_cvae(grid_df, cond_df)
grid_model.fit_diffusion(grid_df, cond_df, risk_df)
```

### Save / Load
```python
ev_model.save("artifacts/ev_world")
grid_model.save("artifacts/grid_world")

ev_model = EVWorldModel.load("artifacts/ev_world")
grid_model = GridWorldModel.load("artifacts/grid_world")
```

### Joint scenarios
```python
pipeline = JointScenarioWorldPipeline(ev_model, grid_model)
joint = pipeline.sample_joint_scenarios(
    conditions_df=cond_df_future,
    n_samples_per_condition=20,
    risk_df=risk_df_future,
    tail_strength=0.7,
)
```

## Assumptions

1. Features are tabular / vectorized daily trajectories or summaries.
2. For trajectory data, flatten each daily trajectory into a fixed-length vector first.
3. Conditions and risk descriptors are user-defined dataframes aligned row-by-row with the training samples.
4. The grid generator can optionally condition on EV baseline outputs during joint sampling.

## Notes

- This code is designed to be **drop-in adaptable** rather than tightly bound to a hidden earlier codebase.
- It emphasizes a clean interface for your `joint_scenarios` pipeline.
- The diffusion block is implemented as a **conditional denoising refiner**, which is computationally lighter and easier to integrate.
