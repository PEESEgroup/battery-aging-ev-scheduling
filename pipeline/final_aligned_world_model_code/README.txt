
This folder contains compatibility-aligned replacements for:
- train_ev_generator.py
- train_grid_generator.py
- build_joint_scenarios.py

They preserve the old class names and public methods used by your project:

- EVGenerator.load(...).sample_sessions_for_horizon(...)
- GridGenerator.load(...).load_data(...)
- GridGenerator.load(...).sample_trajectory(...)
- build_joint_scenarios(config)

Internal generators are upgraded to:
- CVAE main generator
- conditional diffusion tail enhancement

You should keep `world_model_common.py` in the same folder.
