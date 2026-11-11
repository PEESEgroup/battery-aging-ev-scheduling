# Battery Aging and EV Scheduling

Research code for **Reassessing Realizable EV Charging Flexibility Under Battery Aging and Coupled EV–Grid Stress**.

The repository studies unidirectional EV charging, battery degradation, scenario-based optimization, and the difference between offline flexibility potential and delivered flexibility under rolling operation.

## Repository layout

| Directory | Contents |
|---|---|
| `pipeline/` | Data preparation, archived generators, aging models, scheduling, replay and experiment drivers |
| `pipeline/ev_grid_world_model/` | Separate neural world-model implementation with its own README |
| `pipeline/final_world_model_exact_matched/` | Alternative generator implementation and component validation |
| `pipeline/final_aligned_world_model_code/` | Alternative aligned generator snapshot; retained separately |
| `legacy/` | Early small/large-system prototypes and regional preprocessing |
| `tools/` | NYISO and EV data download helper |
| `figures/` | Directional ablation plotting script |
| `docs/` | Reproduction instructions, limitations, code index and source manifest |

Sibling pipeline modules retain their original names and layout to preserve imports. Historical variants are labeled rather than silently substituted for the archived models.

## Installation

Use Python 3.10 or later in a dedicated environment:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Neural models and PINN experiments additionally require:

```bash
python -m pip install -r requirements-neural.txt
```

Requirements list direct dependencies; they are not an exact lockfile for every historical experiment.

## Quick verification without research datasets

```bash
cd pipeline
python run_section_iv_replay.py --check-only
```

This runs the existing checks of aging derivatives, CVaR, offline feasibility and information-prefix behavior using constructed cases. It does not reproduce manuscript results.

## Reproducing experiments

See [the reproduction guide](docs/REPRODUCIBILITY.md) for input locations, commands and external dependencies. Start with [the code index](docs/CODE_INDEX.md) to locate individual experiment drivers.

This is a **source-only release**. Raw charging records, model checkpoints, generated scenarios, experimental outputs, manuscripts and third-party PDFs are excluded. The source manifest records the original relative location and SHA-256 of each copied file.

The archived Section IV replay uses the semi-empirical aging model. It must not be described as independent validation of the calibrated PINN or the learned CVAE–diffusion joint distribution. Some later experiments require external PINN source and weights that are not included; see the reproduction guide.

## Credentials and data

Download helpers obtain keys from environment variables (`EIA_API_KEY`, `NREL_API_KEY`). Optional LLM requests use `OPENAI_API_KEY`, `GOOGLE_API_KEY` or `GEMINI_API_KEY`. Set keys in your shell; no `.env` loader is included. Do not commit credentials or raw individual charging records. Consult each data provider's access and redistribution terms before adding data.

No software license has been selected for this release. Public visibility alone does not grant an open-source license.
