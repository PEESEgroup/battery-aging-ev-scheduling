#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_grid_dataset.py

Build a unified NYISO grid dataset from:
1) isolf      : zonal/system load
2) damlbmp    : zonal day-ahead prices
3) dam_imer   : zonal average / implied marginal emission rates
4) optional   : Gold Book-based long-horizon scenario metadata

Default usage:
    python build_grid_dataset.py

Expected current-directory files:
    *_isolf.csv                or files under raw/load/
    *_damlbmp_zone.csv         or files under raw/price/
    *_dam_imer.csv             or files under raw/emission/

Outputs (default: processed_grid/):
    - grid_hourly_zone.csv
    - grid_hourly_system.csv
    - grid_generator_ready.csv
    - goldbook_scenario_metadata.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class GridBuildConfig:
    base_dir: Path = Path(".")
    out_dir: Path = Path("processed_grid")

    # optional subdirectories if you organize files later
    load_dir: Path = Path("raw/load")
    price_dir: Path = Path("raw/price")
    emission_dir: Path = Path("raw/emission")

    # current filenames / glob patterns
    load_globs: Tuple[str, ...] = ("*isolf.csv", "raw/load/*isolf.csv")
    price_globs: Tuple[str, ...] = ("*damlbmp_zone.csv", "raw/price/*damlbmp_zone.csv")
    emission_globs: Tuple[str, ...] = ("*dam_imer.csv", "raw/emission/*dam_imer.csv")

    # whether to include external interfaces from price files
    keep_external_price_nodes: bool = False

    # if True, create simple Gold Book metadata table
    include_goldbook_metadata: bool = True


# -----------------------------
# Constants / zone mapping
# -----------------------------

ZONE_ALIASES = {
    "CAPITL": "CAPITL",
    "CENTRL": "CENTRL",
    "DUNWOD": "DUNWOD",
    "GENESE": "GENESE",
    "HUD VL": "HUD VL",
    "HUD_VL": "HUD VL",
    "LONGIL": "LONGIL",
    "MHK VL": "MHK VL",
    "MHK_VL": "MHK VL",
    "MILLWD": "MILLWD",
    "N.Y.C.": "N.Y.C.",
    "NYC": "N.Y.C.",
    "NORTH": "NORTH",
    "WEST": "WEST",
    "NYISO": "NYISO",
}

# The 11 NYISO load zones used consistently in load + IMER files
VALID_LOAD_ZONES = [
    "CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUD VL",
    "LONGIL", "MHK VL", "MILLWD", "N.Y.C.", "NORTH", "WEST"
]

# Price file may contain external / interface nodes; keep only zones unless user wants otherwise
EXTERNAL_PRICE_NODES = {"H Q", "NPX", "O H", "PJM"}

# Gold Book metadata distilled from the 2025 Gold Book
# These are intentionally compact "scenario anchor" values for long-horizon scaling / scenario tagging.
GOLDBOOK_SCENARIO_METADATA = [
    {"scenario": "baseline", "avg_temp_trend_degF_per_decade": 0.7, "ldv_ev_sales_target_2035": 0.75,
     "ev_stock_2040_million": 6.285, "managed_charging": "increasing_over_time"},
    {"scenario": "lower", "avg_temp_trend_degF_per_decade": 0.7, "ldv_ev_sales_target_2035": None,
     "ev_stock_2040_million": 4.0, "managed_charging": "greater_share_than_baseline"},
    {"scenario": "higher", "avg_temp_trend_degF_per_decade": 0.7, "ldv_ev_sales_target_2035": 1.0,
     "ev_stock_2040_million": 8.0, "managed_charging": "reduced_share_than_baseline"},
]


# -----------------------------
# Utilities
# -----------------------------

def find_files(base_dir: Path, patterns: Tuple[str, ...]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(base_dir.glob(pat)))
    # deduplicate while preserving order
    seen = set()
    out = []
    for f in files:
        key = str(f.resolve())
        if key not in seen:
            out.append(f)
            seen.add(key)
    return out


def parse_file_date_from_name(path: Path) -> Optional[pd.Timestamp]:
    """
    Parse date from filenames like 20250101isolf.csv, 20250101damlbmp_zone.csv, etc.
    """
    m = re.search(r"(20\d{6})", path.name)
    if not m:
        return None
    return pd.to_datetime(m.group(1), format="%Y%m%d", errors="coerce")


def ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def month_to_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "winter"
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    return "fall"


# -----------------------------
# Load processing
# -----------------------------

def load_isolf_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "Time Stamp" not in df.columns:
        raise ValueError(f"'Time Stamp' column not found in {path}")

    df["timestamp"] = ensure_datetime(df["Time Stamp"])
    long_rows = []

    zone_cols = [c for c in df.columns if c not in {"Time Stamp", "timestamp"}]
    for z in zone_cols:
        zone = ZONE_ALIASES.get(z.strip(), z.strip())
        if zone not in VALID_LOAD_ZONES and zone != "NYISO":
            continue
        tmp = df[["timestamp", z]].copy()
        tmp.columns = ["timestamp", "load_mw"]
        tmp["zone"] = zone
        long_rows.append(tmp)

    out = pd.concat(long_rows, ignore_index=True)
    out["source_file"] = path.name
    out["file_date"] = parse_file_date_from_name(path)
    out["load_mw"] = pd.to_numeric(out["load_mw"], errors="coerce")
    return out.dropna(subset=["timestamp"]).reset_index(drop=True)


def build_load_table(files: List[Path]) -> pd.DataFrame:
    if not files:
        raise ValueError("No load files found.")
    parts = [load_isolf_file(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["timestamp", "zone"]).reset_index(drop=True)
    return df


# -----------------------------
# Price processing
# -----------------------------

def load_price_file(path: Path, keep_external: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    required = ["Time Stamp", "Name", "LBMP ($/MWHr)"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{c} not found in {path}")

    df["timestamp"] = ensure_datetime(df["Time Stamp"])
    df["zone"] = df["Name"].astype(str).str.strip()
    df["zone"] = df["zone"].replace(ZONE_ALIASES)

    if keep_external:
        allowed = set(VALID_LOAD_ZONES) | EXTERNAL_PRICE_NODES
    else:
        allowed = set(VALID_LOAD_ZONES)

    df = df[df["zone"].isin(allowed)].copy()

    out = pd.DataFrame({
        "timestamp": df["timestamp"],
        "zone": df["zone"],
        "lbmp_mwh": pd.to_numeric(df["LBMP ($/MWHr)"], errors="coerce"),
        "loss_mwh": pd.to_numeric(df.get("Marginal Cost Losses ($/MWHr)"), errors="coerce"),
        "congestion_mwh": pd.to_numeric(df.get("Marginal Cost Congestion ($/MWHr)"), errors="coerce"),
    })
    out["source_file"] = path.name
    out["file_date"] = parse_file_date_from_name(path)
    return out.dropna(subset=["timestamp"]).reset_index(drop=True)


def build_price_table(files: List[Path], keep_external: bool = False) -> pd.DataFrame:
    if not files:
        raise ValueError("No price files found.")
    parts = [load_price_file(f, keep_external=keep_external) for f in files]
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["timestamp", "zone"]).reset_index(drop=True)
    return df


# -----------------------------
# Emission processing
# -----------------------------

def load_emission_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    required = ["Timestamp", "Zone"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{c} not found in {path}")

    df["timestamp"] = ensure_datetime(df["Timestamp"])
    df["zone"] = df["Zone"].astype(str).str.strip()
    df["zone"] = df["zone"].replace(ZONE_ALIASES)
    df = df[df["zone"].isin(VALID_LOAD_ZONES)].copy()

    out = pd.DataFrame({
        "timestamp": df["timestamp"],
        "zone": df["zone"],
        "lbmp_proxy_mwh": pd.to_numeric(df.get("LBMP - ($/MWh)"), errors="coerce"),
        "vom_mwh": pd.to_numeric(df.get("VOM - ($/MWh)"), errors="coerce"),
        "carbon_ton_per_mmbtu": pd.to_numeric(df.get("Tons of Carbon per mmBTU"), errors="coerce"),
        "nox_ton_per_mmbtu": pd.to_numeric(df.get("Tons of NOx per mmBTU"), errors="coerce"),
        "ihr_mmbtu_per_mwh": pd.to_numeric(df.get("Implied Heat Rate - IHRi (mmBTU/MWh)"), errors="coerce"),
        "imer_co2_ton_per_mwh": pd.to_numeric(df.get("IMER - CO2 (tons/MWh)"), errors="coerce"),
        "imer_nox_ton_per_mwh": pd.to_numeric(df.get("IMER - NOx (tons/MWh)"), errors="coerce"),
    })
    out["source_file"] = path.name
    out["file_date"] = parse_file_date_from_name(path)
    return out.dropna(subset=["timestamp"]).reset_index(drop=True)


def build_emission_table(files: List[Path]) -> pd.DataFrame:
    if not files:
        raise ValueError("No emission files found.")
    parts = [load_emission_file(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["timestamp", "zone"]).reset_index(drop=True)
    return df


# -----------------------------
# Merge + feature engineering
# -----------------------------

def build_zone_level_dataset(load_df: pd.DataFrame,
                             price_df: pd.DataFrame,
                             emission_df: pd.DataFrame) -> pd.DataFrame:
    # load table includes NYISO system total; remove here for zone-level merge
    zone_load = load_df[load_df["zone"].isin(VALID_LOAD_ZONES)].copy()

    df = zone_load.merge(
        price_df[["timestamp", "zone", "lbmp_mwh", "loss_mwh", "congestion_mwh"]],
        on=["timestamp", "zone"],
        how="left"
    ).merge(
        emission_df[[
            "timestamp", "zone", "lbmp_proxy_mwh", "vom_mwh", "carbon_ton_per_mmbtu",
            "nox_ton_per_mmbtu", "ihr_mmbtu_per_mwh", "imer_co2_ton_per_mwh", "imer_nox_ton_per_mwh"
        ]],
        on=["timestamp", "zone"],
        how="left"
    )

    df = add_grid_features(df)
    return df


def add_grid_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], errors="coerce")

    out["date"] = ts.dt.date
    out["year"] = ts.dt.year
    out["month"] = ts.dt.month
    out["day"] = ts.dt.day
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["is_weekend"] = out["day_of_week"] >= 5
    out["day_of_year"] = ts.dt.dayofyear
    out["season"] = out["month"].apply(month_to_season)

    # cyclical encodings
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)

    return out


def build_system_level_dataset(zone_df: pd.DataFrame, load_df: pd.DataFrame) -> pd.DataFrame:
    # system load from original load file if available
    nyiso_load = load_df[load_df["zone"] == "NYISO"][["timestamp", "load_mw"]].copy()
    nyiso_load = nyiso_load.rename(columns={"load_mw": "system_load_mw"})

    # load-weighted price / emissions
    z = zone_df.copy()
    z["load_mw_for_weight"] = z["load_mw"].clip(lower=0)
    grp = z.groupby("timestamp", as_index=False)

    def wavg(g: pd.DataFrame, value_col: str, weight_col: str = "load_mw_for_weight") -> float:
        x = pd.to_numeric(g[value_col], errors="coerce")
        w = pd.to_numeric(g[weight_col], errors="coerce")
        mask = x.notna() & w.notna() & (w > 0)
        if mask.sum() == 0:
            return np.nan
        return float(np.average(x[mask], weights=w[mask]))

    system_rows = []
    for ts, g in grp:
        system_rows.append({
            "timestamp": ts,
            "weighted_lbmp_mwh": wavg(g, "lbmp_mwh"),
            "weighted_loss_mwh": wavg(g, "loss_mwh"),
            "weighted_congestion_mwh": wavg(g, "congestion_mwh"),
            "weighted_imer_co2_ton_per_mwh": wavg(g, "imer_co2_ton_per_mwh"),
            "weighted_imer_nox_ton_per_mwh": wavg(g, "imer_nox_ton_per_mwh"),
            "weighted_ihr_mmbtu_per_mwh": wavg(g, "ihr_mmbtu_per_mwh"),
            "mean_zone_price_mwh": pd.to_numeric(g["lbmp_mwh"], errors="coerce").mean(),
            "max_zone_price_mwh": pd.to_numeric(g["lbmp_mwh"], errors="coerce").max(),
            "min_zone_price_mwh": pd.to_numeric(g["lbmp_mwh"], errors="coerce").min(),
            "mean_zone_imer_co2_ton_per_mwh": pd.to_numeric(g["imer_co2_ton_per_mwh"], errors="coerce").mean(),
            "total_zonal_load_mw": pd.to_numeric(g["load_mw"], errors="coerce").sum(),
        })

    system_df = pd.DataFrame(system_rows)
    system_df = system_df.merge(nyiso_load, on="timestamp", how="left")
    system_df["system_load_mw"] = system_df["system_load_mw"].fillna(system_df["total_zonal_load_mw"])

    # congestion proxy
    system_df["price_spread_mwh"] = system_df["max_zone_price_mwh"] - system_df["min_zone_price_mwh"]
    system_df["congestion_intensity"] = system_df["weighted_congestion_mwh"].fillna(0.0).abs() + system_df["price_spread_mwh"].fillna(0.0)

    system_df = add_grid_features(system_df)

    # lags useful for generator training
    system_df = system_df.sort_values("timestamp").reset_index(drop=True)
    for lag in [1, 2, 24]:
        for col in ["system_load_mw", "weighted_lbmp_mwh", "weighted_imer_co2_ton_per_mwh", "congestion_intensity"]:
            system_df[f"{col}_lag{lag}"] = system_df[col].shift(lag)

    # next-step targets for world model style training
    for col in ["system_load_mw", "weighted_lbmp_mwh", "weighted_imer_co2_ton_per_mwh", "congestion_intensity"]:
        system_df[f"{col}_next1"] = system_df[col].shift(-1)

    return system_df


def build_goldbook_metadata() -> pd.DataFrame:
    return pd.DataFrame(GOLDBOOK_SCENARIO_METADATA)


# -----------------------------
# Main build
# -----------------------------

def build_all(config: GridBuildConfig) -> Dict[str, pd.DataFrame]:
    load_files = find_files(config.base_dir, config.load_globs)
    price_files = find_files(config.base_dir, config.price_globs)
    emission_files = find_files(config.base_dir, config.emission_globs)

    if len(load_files) == 0:
        raise FileNotFoundError("No load files found matching patterns: " + ", ".join(config.load_globs))
    if len(price_files) == 0:
        raise FileNotFoundError("No price files found matching patterns: " + ", ".join(config.price_globs))
    if len(emission_files) == 0:
        raise FileNotFoundError("No emission files found matching patterns: " + ", ".join(config.emission_globs))

    print(f"[info] load files found     : {len(load_files)}")
    print(f"[info] price files found    : {len(price_files)}")
    print(f"[info] emission files found : {len(emission_files)}")

    load_df = build_load_table(load_files)
    price_df = build_price_table(price_files, keep_external=config.keep_external_price_nodes)
    emission_df = build_emission_table(emission_files)

    zone_df = build_zone_level_dataset(load_df, price_df, emission_df)
    system_df = build_system_level_dataset(zone_df, load_df)

    # generator-ready table: drop timestamps without all core signals and keep training columns
    gen_cols = [
        "timestamp", "system_load_mw", "weighted_lbmp_mwh", "weighted_imer_co2_ton_per_mwh",
        "congestion_intensity", "weighted_loss_mwh", "weighted_congestion_mwh",
        "hour", "day_of_week", "month", "is_weekend",
        "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
        "system_load_mw_lag1", "weighted_lbmp_mwh_lag1", "weighted_imer_co2_ton_per_mwh_lag1", "congestion_intensity_lag1",
        "system_load_mw_lag2", "weighted_lbmp_mwh_lag2", "weighted_imer_co2_ton_per_mwh_lag2", "congestion_intensity_lag2",
        "system_load_mw_lag24", "weighted_lbmp_mwh_lag24", "weighted_imer_co2_ton_per_mwh_lag24", "congestion_intensity_lag24",
        "system_load_mw_next1", "weighted_lbmp_mwh_next1", "weighted_imer_co2_ton_per_mwh_next1", "congestion_intensity_next1",
    ]
    grid_gen_df = system_df[gen_cols].copy()
    grid_gen_df = grid_gen_df.dropna(subset=[
        "system_load_mw", "weighted_lbmp_mwh", "weighted_imer_co2_ton_per_mwh", "congestion_intensity"
    ]).reset_index(drop=True)

    config.out_dir.mkdir(parents=True, exist_ok=True)
    zone_df.to_csv(config.out_dir / "grid_hourly_zone.csv", index=False)
    system_df.to_csv(config.out_dir / "grid_hourly_system.csv", index=False)
    grid_gen_df.to_csv(config.out_dir / "grid_generator_ready.csv", index=False)

    outputs = {
        "grid_hourly_zone": zone_df,
        "grid_hourly_system": system_df,
        "grid_generator_ready": grid_gen_df,
    }

    if config.include_goldbook_metadata:
        gold_df = build_goldbook_metadata()
        gold_df.to_csv(config.out_dir / "goldbook_scenario_metadata.csv", index=False)
        outputs["goldbook_scenario_metadata"] = gold_df

    return outputs


def main() -> None:
    config = GridBuildConfig()
    outputs = build_all(config)

    print(f"[ok] zone-level dataset      -> {config.out_dir / 'grid_hourly_zone.csv'} ({len(outputs['grid_hourly_zone']):,} rows)")
    print(f"[ok] system-level dataset    -> {config.out_dir / 'grid_hourly_system.csv'} ({len(outputs['grid_hourly_system']):,} rows)")
    print(f"[ok] grid-generator dataset  -> {config.out_dir / 'grid_generator_ready.csv'} ({len(outputs['grid_generator_ready']):,} rows)")

    if "goldbook_scenario_metadata" in outputs:
        print(f"[ok] gold book metadata      -> {config.out_dir / 'goldbook_scenario_metadata.csv'} ({len(outputs['goldbook_scenario_metadata']):,} rows)")

    print("[done] build_grid_dataset complete.")


if __name__ == "__main__":
    main()
