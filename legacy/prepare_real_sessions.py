from __future__ import annotations

import argparse
import json
from json import JSONDecoder
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------
# Helpers for loading the ACN-Data export robustly
# --------------------------------------------------

def _safe_find(text: str, needle: str) -> int:
    idx = text.find(needle)
    if idx < 0:
        raise ValueError(f"Could not find required marker: {needle}")
    return idx


def load_acn_export(path: str | Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load an ACN-Data JSON export.

    This loader is intentionally robust to partially truncated files. It tries a
    standard JSON parse first. If that fails, it extracts the top-level _meta
    block and then incrementally decodes objects inside the _items array until
    parsing can no longer continue.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    # Fast path: valid JSON
    try:
        payload = json.loads(text)
        meta = payload.get("_meta", {})
        items = payload.get("_items", [])
        if not isinstance(items, list):
            raise ValueError("_items is not a list")
        return meta, items
    except json.JSONDecodeError:
        pass

    # Fallback: parse meta block + incrementally decode item objects
    meta: Dict[str, Any] = {}
    try:
        meta_key = _safe_find(text, '"_meta"')
        brace_start = text.find("{", meta_key)
        if brace_start >= 0:
            decoder = JSONDecoder()
            meta, _ = decoder.raw_decode(text[brace_start:])
    except Exception:
        meta = {}

    items: List[Dict[str, Any]] = []
    items_key = _safe_find(text, '"_items"')
    array_start = text.find("[", items_key)
    if array_start < 0:
        raise ValueError("Could not locate _items array in file")

    decoder = JSONDecoder()
    i = array_start + 1
    n = len(text)

    while i < n:
        # Skip whitespace, commas, and stray newlines
        while i < n and text[i] in " \r\n\t,":
            i += 1
        if i >= n:
            break
        if text[i] == "]":
            break
        if text[i] != "{":
            # Try to recover from trailing truncation or malformed tail.
            next_obj = text.find("{", i)
            if next_obj < 0:
                break
            i = next_obj

        try:
            obj, end = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                items.append(obj)
            i += end
        except json.JSONDecodeError:
            break

    if not items:
        raise ValueError("No session records could be parsed from the file")

    return meta, items


# --------------------------------------------------
# Session normalization
# --------------------------------------------------

def _parse_gmt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _extract_requested_kwh(user_inputs: Any) -> float:
    if not isinstance(user_inputs, list) or len(user_inputs) == 0:
        return np.nan

    vals: List[float] = []
    for item in user_inputs:
        if isinstance(item, dict) and item.get("kWhRequested") is not None:
            try:
                vals.append(float(item["kWhRequested"]))
            except (TypeError, ValueError):
                continue
    if not vals:
        return np.nan
    return float(vals[-1])


def build_sessions_dataframe(items: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(items).copy()
    if df.empty:
        raise ValueError("Session table is empty")

    required_cols = [
        "connectionTime",
        "disconnectTime",
        "doneChargingTime",
        "kWhDelivered",
        "sessionID",
        "stationID",
        "timezone",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["connection_utc"] = _parse_gmt(df["connectionTime"])
    df["disconnect_utc"] = _parse_gmt(df["disconnectTime"])
    df["done_charging_utc"] = _parse_gmt(df["doneChargingTime"])
    df["kWhDelivered"] = pd.to_numeric(df["kWhDelivered"], errors="coerce")

    # Row-wise timezone conversion because ACN exports can include timezone per row.
    # In this file the timezone appears to be America/Los_Angeles for all shown rows.
    local_conn = []
    local_disc = []
    local_done = []
    for _, row in df[["connection_utc", "disconnect_utc", "done_charging_utc", "timezone"]].iterrows():
        tz = row["timezone"] if pd.notna(row["timezone"]) else "UTC"
        try:
            local_conn.append(row["connection_utc"].tz_convert(tz) if pd.notna(row["connection_utc"]) else pd.NaT)
            local_disc.append(row["disconnect_utc"].tz_convert(tz) if pd.notna(row["disconnect_utc"]) else pd.NaT)
            local_done.append(row["done_charging_utc"].tz_convert(tz) if pd.notna(row["done_charging_utc"]) else pd.NaT)
        except Exception:
            local_conn.append(row["connection_utc"])
            local_disc.append(row["disconnect_utc"])
            local_done.append(row["done_charging_utc"])

    df["connection_local"] = pd.Series(local_conn, index=df.index)
    df["disconnect_local"] = pd.Series(local_disc, index=df.index)
    df["done_charging_local"] = pd.Series(local_done, index=df.index)

    df["requested_kwh"] = df.get("userInputs", pd.Series([None] * len(df))).apply(_extract_requested_kwh)

    # Core derived quantities
    df["session_duration_hours"] = (
        (df["disconnect_utc"] - df["connection_utc"]).dt.total_seconds() / 3600.0
    )
    df["active_charging_hours"] = (
        (df["done_charging_utc"] - df["connection_utc"]).dt.total_seconds() / 3600.0
    )
    df["idle_after_charge_hours"] = (
        (df["disconnect_utc"] - df["done_charging_utc"]).dt.total_seconds() / 3600.0
    )
    df["avg_power_kw"] = df["kWhDelivered"] / df["active_charging_hours"]

    # Temporal features from local timestamps
    df["connection_date"] = pd.to_datetime(df["connection_local"].dt.date)
    df["connection_hour"] = df["connection_local"].dt.hour
    df["disconnect_hour"] = df["disconnect_local"].dt.hour
    df["done_charging_hour"] = df["done_charging_local"].dt.hour
    df["weekday"] = df["connection_local"].dt.dayofweek
    df["is_weekend"] = df["weekday"] >= 5
    df["month"] = df["connection_local"].dt.month
    df["year"] = df["connection_local"].dt.year

    # Flags
    df["cross_day_disconnect"] = (
        df["disconnect_local"].dt.date != df["connection_local"].dt.date
    )
    df["cross_day_done_charging"] = (
        df["done_charging_local"].dt.date != df["connection_local"].dt.date
    )
    df["has_user_inputs"] = df["requested_kwh"].notna()

    return df


# --------------------------------------------------
# Filtering and aggregate statistics
# --------------------------------------------------

def clean_sessions(df: pd.DataFrame, max_duration_hours: float = 72.0) -> pd.DataFrame:
    out = df.copy()

    # Basic validity
    mask = (
        out["connection_utc"].notna()
        & out["disconnect_utc"].notna()
        & out["kWhDelivered"].notna()
        & (out["kWhDelivered"] >= 0)
        & (out["session_duration_hours"] > 0)
        & (out["session_duration_hours"] <= max_duration_hours)
    )

    # doneChargingTime can be missing; when available, impose consistency
    done_ok = (
        out["done_charging_utc"].isna()
        | (
            (out["done_charging_utc"] >= out["connection_utc"])
            & (out["done_charging_utc"] <= out["disconnect_utc"])
        )
    )
    mask &= done_ok

    out = out.loc[mask].copy()

    # Replace missing active charging hours with session duration fallback.
    missing_active = out["active_charging_hours"].isna() | (out["active_charging_hours"] <= 0)
    out.loc[missing_active, "active_charging_hours"] = out.loc[missing_active, "session_duration_hours"]
    out.loc[missing_active, "done_charging_local"] = out.loc[missing_active, "disconnect_local"]
    out.loc[missing_active, "done_charging_utc"] = out.loc[missing_active, "disconnect_utc"]

    out["idle_after_charge_hours"] = (
        (out["disconnect_utc"] - out["done_charging_utc"]).dt.total_seconds() / 3600.0
    )
    out["avg_power_kw"] = out["kWhDelivered"] / out["active_charging_hours"]

    # A loose upper bound for AC Level-2 sessions; keeps implausible rows out.
    out = out.loc[(out["avg_power_kw"] >= 0) & (out["avg_power_kw"] <= 25)].copy()

    return out.reset_index(drop=True)


# --------------------------------------------------
# Distribution tables for simulation input
# --------------------------------------------------

def make_distributions(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # Arrival hour distribution
    arrival_hour = (
        df.groupby("connection_hour").size().rename("count").reset_index()
        .sort_values("connection_hour")
    )
    arrival_hour["probability"] = arrival_hour["count"] / arrival_hour["count"].sum()

    # Stay duration rounded to hour bins, clipped at 24 h for direct simulation use
    stay_bins = np.clip(np.ceil(df["session_duration_hours"]).astype(int), 1, 24)
    stay_dist = (
        stay_bins.value_counts().sort_index().rename_axis("stay_hours").reset_index(name="count")
    )
    stay_dist["probability"] = stay_dist["count"] / stay_dist["count"].sum()

    # Delivered energy rounded to integer kWh bins, clipped at 100 kWh
    energy_bins = np.clip(np.ceil(df["kWhDelivered"]).astype(int), 1, 100)
    energy_dist = (
        energy_bins.value_counts().sort_index().rename_axis("energy_kwh_bin").reset_index(name="count")
    )
    energy_dist["probability"] = energy_dist["count"] / energy_dist["count"].sum()

    # Weekday-weekend arrival profile
    weekday_hour = (
        df.groupby(["is_weekend", "connection_hour"]).size().rename("count").reset_index()
        .sort_values(["is_weekend", "connection_hour"])
    )
    weekday_hour["probability_within_group"] = weekday_hour.groupby("is_weekend")["count"].transform(lambda x: x / x.sum())

    # Daily aggregate energy and sessions
    daily = (
        df.groupby("connection_date")
        .agg(
            sessions=("sessionID", "count"),
            total_energy_kwh=("kWhDelivered", "sum"),
            avg_session_energy_kwh=("kWhDelivered", "mean"),
        )
        .reset_index()
        .sort_values("connection_date")
    )

    return {
        "arrival_hour_distribution": arrival_hour,
        "stay_duration_distribution": stay_dist,
        "energy_distribution": energy_dist,
        "weekday_weekend_arrival_distribution": weekday_hour,
        "daily_summary": daily,
    }


# --------------------------------------------------
# Plotting
# --------------------------------------------------

def _base_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def make_plots(df: pd.DataFrame, out_dir: Path) -> None:
    _base_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Arrival hour histogram
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.hist(df["connection_hour"].dropna(), bins=np.arange(25) - 0.5, edgecolor="white")
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(24))
    ax.set_xlabel("Connection hour (local time)")
    ax.set_ylabel("Number of sessions")
    ax.set_title("Arrival-hour distribution")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "arrival_hour_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # 2) Session duration histogram
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    vals = df["session_duration_hours"].clip(upper=24)
    ax.hist(vals.dropna(), bins=30, edgecolor="white")
    ax.set_xlabel("Session duration (hours, clipped at 24)")
    ax.set_ylabel("Number of sessions")
    ax.set_title("Session-duration distribution")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "session_duration_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # 3) Delivered energy histogram
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.hist(df["kWhDelivered"].clip(upper=80).dropna(), bins=35, edgecolor="white")
    ax.set_xlabel("Delivered energy (kWh, clipped at 80)")
    ax.set_ylabel("Number of sessions")
    ax.set_title("Delivered-energy distribution")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "delivered_energy_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # 4) Weekday vs weekend hourly profile
    hourly = (
        df.groupby(["is_weekend", "connection_hour"]).size().rename("count").reset_index()
    )
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for is_weekend, label in [(False, "Weekday"), (True, "Weekend")]:
        sub = hourly.loc[hourly["is_weekend"] == is_weekend]
        ax.plot(sub["connection_hour"], sub["count"], marker="o", linewidth=1.8, label=label)
    ax.set_xticks(range(24))
    ax.set_xlabel("Connection hour (local time)")
    ax.set_ylabel("Number of sessions")
    ax.set_title("Hourly arrivals: weekday vs weekend")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "weekday_weekend_arrivals.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------
# Summary report
# --------------------------------------------------

def build_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    requested = df_clean["requested_kwh"].dropna()
    summary = {
        "site": meta.get("site"),
        "export_start": meta.get("start"),
        "export_end": meta.get("end"),
        "raw_rows": int(len(df_raw)),
        "clean_rows": int(len(df_clean)),
        "date_min": str(df_clean["connection_date"].min().date()) if not df_clean.empty else None,
        "date_max": str(df_clean["connection_date"].max().date()) if not df_clean.empty else None,
        "n_unique_stations": int(df_clean["stationID"].nunique()) if not df_clean.empty else 0,
        "mean_session_duration_h": float(df_clean["session_duration_hours"].mean()) if not df_clean.empty else np.nan,
        "median_session_duration_h": float(df_clean["session_duration_hours"].median()) if not df_clean.empty else np.nan,
        "mean_delivered_kwh": float(df_clean["kWhDelivered"].mean()) if not df_clean.empty else np.nan,
        "median_delivered_kwh": float(df_clean["kWhDelivered"].median()) if not df_clean.empty else np.nan,
        "mean_active_charging_h": float(df_clean["active_charging_hours"].mean()) if not df_clean.empty else np.nan,
        "mean_idle_after_charge_h": float(df_clean["idle_after_charge_hours"].mean()) if not df_clean.empty else np.nan,
        "mean_avg_power_kw": float(df_clean["avg_power_kw"].mean()) if not df_clean.empty else np.nan,
        "share_weekend": float(df_clean["is_weekend"].mean()) if not df_clean.empty else np.nan,
        "share_cross_day_disconnect": float(df_clean["cross_day_disconnect"].mean()) if not df_clean.empty else np.nan,
        "share_with_requested_kwh": float(df_clean["has_user_inputs"].mean()) if not df_clean.empty else np.nan,
        "mean_requested_kwh_when_available": float(requested.mean()) if not requested.empty else np.nan,
    }
    return pd.DataFrame([summary])


# --------------------------------------------------
# CLI
# --------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ACN-Data session exports for EV charging simulation inputs."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/acndata_sessions.json",
        help="Path to ACN export JSON file.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/acn_prepared",
        help="Directory for cleaned tables and figures.",
    )
    parser.add_argument(
        "--max-duration-hours",
        type=float,
        default=72.0,
        help="Maximum allowed session duration used during cleaning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta, items = load_acn_export(args.input)
    df_raw = build_sessions_dataframe(items)
    df_clean = clean_sessions(df_raw, max_duration_hours=args.max_duration_hours)
    distributions = make_distributions(df_clean)
    summary = build_summary(df_raw, df_clean, meta)

    # Persist outputs
    df_clean.to_csv(out_dir / "acn_sessions_clean.csv", index=False)
    summary.to_csv(out_dir / "acn_summary.csv", index=False)
    with (out_dir / "acn_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    for name, table in distributions.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)

    make_plots(df_clean, out_dir)

    print("Prepared ACN sessions successfully.")
    print(f"Raw rows:   {len(df_raw):,}")
    print(f"Clean rows: {len(df_clean):,}")
    print(f"Output dir: {out_dir}")
    print("Saved files:")
    for p in sorted(out_dir.iterdir()):
        print(f" - {p.name}")


if __name__ == "__main__":
    main()
