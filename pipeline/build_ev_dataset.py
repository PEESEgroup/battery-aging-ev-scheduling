#!/usr/bin/env python3
"""
Build a unified EV-session dataset from ACN-Data JSON and NREL workplace/fleet CSV.

Outputs
-------
1) ev_sessions_unified.csv
   One row per charging session with harmonized fields across sources.
2) ev_hourly_aggregates.csv
   Hourly aggregates useful for EV generator / world model training.
3) ev_daily_aggregates.csv
   Daily aggregates useful for demand and arrival-count modeling.

Designed to be robust to partially downloaded / truncated ACN JSON files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Generic helpers
# -----------------------------


def safe_float(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, str):
        x = x.strip()
        if x == "" or x.upper() == "NULL":
            return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def safe_bool(x: Any) -> Optional[bool]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        v = x.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    return None


def parse_timestamp(value: Any, tz_name: Optional[str] = None) -> pd.Timestamp:
    """
    Parse timestamps from either:
    - ACN style: 'Tue, 01 May 2018 00:31:56 GMT'
    - CSV style: '2016-11-03 10:01:03'

    If the parsed timestamp is timezone-aware and tz_name is provided,
    it is converted to that timezone. If it is naive and tz_name is provided,
    it is localized to that timezone.
    """
    if value is None:
        return pd.NaT
    if isinstance(value, float) and math.isnan(value):
        return pd.NaT
    if isinstance(value, str) and value.strip().upper() == "NULL":
        return pd.NaT

    ts = pd.to_datetime(value, errors="coerce", utc=False)
    if pd.isna(ts):
        return pd.NaT

    try:
        if ts.tzinfo is not None and tz_name:
            return ts.tz_convert(tz_name)
        if ts.tzinfo is None and tz_name:
            return ts.tz_localize(tz_name)
        return ts
    except Exception:
        return ts


def _to_naive_local_series(series: pd.Series) -> pd.Series:
    """
    Convert a Series containing mixed naive / tz-aware timestamps
    into naive local timestamps safe for pandas .dt operations.
    """
    def convert_one(x):
        if pd.isna(x):
            return pd.NaT
        try:
            ts = pd.Timestamp(x)
            if ts.tzinfo is not None:
                return ts.tz_localize(None)
            return ts
        except Exception:
            return pd.NaT

    return series.apply(convert_one)


def choose_latest_user_input(user_inputs: Any) -> Dict[str, Any]:
    """ACN 'userInputs' may be null or a list of revisions. Use the latest revision if possible."""
    if not user_inputs:
        return {}
    if isinstance(user_inputs, dict):
        return user_inputs
    if isinstance(user_inputs, list):
        if len(user_inputs) == 0:
            return {}
        try:
            enriched = []
            for item in user_inputs:
                mod = parse_timestamp(item.get("modifiedAt"), "UTC")
                enriched.append((mod, item))
            enriched.sort(key=lambda x: (pd.isna(x[0]), x[0]))
            return enriched[-1][1]
        except Exception:
            return user_inputs[-1]
    return {}


# -----------------------------
# Robust ACN parser
# -----------------------------


def load_acn_items(acn_path: Path) -> List[Dict[str, Any]]:
    """
    Load ACN items from a valid JSON file.
    If the file is truncated / malformed, salvage complete session objects from the _items array.
    """
    try:
        with acn_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "_items" in payload:
            return payload["_items"]
        if isinstance(payload, list):
            return payload
        raise ValueError("Unsupported ACN JSON structure.")
    except Exception:
        return salvage_acn_items(acn_path)


def salvage_acn_items(acn_path: Path) -> List[Dict[str, Any]]:
    text = acn_path.read_text(encoding="utf-8", errors="ignore")
    key = '"_items"'
    start = text.find(key)
    if start < 0:
        raise ValueError("Could not locate '_items' in ACN file.")

    array_start = text.find("[", start)
    if array_start < 0:
        raise ValueError("Could not locate start of ACN _items array.")

    items: List[Dict[str, Any]] = []
    i = array_start + 1
    n = len(text)
    in_string = False
    escape = False
    brace_depth = 0
    obj_start: Optional[int] = None

    while i < n:
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if brace_depth == 0:
                obj_start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and obj_start is not None:
                obj_text = text[obj_start: i + 1]
                try:
                    items.append(json.loads(obj_text))
                except json.JSONDecodeError:
                    break
                obj_start = None
        elif ch == "]" and brace_depth == 0:
            break
        i += 1

    if not items:
        raise ValueError("Failed to salvage any ACN session objects.")
    return items


# -----------------------------
# Source-specific builders
# -----------------------------


def build_acn_sessions(acn_path: Path) -> pd.DataFrame:
    items = load_acn_items(acn_path)
    rows: List[Dict[str, Any]] = []

    for rec in items:
        tz_name = rec.get("timezone") or None
        ui = choose_latest_user_input(rec.get("userInputs"))

        connection_time = parse_timestamp(rec.get("connectionTime"), tz_name)
        disconnect_time = parse_timestamp(rec.get("disconnectTime"), tz_name)
        done_time = parse_timestamp(rec.get("doneChargingTime"), tz_name)
        requested_departure = parse_timestamp(ui.get("requestedDeparture"), tz_name)

        energy_delivered = safe_float(rec.get("kWhDelivered"))
        energy_requested = safe_float(ui.get("kWhRequested"))
        miles_requested = safe_float(ui.get("milesRequested"))
        minutes_available = safe_float(ui.get("minutesAvailable"))
        wh_per_mile = safe_float(ui.get("WhPerMile"))
        payment_required = safe_bool(ui.get("paymentRequired"))

        parking_hours = None
        if pd.notna(connection_time) and pd.notna(disconnect_time):
            parking_hours = (disconnect_time - connection_time).total_seconds() / 3600.0

        charging_hours = None
        if pd.notna(connection_time) and pd.notna(done_time):
            charging_hours = (done_time - connection_time).total_seconds() / 3600.0

        requested_duration_hours = None
        if pd.notna(connection_time) and pd.notna(requested_departure):
            requested_duration_hours = (requested_departure - connection_time).total_seconds() / 3600.0
        elif minutes_available is not None:
            requested_duration_hours = minutes_available / 60.0

        avg_power_kw = None
        if energy_delivered is not None and charging_hours and charging_hours > 0:
            avg_power_kw = energy_delivered / charging_hours

        rows.append(
            {
                "source": "acn",
                "source_session_id": rec.get("sessionID"),
                "source_record_id": rec.get("_id"),
                "site": rec.get("siteID"),
                "site_name": None,
                "station_id": rec.get("stationID"),
                "space_id": rec.get("spaceID"),
                "cluster_id": rec.get("clusterID"),
                "user_id": ui.get("userID", rec.get("userID")),
                "vehicle_model": None,
                "timezone": tz_name,
                "request_time": connection_time,
                "arrival_time": connection_time,
                "departure_time": disconnect_time,
                "done_charging_time": done_time,
                "requested_departure": requested_departure,
                "request_duration_hours": requested_duration_hours,
                "parking_duration_hours": parking_hours,
                "charging_duration_hours": charging_hours,
                "energy_requested_kwh": energy_requested,
                "energy_delivered_kwh": energy_delivered,
                "max_charge_power_kw": None,
                "avg_charge_power_kw": avg_power_kw,
                "miles_requested": miles_requested,
                "wh_per_mile": wh_per_mile,
                "cost_for_session": None,
                "after_paid": None,
                "payment_required": payment_required,
                "is_claimed": bool(ui),
                "has_actual_departure": pd.notna(disconnect_time),
                "has_actual_energy": energy_delivered is not None,
            }
        )

    return pd.DataFrame(rows)


def build_nrel_sessions(nrel_csv_path: Path, default_timezone: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(nrel_csv_path)

    rows: List[Dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        tz_name = default_timezone

        request_time = parse_timestamp(rec.get("request_entry_time"), tz_name)
        expected_departure = parse_timestamp(rec.get("expected_departure"), tz_name)
        start_charge = parse_timestamp(rec.get("start_charge"), tz_name)
        end_charge = parse_timestamp(rec.get("termin_charge"), tz_name)
        start_date_time = parse_timestamp(rec.get("start_date_time"), tz_name)
        end_date_time = parse_timestamp(rec.get("end_date_time"), tz_name)

        arrival_time = start_charge if pd.notna(start_charge) else request_time
        departure_time = (
            end_charge
            if pd.notna(end_charge)
            else end_date_time
            if pd.notna(end_date_time)
            else expected_departure
        )
        done_time = end_charge if pd.notna(end_charge) else departure_time

        energy_requested = safe_float(rec.get("kwh_requested"))
        energy_delivered = safe_float(rec.get("energy_charged"))
        max_power_kw = safe_float(rec.get("max_charge_power"))
        cost_for_session = safe_float(rec.get("cost_for_session"))
        after_paid = safe_bool(rec.get("afterPaid"))
        controlled_duration = safe_float(rec.get("controlled_duration"))

        parking_hours = None
        if pd.notna(arrival_time) and pd.notna(departure_time):
            parking_hours = (departure_time - arrival_time).total_seconds() / 3600.0

        charging_hours = None
        if pd.notna(arrival_time) and pd.notna(done_time):
            charging_hours = (done_time - arrival_time).total_seconds() / 3600.0

        requested_duration_hours = None
        if pd.notna(request_time) and pd.notna(expected_departure):
            requested_duration_hours = (expected_departure - request_time).total_seconds() / 3600.0
        elif controlled_duration is not None:
            requested_duration_hours = controlled_duration / 60.0

        avg_power_kw = None
        if energy_delivered is not None and charging_hours and charging_hours > 0:
            avg_power_kw = energy_delivered / charging_hours

        rows.append(
            {
                "source": "nrel",
                "source_session_id": None,
                "source_record_id": None,
                "site": None,
                "site_name": "nrel_workplace_or_fleet",
                "station_id": rec.get("station"),
                "space_id": None,
                "cluster_id": None,
                "user_id": rec.get("driverId"),
                "vehicle_model": rec.get("vehicle_model"),
                "timezone": tz_name,
                "request_time": request_time,
                "arrival_time": arrival_time,
                "departure_time": departure_time,
                "done_charging_time": done_time,
                "requested_departure": expected_departure,
                "request_duration_hours": requested_duration_hours,
                "parking_duration_hours": parking_hours,
                "charging_duration_hours": charging_hours,
                "energy_requested_kwh": energy_requested,
                "energy_delivered_kwh": energy_delivered,
                "max_charge_power_kw": max_power_kw,
                "avg_charge_power_kw": avg_power_kw,
                "miles_requested": safe_float(rec.get("miles_requested")),
                "wh_per_mile": None,
                "cost_for_session": cost_for_session,
                "after_paid": after_paid,
                "payment_required": after_paid,
                "is_claimed": True,
                "has_actual_departure": pd.notna(departure_time),
                "has_actual_energy": energy_delivered is not None,
            }
        )

    return pd.DataFrame(rows)


# -----------------------------
# Feature engineering
# -----------------------------


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    base_time = out["arrival_time"].copy()
    if "request_time" in out.columns:
        base_time = base_time.where(base_time.notna(), out["request_time"])

    base_time = _to_naive_local_series(base_time)
    arrival_time = _to_naive_local_series(out["arrival_time"])
    departure_time = _to_naive_local_series(out["departure_time"])
    request_time = _to_naive_local_series(out["request_time"])

    out["date"] = base_time.dt.date
    out["year"] = base_time.dt.year
    out["month"] = base_time.dt.month
    out["day"] = base_time.dt.day
    out["day_of_week"] = base_time.dt.dayofweek
    out["day_name"] = base_time.dt.day_name()
    out["is_weekend"] = out["day_of_week"].isin([5, 6])

    out["arrival_hour"] = arrival_time.dt.hour + arrival_time.dt.minute / 60.0
    out["departure_hour"] = departure_time.dt.hour + departure_time.dt.minute / 60.0
    out["request_hour"] = request_time.dt.hour + request_time.dt.minute / 60.0

    return out


def filter_reasonable_sessions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "parking_duration_hours" in out.columns:
        out = out[(out["parking_duration_hours"].isna()) | (out["parking_duration_hours"] >= 0)]
        out = out[(out["parking_duration_hours"].isna()) | (out["parking_duration_hours"] <= 7 * 24)]
    if "charging_duration_hours" in out.columns:
        out = out[(out["charging_duration_hours"].isna()) | (out["charging_duration_hours"] >= 0)]
    if "energy_delivered_kwh" in out.columns:
        out = out[(out["energy_delivered_kwh"].isna()) | (out["energy_delivered_kwh"] >= 0)]
    if "energy_requested_kwh" in out.columns:
        out = out[(out["energy_requested_kwh"].isna()) | (out["energy_requested_kwh"] >= 0)]

    return out.reset_index(drop=True)


def build_hourly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()

    arrival_time = _to_naive_local_series(temp["arrival_time"])
    request_time = _to_naive_local_series(temp["request_time"])

    ts = arrival_time.where(arrival_time.notna(), request_time)
    temp["hour_ts"] = ts.dt.floor("h")

    agg = (
        temp.dropna(subset=["hour_ts"])
        .groupby(["source", "hour_ts"], dropna=False)
        .agg(
            n_sessions=("source", "size"),
            n_claimed=("is_claimed", "sum"),
            n_actual_energy=("has_actual_energy", "sum"),
            mean_arrival_hour=("arrival_hour", "mean"),
            mean_requested_kwh=("energy_requested_kwh", "mean"),
            mean_delivered_kwh=("energy_delivered_kwh", "mean"),
            total_delivered_kwh=("energy_delivered_kwh", "sum"),
            mean_parking_duration_h=("parking_duration_hours", "mean"),
            mean_charging_duration_h=("charging_duration_hours", "mean"),
            mean_max_power_kw=("max_charge_power_kw", "mean"),
        )
        .reset_index()
    )

    agg["year"] = agg["hour_ts"].dt.year
    agg["month"] = agg["hour_ts"].dt.month
    agg["day_of_week"] = agg["hour_ts"].dt.dayofweek
    agg["hour"] = agg["hour_ts"].dt.hour
    agg["is_weekend"] = agg["day_of_week"].isin([5, 6])

    return agg


def build_daily_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()

    arrival_time = _to_naive_local_series(temp["arrival_time"])
    request_time = _to_naive_local_series(temp["request_time"])

    ts = arrival_time.where(arrival_time.notna(), request_time)
    temp["day_ts"] = ts.dt.floor("d")

    agg = (
        temp.dropna(subset=["day_ts"])
        .groupby(["source", "day_ts"], dropna=False)
        .agg(
            n_sessions=("source", "size"),
            n_unique_users=("user_id", pd.Series.nunique),
            n_unique_stations=("station_id", pd.Series.nunique),
            total_requested_kwh=("energy_requested_kwh", "sum"),
            total_delivered_kwh=("energy_delivered_kwh", "sum"),
            mean_requested_kwh=("energy_requested_kwh", "mean"),
            mean_delivered_kwh=("energy_delivered_kwh", "mean"),
            mean_arrival_hour=("arrival_hour", "mean"),
            mean_departure_hour=("departure_hour", "mean"),
            mean_parking_duration_h=("parking_duration_hours", "mean"),
            mean_charging_duration_h=("charging_duration_hours", "mean"),
        )
        .reset_index()
    )

    agg["year"] = agg["day_ts"].dt.year
    agg["month"] = agg["day_ts"].dt.month
    agg["day_of_week"] = agg["day_ts"].dt.dayofweek
    agg["is_weekend"] = agg["day_of_week"].isin([5, 6])

    return agg


# -----------------------------
# Main
# -----------------------------


def build_dataset(
    acn_path: Optional[Path],
    nrel_path: Optional[Path],
    out_dir: Path,
    nrel_timezone: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames: List[pd.DataFrame] = []

    if acn_path is not None:
        acn_df = build_acn_sessions(acn_path)
        frames.append(acn_df)

    if nrel_path is not None:
        nrel_df = build_nrel_sessions(nrel_path, nrel_timezone)
        frames.append(nrel_df)

    if not frames:
        raise ValueError("At least one of ACN or NREL input must be provided.")

    sessions = pd.concat(frames, ignore_index=True, sort=False)
    sessions = add_calendar_features(sessions)
    sessions = filter_reasonable_sessions(sessions)

    sort_cols = [c for c in ["source", "arrival_time", "request_time", "station_id"] if c in sessions.columns]
    sessions = sessions.sort_values(sort_cols).reset_index(drop=True)

    hourly = build_hourly_aggregates(sessions)
    daily = build_daily_aggregates(sessions)

    out_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(out_dir / "ev_sessions_unified.csv", index=False)
    hourly.to_csv(out_dir / "ev_hourly_aggregates.csv", index=False)
    daily.to_csv(out_dir / "ev_daily_aggregates.csv", index=False)

    return sessions, hourly, daily


def main() -> None:
    acn_path = Path("acndata_sessions.json")
    nrel_path = Path("dataSet_Oct2021.csv")
    out_dir = Path("processed")

    # 按你的当前设定先保留这个时区；若后面发现 NREL 原始数据不是 Denver 时间，再改
    nrel_timezone = "America/Denver"

    sessions, hourly, daily = build_dataset(
        acn_path=acn_path,
        nrel_path=nrel_path,
        out_dir=out_dir,
        nrel_timezone=nrel_timezone,
    )

    print(f"[ok] sessions: {len(sessions):,} -> {out_dir / 'ev_sessions_unified.csv'}")
    print(f"[ok] hourly aggregates: {len(hourly):,} -> {out_dir / 'ev_hourly_aggregates.csv'}")
    print(f"[ok] daily aggregates: {len(daily):,} -> {out_dir / 'ev_daily_aggregates.csv'}")
    print("[done] Build complete.")


if __name__ == "__main__":
    main()

