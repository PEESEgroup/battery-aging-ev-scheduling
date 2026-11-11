import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def hhmm_to_hour(x):
    """
    Convert HHMM integer-like values to decimal hours.
    Examples:
        730 -> 7.5
        45  -> 0.75
        2400/0 -> 0.0
    """
    if pd.isna(x):
        return np.nan
    try:
        x = int(float(x))
    except Exception:
        return np.nan
    if x >= 2400:
        x = x % 2400
    h = x // 100
    m = x % 100
    if h > 23 or m > 59:
        return np.nan
    return h + m / 60.0


def weighted_quantile(values, quantiles, sample_weight=None):
    values = np.asarray(values)
    quantiles = np.asarray(quantiles)

    if sample_weight is None:
        sample_weight = np.ones_like(values, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)

    valid = np.isfinite(values) & np.isfinite(sample_weight)
    values = values[valid]
    sample_weight = sample_weight[valid]

    if len(values) == 0:
        return np.full(len(quantiles), np.nan)

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    weighted_cdf = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_cdf = weighted_cdf / np.sum(sample_weight)
    return np.interp(quantiles, weighted_cdf, values)


def plot_hist(series, weights, bins, title, xlabel, output_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(series, bins=bins, weights=weights, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Weighted count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def find_day_col(df, candidates=("TDAYDATE", "TDAY", "TRAVDAY")):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -----------------------------
# Main processing
# -----------------------------
def prepare_nhts_us(
    hh_path,
    trip_path,
    veh_path,
    out_dir,
    min_stay_hours=0.25,
    max_stay_hours=24.0,
    kwh_per_mile=0.30,
    include_overnight=True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading files...")
    hh = pd.read_csv(hh_path, low_memory=False)
    trip = pd.read_csv(trip_path, low_memory=False)
    veh = pd.read_csv(veh_path, low_memory=False)

    # -----------------------------
    # 1) Select and standardize columns
    # -----------------------------
    trip_day_col = find_day_col(trip)

    trip_base_cols = [
        "HOUSEID", "PERSONID", "TRIPID", "SEQ_TRIPID",
        "STRTTIME", "ENDTIME", "TRPMILES", "VEHID",
        "TRPHHVEH", "TRPTRANS", "DWELTIME", "WTTRDFIN",
        "CENSUS_R", "CENSUS_D",
        "CDIVMSAR",   # ⭐ finer regional unit
        "URBAN", "URBRUR", "MSASIZE"
    ]
    trip_cols = [c for c in trip_base_cols if c in trip.columns]

    if trip_day_col is not None and trip_day_col not in trip_cols:
        trip_cols.append(trip_day_col)

    trip = trip[trip_cols].copy()

    # Only trip keeps the daily identifier used later
    if trip_day_col is not None:
        trip = trip.rename(columns={trip_day_col: "DAY_ID"})
    else:
        trip["DAY_ID"] = 1

    # Rename weights
    if "WTTRDFIN" in trip.columns:
        trip = trip.rename(columns={"WTTRDFIN": "trip_weight"})
    else:
        trip["trip_weight"] = 1.0

    # -----------------------------
    # 2) Basic cleaning
    # -----------------------------
    print("Basic cleaning...")

    if "TRPHHVEH" in trip.columns:
        trip = trip[trip["TRPHHVEH"] == 1]

    trip["TRPMILES"] = pd.to_numeric(trip["TRPMILES"], errors="coerce")
    trip["VEHID"] = pd.to_numeric(trip["VEHID"], errors="coerce")
    trip["trip_weight"] = pd.to_numeric(trip["trip_weight"], errors="coerce")

    trip = trip[trip["TRPMILES"].fillna(0) > 0]
    trip = trip[trip["VEHID"].fillna(-1) > 0]

    # Time conversion
    trip["start_hour"] = trip["STRTTIME"].apply(hhmm_to_hour)
    trip["end_hour"] = trip["ENDTIME"].apply(hhmm_to_hour)

    trip = trip.dropna(subset=["start_hour", "end_hour", "TRPMILES"])

    # -----------------------------
    # 3) Region key: use CDIVMSAR first, fallback to Census division
    # -----------------------------
    if "CDIVMSAR" in trip.columns:
        trip["CDIVMSAR"] = pd.to_numeric(trip["CDIVMSAR"], errors="coerce")
    else:
        trip["CDIVMSAR"] = np.nan

    has_cdivmsar = trip["CDIVMSAR"].notna()

    trip["region_key"] = np.where(
        has_cdivmsar,
        "CDIV_" + trip["CDIVMSAR"].astype("Int64").astype(str),
        np.where(
            ("CENSUS_R" in trip.columns) & ("CENSUS_D" in trip.columns),
            "R" + trip["CENSUS_R"].astype("Int64").astype(str) +
            "_D" + trip["CENSUS_D"].astype("Int64").astype(str),
            "US"
        )
    )

    # Weight
    trip["weight"] = trip["trip_weight"].fillna(1.0)

    # Energy proxy
    trip["energy_kwh_trip"] = trip["TRPMILES"] * kwh_per_mile

    # -----------------------------
    # 4) Inter-trip parking opportunities
    # -----------------------------
    print("Building inter-trip parking opportunities...")
    trip = trip.sort_values(
        ["HOUSEID", "PERSONID", "DAY_ID", "start_hour"]
    ).reset_index(drop=True)

    trip["next_start_hour"] = trip.groupby(
        ["HOUSEID", "PERSONID", "DAY_ID"]
    )["start_hour"].shift(-1)

    trip["intertrip_stay_hours"] = trip["next_start_hour"] - trip["end_hour"]

    intertrip = trip.dropna(subset=["intertrip_stay_hours"]).copy()
    intertrip = intertrip[
        (intertrip["intertrip_stay_hours"] >= min_stay_hours) &
        (intertrip["intertrip_stay_hours"] <= max_stay_hours)
    ].copy()

    intertrip["connection_hour"] = intertrip["end_hour"]
    intertrip["session_duration_hours"] = intertrip["intertrip_stay_hours"]
    intertrip["session_energy_kwh"] = intertrip["energy_kwh_trip"].clip(lower=0.5)

    intertrip_out = intertrip[[
        "HOUSEID", "PERSONID", "VEHID", "DAY_ID", "region_key", "weight",
        "connection_hour", "session_duration_hours", "session_energy_kwh"
    ]].copy()
    intertrip_out["session_type"] = "intertrip"

    # -----------------------------
    # 5) Overnight charging opportunities
    # -----------------------------
    overnight = pd.DataFrame()

    if include_overnight:
        print("Building overnight charging sessions...")

        veh_day = trip.groupby(
            ["HOUSEID", "VEHID", "DAY_ID"], as_index=False
        ).agg(
            day_energy_kwh=("energy_kwh_trip", "sum"),
            last_arrival=("end_hour", "max"),
            weight=("weight", "first"),
            region_key=("region_key", "first"),
        )

        veh_day = veh_day.sort_values(["HOUSEID", "VEHID", "DAY_ID"]).reset_index(drop=True)

        first_depart = trip.groupby(
            ["HOUSEID", "VEHID", "DAY_ID"], as_index=False
        ).agg(
            first_departure=("start_hour", "min")
        )

        next_map = first_depart.copy()
        next_map["DAY_ID_prev"] = next_map["DAY_ID"] - 1
        next_map = next_map[["HOUSEID", "VEHID", "DAY_ID_prev", "first_departure"]]

        veh_day = veh_day.merge(
            next_map,
            left_on=["HOUSEID", "VEHID", "DAY_ID"],
            right_on=["HOUSEID", "VEHID", "DAY_ID_prev"],
            how="left"
        )

        veh_day["next_day_first_departure"] = veh_day["first_departure"].fillna(7.0)

        veh_day["overnight_stay_hours"] = (
            (24.0 - veh_day["last_arrival"]) + veh_day["next_day_first_departure"]
        )
        veh_day["overnight_stay_hours"] = veh_day["overnight_stay_hours"].clip(
            lower=min_stay_hours, upper=max_stay_hours
        )

        overnight = veh_day.copy()
        overnight["connection_hour"] = overnight["last_arrival"]
        overnight["session_duration_hours"] = overnight["overnight_stay_hours"]
        overnight["session_energy_kwh"] = overnight["day_energy_kwh"].clip(lower=1.0)

        overnight = overnight[[
            "HOUSEID", "VEHID", "DAY_ID", "region_key", "weight",
            "connection_hour", "session_duration_hours", "session_energy_kwh"
        ]].copy()
        overnight["session_type"] = "overnight"

    # -----------------------------
    # 6) Final session table
    # -----------------------------
    sessions = pd.concat([intertrip_out, overnight], ignore_index=True, sort=False)

    sessions = sessions.dropna(subset=[
        "connection_hour", "session_duration_hours", "session_energy_kwh"
    ])
    sessions = sessions[
        (sessions["session_duration_hours"] >= min_stay_hours) &
        (sessions["session_duration_hours"] <= max_stay_hours) &
        (sessions["session_energy_kwh"] > 0)
    ].copy()

    sessions.to_csv(out_dir / "nhts_charging_sessions_us.csv", index=False)

    # -----------------------------
    # 7) Save empirical distributions
    # -----------------------------
    print("Saving empirical distributions...")

    sessions[["connection_hour", "weight"]].to_csv(
        out_dir / "arrival_dist_us.csv", index=False
    )
    sessions[["session_duration_hours", "weight"]].to_csv(
        out_dir / "stay_dist_us.csv", index=False
    )
    sessions[["session_energy_kwh", "weight"]].to_csv(
        out_dir / "energy_dist_us.csv", index=False
    )

    sessions[["region_key", "connection_hour", "weight"]].to_csv(
        out_dir / "arrival_dist_by_region.csv", index=False
    )
    sessions[["region_key", "session_duration_hours", "weight"]].to_csv(
        out_dir / "stay_dist_by_region.csv", index=False
    )
    sessions[["region_key", "session_energy_kwh", "weight"]].to_csv(
        out_dir / "energy_dist_by_region.csv", index=False
    )

    # Regional summary
    regional_summary = sessions.groupby("region_key").apply(
        lambda g: pd.Series({
            "n_sessions": len(g),
            "weighted_sessions": g["weight"].sum(),
            "arrival_p25": weighted_quantile(g["connection_hour"], [0.25], g["weight"])[0],
            "arrival_p50": weighted_quantile(g["connection_hour"], [0.50], g["weight"])[0],
            "arrival_p75": weighted_quantile(g["connection_hour"], [0.75], g["weight"])[0],
            "stay_p25": weighted_quantile(g["session_duration_hours"], [0.25], g["weight"])[0],
            "stay_p50": weighted_quantile(g["session_duration_hours"], [0.50], g["weight"])[0],
            "stay_p75": weighted_quantile(g["session_duration_hours"], [0.75], g["weight"])[0],
            "energy_p25": weighted_quantile(g["session_energy_kwh"], [0.25], g["weight"])[0],
            "energy_p50": weighted_quantile(g["session_energy_kwh"], [0.50], g["weight"])[0],
            "energy_p75": weighted_quantile(g["session_energy_kwh"], [0.75], g["weight"])[0],
        })
    ).reset_index()

    regional_summary.to_csv(out_dir / "regional_summary_stats.csv", index=False)

    # -----------------------------
    # 8) Diagnostic plots
    # -----------------------------
    print("Saving diagnostic plots...")

    plot_hist(
        sessions["connection_hour"],
        sessions["weight"],
        bins=np.arange(0, 25, 1),
        title="NHTS-derived charging session arrival time",
        xlabel="Connection hour",
        output_path=out_dir / "arrival_hist_us.png",
    )

    plot_hist(
        sessions["session_duration_hours"],
        sessions["weight"],
        bins=np.arange(0, 37, 1),
        title="NHTS-derived charging session duration",
        xlabel="Session duration (hours)",
        output_path=out_dir / "stay_hist_us.png",
    )

    energy_q99 = sessions["session_energy_kwh"].quantile(0.99)
    max_bin = min(80, int(np.ceil(energy_q99)) + 2)
    if max_bin < 4:
        max_bin = 4

    plot_hist(
        sessions["session_energy_kwh"],
        sessions["weight"],
        bins=np.arange(0, max_bin, 2),
        title="NHTS-derived charging energy demand",
        xlabel="Session energy proxy (kWh)",
        output_path=out_dir / "energy_hist_us.png",
    )

    if "session_type" in sessions.columns:
        mix = sessions.groupby("session_type")["weight"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(mix["session_type"], mix["weight"], edgecolor="black")
        ax.set_title("Session type mix")
        ax.set_ylabel("Weighted sessions")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / "session_type_mix.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    readme = """
Outputs:
- nhts_charging_sessions_us.csv
- arrival_dist_us.csv
- stay_dist_us.csv
- energy_dist_us.csv
- arrival_dist_by_region.csv
- stay_dist_by_region.csv
- energy_dist_by_region.csv
- regional_summary_stats.csv

Notes:
1. region_key uses CDIVMSAR when available.
2. If CDIVMSAR is missing, fallback is Census region/division.
"""
    (out_dir / "README_prepare_nhts_us.txt").write_text(readme.strip(), encoding="utf-8")

    print(f"Done. Outputs saved to: {out_dir}")


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="Prepare NHTS 2022 data for EV scheduling studies")
        parser.add_argument("--hh", required=True, help="Path to hhv2pub.csv")
        parser.add_argument("--trip", required=True, help="Path to tripv2pub.csv")
        parser.add_argument("--veh", required=True, help="Path to vehv2pub.csv")
        parser.add_argument("--out", required=True, help="Output directory")
        parser.add_argument("--min_stay_hours", type=float, default=0.25)
        parser.add_argument("--max_stay_hours", type=float, default=24.0)
        parser.add_argument("--kwh_per_mile", type=float, default=0.30)
        parser.add_argument("--no_overnight", action="store_true")
        args = parser.parse_args()

    print("=========================================")
    print("Preparing NHTS-based EV charging sessions")
    print("=========================================")
    print(f"Household file : {args.hh}")
    print(f"Trip file      : {args.trip}")
    print(f"Vehicle file   : {args.veh}")
    print(f"Output dir     : {args.out}")
    print(f"Min stay hours : {args.min_stay_hours}")
    print(f"Max stay hours : {args.max_stay_hours}")
    print(f"kWh per mile   : {args.kwh_per_mile}")
    print(f"Include overnight sessions: {not args.no_overnight}")
    print("----------------------------------------")

    for file_path in [args.hh, args.trip, args.veh]:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

    prepare_nhts_us(
        hh_path=args.hh,
        trip_path=args.trip,
        veh_path=args.veh,
        out_dir=args.out,
        min_stay_hours=args.min_stay_hours,
        max_stay_hours=args.max_stay_hours,
        kwh_per_mile=args.kwh_per_mile,
        include_overnight=not args.no_overnight,
    )

    print("========================================")
    print("NHTS preparation finished successfully.")
    print("========================================")


if __name__ == "__main__":
    args = argparse.Namespace(
        hh=r"data\trip data\hhv2pub.csv",
        trip=r"data\trip data\tripv2pub.csv",
        veh=r"data\trip data\vehv2pub.csv",
        out=r"data\trip data\nhts_prepared_cdivmsar",
        min_stay_hours=0.25,
        max_stay_hours=24.0,
        kwh_per_mile=0.30,
        no_overnight=False
    )
    main(args)