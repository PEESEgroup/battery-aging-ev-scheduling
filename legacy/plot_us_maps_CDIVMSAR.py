import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go


# =================================================
# 1. Division anchors
# =================================================
DIVISION_META = {
    1: {"name": "New England",         "lat": 42.5, "lon": -71.8},
    2: {"name": "Mid-Atlantic",        "lat": 40.9, "lon": -75.5},
    3: {"name": "East North Central",  "lat": 41.7, "lon": -87.0},
    4: {"name": "West North Central",  "lat": 39.8, "lon": -96.5},
    5: {"name": "South Atlantic",      "lat": 33.8, "lon": -81.2},
    6: {"name": "East South Central",  "lat": 33.6, "lon": -86.7},
    7: {"name": "West South Central",  "lat": 31.0, "lon": -97.5},
    8: {"name": "Mountain",            "lat": 39.0, "lon": -109.5},
    9: {"name": "Pacific",             "lat": 37.5, "lon": -121.0},
}


# =================================================
# 2. I/O helpers
# =================================================
def _save_plotly(fig, out_base):
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(str(out_base.with_suffix(".html")))

    try:
        fig.write_image(str(out_base.with_suffix(".png")), scale=3, width=1400, height=850)
        fig.write_image(str(out_base.with_suffix(".pdf")), scale=3, width=1400, height=850)
        print(f"Saved: {out_base.with_suffix('.png')} / .pdf / .html")
    except Exception as e:
        print(f"Static export unavailable ({e}). Saved HTML only: {out_base.with_suffix('.html')}")


# =================================================
# 3. CDIVMSAR position utilities
# =================================================
def _division_anchor_from_row(row):
    """
    Determine division anchor from available columns.
    Prefer CENSUS_D. If unavailable, infer from CDIVMSAR first digit.
    """
    if pd.notna(row.get("CENSUS_D", np.nan)):
        d = int(row["CENSUS_D"])
        if d in DIVISION_META:
            return d

    cdiv = row.get("CDIVMSAR", np.nan)
    if pd.notna(cdiv):
        cdiv = str(int(cdiv))
        if len(cdiv) >= 1:
            d = int(cdiv[0])
            if d in DIVISION_META:
                return d

    return None


def _assign_cdivmsar_positions(region_df, radius_deg=1.6):
    """
    Spread CDIVMSAR units around each Census Division anchor.
    This is a schematic map, not literal geography.
    """
    out_rows = []

    for division_id, g in region_df.groupby("division_id"):
        base_lat = DIVISION_META[division_id]["lat"]
        base_lon = DIVISION_META[division_id]["lon"]

        g = g.sort_values("region_key").reset_index(drop=True)
        n = len(g)

        if n == 1:
            offsets = [(0.0, 0.0)]
        else:
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            offsets = []
            for i, ang in enumerate(angles):
                r = radius_deg if n <= 8 else (radius_deg if i < 8 else radius_deg * 1.8)
                dlat = r * np.sin(ang) * 0.55
                dlon = r * np.cos(ang)
                offsets.append((dlat, dlon))

        for (_, row), (dlat, dlon) in zip(g.iterrows(), offsets):
            out_rows.append({
                "region_key": row["region_key"],
                "division_id": division_id,
                "lat": base_lat + dlat,
                "lon": base_lon + dlon,
            })

    return pd.DataFrame(out_rows)


# =================================================
# 4. Build trip summary at CDIVMSAR level
# =================================================
def build_trip_region_summary(hh_path, trip_path):
    hh = pd.read_csv(hh_path, low_memory=False)
    trip = pd.read_csv(trip_path, low_memory=False)

    hh_cols = [c for c in ["HOUSEID", "CENSUS_D"] if c in hh.columns]
    trip_cols = [c for c in [
        "HOUSEID", "TRPMILES", "WTTRDFIN", "TRPHHVEH",
        "CDIVMSAR", "CENSUS_D"
    ] if c in trip.columns]

    hh = hh[hh_cols].copy()
    trip = trip[trip_cols].copy()

    # Merge household only if trip lacks CENSUS_D
    if "CENSUS_D" not in trip.columns and "CENSUS_D" in hh.columns:
        trip = trip.merge(hh, on="HOUSEID", how="left")

    if "TRPHHVEH" in trip.columns:
        trip = trip[trip["TRPHHVEH"] == 1]

    trip["TRPMILES"] = pd.to_numeric(trip["TRPMILES"], errors="coerce")
    trip["WTTRDFIN"] = pd.to_numeric(trip["WTTRDFIN"], errors="coerce").fillna(1.0)
    trip["CDIVMSAR"] = pd.to_numeric(trip["CDIVMSAR"], errors="coerce")

    if "CENSUS_D" in trip.columns:
        trip["CENSUS_D"] = pd.to_numeric(trip["CENSUS_D"], errors="coerce")

    trip = trip[trip["TRPMILES"].fillna(0) > 0].copy()
    trip = trip[trip["CDIVMSAR"].notna()].copy()

    trip["region_key"] = "CDIV_" + trip["CDIVMSAR"].astype("Int64").astype(str)

    # Infer division anchor
    trip["division_id"] = trip.apply(_division_anchor_from_row, axis=1)
    trip = trip[trip["division_id"].notna()].copy()
    trip["division_id"] = trip["division_id"].astype(int)

    def wavg(x, vcol):
        w = x["WTTRDFIN"].to_numpy()
        v = x[vcol].to_numpy()
        m = np.isfinite(w) & np.isfinite(v)
        if m.sum() == 0:
            return np.nan
        return np.average(v[m], weights=w[m])

    rows = []
    for rk, g in trip.groupby("region_key"):
        division_id = int(g["division_id"].iloc[0])
        rows.append({
            "region_key": rk,
            "cdivmsar_code": int(g["CDIVMSAR"].iloc[0]),
            "division_id": division_id,
            "division_name": DIVISION_META[division_id]["name"],
            "weighted_trip_count": g["WTTRDFIN"].sum(),
            "mean_trip_miles": wavg(g, "TRPMILES"),
            "p50_trip_miles": np.nanpercentile(g["TRPMILES"], 50),
            "p75_trip_miles": np.nanpercentile(g["TRPMILES"], 75),
        })

    trip_summary = pd.DataFrame(rows)

    # Only merge positioning columns, avoid duplicate division_name
    pos_df = _assign_cdivmsar_positions(
        trip_summary[["region_key", "division_id"]].drop_duplicates()
    )

    trip_summary = trip_summary.merge(
        pos_df,
        on=["region_key", "division_id"],
        how="left"
    )

    return trip_summary


# =================================================
# 5. Build charging summary at CDIVMSAR level
# =================================================
def build_charging_region_summary(nhts_prepared_dir, trip_summary_df):
    nhts_prepared_dir = Path(nhts_prepared_dir)
    regional = pd.read_csv(nhts_prepared_dir / "regional_summary_stats.csv")

    region_meta = trip_summary_df[[
        "region_key", "division_id", "division_name", "lat", "lon", "cdivmsar_code"
    ]].drop_duplicates()

    regional = regional.merge(region_meta, on="region_key", how="inner")

    required_cols = ["region_key", "division_name", "lat", "lon"]
    missing = [c for c in required_cols if c not in regional.columns]
    if missing:
        raise ValueError(f"Missing required columns in charging_summary after merge: {missing}")

    return regional


# =================================================
# 6. Plot trip map
# =================================================
def plot_us_trip_map(trip_summary_df, out_base="figures/us_trip_map_cdivmsar", annotate_top_n=10):
    df = trip_summary_df.copy()

    required_cols = [
        "region_key", "division_name", "lat", "lon",
        "weighted_trip_count", "mean_trip_miles"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in trip_summary_df: {missing}")

    df["label"] = (
        df["region_key"] +
        "<br>Division: " + df["division_name"] +
        "<br>Weighted trips: " + df["weighted_trip_count"].round(0).astype(int).astype(str) +
        "<br>Mean trip miles: " + df["mean_trip_miles"].round(2).astype(str)
    )

    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="lon",
        scope="usa",
        size="weighted_trip_count",
        color="mean_trip_miles",
        size_max=40,
        hover_name="region_key",
        hover_data={
            "division_name": True,
            "lat": False,
            "lon": False,
            "weighted_trip_count": ":,.0f",
            "mean_trip_miles": ":.2f",
            "p50_trip_miles": ":.2f",
            "p75_trip_miles": ":.2f",
            "cdivmsar_code": True,
        },
        color_continuous_scale="YlOrRd",
    )

    fig.update_traces(
        marker=dict(line=dict(color="black", width=0.7), opacity=0.88),
        text=None,
    )

    # Annotate only top-N to avoid clutter
    top_df = df.sort_values("weighted_trip_count", ascending=False).head(annotate_top_n)
    fig.add_trace(
        go.Scattergeo(
            lon=top_df["lon"],
            lat=top_df["lat"],
            mode="text",
            text=top_df["region_key"],
            textfont=dict(size=11, color="black"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_geos(
        projection_type="albers usa",
        showland=True,
        landcolor="rgb(243,243,243)",
        coastlinecolor="rgb(200,200,200)",
        subunitcolor="rgb(180,180,180)",
        showcountries=False,
        showsubunits=True,
        bgcolor="white",
    )

    fig.update_layout(
        title=dict(
            text="Fine-grained regional travel intensity across the United States",
            x=0.02, xanchor="left", y=0.96
        ),
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=15),
        coloraxis_colorbar=dict(title="Mean trip<br>miles"),
        annotations=[
            dict(
                x=0.02, y=0.02, xref="paper", yref="paper",
                text=(
                    "Bubble size = weighted trip volume; color = mean trip miles. "
                    "Public NHTS does not provide literal route geometry. "
                    "CDIVMSAR units are shown as schematic locations within each Census Division."
                ),
                showarrow=False, align="left", font=dict(size=11)
            )
        ]
    )

    _save_plotly(fig, out_base)


# =================================================
# 7. Plot charging map
# =================================================
def plot_us_charging_map(charging_summary_df, out_base="figures/us_charging_map_cdivmsar", annotate_top_n=10):
    df = charging_summary_df.copy()

    required_cols = ["region_key", "division_name", "lat", "lon"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in charging_summary_df: {missing}")

    size_col = "weighted_sessions" if "weighted_sessions" in df.columns else "n_sessions"
    color_col = "energy_p50" if "energy_p50" in df.columns else "stay_p50"

    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="lon",
        scope="usa",
        size=size_col,
        color=color_col,
        size_max=40,
        hover_name="region_key",
        hover_data={
            "division_name": True,
            "lat": False,
            "lon": False,
            "n_sessions": ":,.0f" if "n_sessions" in df.columns else False,
            "weighted_sessions": ":,.0f" if "weighted_sessions" in df.columns else False,
            "arrival_p50": ":.2f" if "arrival_p50" in df.columns else False,
            "stay_p50": ":.2f" if "stay_p50" in df.columns else False,
            "energy_p50": ":.2f" if "energy_p50" in df.columns else False,
            "cdivmsar_code": True if "cdivmsar_code" in df.columns else False,
        },
        color_continuous_scale="Viridis",
    )

    fig.update_traces(
        marker=dict(line=dict(color="black", width=0.7), opacity=0.90),
        text=None,
    )

    top_df = df.sort_values(size_col, ascending=False).head(annotate_top_n)
    fig.add_trace(
        go.Scattergeo(
            lon=top_df["lon"],
            lat=top_df["lat"],
            mode="text",
            text=top_df["region_key"],
            textfont=dict(size=11, color="black"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_geos(
        projection_type="albers usa",
        showland=True,
        landcolor="rgb(243,243,243)",
        coastlinecolor="rgb(200,200,200)",
        subunitcolor="rgb(180,180,180)",
        showcountries=False,
        showsubunits=True,
        bgcolor="white",
    )

    color_title = "Median session<br>energy (kWh)" if color_col == "energy_p50" else "Median stay<br>duration (h)"

    fig.update_layout(
        title=dict(
            text="Fine-grained NHTS-derived EV charging opportunity landscape across the United States",
            x=0.02, xanchor="left", y=0.96
        ),
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=15),
        coloraxis_colorbar=dict(title=color_title),
        annotations=[
            dict(
                x=0.02, y=0.02, xref="paper", yref="paper",
                text=(
                    "Bubble size = weighted charging opportunities; color = median charging energy. "
                    "CDIVMSAR units are visualized as fine-grained schematic locations nested within larger Census Divisions."
                ),
                showarrow=False, align="left", font=dict(size=11)
            )
        ]
    )

    _save_plotly(fig, out_base)


# =================================================
# 8. Main
# =================================================
def main():
    hh_path = "data/trip data/hhv2pub.csv"
    trip_path = "data/trip data/tripv2pub.csv"
    nhts_prepared_dir = "data/trip data/nhts_prepared_cdivmsar"

    trip_summary = build_trip_region_summary(hh_path, trip_path)
    trip_summary.to_csv(
        Path(nhts_prepared_dir) / "trip_region_summary_cdivmsar.csv",
        index=False
    )

    plot_us_trip_map(
        trip_summary,
        out_base=Path(nhts_prepared_dir) / "fig_trip_map_us_cdivmsar",
        annotate_top_n=12
    )

    charging_summary = build_charging_region_summary(nhts_prepared_dir, trip_summary)
    charging_summary.to_csv(
        Path(nhts_prepared_dir) / "charging_region_summary_cdivmsar.csv",
        index=False
    )

    plot_us_charging_map(
        charging_summary,
        out_base=Path(nhts_prepared_dir) / "fig_charging_map_us_cdivmsar",
        annotate_top_n=12
    )


if __name__ == "__main__":
    main()