
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

DIVISION_META = {
    "R1_D1": {"name": "New England",         "lat": 42.5, "lon": -71.8},
    "R1_D2": {"name": "Mid-Atlantic",        "lat": 40.9, "lon": -75.5},
    "R2_D3": {"name": "East North Central",  "lat": 41.7, "lon": -87.0},
    "R2_D4": {"name": "West North Central",  "lat": 39.8, "lon": -96.5},
    "R3_D5": {"name": "South Atlantic",      "lat": 33.8, "lon": -81.2},
    "R3_D6": {"name": "East South Central",  "lat": 33.6, "lon": -86.7},
    "R3_D7": {"name": "West South Central",  "lat": 31.0, "lon": -97.5},
    "R4_D8": {"name": "Mountain",            "lat": 39.0, "lon": -109.5},
    "R4_D9": {"name": "Pacific",             "lat": 37.5, "lon": -121.0},
}

def _save_plotly(fig, out_base):
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_base.with_suffix(".html")))
    try:
        fig.write_image(str(out_base.with_suffix(".png")), scale=3, width=1300, height=800)
        fig.write_image(str(out_base.with_suffix(".pdf")), scale=3, width=1300, height=800)
        print(f"Saved: {out_base.with_suffix('.png')} and .pdf/.html")
    except Exception as e:
        print(f"Static export unavailable ({e}). Saved HTML only: {out_base.with_suffix('.html')}")

def build_trip_region_summary(hh_path, trip_path):
    hh = pd.read_csv(hh_path, low_memory=False)
    trip = pd.read_csv(trip_path, low_memory=False)

    hh_cols = [c for c in ["HOUSEID", "CENSUS_R", "CENSUS_D"] if c in hh.columns]
    trip_cols = [c for c in ["HOUSEID", "TRPMILES", "WTTRDFIN", "TRPHHVEH", "TRVLCMIN"] if c in trip.columns]

    hh = hh[hh_cols].copy()
    trip = trip[trip_cols].copy()

    trip = trip.merge(hh, on="HOUSEID", how="left")

    if "TRPHHVEH" in trip.columns:
        trip = trip[trip["TRPHHVEH"] == 1]

    trip["TRPMILES"] = pd.to_numeric(trip["TRPMILES"], errors="coerce")
    trip["WTTRDFIN"] = pd.to_numeric(trip["WTTRDFIN"], errors="coerce").fillna(1.0)
    trip = trip[trip["TRPMILES"].fillna(0) > 0].copy()

    trip["region_key"] = (
        "R" + trip["CENSUS_R"].astype("Int64").astype(str) +
        "_D" + trip["CENSUS_D"].astype("Int64").astype(str)
    )

    def wavg(x, vcol):
        w = x["WTTRDFIN"].to_numpy()
        v = x[vcol].to_numpy()
        m = np.isfinite(w) & np.isfinite(v)
        if m.sum() == 0:
            return np.nan
        return np.average(v[m], weights=w[m])

    rows = []
    for rk, g in trip.groupby("region_key"):
        if rk not in DIVISION_META:
            continue
        rows.append({
            "region_key": rk,
            "division_name": DIVISION_META[rk]["name"],
            "lat": DIVISION_META[rk]["lat"],
            "lon": DIVISION_META[rk]["lon"],
            "weighted_trip_count": g["WTTRDFIN"].sum(),
            "mean_trip_miles": wavg(g, "TRPMILES"),
            "p50_trip_miles": np.nanpercentile(g["TRPMILES"], 50),
            "p75_trip_miles": np.nanpercentile(g["TRPMILES"], 75),
        })
    return pd.DataFrame(rows)

def build_charging_region_summary(nhts_prepared_dir):
    nhts_prepared_dir = Path(nhts_prepared_dir)
    regional = pd.read_csv(nhts_prepared_dir / "regional_summary_stats.csv")

    regional["division_name"] = regional["region_key"].map(lambda x: DIVISION_META.get(x, {}).get("name", x))
    regional["lat"] = regional["region_key"].map(lambda x: DIVISION_META.get(x, {}).get("lat", np.nan))
    regional["lon"] = regional["region_key"].map(lambda x: DIVISION_META.get(x, {}).get("lon", np.nan))
    regional = regional.dropna(subset=["lat", "lon"]).copy()
    return regional

def plot_us_trip_map(trip_summary_df, out_base="figures/us_trip_map"):
    df = trip_summary_df.copy()
    df["label"] = (
        df["division_name"] +
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
        size_max=55,
        hover_name="division_name",
        hover_data={
            "lat": False,
            "lon": False,
            "weighted_trip_count": ":,.0f",
            "mean_trip_miles": ":.2f",
            "p50_trip_miles": ":.2f",
            "p75_trip_miles": ":.2f",
        },
        color_continuous_scale="YlOrRd",
    )

    fig.update_traces(
        marker=dict(line=dict(color="black", width=0.8), opacity=0.9),
        text=df["division_name"],
        textposition="top center",
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
            text="Regional travel intensity across the United States",
            x=0.02, xanchor="left", y=0.96
        ),
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=16),
        coloraxis_colorbar=dict(
            title="Mean trip<br>miles"
        ),
        legend_title_text="",
        annotations=[
            dict(
                x=0.02, y=0.02, xref="paper", yref="paper",
                text=("Bubble size = weighted trip volume; color = mean trip miles. "
                      "Public NHTS does not provide true route geometry, so this panel "
                      "shows regional travel intensity rather than literal road trajectories."),
                showarrow=False, align="left", font=dict(size=12)
            )
        ]
    )

    _save_plotly(fig, out_base)

def plot_us_charging_map(charging_summary_df, out_base="figures/us_charging_map"):
    df = charging_summary_df.copy()
    # normalize session count proxy if n_sessions too huge/small
    size_col = "weighted_sessions" if "weighted_sessions" in df.columns else "n_sessions"
    color_col = "energy_p50" if "energy_p50" in df.columns else "stay_p50"

    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="lon",
        scope="usa",
        size=size_col,
        color=color_col,
        size_max=55,
        hover_name="division_name",
        hover_data={
            "lat": False,
            "lon": False,
            "n_sessions": ":,.0f" if "n_sessions" in df.columns else False,
            "weighted_sessions": ":,.0f" if "weighted_sessions" in df.columns else False,
            "arrival_p50": ":.2f" if "arrival_p50" in df.columns else False,
            "stay_p50": ":.2f" if "stay_p50" in df.columns else False,
            "energy_p50": ":.2f" if "energy_p50" in df.columns else False,
        },
        color_continuous_scale="Viridis",
    )

    fig.update_traces(
        marker=dict(line=dict(color="black", width=0.8), opacity=0.92),
        text=df["division_name"],
        textposition="top center",
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
            text="NHTS-derived EV charging opportunity landscape across the United States",
            x=0.02, xanchor="left", y=0.96
        ),
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=16),
        coloraxis_colorbar=dict(title=color_title),
        annotations=[
            dict(
                x=0.02, y=0.02, xref="paper", yref="paper",
                text=("Bubble size = weighted charging opportunities; color = median charging energy. "
                      "This map summarizes generated EV charging opportunities derived from NHTS travel behavior."),
                showarrow=False, align="left", font=dict(size=12)
            )
        ]
    )

    _save_plotly(fig, out_base)

def main():
    # Example paths; replace with your local paths if needed
    hh_path = "data/trip data/hhv2pub.csv"
    trip_path = "data/trip data/tripv2pub.csv"
    nhts_prepared_dir = "data/trip data/nhts_prepared"

    trip_summary = build_trip_region_summary(hh_path, trip_path)
    trip_summary.to_csv(Path(nhts_prepared_dir) / "trip_region_summary.csv", index=False)
    plot_us_trip_map(trip_summary, out_base=Path(nhts_prepared_dir) / "fig_trip_map_us")

    charging_summary = build_charging_region_summary(nhts_prepared_dir)
    plot_us_charging_map(charging_summary, out_base=Path(nhts_prepared_dir) / "fig_charging_map_us")

if __name__ == "__main__":
    main()
