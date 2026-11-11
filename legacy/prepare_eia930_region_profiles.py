import os

import requests
import pandas as pd
import numpy as np

API_KEY = os.environ["EIA_API_KEY"]

# 选择几个代表区域（非常重要）
REGION_MAP = {
    "PACIFIC": "CISO",   # California ISO
    "TEXAS": "ERCO",     # ERCOT
    "MIDATL": "PJM",     # PJM
    "NE": "ISNE",        # ISO New England
    "MIDWEST": "MISO",   # MISO
}

TIME_HORIZON = 24


def fetch_eia930(region_code):
    url = f"https://api.eia.gov/v2/electricity/rto/region-data/data/"

    params = {
        "api_key": API_KEY,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": region_code,
        "facets[type][]": "D",  # Demand
        "start": "2023-01-01",
        "end": "2023-01-02",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
    }

    r = requests.get(url, params=params)
    data = r.json()["response"]["data"]

    df = pd.DataFrame(data)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df


def build_available_cap(df, kappa=200, beta=0.8):
    demand = df["value"].values[:TIME_HORIZON]

    dmin, dmax = demand.min(), demand.max()

    if dmax - dmin < 1e-6:
        norm = np.zeros_like(demand)
    else:
        norm = (demand - dmin) / (dmax - dmin)

    cap = kappa * (1 - beta * norm)
    cap = np.maximum(cap, 0.25 * kappa)

    return cap


def main():
    rows = []

    for name, code in REGION_MAP.items():
        print(f"Fetching {name} ({code})")

        df = fetch_eia930(code)

        cap = build_available_cap(df)

        for h in range(TIME_HORIZON):
            rows.append({
                "region_key": name,
                "hour": h,
                "demand_mw": df["value"].values[h],
                "available_cap": cap[h],
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv("regional_grid_profiles.csv", index=False)

    print("Saved: regional_grid_profiles.csv")


if __name__ == "__main__":
    main()
