import numpy as np
import pandas as pd

arrival_region = pd.read_csv(r"data\trip data\nhts_prepared\arrival_dist_by_region.csv")
print(arrival_region["region_key"].drop_duplicates().sort_values().tolist())

arrival = pd.read_csv(r"data\trip data\nhts_prepared_cdivmsar\arrival_dist_by_region.csv")
print(sorted(arrival["region_key"].drop_duplicates().tolist())[:30])
print(arrival["region_key"].nunique())

summary = pd.read_csv(r"data\trip data\nhts_prepared_cdivmsar\regional_summary_stats.csv")
print(summary.head())

