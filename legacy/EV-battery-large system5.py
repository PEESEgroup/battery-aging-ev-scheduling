##real-world nhts_region data generation来代替完全的synthetic data
##calculate system decision bias
##more detailed CDIV_XX regions
##EIA-930 grid constraint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

# =========================
# 1. Basic settings
# =========================
np.random.seed(42)

TIME_HORIZON = 24
N_EV = 2000
DT = 1.0
P_MAX_EV = 7.0

# Health model weights
ALPHA_THROUGHPUT = 0.8
BETA_HIGH_SOC = 6.0
GAMMA_HIGH_POWER = 3.5
LAMBDA_HEALTH = 2.8

# Flexibility penalty coefficient:
# maps health burden into effective flexibility loss
ALPHA_FLEX = 0.02

hours = np.arange(TIME_HORIZON)

# =========================
# 2. Time-varying signals
# =========================
# price: 夜间便宜，傍晚高
price = np.array([
    0.07, 0.07, 0.07, 0.08, 0.09, 0.11,
    0.14, 0.19, 0.26, 0.30, 0.32, 0.28,
    0.22, 0.18, 0.15, 0.17, 0.22, 0.34,
    0.40, 0.38, 0.26, 0.17, 0.10, 0.08
], dtype=float)

# carbon: 中午更低，傍晚和夜间更高
carbon = np.array([
    0.64, 0.62, 0.60, 0.54, 0.44, 0.30,
    0.20, 0.12, 0.07, 0.05, 0.04, 0.04,
    0.05, 0.06, 0.08, 0.14, 0.24, 0.40,
    0.56, 0.62, 0.66, 0.68, 0.68, 0.66
], dtype=float)

# baseline capacity profile
base_available_cap = np.array([
    145, 145, 145, 145, 150, 155,
    170, 185, 215, 225, 225, 220,
    210, 210, 205, 195, 170, 145,
    125, 120, 120, 120, 125, 140
], dtype=float)

# =========================
# CDIVMSAR -> aggregated real grid region mapping
# based on regional_grid_profiles.csv
# Available grid regions:
# PACIFIC, TEXAS, MIDATL, NE, MIDWEST
# =========================
GRID_REGION_MAP = {
    # ---------- Northeast ----------
    "CDIV_11": "NE",
    "CDIV_12": "MIDATL",
    "CDIV_13": "MIDATL",

    # ---------- Midwest ----------
    "CDIV_21": "MIDWEST",
    "CDIV_22": "MIDWEST",
    "CDIV_23": "MIDWEST",

    # ---------- South ----------
    "CDIV_31": "MIDATL",
    "CDIV_32": "MIDATL",
    "CDIV_33": "TEXAS",

    # ---------- West ----------
    "CDIV_41": "MIDWEST",
    "CDIV_42": "PACIFIC",
}


# =========================
# 3. EV data structure
# =========================
@dataclass
class EV:
    ev_id: int
    arrival: int
    departure: int
    init_soc: float
    target_soc: float
    battery_kwh: float
    max_power: float

    @property
    def required_energy(self):
        return max(0.0, (self.target_soc - self.init_soc) * self.battery_kwh)

# =========================
# 4. Generate synthetic EV fleet
# =========================
def generate_evs(n_ev: int):
    evs = []
    for i in range(n_ev):
        # 50% daytime / 50% evening
        if np.random.rand() < 0.50:
            arrival = int(np.clip(np.random.normal(loc=8.5, scale=1.8), 0, 21))
            stay = int(np.clip(np.random.normal(loc=8.5, scale=1.6), 4, 11))
        else:
            arrival = int(np.clip(np.random.normal(loc=17.8, scale=2.0), 0, 22))
            stay = int(np.clip(np.random.normal(loc=7.0, scale=1.8), 4, 11))

        departure = min(arrival + stay, TIME_HORIZON)

        init_soc = float(np.clip(np.random.normal(loc=0.30, scale=0.12), 0.05, 0.65))
        target_soc = float(np.clip(np.random.normal(loc=0.88, scale=0.05), 0.75, 0.95))
        battery_kwh = float(np.clip(np.random.normal(loc=65, scale=10), 45, 90))

        evs.append(
            EV(
                ev_id=i,
                arrival=arrival,
                departure=departure,
                init_soc=init_soc,
                target_soc=target_soc,
                battery_kwh=battery_kwh,
                max_power=P_MAX_EV
            )
        )
    return evs

def generate_evs_from_acn(
    n_ev: int,
    acn_csv_path: str,
    battery_kwh_range=(50.0, 80.0),
    target_soc_range=(0.80, 0.95),
    min_init_soc=0.05,
    max_init_soc=0.75,
    seed: int = 42,
):
    """
    Generate EV fleet from cleaned ACN session data.

    Parameters
    ----------
    n_ev : int
        Number of EVs to generate.
    acn_csv_path : str
        Path to cleaned ACN CSV file produced by prepare_real_sessions.py.
    battery_kwh_range : tuple
        Min and max battery capacity (kWh) used to map session energy into SOC states.
    target_soc_range : tuple
        Range for target SOC sampling.
    min_init_soc, max_init_soc : float
        Bounds for initial SOC.
    seed : int
        Random seed.

    Returns
    -------
    list[EV]
        A list of EV objects compatible with the current simulation framework.
    """
    rng = np.random.default_rng(seed)

    df = pd.read_csv(acn_csv_path)

    required_cols = ["connection_hour", "session_duration_hours", "kWhDelivered"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in ACN CSV: {col}")

    # Keep only valid rows
    df = df.copy()
    df = df.dropna(subset=["connection_hour", "session_duration_hours", "kWhDelivered"])
    df = df[df["session_duration_hours"] > 0]
    df = df[df["kWhDelivered"] > 0]

    # Optional requested energy if available
    use_requested = "requested_kwh" in df.columns and df["requested_kwh"].notna().sum() > 0

    evs = []

    sampled_idx = rng.choice(df.index.to_numpy(), size=n_ev, replace=True)

    for ev_id, idx in enumerate(sampled_idx):
        row = df.loc[idx]

        # -------------------------
        # 1. Arrival / departure
        # -------------------------
        arrival = int(np.clip(round(row["connection_hour"]), 0, TIME_HORIZON - 1))

        session_duration = float(row["session_duration_hours"])
        stay = max(1, int(round(session_duration)))
        departure = min(arrival + stay, TIME_HORIZON)

        # If departure collapses, repair
        if departure <= arrival:
            departure = min(arrival + 1, TIME_HORIZON)

        # -------------------------
        # 2. Session energy demand
        # -------------------------
        if use_requested and pd.notna(row.get("requested_kwh", np.nan)):
            energy_need = float(row["requested_kwh"])
        else:
            energy_need = float(row["kWhDelivered"])

        energy_need = max(1.0, energy_need)

        # -------------------------
        # 3. Battery capacity
        # -------------------------
        battery_kwh = float(rng.uniform(*battery_kwh_range))

        # Ensure the chosen battery is large enough to accommodate the sampled session
        # while keeping target_soc in a realistic range.
        min_battery_needed = energy_need / max(target_soc_range[1] - min_init_soc, 1e-3)
        if battery_kwh < min_battery_needed:
            battery_kwh = min(max(min_battery_needed, battery_kwh_range[0]), 100.0)

        # -------------------------
        # 4. Target SOC
        # -------------------------
        target_soc = float(rng.uniform(*target_soc_range))

        # -------------------------
        # 5. Infer initial SOC from energy demand
        # required_energy = (target_soc - init_soc) * battery_kwh
        # => init_soc = target_soc - required_energy / battery_kwh
        # -------------------------
        init_soc = target_soc - energy_need / battery_kwh

        # If inferred init_soc is too low/high, repair by adjusting battery or target
        if init_soc < min_init_soc:
            init_soc = min_init_soc
            # repair target SOC to preserve energy_need as much as possible
            target_soc = init_soc + energy_need / battery_kwh

        if target_soc > 0.98:
            target_soc = 0.98
            init_soc = target_soc - energy_need / battery_kwh

        if init_soc > max_init_soc:
            init_soc = max_init_soc
            target_soc = init_soc + energy_need / battery_kwh

        # Final clipping
        init_soc = float(np.clip(init_soc, min_init_soc, max_init_soc))
        target_soc = float(np.clip(target_soc, init_soc + 0.02, 0.98))

        # Recompute required energy implied by repaired SOCs
        implied_energy = (target_soc - init_soc) * battery_kwh

        # If implied energy is still too far from sampled session energy, adjust battery size once more
        if implied_energy < 0.85 * energy_need:
            battery_kwh = min(100.0, energy_need / max(target_soc - init_soc, 1e-3))
            implied_energy = (target_soc - init_soc) * battery_kwh

        # -------------------------
        # 6. Max power
        # Optionally infer from active charging if available
        # -------------------------
        max_power = P_MAX_EV
        if "active_charging_hours" in df.columns and pd.notna(row.get("active_charging_hours", np.nan)):
            active_h = float(row["active_charging_hours"])
            if active_h > 0.25:
                avg_session_power = energy_need / active_h
                # cap inferred power to a practical L2-like range
                max_power = float(np.clip(max(avg_session_power * 1.2, 3.0), 3.0, P_MAX_EV))

        evs.append(
            EV(
                ev_id=ev_id,
                arrival=arrival,
                departure=departure,
                init_soc=init_soc,
                target_soc=target_soc,
                battery_kwh=float(battery_kwh),
                max_power=max_power,
            )
        )

    return evs

# =========================
# 4.1 Regional grid constraint loader
# =========================
def load_regional_grid_profiles(grid_profile_csv: str):
    """
    Load regional hourly grid profiles from CSV.

    Required columns:
    - region_key
    - hour
    - available_cap
    """
    df = pd.read_csv(grid_profile_csv)

    required_cols = ["region_key", "hour", "available_cap"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"regional_grid_profiles.csv is missing required columns: {missing}")

    df = df.copy()
    df["region_key"] = df["region_key"].astype(str).str.strip().str.upper()
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["available_cap"] = pd.to_numeric(df["available_cap"], errors="coerce")
    df = df.dropna(subset=["region_key", "hour", "available_cap"])

    df = df[(df["hour"] >= 0) & (df["hour"] < TIME_HORIZON)].copy()
    df["hour"] = df["hour"].astype(int)

    return df


def map_to_grid_region(region_key: str) -> str:
    """
    Map behavioral region key (e.g., CDIV_11) to aggregated real grid region
    used in regional_grid_profiles.csv.
    """
    region_key = str(region_key).strip()
    if region_key in GRID_REGION_MAP:
        return GRID_REGION_MAP[region_key]
    return region_key


def get_region_available_cap(region_key: str, grid_profiles_df: pd.DataFrame, cap_scale: float = 1.0):
    """
    Return hourly available capacity for one behavioral region.
    region_key may be CDIV_XX and will be mapped to a real grid-region key.
    """
    grid_region_key = map_to_grid_region(region_key)
    grid_region_key = str(grid_region_key).strip().upper()

    df_r = grid_profiles_df[grid_profiles_df["region_key"] == grid_region_key].copy()

    if len(df_r) == 0:
        raise ValueError(
            f"No grid profile found for behavioral region {region_key} "
            f"(mapped to grid region {grid_region_key})"
        )

    df_r = df_r.sort_values("hour")

    expected_hours = set(range(TIME_HORIZON))
    actual_hours = set(df_r["hour"].tolist())
    missing_hours = sorted(expected_hours - actual_hours)
    if missing_hours:
        raise ValueError(
            f"Grid profile for {grid_region_key} is incomplete. Missing hours: {missing_hours}"
        )

    cap = df_r["available_cap"].to_numpy(dtype=float)

    if len(cap) != TIME_HORIZON:
        raise ValueError(
            f"Grid profile for {grid_region_key} should have {TIME_HORIZON} rows, got {len(cap)}"
        )

    return cap * cap_scale


def validate_region_keys_against_grid(region_keys, grid_profiles_df):
    available_grid_regions = set(
        grid_profiles_df["region_key"].dropna().astype(str).str.strip().str.upper().unique().tolist()
    )

    missing = []
    mapping_rows = []

    for rk in region_keys:
        mapped = map_to_grid_region(rk)
        mapped = str(mapped).strip().upper()
        mapping_rows.append((rk, mapped))
        if mapped not in available_grid_regions:
            missing.append((rk, mapped))

    print("\n=== Behavioral region -> Grid region mapping ===")
    for rk, mapped in mapping_rows:
        print(f"{rk} -> {mapped}")

    if missing:
        raise ValueError(
            "The following behavioral region_keys cannot be mapped to available grid regions:\n"
            + "\n".join([f"  {rk} -> {mapped}" for rk, mapped in missing])
        )

def weighted_sample(values, weights, size, rng):
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]

    if len(values) == 0:
        raise ValueError("No valid samples available for weighted sampling.")

    probs = weights / weights.sum()
    idx = rng.choice(np.arange(len(values)), size=size, replace=True, p=probs)
    return values[idx]


def generate_evs_from_nhts_region(
    n_ev: int,
    region_key: str,
    nhts_dir: str,
    battery_kwh_range=(50.0, 85.0),
    target_soc_range=(0.80, 0.95),
    min_init_soc=0.05,
    max_init_soc=0.75,
    default_departure_hour=7.0,
    seed: int = 42,
):
    """
    Generate EV fleet from region-level NHTS-derived charging opportunity distributions.

    Parameters
    ----------
    n_ev : int
        Number of EVs to generate.
    region_key : str
        Region identifier such as 'R1_D1'. Must exist in the prepared NHTS outputs.
    nhts_dir : str
        Directory containing:
            arrival_dist_by_region.csv
            stay_dist_by_region.csv
            energy_dist_by_region.csv
    battery_kwh_range : tuple
        Sampled battery capacity range.
    target_soc_range : tuple
        Sampled target SOC range.
    min_init_soc, max_init_soc : float
        Bounds for inferred initial SOC.
    default_departure_hour : float
        Used only if repaired stay produces invalid departure.
    seed : int
        Random seed.

    Returns
    -------
    list[EV]
    """
    rng = np.random.default_rng(seed)
    nhts_dir = Path(nhts_dir)

    arrival_df = pd.read_csv(nhts_dir / "arrival_dist_by_region.csv")
    stay_df = pd.read_csv(nhts_dir / "stay_dist_by_region.csv")
    energy_df = pd.read_csv(nhts_dir / "energy_dist_by_region.csv")

    # Filter one region
    arr_r = arrival_df[arrival_df["region_key"] == region_key].copy()
    stay_r = stay_df[stay_df["region_key"] == region_key].copy()
    energy_r = energy_df[energy_df["region_key"] == region_key].copy()

    if len(arr_r) == 0 or len(stay_r) == 0 or len(energy_r) == 0:
        raise ValueError(f"No NHTS regional distribution found for region_key={region_key}")

    # Draw empirical samples
    sampled_arrivals = weighted_sample(
        arr_r["connection_hour"].to_numpy(),
        arr_r["weight"].to_numpy(),
        n_ev,
        rng
    )

    sampled_stays = weighted_sample(
        stay_r["session_duration_hours"].to_numpy(),
        stay_r["weight"].to_numpy(),
        n_ev,
        rng
    )

    sampled_energies = weighted_sample(
        energy_r["session_energy_kwh"].to_numpy(),
        energy_r["weight"].to_numpy(),
        n_ev,
        rng
    )

    evs = []

    for i in range(n_ev):
        arrival_hour = float(sampled_arrivals[i])
        stay_hours = float(sampled_stays[i])
        energy_need = float(sampled_energies[i])

        # ----- Arrival / departure -----
        arrival = int(np.clip(np.floor(arrival_hour), 0, TIME_HORIZON - 1))

        # round stay to at least 1 hour for current hourly simulator
        stay = max(1, int(np.round(stay_hours)))
        departure = min(arrival + stay, TIME_HORIZON)

        if departure <= arrival:
            departure = min(max(arrival + 1, int(np.ceil(default_departure_hour))), TIME_HORIZON)

        # ----- Battery capacity -----
        battery_kwh = float(rng.uniform(*battery_kwh_range))

        # Ensure enough battery size so inferred SOC is feasible
        min_battery_needed = energy_need / max(target_soc_range[1] - min_init_soc, 1e-3)
        if battery_kwh < min_battery_needed:
            battery_kwh = min(max(min_battery_needed, battery_kwh_range[0]), 120.0)

        # ----- Target SOC -----
        target_soc = float(rng.uniform(*target_soc_range))

        # ----- Infer initial SOC from session energy -----
        init_soc = target_soc - energy_need / battery_kwh

        if init_soc < min_init_soc:
            init_soc = min_init_soc
            target_soc = init_soc + energy_need / battery_kwh

        if target_soc > 0.98:
            target_soc = 0.98
            init_soc = target_soc - energy_need / battery_kwh

        if init_soc > max_init_soc:
            init_soc = max_init_soc
            target_soc = init_soc + energy_need / battery_kwh

        init_soc = float(np.clip(init_soc, min_init_soc, max_init_soc))
        target_soc = float(np.clip(target_soc, init_soc + 0.02, 0.98))

        # One more repair if the implied session energy is too small
        implied_energy = (target_soc - init_soc) * battery_kwh
        if implied_energy < 0.85 * energy_need:
            battery_kwh = min(120.0, energy_need / max(target_soc - init_soc, 1e-3))
            implied_energy = (target_soc - init_soc) * battery_kwh

        # ----- Max power -----
        # Keep current framework simple: L2-like charging by default
        max_power = P_MAX_EV

        evs.append(
            EV(
                ev_id=i,
                arrival=arrival,
                departure=departure,
                init_soc=init_soc,
                target_soc=target_soc,
                battery_kwh=float(battery_kwh),
                max_power=float(max_power),
            )
        )

    return evs


def inspect_generated_evs(evs, n_show=10):
    rows = []
    for ev in evs[:n_show]:
        rows.append({
            "ev_id": ev.ev_id,
            "arrival": ev.arrival,
            "departure": ev.departure,
            "stay_h": ev.departure - ev.arrival,
            "init_soc": round(ev.init_soc, 3),
            "target_soc": round(ev.target_soc, 3),
            "battery_kwh": round(ev.battery_kwh, 2),
            "required_energy": round(ev.required_energy, 2),
            "max_power": round(ev.max_power, 2),
        })
    return pd.DataFrame(rows)

# =========================
# 5. Health cost model
# =========================
def marginal_health_cost(curr_soc, charge_power, max_power):
    throughput_term = ALPHA_THROUGHPUT

    high_soc_term = 0.0
    if curr_soc > 0.65:
        high_soc_term = BETA_HIGH_SOC * ((curr_soc - 0.65) / 0.35) ** 2

    power_ratio = charge_power / max_power if max_power > 0 else 0.0
    high_power_term = GAMMA_HIGH_POWER * (power_ratio ** 2)

    return throughput_term + high_soc_term + high_power_term

# =========================
# 6. Strategy-specific slot signal
# =========================
def slot_signal(strategy, t, ev, curr_soc, remaining_energy):
    """
    Lower signal = more preferred slot
    """
    slack = max(ev.departure - t, 1)
    urgency = (remaining_energy / slack) / max(ev.max_power, 1e-6)

    if strategy == "cost":
        return price[t] - 0.16 * urgency

    elif strategy == "carbon":
        return carbon[t] - 0.16 * urgency

    elif strategy == "health":
        proxy_power = 2.0 if curr_soc >= 0.65 else 2.5
        hour_bias = 0.03 * (t - ev.arrival)
        return (
            0.10 * price[t]
            + LAMBDA_HEALTH * marginal_health_cost(curr_soc, proxy_power, ev.max_power)
            + hour_bias
            - 0.06 * urgency
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

# =========================
# 7. Core simulation
# =========================
def run_simulation(evs, strategy="cost", available_cap=None, alpha_flex=ALPHA_FLEX):
    if available_cap is None:
        available_cap = base_available_cap.copy()

    n = len(evs)

    remaining_energy = np.array([ev.required_energy for ev in evs], dtype=float)
    requested_energy = remaining_energy.copy()
    soc = np.array([ev.init_soc for ev in evs], dtype=float)

    charge_matrix = np.zeros((n, TIME_HORIZON))
    energy_matrix = np.zeros((n, TIME_HORIZON))
    health_matrix = np.zeros((n, TIME_HORIZON))

    cap_left = available_cap.copy()

    for _ in range(40):
        progress = False

        order = sorted(
            range(n),
            key=lambda i: (
                evs[i].departure,
                -(remaining_energy[i] / max(evs[i].departure - evs[i].arrival, 1))
            )
        )

        for i in order:
            if remaining_energy[i] <= 1e-6:
                continue

            ev = evs[i]
            feasible_hours = [t for t in range(ev.arrival, ev.departure) if cap_left[t] > 1e-6]
            if not feasible_hours:
                continue

            ranked_hours = sorted(
                feasible_hours,
                key=lambda t: (slot_signal(strategy, t, ev, soc[i], remaining_energy[i]), t)
            )

            for t in ranked_hours:
                if remaining_energy[i] <= 1e-6:
                    break
                if cap_left[t] <= 1e-6:
                    continue

                if strategy == "health":
                    power_cap = 2.5 if soc[i] < 0.65 else 1.5
                else:
                    power_cap = ev.max_power

                feasible_power = min(power_cap, cap_left[t], remaining_energy[i] / DT)
                if feasible_power <= 1e-6:
                    continue

                delivered = feasible_power * DT

                charge_matrix[i, t] += feasible_power
                energy_matrix[i, t] += delivered
                cap_left[t] -= feasible_power
                remaining_energy[i] -= delivered

                prev_soc = soc[i]
                soc[i] = min(1.0, soc[i] + delivered / ev.battery_kwh)

                h_cost = marginal_health_cost(prev_soc, feasible_power, ev.max_power) * delivered
                health_matrix[i, t] += h_cost

                progress = True

        if not progress:
            break

    agg_power = charge_matrix.sum(axis=0)
    agg_energy = energy_matrix.sum(axis=0)

    total_cost = float(np.sum(agg_energy * price))
    total_carbon = float(np.sum(agg_energy * carbon))
    total_health = float(np.sum(health_matrix))

    served_energy = float(agg_energy.sum())
    requested_energy_total = float(requested_energy.sum())
    unserved_energy = float(remaining_energy.sum())

    # ---- New flexibility metrics ----
    # ---- Flexibility and system-bias metrics ----
    naive_flexibility = served_energy
    sustainable_flexibility = max(0.0, served_energy - alpha_flex * total_health)

    flexibility_overestimation = naive_flexibility - sustainable_flexibility
    overestimation_ratio = (
        flexibility_overestimation / naive_flexibility
        if naive_flexibility > 0 else np.nan
    )

    # System decision bias proxies
    # 1) If the system assumes naive flexibility, this is the hidden flexibility gap
    flexibility_gap = naive_flexibility - sustainable_flexibility

    # 2) Convert flexibility gap into an equivalent capacity need
    # Here we normalize by time horizon to get a kW-equivalent proxy
    equivalent_capacity_gap = flexibility_gap / TIME_HORIZON

    # 3) Effective peak after correcting for degradation-induced flexibility loss
    effective_peak_load = agg_power.max() + equivalent_capacity_gap

    # 4) Capacity underestimation ratio
    capacity_underestimation_ratio = (
        equivalent_capacity_gap / agg_power.max()
        if agg_power.max() > 0 else np.nan
    )

    results = {
        "strategy": strategy,
        "agg_power": agg_power,
        "agg_energy": agg_energy,
        "charge_matrix": charge_matrix,
        "energy_matrix": energy_matrix,
        "health_matrix": health_matrix,
        "total_cost": total_cost,
        "total_carbon": total_carbon,
        "total_health": total_health,
        "peak_load": float(agg_power.max()),
        "load_variance": float(np.var(agg_power)),
        "served_energy": served_energy,
        "unserved_energy": unserved_energy,
        "service_ratio": served_energy / requested_energy_total if requested_energy_total > 0 else 0.0,
        "cost_per_served_kWh": total_cost / served_energy if served_energy > 0 else np.nan,
        "carbon_per_served_kWh": total_carbon / served_energy if served_energy > 0 else np.nan,
        "health_per_served_kWh": total_health / served_energy if served_energy > 0 else np.nan,
        "naive_flexibility": naive_flexibility,
        "sustainable_flexibility": sustainable_flexibility,
        "flexibility_overestimation": flexibility_overestimation,
        "overestimation_ratio": overestimation_ratio,
        "ev_health": health_matrix.sum(axis=1),
        "ev_energy": energy_matrix.sum(axis=1),
        "available_cap": available_cap.copy(),
        "alpha_flex": alpha_flex,
        "flexibility_gap": flexibility_gap,
        "equivalent_capacity_gap": equivalent_capacity_gap,
        "effective_peak_load": effective_peak_load,
        "capacity_underestimation_ratio": capacity_underestimation_ratio,
    }
    return results

# =========================
# 8. Run all strategies
# =========================
def run_all(
    cap_scale=1.0,
    alpha_flex=ALPHA_FLEX,
    ev_mode="synthetic",
    acn_csv_path=None,
    nhts_dir=None,
    region_key=None,
    grid_profiles_df=None,
    seed=42,
):
    if ev_mode == "synthetic":
        evs = generate_evs(N_EV)

    elif ev_mode == "acn":
        if acn_csv_path is None:
            raise ValueError("acn_csv_path must be provided when ev_mode='acn'")
        evs = generate_evs_from_acn(
            N_EV,
            acn_csv_path=acn_csv_path,
            seed=seed
        )

    elif ev_mode == "nhts_region":
        if nhts_dir is None or region_key is None:
            raise ValueError("nhts_dir and region_key must be provided when ev_mode='nhts_region'")
        evs = generate_evs_from_nhts_region(
            n_ev=N_EV,
            region_key=region_key,
            nhts_dir=nhts_dir,
            seed=seed
        )

    else:
        raise ValueError(f"Unknown ev_mode: {ev_mode}")

    # ---- real regional grid constraint ----
    if grid_profiles_df is not None and region_key is not None:
        available_cap = get_region_available_cap(
            region_key=region_key,
            grid_profiles_df=grid_profiles_df,
            cap_scale=cap_scale
        )
    else:
        # fallback to stylized profile
        available_cap = base_available_cap.copy()
        available_cap[0:7] *= 0.9 * cap_scale
        available_cap[8:16] *= 1.2 * cap_scale
        available_cap[17:23] *= 0.7 * cap_scale

    out_cost = run_simulation(evs, strategy="cost", available_cap=available_cap, alpha_flex=alpha_flex)
    out_carbon = run_simulation(evs, strategy="carbon", available_cap=available_cap, alpha_flex=alpha_flex)
    out_health = run_simulation(evs, strategy="health", available_cap=available_cap, alpha_flex=alpha_flex)

    return evs, {
        "cost": out_cost,
        "carbon": out_carbon,
        "health": out_health
    }

REGION_NAME_MAP = {
    "R1_D1": "New England",
    "R1_D2": "Mid-Atlantic",
    "R2_D3": "East North Central",
    "R2_D4": "West North Central",
    "R3_D5": "South Atlantic",
    "R3_D6": "East South Central",
    "R3_D7": "West South Central",
    "R4_D8": "Mountain",
    "R4_D9": "Pacific",
}

def run_multi_region_experiments(
    region_keys,
    nhts_dir,
    grid_profile_csv=None,
    n_ev=2000,
    alpha_flex=ALPHA_FLEX,
    seed=42,
    cap_scale=1.0,
):
    """
    Run baseline simulations for multiple NHTS-derived regions
    under real regional grid constraints if grid_profile_csv is provided.
    """
    all_outputs = {}
    summary_rows = []

    # check NHTS region keys
    arrival_df = pd.read_csv(Path(nhts_dir) / "arrival_dist_by_region.csv")
    available_regions = set(arrival_df["region_key"].dropna().unique().tolist())

    missing_regions = [rk for rk in region_keys if rk not in available_regions]
    if len(missing_regions) > 0:
        raise ValueError(
            f"The following region_keys are not found in {nhts_dir}/arrival_dist_by_region.csv: {missing_regions}"
        )

    # optional real regional grid profiles
    grid_profiles_df = None
    if grid_profile_csv is not None:
        grid_profiles_df = load_regional_grid_profiles(grid_profile_csv)
        validate_region_keys_against_grid(region_keys, grid_profiles_df)

    for region_key in region_keys:
        print(f"\nRunning region: {region_key} -> grid region: {map_to_grid_region(region_key)}")

        evs = generate_evs_from_nhts_region(
            n_ev=n_ev,
            region_key=region_key,
            nhts_dir=nhts_dir,
            seed=seed
        )

        # ---- use real regional grid constraint if available ----
        if grid_profiles_df is not None:
            available_cap = get_region_available_cap(
                region_key=region_key,
                grid_profiles_df=grid_profiles_df,
                cap_scale=cap_scale
            )
        else:
            available_cap = base_available_cap.copy()
            available_cap[0:7] *= 0.9 * cap_scale
            available_cap[8:16] *= 1.2 * cap_scale
            available_cap[17:23] *= 0.7 * cap_scale

        region_outputs = {}

        for strategy in ["cost", "carbon", "health"]:
            out = run_simulation(
                evs,
                strategy=strategy,
                available_cap=available_cap,
                alpha_flex=alpha_flex
            )
            region_outputs[strategy] = out

            summary_rows.append({
                "region_key": region_key,
                "region_label": make_region_label(region_key),
                "strategy": strategy,
                "served_energy_kWh": out["served_energy"],
                "unserved_energy_kWh": out["unserved_energy"],
                "service_ratio": out["service_ratio"],
                "cost_per_served_kWh": out["cost_per_served_kWh"],
                "carbon_per_served_kWh": out["carbon_per_served_kWh"],
                "health_per_served_kWh": out["health_per_served_kWh"],
                "naive_flexibility": out["naive_flexibility"],
                "sustainable_flexibility": out["sustainable_flexibility"],
                "flexibility_overestimation": out["flexibility_overestimation"],
                "overestimation_ratio": out["overestimation_ratio"],
                "equivalent_capacity_gap_kW": out.get("equivalent_capacity_gap", np.nan),
                "effective_peak_load_kW": out.get("effective_peak_load", np.nan),
                "capacity_underestimation_ratio": out.get("capacity_underestimation_ratio", np.nan),
                "peak_load_kW": out["peak_load"],
            })

        all_outputs[region_key] = region_outputs

    summary_df = pd.DataFrame(summary_rows)
    return all_outputs, summary_df


# =========================
# 9. Multi-scenario analysis
# =========================

"""
def run_congestion_scenarios(cap_scales=(0.85, 1.0, 1.15), alpha_flex=ALPHA_FLEX):
    scenario_outputs = {}
    for cap_scale in cap_scales:
        _, outputs = run_all(cap_scale=cap_scale, alpha_flex=alpha_flex)
        scenario_outputs[f"cap_x_{cap_scale:.2f}"] = outputs
    return scenario_outputs
"""

def run_congestion_scenarios(
    cap_scales=(0.60, 0.85, 1.0, 1.15, 1.30),
    alpha_flex=ALPHA_FLEX,
    peak_hours=(17, 23),
    ev_mode="synthetic",
    acn_csv_path=None,
    nhts_dir=None,
    region_key=None,
    grid_profile_csv=None,
    seed=42,
):
    scenario_outputs = {}

    grid_profiles_df = None
    if grid_profile_csv is not None:
        grid_profiles_df = load_regional_grid_profiles(grid_profile_csv)

    for cap_scale in cap_scales:
        if ev_mode == "synthetic":
            evs = generate_evs(N_EV)

        elif ev_mode == "acn":
            if acn_csv_path is None:
                raise ValueError("acn_csv_path must be provided when ev_mode='acn'")
            evs = generate_evs_from_acn(N_EV, acn_csv_path=acn_csv_path, seed=seed)

        elif ev_mode == "nhts_region":
            if nhts_dir is None or region_key is None:
                raise ValueError("nhts_dir and region_key must be provided when ev_mode='nhts_region'")
            evs = generate_evs_from_nhts_region(
                n_ev=N_EV,
                region_key=region_key,
                nhts_dir=nhts_dir,
                seed=seed
            )
        else:
            raise ValueError(f"Unknown ev_mode: {ev_mode}")

        if grid_profiles_df is not None and region_key is not None:
            available_cap = get_region_available_cap(region_key, grid_profiles_df, cap_scale=1.0)
            start, end = peak_hours
            available_cap[start:end] *= cap_scale
        else:
            available_cap = base_available_cap.copy()
            start, end = peak_hours
            available_cap[start:end] *= cap_scale

        out_cost = run_simulation(evs, strategy="cost", available_cap=available_cap, alpha_flex=alpha_flex)
        out_carbon = run_simulation(evs, strategy="carbon", available_cap=available_cap, alpha_flex=alpha_flex)
        out_health = run_simulation(evs, strategy="health", available_cap=available_cap, alpha_flex=alpha_flex)

        scenario_outputs[f"peak_cap_x_{cap_scale:.2f}"] = {
            "cost": out_cost,
            "carbon": out_carbon,
            "health": out_health
        }

    return scenario_outputs

# =========================
# 10. Summaries
# =========================
def summarize(outputs):
    rows = []
    for name, out in outputs.items():
        rows.append({
            "strategy": name,
            "total_cost_$": out["total_cost"],
            "total_carbon_kg": out["total_carbon"],
            "total_health_index": out["total_health"],
            "served_energy_kWh": out["served_energy"],
            "unserved_energy_kWh": out["unserved_energy"],
            "service_ratio": out["service_ratio"],
            "cost_per_served_kWh": out["cost_per_served_kWh"],
            "carbon_per_served_kWh": out["carbon_per_served_kWh"],
            "health_per_served_kWh": out["health_per_served_kWh"],
            "naive_flexibility": out["naive_flexibility"],
            "sustainable_flexibility": out["sustainable_flexibility"],
            "flexibility_overestimation": out["flexibility_overestimation"],
            "overestimation_ratio": out["overestimation_ratio"],
            "peak_power_kW": out["peak_load"],
            "effective_peak_load_kW": out["effective_peak_load"],
            "equivalent_capacity_gap_kW": out["equivalent_capacity_gap"],
            "capacity_underestimation_ratio": out["capacity_underestimation_ratio"],
        })
    df = pd.DataFrame(rows)
    return df.sort_values("strategy").reset_index(drop=True)

def summarize_scenarios(scenario_outputs):
    rows = []
    for scen_name, outputs in scenario_outputs.items():
        for strat_name, out in outputs.items():
            rows.append({
                "scenario": scen_name,
                "strategy": strat_name,
                "served_energy_kWh": out["served_energy"],
                "total_health_index": out["total_health"],
                "naive_flexibility": out["naive_flexibility"],
                "sustainable_flexibility": out["sustainable_flexibility"],
                "overestimation_ratio": out["overestimation_ratio"],
                "cost_per_served_kWh": out["cost_per_served_kWh"],
                "carbon_per_served_kWh": out["carbon_per_served_kWh"],
                "health_per_served_kWh": out["health_per_served_kWh"],
            })
    return pd.DataFrame(rows)

def summarize_multi_region(summary_df):
    """
    Aggregate across regions by strategy.
    """
    agg = summary_df.groupby("strategy").agg({
        "served_energy_kWh": "mean",
        "unserved_energy_kWh": "mean",
        "service_ratio": "mean",
        "cost_per_served_kWh": "mean",
        "carbon_per_served_kWh": "mean",
        "health_per_served_kWh": "mean",
        "naive_flexibility": "mean",
        "sustainable_flexibility": "mean",
        "flexibility_overestimation": "mean",
        "overestimation_ratio": "mean",
        "equivalent_capacity_gap_kW": "mean",
        "effective_peak_load_kW": "mean",
        "capacity_underestimation_ratio": "mean",
        "peak_load_kW": "mean",
    }).reset_index()

    return agg

def make_region_label(region_key: str) -> str:
    """
    Human-readable label for region keys.
    Supports:
    - CDIVMSAR-based keys: CDIV_11, CDIV_21, ...
    - fallback old census division keys: R1_D1, ...
    """
    if region_key.startswith("CDIV_"):
        code = region_key.replace("CDIV_", "")
        return f"CDIVMSAR {code}"

    if region_key in REGION_NAME_MAP:
        return REGION_NAME_MAP[region_key]

    return region_key


# =========================
# 11. Plotting (publication-ready)
# =========================
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

COLORS = {
    "cost": "#1f3c88",      # deep blue
    "carbon": "#2a9d8f",    # teal
    "health": "#c76d3a",    # muted orange
    "naive": "#8da0cb",
    "sustainable": "#66c2a5",
    "cap": "#7f7f7f",
    "price": "#4c78a8",
    "carbon_signal": "#59a14f",
}

def set_pub_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def save_figure(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")

def style_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, linestyle="--", alpha=0.22)
    ax.set_axisbelow(True)

def annotate_bars(ax, bars, fmt="{:.3f}", rotation=0, fontsize=8, y_offset_frac=0.01):
    ymin, ymax = ax.get_ylim()
    y_offset = (ymax - ymin) * y_offset_frac
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + y_offset,
            fmt.format(h),
            ha="center",
            va="bottom",
            rotation=rotation,
            fontsize=fontsize
        )

def plot_inputs_and_profiles(outputs):
    """
    Fig. 1: external signals + aggregate charging profiles
    """
    set_pub_style()
    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 5.8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.15]}
    )

    # Panel (a): external signals
    ax = axes[0]
    ax2 = ax.twinx()

    l1 = ax.plot(hours, price, color=COLORS["price"], marker="o", linewidth=2, label="Electricity price")
    l2 = ax2.plot(hours, carbon, color=COLORS["carbon_signal"], marker="s", linewidth=2, label="Carbon intensity")
    cap_ref = np.asarray(outputs["cost"]["available_cap"], dtype=float)
    l3 = ax.bar(hours, cap_ref, color=COLORS["cap"], alpha=0.18, width=0.8, label="Available capacity")

    ax.set_ylabel("Price ($/kWh)")
    ax2.set_ylabel("Carbon (kg CO$_2$/kWh)")
    ax.set_title("Input signals and charging outcomes", loc="left", fontweight="bold")
    style_axis(ax, grid_axis="y")
    ax2.spines["top"].set_visible(False)

    handles = l1 + l2 + [l3]
    labels = ["Electricity price", "Carbon intensity", "Available capacity"]
    ax.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))

    # Panel (b): aggregate charging profiles
    ax = axes[1]
    for s in ["cost", "carbon", "health"]:
        ax.plot(
            hours,
            outputs[s]["agg_power"],
            color=COLORS[s],
            linewidth=2.2,
            marker="o",
            label=s.capitalize()
        )
    ax.plot(hours, cap_ref, color="black", linestyle="--", linewidth=1.4, label="Capacity limit")

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Aggregate charging power (kW)")
    style_axis(ax, grid_axis="both")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.set_xticks(np.arange(0, 24, 2))

    fig.tight_layout()
    save_figure(fig, "fig1_inputs_and_profiles")
    plt.show()

def plot_unit_metrics(outputs):
    """
    Fig. 2: unit-normalized metrics, better for paper than total metrics
    """
    set_pub_style()

    metrics = [
        ("cost_per_served_kWh", "Cost per served kWh"),
        ("carbon_per_served_kWh", "Carbon per served kWh"),
        ("health_per_served_kWh", "Health per served kWh"),
        ("overestimation_ratio", "Overestimation ratio"),
    ]
    strategies = ["cost", "carbon", "health"]

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2))
    for ax, (metric, title) in zip(axes, metrics):
        vals = [outputs[s][metric] for s in strategies]
        bars = ax.bar(
            np.arange(len(strategies)),
            vals,
            color=[COLORS[s] for s in strategies],
            width=0.65
        )
        ax.set_xticks(np.arange(len(strategies)))
        ax.set_xticklabels([s.capitalize() for s in strategies], rotation=0)
        ax.set_title(title, pad=8)
        style_axis(ax, grid_axis="y")

        if metric == "overestimation_ratio":
            annotate_bars(ax, bars, fmt="{:.2%}", fontsize=8, y_offset_frac=0.02)
        else:
            annotate_bars(ax, bars, fmt="{:.3f}", fontsize=8, y_offset_frac=0.02)

    fig.suptitle("Comparison of strategy-level performance metrics", y=1.05, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig2_unit_metrics")
    plt.show()

def plot_health_distribution(outputs):
    """
    Fig. 3: ECDF-style health burden plot, cleaner than sorted raw line
    """
    set_pub_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))

    for s in ["cost", "carbon", "health"]:
        x = np.sort(outputs[s]["ev_health"])
        y = np.linspace(0, 1, len(x))
        ax.plot(x, y, color=COLORS[s], linewidth=2.2, label=s.capitalize())

    ax.set_xlabel("Vehicle-level health burden")
    ax.set_ylabel("Cumulative fraction of EVs")
    ax.set_title("Distribution of battery health burden", loc="left", fontweight="bold")
    style_axis(ax, grid_axis="both")
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    save_figure(fig, "fig3_health_distribution")
    plt.show()

def plot_cost_health_carbon_scatter(outputs):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.4))

    strategies = ["cost", "carbon", "health"]
    x_vals = [outputs[s]["health_per_served_kWh"] for s in strategies]
    y_vals = [outputs[s]["cost_per_served_kWh"] for s in strategies]

    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    x_range = x_max - x_min
    y_range = y_max - y_min

    # 先画点
    for s in strategies:
        x = outputs[s]["health_per_served_kWh"]
        y = outputs[s]["cost_per_served_kWh"]

        ax.scatter(
            x, y,
            s=180,
            color=COLORS[s],
            edgecolor="black",
            linewidth=0.7,
            zorder=3
        )

    # 给坐标轴留白，避免标签贴边
    ax.set_xlim(x_min - 0.22 * x_range, x_max + 0.28 * x_range)
    ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.18 * y_range)

    # 为每个点单独设定标注位置（相对点）
    label_specs = {
        "health": {
            "dx": -0.12 * x_range,
            "dy":  0.08 * y_range,
            "ha": "right",
            "va": "center",
        },
        "cost": {
            "dx":  0.05 * x_range,
            "dy": -0.02 * y_range,
            "ha": "left",
            "va": "center",
        },
        "carbon": {
            "dx":  0.05 * x_range,
            "dy":  0.10 * y_range,
            "ha": "left",
            "va": "bottom",
        },
    }

    # 再画文字标注
    for s in strategies:
        x = outputs[s]["health_per_served_kWh"]
        y = outputs[s]["cost_per_served_kWh"]
        c = outputs[s]["carbon_per_served_kWh"]

        spec = label_specs[s]
        tx = x + spec["dx"]
        ty = y + spec["dy"]

        ax.annotate(
            f"{s.capitalize()}\nC={c:.3f}",
            xy=(x, y),
            xytext=(tx, ty),
            textcoords="data",
            ha=spec["ha"],
            va=spec["va"],
            fontsize=9,
            arrowprops=dict(
                arrowstyle="-",
                lw=0.8,
                color="0.35",
                shrinkA=4,
                shrinkB=4
            ),
            zorder=4
        )

    ax.set_xlabel("Health burden per served kWh")
    ax.set_ylabel("Cost per served kWh ($/kWh)")
    ax.set_title("Trade-off between cost and battery health", loc="left", fontweight="bold")
    style_axis(ax, grid_axis="both")

    fig.tight_layout()
    save_figure(fig, "fig4_tradeoff_scatter_fixed_v2")
    plt.show()

def plot_flexibility_misestimation(outputs):
    """
    Fig. 5: naive vs sustainable flexibility + overestimation ratio
    """
    set_pub_style()
    strategies = ["cost", "carbon", "health"]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), gridspec_kw={"width_ratios": [1.4, 1]})

    # Panel (a): paired bars
    ax = axes[0]
    x = np.arange(len(strategies))
    width = 0.34

    naive_vals = [outputs[s]["naive_flexibility"] for s in strategies]
    sust_vals = [outputs[s]["sustainable_flexibility"] for s in strategies]

    bars1 = ax.bar(x - width/2, naive_vals, width=width, color=COLORS["naive"], label="Naive flexibility")
    bars2 = ax.bar(x + width/2, sust_vals, width=width, color=COLORS["sustainable"], label="Sustainable flexibility")

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in strategies])
    ax.set_ylabel("Flexibility (kWh-equivalent)")
    ax.set_title("Flexibility misestimation", loc="left", fontweight="bold")
    style_axis(ax, grid_axis="y")
    ax.legend(frameon=False, loc="upper right")

    # only annotate ratio above sustainable bars
    ratios = [outputs[s]["overestimation_ratio"] for s in strategies]
    for xi, r, y in zip(x, ratios, sust_vals):
        ax.text(xi, y + 18, f"{r:.1%}", ha="center", va="bottom", fontsize=8)

    # Panel (b): overestimation ratio
    ax = axes[1]
    bars = ax.bar(
        np.arange(len(strategies)),
        ratios,
        color=[COLORS[s] for s in strategies],
        width=0.65
    )
    ax.set_xticks(np.arange(len(strategies)))
    ax.set_xticklabels([s.capitalize() for s in strategies])
    ax.set_ylabel("Overestimation ratio")
    style_axis(ax, grid_axis="y")
    annotate_bars(ax, bars, fmt="{:.1%}", fontsize=8, y_offset_frac=0.02)

    fig.tight_layout()
    save_figure(fig, "fig5_flexibility_misestimation")
    plt.show()

def plot_congestion_effects(scenario_outputs):
    set_pub_style()

    scenario_names = list(scenario_outputs.keys())
    scenario_labels = [s.replace("cap_x_", "×") for s in scenario_names]
    strategies = ["cost", "carbon", "health"]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    for s in strategies:
        vals = [scenario_outputs[sc][s]["flexibility_overestimation"] for sc in scenario_names]
        ax.plot(
            scenario_labels,
            vals,
            marker="o",
            linewidth=2.2,
            color=COLORS[s],
            label=s.capitalize()
        )
        for xi, yi in zip(scenario_labels, vals):
            ax.text(xi, yi, f"{yi:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Capacity scaling factor")
    ax.set_ylabel("Flexibility overestimation (kWh-equivalent)")
    ax.set_title("Congestion increases the overestimation of EV fleet flexibility",
                 loc="left", fontweight="bold")
    style_axis(ax, grid_axis="y")
    ax.legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, "fig6_congestion_effects_overestimation")
    plt.show()

def plot_system_decision_bias(outputs):
    """
    Fig. 6: system decision bias caused by ignoring degradation
    """
    set_pub_style()
    strategies = ["cost", "carbon", "health"]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})

    # Panel (a): peak load vs effective peak load
    ax = axes[0]
    x = np.arange(len(strategies))
    width = 0.34

    peak_vals = [outputs[s]["peak_load"] for s in strategies]
    eff_peak_vals = [outputs[s]["effective_peak_load"] for s in strategies]

    bars1 = ax.bar(x - width/2, peak_vals, width=width, color=COLORS["naive"], label="Observed peak load")
    bars2 = ax.bar(x + width/2, eff_peak_vals, width=width, color=COLORS["sustainable"], label="Degradation-adjusted peak load")

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in strategies])
    ax.set_ylabel("System capacity proxy (kW)")
    ax.set_title("System decision bias", loc="left", fontweight="bold")
    style_axis(ax, grid_axis="y")
    ax.legend(frameon=False, loc="upper left")

    # Panel (b): capacity underestimation ratio
    ax = axes[1]
    vals = [outputs[s]["capacity_underestimation_ratio"] for s in strategies]
    bars = ax.bar(
        np.arange(len(strategies)),
        vals,
        color=[COLORS[s] for s in strategies],
        width=0.65
    )
    ax.set_xticks(np.arange(len(strategies)))
    ax.set_xticklabels([s.capitalize() for s in strategies])
    ax.set_ylabel("Capacity underestimation ratio")
    style_axis(ax, grid_axis="y")
    annotate_bars(ax, bars, fmt="{:.1%}", fontsize=8, y_offset_frac=0.02)

    fig.tight_layout()
    save_figure(fig, "fig6_system_decision_bias")
    plt.show()

def plot_multi_region_overestimation(summary_df):
    set_pub_style()

    plot_df = summary_df.copy()

    if "region_label" not in plot_df.columns:
        plot_df["region_label"] = plot_df["region_key"].apply(make_region_label)

    regions = plot_df["region_label"].drop_duplicates().tolist()
    strategies = ["cost", "carbon", "health"]

    x = np.arange(len(regions))
    width = 0.24

    fig, ax = plt.subplots(figsize=(max(10.5, 0.8 * len(regions)), 4.8))

    for i, strategy in enumerate(strategies):
        vals = []
        for region in regions:
            row = plot_df[
                (plot_df["region_label"] == region) &
                (plot_df["strategy"] == strategy)
            ]
            vals.append(row["overestimation_ratio"].iloc[0] if len(row) > 0 else np.nan)

        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            color=COLORS[strategy],
            label=strategy.capitalize()
        )

    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=35, ha="right")
    ax.set_ylabel("Overestimation ratio")
    ax.set_title("Regional variation in EV flexibility misestimation", loc="left", fontweight="bold")
    style_axis(ax, grid_axis="y")
    ax.legend(frameon=False, ncol=3, loc="upper left")

    fig.tight_layout()
    save_figure(fig, "fig_multi_region_overestimation_cdivmsar")
    plt.show()

def plot_multi_region_capacity_bias(summary_df):
    set_pub_style()

    plot_df = summary_df.copy()

    if "region_label" not in plot_df.columns:
        plot_df["region_label"] = plot_df["region_key"].apply(make_region_label)

    regions = plot_df["region_label"].drop_duplicates().tolist()
    strategies = ["cost", "carbon", "health"]

    fig, ax = plt.subplots(figsize=(max(10.5, 0.8 * len(regions)), 4.8))

    for strategy in strategies:
        vals = []
        for region in regions:
            row = plot_df[
                (plot_df["region_label"] == region) &
                (plot_df["strategy"] == strategy)
            ]
            vals.append(row["capacity_underestimation_ratio"].iloc[0] if len(row) > 0 else np.nan)

        ax.plot(
            regions,
            vals,
            marker="o",
            linewidth=2.0,
            color=COLORS[strategy],
            label=strategy.capitalize()
        )

    ax.set_ylabel("Capacity underestimation ratio")
    ax.set_title("Regional system decision bias under battery degradation", loc="left", fontweight="bold")
    style_axis(ax, grid_axis="y")
    ax.legend(frameon=False)
    plt.xticks(rotation=35, ha="right")

    fig.tight_layout()
    save_figure(fig, "fig_multi_region_capacity_bias_cdivmsar")
    plt.show()

def plot_results(outputs):
    plot_inputs_and_profiles(outputs)
    plot_unit_metrics(outputs)
    plot_health_distribution(outputs)
    plot_cost_health_carbon_scatter(outputs)
    plot_flexibility_misestimation(outputs)
    plot_system_decision_bias(outputs)


# =========================
# 12. Main
# =========================
if __name__ == "__main__":
    nhts_dir = r"data\trip data\nhts_prepared_cdivmsar"
    grid_profile_csv = r"data\trip data\regional_grid_profiles.csv"

    # load behavioral regions
    arrival_df = pd.read_csv(Path(nhts_dir) / "arrival_dist_by_region.csv")
    all_region_keys = sorted(arrival_df["region_key"].dropna().unique().tolist())
    cdiv_regions = [rk for rk in all_region_keys if rk.startswith("CDIV_")]

    print("\n=== Available CDIVMSAR regions ===")
    print(cdiv_regions)

    mapped_cdiv_regions = [rk for rk in cdiv_regions if rk in GRID_REGION_MAP]
    print("\n=== Mapped CDIVMSAR regions ===")
    print(mapped_cdiv_regions)

    # recommended representative regions
    region_keys = [
        "CDIV_11",  # NE
        "CDIV_12",  # MIDATL
        "CDIV_21",  # MIDWEST
        "CDIV_33",  # TEXAS
        "CDIV_42",  # PACIFIC
    ]

    grid_profiles_df = load_regional_grid_profiles(grid_profile_csv)

    print("\n=== Available grid regions in CSV ===")
    print(sorted(grid_profiles_df["region_key"].unique().tolist()))

    validate_region_keys_against_grid(region_keys, grid_profiles_df)

    all_outputs, summary_df = run_multi_region_experiments(
        region_keys=region_keys,
        nhts_dir=nhts_dir,
        grid_profile_csv=grid_profile_csv,
        n_ev=2000,
        alpha_flex=ALPHA_FLEX,
        seed=42,
        cap_scale=1.0,
    )

    print("\n=== Multi-region summary ===")
    print(summary_df.round(4))

    summary_df.to_csv("multi_region_summary_cdivmsar_realgrid.csv", index=False)

    agg_df = summarize_multi_region(summary_df)
    print("\n=== Cross-region average summary ===")
    print(agg_df.round(4))

    agg_df.to_csv("multi_region_average_summary_cdivmsar_realgrid.csv", index=False)

    plot_multi_region_overestimation(summary_df)
    plot_multi_region_capacity_bias(summary_df)

    # inspect one region in detail
    demo_region = region_keys[0]
    evs, outputs = run_all(
        cap_scale=1.0,
        alpha_flex=ALPHA_FLEX,
        ev_mode="nhts_region",
        nhts_dir=nhts_dir,
        region_key=demo_region,
        grid_profiles_df=grid_profiles_df,
        seed=42,
    )
    plot_results(outputs)

