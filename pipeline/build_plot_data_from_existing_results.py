import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


# =========================================================
# 1. 基础工具
# =========================================================

def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def compute_cvar(values, alpha=0.95):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    var = np.quantile(arr, alpha)
    tail = arr[arr >= var]
    if len(tail) == 0:
        return float(var)
    return float(np.mean(tail))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_if_exists(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return None


def lambda_to_tag(lambda_value: float) -> str:
    return f"lambda_{str(lambda_value).replace('.', '_')}"


def tag_to_lambda(tag_name: str) -> float:
    # lambda_0_3 -> 0.3
    s = tag_name.replace("lambda_", "").replace("_", ".")
    return float(s)


# =========================================================
# 2. 读取单个 lambda 目录
# =========================================================

def read_one_lambda_result_folder(result_dir: Union[str, Path], risk_alpha=0.95):
    """
    Read one lambda-specific folder, e.g.
    lambda_sweep_results_joint/lambda_0_3/
    """
    result_dir = Path(result_dir)

    summary_csv_path = result_dir / "stochastic_results_summary.csv"
    risk_json_path = result_dir / "stochastic_risk_summary.json"

    out = {
        "summary": {},
        "risk_samples": [],
        "contribution": {
            "degradation_loss": np.nan,
            "congestion_loss": np.nan,
            "behavior_loss": np.nan,
        },
    }

    # -----------------------------------------------------
    # A. risk summary json
    # -----------------------------------------------------
    raw_risk_json = {}
    risk_summary = {}
    contribution_summary = {}

    if risk_json_path.exists():
        raw_risk_json = load_json(risk_json_path)
        risk_summary = raw_risk_json.get("risk_summary", {})
        contribution_summary = raw_risk_json.get("contribution_summary", {})

    # -----------------------------------------------------
    # B. summary csv
    # -----------------------------------------------------
    summary_csv = read_csv_if_exists(summary_csv_path)
    summary_csv_row = {}

    if summary_csv is not None and not summary_csv.empty:
        numeric_cols = summary_csv.select_dtypes(include=[np.number]).columns.tolist()
        if len(summary_csv) == 1:
            summary_csv_row = summary_csv.iloc[0].to_dict()
        else:
            summary_csv_row = {c: summary_csv[c].mean() for c in numeric_cols}

    # -----------------------------------------------------
    # C. build summary
    # -----------------------------------------------------
    out["summary"]["service_ratio"] = safe_float(
        risk_summary.get("expected_service_ratio", np.nan)
    )
    out["summary"]["flexibility_realized"] = safe_float(
        risk_summary.get("expected_service_ratio", np.nan)
    )
    out["summary"]["cvar_unmet_demand"] = safe_float(
        risk_summary.get("cvar_unmet_energy_kwh", np.nan)
    )

    # From risk_summary for Pareto use
    out["summary"]["lambda_risk"] = safe_float(risk_summary.get("lambda", np.nan))
    out["summary"]["expected_objective"] = safe_float(
        risk_summary.get("expected_objective", np.nan)
    )

    # Means for scenario_summary / pareto_df
    out["summary"]["cost_per_served_kwh"] = np.nan
    out["summary"]["carbon_per_served_kwh"] = np.nan
    out["summary"]["degradation_per_served_kwh"] = np.nan
    out["summary"]["peak_load"] = safe_float(
        risk_summary.get("expected_peak_ev_power_kw", np.nan)
    )
    out["summary"]["congestion_exposure"] = safe_float(
        risk_summary.get("expected_congestion_exposure", np.nan)
    )

    # contribution summary
    out["contribution"]["degradation_loss"] = safe_float(
        contribution_summary.get("degradation_loss", np.nan)
    )
    out["contribution"]["congestion_loss"] = safe_float(
        contribution_summary.get("congestion_loss", np.nan)
    )
    out["contribution"]["behavior_loss"] = safe_float(
        contribution_summary.get("behavior_loss", np.nan)
    )

    # -----------------------------------------------------
    # D. read scenario_xx for risk samples and per-served metrics
    # -----------------------------------------------------
    scenario_dirs = sorted([
        p for p in result_dir.iterdir()
        if p.is_dir() and p.name.startswith("scenario_")
    ])

    sample_rows = []

    for i, sdir in enumerate(scenario_dirs):
        obj_path = sdir / "objective_summary.json"
        obj_json = {}
        if obj_path.exists():
            obj_json = load_json(obj_path)

        obj = obj_json.get("objective_summary", {})

        total_required = safe_float(obj.get("total_required_energy_kwh", np.nan))
        total_served = safe_float(obj.get("total_served_energy_kwh", np.nan))
        total_unmet = safe_float(obj.get("total_unmet_energy_kwh", np.nan))
        service_ratio = safe_float(obj.get("service_ratio", np.nan))
        total_energy_cost = safe_float(obj.get("total_energy_cost", np.nan))
        total_emission_ton = safe_float(obj.get("total_emission_ton", np.nan))
        total_health_penalty = safe_float(obj.get("total_health_penalty", np.nan))
        peak_ev_power_kw = safe_float(obj.get("peak_ev_power_kw", np.nan))
        weighted_objective_proxy = safe_float(obj.get("weighted_objective_proxy", np.nan))

        cost_per_served_kwh = np.nan
        carbon_per_served_kwh = np.nan
        degradation_per_served_kwh = np.nan

        if np.isfinite(total_served) and total_served > 0:
            if np.isfinite(total_energy_cost):
                cost_per_served_kwh = total_energy_cost / total_served
            if np.isfinite(total_emission_ton):
                carbon_per_served_kwh = total_emission_ton / total_served
            if np.isfinite(total_health_penalty):
                degradation_per_served_kwh = total_health_penalty / total_served

        sample_rows.append({
            "sample_id": i,
            "scenario_subdir": sdir.name,
            "unmet_demand": total_unmet,
            "total_loss": weighted_objective_proxy,
            "service_ratio": service_ratio,
            "cost_per_served_kwh": cost_per_served_kwh,
            "carbon_per_served_kwh": carbon_per_served_kwh,
            "degradation_per_served_kwh": degradation_per_served_kwh,
            "peak_load": peak_ev_power_kw,
            "required_energy_kwh": total_required,
            "served_energy_kwh": total_served,
        })

    out["risk_samples"] = sample_rows

    # -----------------------------------------------------
    # E. backfill means from scenario samples
    # -----------------------------------------------------
    if len(sample_rows) > 0:
        sample_df = pd.DataFrame(sample_rows)

        def mean_or_nan(col):
            if col not in sample_df.columns:
                return np.nan
            vals = sample_df[col].dropna().values
            if len(vals) == 0:
                return np.nan
            return float(np.mean(vals))

        out["summary"]["cost_per_served_kwh"] = mean_or_nan("cost_per_served_kwh")
        out["summary"]["carbon_per_served_kwh"] = mean_or_nan("carbon_per_served_kwh")
        out["summary"]["degradation_per_served_kwh"] = mean_or_nan("degradation_per_served_kwh")

        if not np.isfinite(out["summary"].get("service_ratio", np.nan)):
            out["summary"]["service_ratio"] = mean_or_nan("service_ratio")
            out["summary"]["flexibility_realized"] = mean_or_nan("service_ratio")

        if not np.isfinite(out["summary"].get("peak_load", np.nan)):
            out["summary"]["peak_load"] = mean_or_nan("peak_load")

        if not np.isfinite(out["summary"].get("cvar_unmet_demand", np.nan)):
            unmet_vals = sample_df["unmet_demand"].dropna().values
            if len(unmet_vals) > 0:
                out["summary"]["cvar_unmet_demand"] = compute_cvar(unmet_vals, alpha=risk_alpha)

    return out


# =========================================================
# 3. 读取一个场景集的 lambda sweep 结果目录
# =========================================================

def read_one_lambda_sweep_root(
    root_dir: Union[str, Path],
    representative_lambda: float = 0.3,
    risk_alpha: float = 0.95
):
    """
    Example root_dir:
        lambda_sweep_results_joint/
            lambda_0_0/
            lambda_0_3/
            lambda_0_7/
            lambda_sweep_summary.csv
    """
    root_dir = Path(root_dir)

    lambda_dirs = sorted([
        p for p in root_dir.iterdir()
        if p.is_dir() and p.name.startswith("lambda_")
    ])

    if len(lambda_dirs) == 0:
        raise FileNotFoundError(f"No lambda_* subdirectories found under {root_dir}")

    # all lambdas -> for pareto_df
    lambda_results = {}
    for ldir in lambda_dirs:
        lam = tag_to_lambda(ldir.name)
        lambda_results[lam] = read_one_lambda_result_folder(ldir, risk_alpha=risk_alpha)

    # representative lambda -> for scenario_summary / contrib_df / risk_df
    if representative_lambda in lambda_results:
        rep_lam = representative_lambda
    else:
        rep_lam = sorted(lambda_results.keys())[0]

    rep_result = lambda_results[rep_lam]

    return {
        "representative_lambda": rep_lam,
        "representative_result": rep_result,
        "lambda_results": lambda_results,
    }


# =========================================================
# 4. 提取 DataFrame 行
# =========================================================

def extract_summary_row(scenario_name, representative_lambda, result):
    s = result["summary"]
    return {
        "scenario": scenario_name,
        "representative_lambda": representative_lambda,
        "service_ratio": safe_float(s.get("service_ratio", np.nan)),
        "cost_per_served_kwh": safe_float(s.get("cost_per_served_kwh", np.nan)),
        "carbon_per_served_kwh": safe_float(s.get("carbon_per_served_kwh", np.nan)),
        "degradation_per_served_kwh": safe_float(s.get("degradation_per_served_kwh", np.nan)),
        "peak_load": safe_float(s.get("peak_load", np.nan)),
        "congestion_exposure": safe_float(s.get("congestion_exposure", np.nan)),
        "cvar_unmet_demand": safe_float(s.get("cvar_unmet_demand", np.nan)),
        "flexibility_realized": safe_float(s.get("flexibility_realized", np.nan)),
        "flexibility_naive": np.nan,
    }


def extract_pareto_rows(scenario_name, lambda_results):
    rows = []
    for lam, result in sorted(lambda_results.items(), key=lambda x: x[0]):
        s = result["summary"]
        rows.append({
            "scenario": scenario_name,
            "lambda_risk": float(lam),
            "service_ratio": safe_float(s.get("service_ratio", np.nan)),
            "degradation_per_served_kwh": safe_float(s.get("degradation_per_served_kwh", np.nan)),
            "cost_per_served_kwh": safe_float(s.get("cost_per_served_kwh", np.nan)),
            "cvar_unmet_demand": safe_float(s.get("cvar_unmet_demand", np.nan)),
        })
    return rows


def extract_contrib_row(scenario_name, representative_lambda, result):
    c = result.get("contribution", {})
    return {
        "scenario": scenario_name,
        "representative_lambda": representative_lambda,
        "degradation_loss": safe_float(c.get("degradation_loss", np.nan)),
        "congestion_loss": safe_float(c.get("congestion_loss", np.nan)),
        "behavior_loss": safe_float(c.get("behavior_loss", np.nan)),
    }


def extract_risk_rows(scenario_name, representative_lambda, result):
    rows = []
    for sample in result.get("risk_samples", []):
        rows.append({
            "scenario": scenario_name,
            "representative_lambda": representative_lambda,
            "sample_id": sample.get("sample_id", np.nan),
            "scenario_subdir": sample.get("scenario_subdir", ""),
            "unmet_demand": safe_float(sample.get("unmet_demand", np.nan)),
            "total_loss": safe_float(sample.get("total_loss", np.nan)),
        })
    return rows


# =========================================================
# 5. baseline-relative normalization
# =========================================================

def add_baseline_relative_columns(scenario_summary: pd.DataFrame, baseline_scenario: str = "joint") -> pd.DataFrame:
    """
    Add baseline-relative normalized columns.

    higher-better:
        rel = scenario / baseline

    lower-better:
        rel = baseline / scenario
    """
    df = scenario_summary.copy()

    if baseline_scenario not in df["scenario"].values:
        return df

    baseline_row = df.loc[df["scenario"] == baseline_scenario].iloc[0]

    def safe_ratio(num, den):
        num = safe_float(num, default=np.nan)
        den = safe_float(den, default=np.nan)
        if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-12:
            return np.nan
        return float(num / den)

    # higher-better
    df["service_ratio_rel"] = df["service_ratio"].apply(
        lambda x: safe_ratio(x, baseline_row["service_ratio"])
    )

    # lower-better
    df["cost_per_served_kwh_rel"] = df["cost_per_served_kwh"].apply(
        lambda x: safe_ratio(baseline_row["cost_per_served_kwh"], x)
    )
    df["carbon_per_served_kwh_rel"] = df["carbon_per_served_kwh"].apply(
        lambda x: safe_ratio(baseline_row["carbon_per_served_kwh"], x)
    )
    df["degradation_per_served_kwh_rel"] = df["degradation_per_served_kwh"].apply(
        lambda x: safe_ratio(baseline_row["degradation_per_served_kwh"], x)
    )
    df["peak_load_rel"] = df["peak_load"].apply(
        lambda x: safe_ratio(baseline_row["peak_load"], x)
    )
    df["cvar_unmet_demand_rel"] = df["cvar_unmet_demand"].apply(
        lambda x: safe_ratio(baseline_row["cvar_unmet_demand"], x)
    )

    return df


# =========================================================
# 6. 主函数：从 lambda sweep 目录构建 4 张表
# =========================================================

def build_plot_data_from_lambda_sweep_directories(
    joint_dir="lambda_sweep_results_joint",
    evening_peak_dir="lambda_sweep_results_evening_peak_cluster",
    sustained_high_price_dir="lambda_sweep_results_sustained_high_price",
    capacity_shortage_dir="lambda_sweep_results_capacity_shortage",
    representative_lambda=0.3,
    risk_alpha=0.95,
):
    scenario_roots = {
        "joint": joint_dir,
        "evening_peak_cluster": evening_peak_dir,
        "sustained_high_price": sustained_high_price_dir,
        "capacity_shortage": capacity_shortage_dir,
    }

    loaded = {}
    for scenario_name, root_dir in scenario_roots.items():
        loaded[scenario_name] = read_one_lambda_sweep_root(
            root_dir=root_dir,
            representative_lambda=representative_lambda,
            risk_alpha=risk_alpha,
        )

    scenario_summary_rows = []
    pareto_rows = []
    contrib_rows = []
    risk_rows = []

    for scenario_name, item in loaded.items():
        rep_lam = item["representative_lambda"]
        rep_result = item["representative_result"]
        lambda_results = item["lambda_results"]

        scenario_summary_rows.append(
            extract_summary_row(scenario_name, rep_lam, rep_result)
        )
        pareto_rows.extend(
            extract_pareto_rows(scenario_name, lambda_results)
        )
        contrib_rows.append(
            extract_contrib_row(scenario_name, rep_lam, rep_result)
        )
        risk_rows.extend(
            extract_risk_rows(scenario_name, rep_lam, rep_result)
        )

    scenario_summary = pd.DataFrame(scenario_summary_rows)
    pareto_df = pd.DataFrame(pareto_rows)
    contrib_df = pd.DataFrame(contrib_rows)
    risk_df = pd.DataFrame(risk_rows)

    # use joint baseline as naive flexibility reference
    if "joint" in scenario_summary["scenario"].values:
        joint_ref = float(
            scenario_summary.loc[
                scenario_summary["scenario"] == "joint", "service_ratio"
            ].iloc[0]
        )
        scenario_summary["flexibility_naive"] = joint_ref
    else:
        scenario_summary["flexibility_naive"] = 1.0

    # baseline-relative normalized columns
    scenario_summary = add_baseline_relative_columns(
        scenario_summary,
        baseline_scenario="joint"
    )

    return scenario_summary, pareto_df, contrib_df, risk_df


# =========================================================
# 7. 保存
# =========================================================

def save_plot_dataframes(
    scenario_summary,
    pareto_df,
    contrib_df,
    risk_df,
    out_dir="plot_data"
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_summary.to_csv(out_dir / "scenario_summary.csv", index=False)
    pareto_df.to_csv(out_dir / "pareto_df.csv", index=False)
    contrib_df.to_csv(out_dir / "contrib_df.csv", index=False)
    risk_df.to_csv(out_dir / "risk_df.csv", index=False)


# =========================================================
# 8. 直接运行
# =========================================================

if __name__ == "__main__":
    scenario_summary, pareto_df, contrib_df, risk_df = build_plot_data_from_lambda_sweep_directories(
        joint_dir="lambda_sweep_results_joint",
        evening_peak_dir="lambda_sweep_results_evening_peak_cluster",
        sustained_high_price_dir="lambda_sweep_results_sustained_high_price",
        capacity_shortage_dir="lambda_sweep_results_capacity_shortage",
        representative_lambda=0.3,
        risk_alpha=0.95
    )

    print("scenario_summary:")
    print(scenario_summary)

    print("\npareto_df:")
    print(pareto_df)

    print("\ncontrib_df:")
    print(contrib_df)

    print("\nrisk_df:")
    print(risk_df.head())

    save_plot_dataframes(
        scenario_summary,
        pareto_df,
        contrib_df,
        risk_df,
        out_dir="plot_data"
    )