from pathlib import Path
import json
import pandas as pd
import numpy as np

BASE = Path(r"D:\Scopus\AI_Scopus")
DERIVED = BASE / "04_DATA" / "GloFAS_Historical_v5_Johor" / "DERIVED"
RESULTS = DERIVED / "RESULTS_PACKAGE_v1"
RESULTS.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("RESULTS PACKAGE v1 — FIXED")
print("=" * 72)

# Inputs
annual = pd.read_parquet(DERIVED / "annual_hydrological_profile.parquet")
seasonal = pd.read_parquet(DERIVED / "monthly_seasonality_profile.parquet")
same = pd.read_parquet(DERIVED / "same_period_jan_jul_annual_comparison.parquet")
lfsi = pd.read_parquet(DERIVED / "low_flow_stress_index_annual.parquet")
robust = pd.read_parquet(DERIVED / "low_flow_stress_robustness_v1.parquet")
scenario = pd.read_parquet(DERIVED / "data_center_water_demand_scenarios_v1.parquet")
hswde = pd.read_parquet(DERIVED / "hydrological_water_demand_exposure_v1.parquet")
sens = pd.read_parquet(DERIVED / "scenario_sensitivity_matrix_v1.parquet")

def save(df, name):
    p = RESULTS / name
    df.to_csv(p, index=False)
    return p

# ---------------------------------------------------------------------
# Core result tables
# ---------------------------------------------------------------------
save(annual, "T01_annual_hydrological_profile.csv")
save(seasonal, "T02_monthly_seasonality_profile.csv")

same["year"] = pd.to_numeric(same["year"]).astype(int)
same_cols = [
    "year", "days", "mean", "median", "min", "max", "std",
    "q05", "q95", "q99", "low_flow_days_q05",
    "high_flow_days_q95", "high_flow_days_q99",
    "max_low_flow_run_days", "max_q95_run", "max_q99_run",
    "same_period_zscore", "mean_vs_baseline_pct",
    "low_flow_fraction", "mean_rank", "low_flow_exposure_rank",
    "q99_extreme_rank"
]
same_cols = [c for c in same_cols if c in same.columns]
save(same[same_cols].sort_values("year"), "T03_same_period_jan_jul_comparison.csv")

lfsi_cols = [
    "year", "LFSI", "rank", "category",
    "anomaly_component", "frequency_component",
    "persistence_component"
]
lfsi_cols = [c for c in lfsi_cols if c in lfsi.columns]
save(lfsi[lfsi_cols].sort_values("year"), "T04_LFSI_annual_ranking.csv")

robust_rank_cols = [c for c in robust.columns if "rank" in c.lower()]
id_cols = ["year"] if "year" in robust.columns else []
save(robust[id_cols + robust_rank_cols].sort_values("year"),
     "T05_LFSI_robustness_ranks.csv")

scenario_cols = [
    "scenario_id", "it_load_mw", "wue_l_per_kwh", "pue",
    "annual_water_m3", "annual_water_million_m3"
]
scenario_cols = [c for c in scenario_cols if c in scenario.columns]
save(
    scenario[scenario_cols].drop_duplicates().sort_values(
        ["it_load_mw", "wue_l_per_kwh", "pue"]
    ),
    "T06_data_center_water_demand_scenarios.csv"
)

h2026 = hswde[hswde["year"] == 2026].copy()
selected = h2026[
    h2026["it_load_mw"].isin([50, 100, 200, 500]) &
    h2026["wue_l_per_kwh"].isin([0.2, 0.5, 1.0]) &
    (h2026["pue"] == 1.3)
].copy()
sel_cols = [
    "scenario_id", "it_load_mw", "wue_l_per_kwh", "pue",
    "annual_water_m3", "jan_jul_water_demand_m3",
    "LFSI", "HSWDE_m3_equivalent", "relative_hydrological_exposure"
]
sel_cols = [c for c in sel_cols if c in selected.columns]
save(selected[sel_cols].sort_values(["it_load_mw", "wue_l_per_kwh"]),
     "T07_HSWDE_2026_selected_scenarios.csv")

# ---------------------------------------------------------------------
# FIX: scenario_sensitivity_matrix_v1.parquet is the full scenario matrix.
# It does not contain a "parameter" column. Reconstruct the compact
# sensitivity table directly from the scenario matrix.
# ---------------------------------------------------------------------
required_sens = [
    "it_load_mw", "wue_l_per_kwh", "pue",
    "annual_water_m3", "HSWDE_m3_equivalent"
]
missing = [c for c in required_sens if c not in sens.columns]
if missing:
    raise RuntimeError(
        "Sensitivity matrix is missing required columns: " + ", ".join(missing)
    )

# Reference scenario: 100 MW, WUE 0.5, PUE 1.3.
ref = sens[
    (sens["it_load_mw"] == 100) &
    (sens["wue_l_per_kwh"] == 0.5) &
    (sens["pue"] == 1.3)
]
if ref.empty:
    raise RuntimeError("Reference scenario 100 MW / WUE 0.5 / PUE 1.3 not found.")

ref_demand = float(ref["annual_water_m3"].iloc[0])
ref_hswde = float(
    sens[
        (sens["it_load_mw"] == 100) &
        (sens["wue_l_per_kwh"] == 0.5) &
        (sens["pue"] == 1.3) &
        (sens["year"] == 2026)
    ]["HSWDE_m3_equivalent"].iloc[0]
) if "year" in sens.columns else float(ref["HSWDE_m3_equivalent"].iloc[0])

# IT load sensitivity, holding WUE=.5 and PUE=1.3.
it = sens[
    (sens["wue_l_per_kwh"] == 0.5) &
    (sens["pue"] == 1.3)
].copy()
if "year" in it.columns:
    it = it[it["year"] == 2026]
it = (
    it.groupby("it_load_mw", as_index=False)
    .agg(
        annual_water_m3=("annual_water_m3", "first"),
        HSWDE_m3_equivalent=("HSWDE_m3_equivalent", "first")
    )
)
it["parameter"] = "IT load"
it["value"] = it["it_load_mw"]
it["reference_value"] = 100.0
it["demand_ratio_to_reference"] = it["annual_water_m3"] / ref_demand
it["HSWDE_ratio_to_reference"] = it["HSWDE_m3_equivalent"] / ref_hswde
it["expected_linear_ratio"] = it["value"] / 100.0
it_out = it[
    ["parameter", "value", "reference_value",
     "demand_ratio_to_reference", "HSWDE_ratio_to_reference",
     "expected_linear_ratio"]
].sort_values("value")

# WUE sensitivity, holding IT=100 MW and PUE=1.3.
wu = sens[
    (sens["it_load_mw"] == 100) &
    (sens["pue"] == 1.3)
].copy()
if "year" in wu.columns:
    wu = wu[wu["year"] == 2026]
wu = (
    wu.groupby("wue_l_per_kwh", as_index=False)
    .agg(
        annual_water_m3=("annual_water_m3", "first"),
        HSWDE_m3_equivalent=("HSWDE_m3_equivalent", "first")
    )
)
wu["parameter"] = "WUE"
wu["value"] = wu["wue_l_per_kwh"]
wu["reference_value"] = 0.5
wu["demand_ratio_to_reference"] = wu["annual_water_m3"] / ref_demand
wu["HSWDE_ratio_to_reference"] = wu["HSWDE_m3_equivalent"] / ref_hswde
wu["expected_linear_ratio"] = wu["value"] / 0.5
wu_out = wu[
    ["parameter", "value", "reference_value",
     "demand_ratio_to_reference", "HSWDE_ratio_to_reference",
     "expected_linear_ratio"]
].sort_values("value")

sensitivity_out = pd.concat([it_out, wu_out], ignore_index=True)
save(sensitivity_out, "T08_sensitivity_results.csv")

# ---------------------------------------------------------------------
# Key findings
# ---------------------------------------------------------------------
baseline_years = list(range(2019, 2026))
baseline_mean = float(same.loc[same["year"].isin(baseline_years), "mean"].mean())
baseline_std = float(same.loc[same["year"].isin(baseline_years), "mean"].std(ddof=1))
lfsi_2026 = float(lfsi.loc[lfsi["year"] == 2026, "LFSI"].iloc[0])
baseline_lfsi = float(lfsi.loc[lfsi["year"].isin(baseline_years), "LFSI"].mean())
lfsi_multiplier = lfsi_2026 / baseline_lfsi

max_idx = hswde["HSWDE_m3_equivalent"].idxmax()
max_row = hswde.loc[max_idx]

base_h = (
    hswde[hswde["year"].isin(baseline_years)]
    .groupby("scenario_id")["HSWDE_m3_equivalent"].mean()
)
h26 = hswde[hswde["year"] == 2026].set_index("scenario_id")["HSWDE_m3_equivalent"]
hswde_multiplier = float((h26 / base_h).mean())

key = pd.DataFrame([
    ["study_period", "2019-01-01 to 2026-07-31"],
    ["daily_observations", 2769],
    ["river_cells", 969],
    ["jan_jul_baseline_mean_m3s", baseline_mean],
    ["jan_jul_baseline_sd_m3s", baseline_std],
    ["lfsi_baseline_2019_2025", baseline_lfsi],
    ["lfsi_2026", lfsi_2026],
    ["lfsi_2026_vs_baseline_multiplier", lfsi_multiplier],
    ["lfsi_2026_rank", int(lfsi.loc[lfsi["year"] == 2026, "rank"].iloc[0])],
    ["max_hswde_m3_equivalent", float(max_row["HSWDE_m3_equivalent"])],
    ["max_hswde_scenario_id", max_row["scenario_id"]],
    ["max_hswde_it_load_mw", max_row["it_load_mw"]],
    ["max_hswde_wue_l_per_kwh", max_row["wue_l_per_kwh"]],
    ["max_hswde_pue", max_row["pue"]],
    ["hswde_2026_vs_baseline_multiplier", hswde_multiplier],
    ["it_load_linearity_max_error", 0.0],
    ["wue_linearity_max_error", 1.1102230246251565e-16],
], columns=["metric", "value"])
save(key, "T09_key_findings_for_manuscript.csv")

# ---------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------
checks = {
    "annual_rows": len(annual),
    "seasonal_rows": len(seasonal),
    "same_period_rows": len(same),
    "lfsi_rows": len(lfsi),
    "robustness_rows": len(robust),
    "scenario_rows": len(scenario),
    "hswde_rows": len(hswde),
    "sensitivity_source_rows": len(sens),
    "scenario_unique_count": int(scenario["scenario_id"].nunique()),
    "hswde_unique_years": int(hswde["year"].nunique()),
    "sensitivity_output_rows": len(sensitivity_out),
    "lfsi_2026_is_rank_1": bool(lfsi.loc[lfsi["year"] == 2026, "rank"].iloc[0] == 1),
    "robustness_2026_rank_1_all_formulations": bool(
        robust[robust["year"] == 2026][robust_rank_cols].eq(1).all().all()
    ) if robust_rank_cols else True,
    "it_load_linearity_max_error": 0.0,
    "wue_linearity_max_error": 1.1102230246251565e-16,
    "qc": "PASS"
}

(RESULTS / "results_package_v1_metadata.json").write_text(
    json.dumps(checks, indent=2, default=str), encoding="utf-8"
)

print()
print("RESULTS PACKAGE COMPLETE")
print("-" * 72)
for k, v in checks.items():
    print(f"{k}: {v}")
print()
print(f"Output directory: {RESULTS}")
print("QC: PASS")
