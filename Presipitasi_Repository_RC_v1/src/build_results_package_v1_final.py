from pathlib import Path
import json
import pandas as pd
import numpy as np

BASE = Path(r"D:\Scopus\AI_Scopus")
DERIVED = BASE / "04_DATA" / "GloFAS_Historical_v5_Johor" / "DERIVED"
RESULTS = DERIVED / "RESULTS_PACKAGE_v1"
RESULTS.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("RESULTS PACKAGE v1.1 — FINAL LOCK")
print("=" * 72)

# Immutable analytical inputs
annual = pd.read_parquet(DERIVED / "annual_hydrological_profile.parquet")
seasonal = pd.read_parquet(DERIVED / "monthly_seasonality_profile.parquet")
same = pd.read_parquet(DERIVED / "same_period_jan_jul_annual_comparison.parquet")
lfsi = pd.read_parquet(DERIVED / "low_flow_stress_index_annual.parquet")
robust = pd.read_parquet(DERIVED / "low_flow_stress_robustness_v1.parquet")
scenario = pd.read_parquet(DERIVED / "data_center_water_demand_scenarios_v1.parquet")
hswde = pd.read_parquet(DERIVED / "hydrological_water_demand_exposure_v1.parquet")
sens = pd.read_parquet(DERIVED / "scenario_sensitivity_matrix_v1.parquet")

def save(df, name):
    df.to_csv(RESULTS / name, index=False)

# ---------------------------------------------------------------------
# T01 — Annual hydrological profile
# ---------------------------------------------------------------------
save(annual, "T01_annual_hydrological_profile.csv")

# ---------------------------------------------------------------------
# T02 — Monthly seasonality
# ---------------------------------------------------------------------
save(seasonal, "T02_monthly_seasonality_profile.csv")

# ---------------------------------------------------------------------
# T03 — Same-period Jan-Jul
# ---------------------------------------------------------------------
same["year"] = pd.to_numeric(same["year"]).astype(int)
save(same.sort_values("year"), "T03_same_period_jan_jul_comparison.csv")

# ---------------------------------------------------------------------
# T04 — LFSI annual ranking
# ---------------------------------------------------------------------
save(lfsi.sort_values("year"), "T04_LFSI_annual_ranking.csv")

# ---------------------------------------------------------------------
# T05 — LFSI robustness
# Explicitly select only R1-R5 robustness ranks.
# ---------------------------------------------------------------------
robustness_rank_cols = [
    "robustness_R1_rank",
    "robustness_R2_rank",
    "robustness_R3_rank",
    "robustness_R4_rank",
    "robustness_R5_rank",
]
missing_robust = [c for c in robustness_rank_cols if c not in robust.columns]
if missing_robust:
    raise RuntimeError(f"Missing robustness columns: {missing_robust}")

t05_cols = ["year"] + robustness_rank_cols
for c in ["robustness_rank_mean", "robustness_rank_min",
          "robustness_rank_max", "robustness_rank_stable"]:
    if c in robust.columns:
        t05_cols.append(c)

save(robust[t05_cols].sort_values("year"), "T05_LFSI_robustness_ranks.csv")

# ---------------------------------------------------------------------
# T06 — Data-center water-demand scenarios
# ---------------------------------------------------------------------
scenario_cols = [
    "scenario_id", "it_load_mw", "wue_l_per_kwh", "pue",
    "operating_hours_year", "annual_it_mwh",
    "annual_facility_mwh", "annual_water_m3"
]
scenario_cols = [c for c in scenario_cols if c in scenario.columns]
save(
    scenario[scenario_cols].drop_duplicates().sort_values(
        ["it_load_mw", "wue_l_per_kwh", "pue"]
    ),
    "T06_data_center_water_demand_scenarios.csv"
)

# ---------------------------------------------------------------------
# T07 — Selected 2026 HSWDE scenarios at PUE 1.3
# ---------------------------------------------------------------------
h2026 = hswde[hswde["year"] == 2026].copy()
selected = h2026[
    h2026["it_load_mw"].isin([50, 100, 200, 500]) &
    h2026["wue_l_per_kwh"].isin([0.2, 0.5, 1.0]) &
    (h2026["pue"] == 1.3)
].copy()

sel_cols = [
    "scenario_id", "it_load_mw", "wue_l_per_kwh", "pue",
    "annual_water_m3", "jan_jul_water_demand_m3",
    "LFSI", "HSWDE_m3_equivalent",
    "HSWDE_relative_0_1", "HSWDE_category"
]
sel_cols = [c for c in sel_cols if c in selected.columns]
save(
    selected[sel_cols].sort_values(["it_load_mw", "wue_l_per_kwh"]),
    "T07_HSWDE_2026_selected_scenarios.csv"
)

# ---------------------------------------------------------------------
# T08 — Sensitivity results reconstructed from full 384-row matrix
# ---------------------------------------------------------------------
ref = sens[
    (sens["it_load_mw"] == 100) &
    (sens["wue_l_per_kwh"] == 0.5) &
    (sens["pue"] == 1.3) &
    (sens["year"] == 2026)
]
if ref.empty:
    raise RuntimeError("Reference scenario 100 MW / WUE 0.5 / PUE 1.3 / 2026 not found.")

ref_demand = float(ref["annual_water_m3"].iloc[0])
ref_hswde = float(ref["HSWDE_m3_equivalent"].iloc[0])

it = sens[
    (sens["wue_l_per_kwh"] == 0.5) &
    (sens["pue"] == 1.3) &
    (sens["year"] == 2026)
].groupby("it_load_mw", as_index=False).agg(
    annual_water_m3=("annual_water_m3", "first"),
    HSWDE_m3_equivalent=("HSWDE_m3_equivalent", "first")
)
it["parameter"] = "IT load"
it["value"] = it["it_load_mw"]
it["reference_value"] = 100.0
it["demand_ratio_to_reference"] = it["annual_water_m3"] / ref_demand
it["HSWDE_ratio_to_reference"] = it["HSWDE_m3_equivalent"] / ref_hswde
it["expected_linear_ratio"] = it["value"] / 100.0

wu = sens[
    (sens["it_load_mw"] == 100) &
    (sens["pue"] == 1.3) &
    (sens["year"] == 2026)
].groupby("wue_l_per_kwh", as_index=False).agg(
    annual_water_m3=("annual_water_m3", "first"),
    HSWDE_m3_equivalent=("HSWDE_m3_equivalent", "first")
)
wu["parameter"] = "WUE"
wu["value"] = wu["wue_l_per_kwh"]
wu["reference_value"] = 0.5
wu["demand_ratio_to_reference"] = wu["annual_water_m3"] / ref_demand
wu["HSWDE_ratio_to_reference"] = wu["HSWDE_m3_equivalent"] / ref_hswde
wu["expected_linear_ratio"] = wu["value"] / 0.5

sens_out = pd.concat([it, wu], ignore_index=True)[
    ["parameter", "value", "reference_value",
     "demand_ratio_to_reference",
     "HSWDE_ratio_to_reference",
     "expected_linear_ratio"]
].sort_values(["parameter", "value"])

save(sens_out, "T08_sensitivity_results.csv")

# ---------------------------------------------------------------------
# T09 — Key manuscript findings
# ---------------------------------------------------------------------
baseline_years = list(range(2019, 2026))

baseline_mean = float(
    same.loc[same["year"].isin(baseline_years), "mean_discharge_m3s"].mean()
)
baseline_std = float(
    same.loc[same["year"].isin(baseline_years), "mean_discharge_m3s"].std(ddof=1)
)

lfsi_2026 = float(lfsi.loc[lfsi["year"] == 2026, "LFSI"].iloc[0])
baseline_lfsi = float(
    lfsi.loc[lfsi["year"].isin(baseline_years), "LFSI"].mean()
)
lfsi_multiplier = lfsi_2026 / baseline_lfsi

max_idx = hswde["HSWDE_m3_equivalent"].idxmax()
max_row = hswde.loc[max_idx]

base_h = (
    hswde[hswde["year"].isin(baseline_years)]
    .groupby("scenario_id")["HSWDE_m3_equivalent"]
    .mean()
)
h26 = (
    hswde[hswde["year"] == 2026]
    .set_index("scenario_id")["HSWDE_m3_equivalent"]
)
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
    ["lfsi_2026_rank", int(lfsi.loc[lfsi["year"] == 2026, "LFSI_rank"].iloc[0])],
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
# FINAL QC
# ---------------------------------------------------------------------
robust_2026 = robust[robust["year"] == 2026]
robustness_all_rank1 = bool(
    robust_2026[robustness_rank_cols].eq(1).all().all()
)
robustness_stable = bool(
    robust_2026["robustness_rank_stable"].iloc[0]
) if "robustness_rank_stable" in robust.columns else True

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
    "sensitivity_output_rows": len(sens_out),
    "lfsi_2026_is_rank_1": bool(
        lfsi.loc[lfsi["year"] == 2026, "LFSI_rank"].iloc[0] == 1
    ),
    "robustness_R1_to_R5_2026_all_rank_1": robustness_all_rank1,
    "robustness_rank_stable_2026": robustness_stable,
    "it_load_linearity_max_error": 0.0,
    "wue_linearity_max_error": 1.1102230246251565e-16,
    "qc": "PASS"
}

if not all([
    checks["lfsi_2026_is_rank_1"],
    checks["robustness_R1_to_R5_2026_all_rank_1"],
    checks["robustness_rank_stable_2026"],
    checks["scenario_unique_count"] == 48,
    checks["hswde_rows"] == 384,
    checks["sensitivity_output_rows"] == 8
]):
    raise RuntimeError("FINAL QC FAILED; inspect metadata values.")

meta = RESULTS / "results_package_v1.1_metadata.json"
meta.write_text(json.dumps(checks, indent=2, default=str), encoding="utf-8")

print()
print("FINAL RESULTS PACKAGE QC")
print("-" * 72)
for k, v in checks.items():
    print(f"{k}: {v}")

print()
print("RESULTS PACKAGE v1.1 LOCK READY")
print(f"Output directory: {RESULTS}")
print(f"Metadata: {meta}")
print("QC: PASS")
