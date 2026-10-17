import pandas as pd
import numpy as np
import json
from pathlib import Path

# ============================================================
# HYDROLOGICAL STRESS-WEIGHTED WATER DEMAND EXPOSURE v1
# ============================================================
#
# Purpose:
#   Couple the independently derived:
#
#       1. Low-Flow Stress Index (LFSI)
#       2. Data-center water-demand scenarios
#
#   into a descriptive exposure metric.
#
# IMPORTANT:
#   HSWDE is NOT:
#       - water shortage
#       - water deficit
#       - actual water availability
#       - actual withdrawal
#       - actual consumption
#       - regulatory allocation
#
#   It is a scenario-weighted exposure indicator:
#
#       HSWDE = Jan-Jul water demand * LFSI
#
# This is intended for comparative analysis across years and
# data-center scenarios.
#
# ============================================================

BASE = Path(r"D:\Scopus\AI_Scopus")

DERIVED = (
    BASE
    / r"04_DATA\GloFAS_Historical_v5_Johor\DERIVED"
)

LFSI_FILE = (
    DERIVED
    / "low_flow_stress_index_annual.parquet"
)

DEMAND_FILE = (
    DERIVED
    / "data_center_water_demand_scenarios_v1.parquet"
)

OUT_FILE = (
    DERIVED
    / "hydrological_water_demand_exposure_v1.parquet"
)

META_FILE = (
    DERIVED
    / "hydrological_water_demand_exposure_v1_metadata.json"
)


# ============================================================
# READ INPUTS
# ============================================================

print("=" * 72)
print("HYDROLOGICAL STRESS-WEIGHTED WATER DEMAND EXPOSURE v1")
print("=" * 72)

print()
print("Reading LFSI dataset...")

lfsi = pd.read_parquet(
    LFSI_FILE
)

print(
    "LFSI rows:",
    len(lfsi)
)

print()
print("Reading data-center demand scenarios...")

demand = pd.read_parquet(
    DEMAND_FILE
)

print(
    "Demand scenarios:",
    len(demand)
)


# ============================================================
# INPUT QC
# ============================================================

expected_years = list(range(2019, 2027))

expected_days_by_year = {
    2019: 212,
    2020: 213,
    2021: 212,
    2022: 212,
    2023: 212,
    2024: 213,
    2025: 212,
    2026: 212
}

required_lfsi = [
    "year",
    "days",
    "LFSI",
    "LFSI_rank",
    "stress_category"
]

required_demand = [
    "scenario_id",
    "it_load_mw",
    "wue_l_per_kwh",
    "pue",
    "annual_water_m3"
]

missing_lfsi = [
    c
    for c in required_lfsi
    if c not in lfsi.columns
]

missing_demand = [
    c
    for c in required_demand
    if c not in demand.columns
]

if missing_lfsi:
    raise RuntimeError(
        f"Missing LFSI columns: {missing_lfsi}"
    )

if missing_demand:
    raise RuntimeError(
        f"Missing demand columns: {missing_demand}"
    )

if lfsi["year"].tolist() != expected_years:
    raise RuntimeError(
        "Expected LFSI years 2019-2026 exactly."
    )

if lfsi["year"].duplicated().any():
    raise RuntimeError(
        "Duplicate LFSI year detected."
    )

if len(demand) != 48:
    raise RuntimeError(
        f"Expected 48 demand scenarios, found {len(demand)}."
    )

if demand["scenario_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate scenario_id detected."
    )

if lfsi["LFSI"].isna().any():
    raise RuntimeError(
        "NaN detected in LFSI."
    )

if not lfsi["LFSI"].between(0, 1).all():
    raise RuntimeError(
        "LFSI outside [0,1]."
    )


# ============================================================
# COMMON WINDOW VALIDATION
# ============================================================

for row in lfsi.itertuples():

    expected_days = expected_days_by_year[
        int(row.year)
    ]

    if int(row.days) != expected_days:

        raise RuntimeError(
            f"Unexpected Jan-Jul days for {row.year}: "
            f"{row.days}; expected {expected_days}"
        )

if int(lfsi["days"].sum()) != 1698:

    raise RuntimeError(
        "Expected 1698 Jan-Jul observations."
    )


# ============================================================
# CROSS JOIN:
#   48 SCENARIOS x 8 YEARS = 384 ROWS
# ============================================================

lfsi_small = lfsi[
    [
        "year",
        "days",
        "LFSI",
        "LFSI_rank",
        "stress_category"
    ]
].copy()

demand_small = demand[
    [
        "scenario_id",
        "it_load_mw",
        "wue_l_per_kwh",
        "pue",
        "operating_hours_year",
        "annual_it_mwh",
        "annual_facility_mwh",
        "annual_water_m3"
    ]
].copy()

lfsi_small["_key"] = 1

demand_small["_key"] = 1

exposure = (
    demand_small
    .merge(
        lfsi_small,
        on="_key",
        how="inner"
    )
    .drop(
        columns="_key"
    )
)

expected_rows = 48 * 8

if len(exposure) != expected_rows:

    raise RuntimeError(
        f"Expected {expected_rows} rows, "
        f"found {len(exposure)}."
    )


# ============================================================
# JAN-JUL DEMAND
# ============================================================
#
# 2026 and normal years:
#   212 days
#
# Leap years 2020 and 2024:
#   213 days
#
# This preserves the same temporal window used by LFSI.
# ============================================================

exposure["jan_jul_days"] = (
    exposure["days"]
)

exposure["jan_jul_fraction_of_year"] = (
    exposure["jan_jul_days"]
    / 365.0
)

exposure["jan_jul_water_demand_m3"] = (
    exposure["annual_water_m3"]
    * exposure["jan_jul_fraction_of_year"]
)


# ============================================================
# DAILY DEMAND
# ============================================================

exposure["daily_water_demand_m3"] = (
    exposure["annual_water_m3"]
    / 365.0
)


# ============================================================
# HSWDE
# ============================================================
#
# Hydrological Stress-Weighted Water Demand Exposure
#
#     HSWDE = Jan-Jul demand * LFSI
#
# This is a relative exposure metric, not a physical deficit.
# ============================================================

exposure["HSWDE_m3_equivalent"] = (
    exposure["jan_jul_water_demand_m3"]
    * exposure["LFSI"]
)


# ============================================================
# RELATIVE EXPOSURE INDEX
# ============================================================

max_hswde = float(
    exposure["HSWDE_m3_equivalent"].max()
)

if max_hswde <= 0:

    raise RuntimeError(
        "Maximum HSWDE must be positive."
    )

exposure["HSWDE_relative_0_1"] = (
    exposure["HSWDE_m3_equivalent"]
    / max_hswde
)


# ============================================================
# HYDROLOGICAL STRESS CATEGORY
# ============================================================

def classify_exposure(value):

    if value >= 0.75:
        return "Very High"

    if value >= 0.50:
        return "High"

    if value >= 0.25:
        return "Moderate"

    return "Low"


exposure["HSWDE_category"] = (
    exposure["HSWDE_relative_0_1"]
    .apply(classify_exposure)
)


# ============================================================
# SCENARIO-SPECIFIC YEAR RANK
# ============================================================
#
# For each demand scenario, rank years by HSWDE.
#
# This answers:
#   Under the same data-center scenario, which year had the
#   greatest hydrological-stress-weighted exposure?
# ============================================================

exposure["HSWDE_year_rank_within_scenario"] = (
    exposure
    .groupby("scenario_id")[
        "HSWDE_m3_equivalent"
    ]
    .rank(
        method="min",
        ascending=False
    )
    .astype(int)
)


# ============================================================
# YEAR-SPECIFIC SCENARIO RANK
# ============================================================
#
# For each year, rank data-center scenarios by exposure.
#
# This answers:
#   Under the same hydrological condition, which demand
#   scenarios create the largest exposure?
# ============================================================

exposure["HSWDE_scenario_rank_within_year"] = (
    exposure
    .groupby("year")[
        "HSWDE_m3_equivalent"
    ]
    .rank(
        method="min",
        ascending=False
    )
    .astype(int)
)


# ============================================================
# DEMAND INTENSITY RATIO
# ============================================================
#
# Purely descriptive:
#
#   HSWDE / Jan-Jul demand = LFSI
#
# This verifies the coupling mathematically.
# ============================================================

exposure["HSWDE_ratio_to_jan_jul_demand"] = (
    exposure["HSWDE_m3_equivalent"]
    / exposure["jan_jul_water_demand_m3"]
)


# ============================================================
# QC
# ============================================================

print()
print("Running coupling QC...")

if exposure["HSWDE_m3_equivalent"].isna().any():

    raise RuntimeError(
        "NaN detected in HSWDE."
    )

if not (
    exposure["HSWDE_m3_equivalent"] >= 0
).all():

    raise RuntimeError(
        "Negative HSWDE detected."
    )

if not (
    exposure["HSWDE_relative_0_1"]
    .between(0, 1)
).all():

    raise RuntimeError(
        "Relative HSWDE outside [0,1]."
    )


# Coupling identity:
# HSWDE / Jan-Jul demand must equal LFSI.

coupling_error = (
    exposure["HSWDE_ratio_to_jan_jul_demand"]
    - exposure["LFSI"]
).abs().max()

if float(coupling_error) > 1e-10:

    raise RuntimeError(
        "HSWDE coupling identity failed."
    )


# 2026 must remain highest year for every scenario because
# LFSI 2026 is the highest hydrological stress year.

rank_2026 = exposure.loc[
    exposure["year"] == 2026,
    "HSWDE_year_rank_within_scenario"
]

if not (rank_2026 == 1).all():

    raise RuntimeError(
        "2026 is not rank #1 for every demand scenario."
    )


# Highest-demand scenario must also be highest exposure
# within each year.

max_scenario_per_year = (
    exposure
    .groupby("year")[
        "HSWDE_m3_equivalent"
    ]
    .transform("max")
)

if not (
    exposure.loc[
        exposure["HSWDE_m3_equivalent"]
        == max_scenario_per_year,
        "HSWDE_scenario_rank_within_year"
    ]
    == 1
).all():

    raise RuntimeError(
        "Scenario ranking QC failed."
    )


# ============================================================
# SAVE
# ============================================================

exposure.to_parquet(
    OUT_FILE,
    index=False
)


# ============================================================
# METADATA
# ============================================================

metadata = {

    "method":
        "Hydrological Stress-Weighted Water Demand Exposure v1",

    "formula":
        "HSWDE = Jan-Jul water demand * LFSI",

    "source_lfsi":
        str(LFSI_FILE),

    "source_demand":
        str(DEMAND_FILE),

    "study_window":
        "January-July 2019-2026",

    "scenario_count":
        48,

    "year_count":
        8,

    "expected_rows":
        384,

    "jan_jul_days":
        expected_days_by_year,

    "interpretation":
        "HSWDE is a descriptive scenario exposure metric. "
        "It does not represent actual water shortage, "
        "physical water deficit, water availability, "
        "regulatory allocation, or observed data-center "
        "water consumption.",

    "coupling_identity":
        "HSWDE / Jan-Jul demand = LFSI",

    "2026_hydrological_status":
        "Highest LFSI rank among 2019-2026.",

    "qc": {

        "status":
            "PASS",

        "rows":
            int(len(exposure)),

        "expected_rows":
            384,

        "coupling_max_absolute_error":
            float(coupling_error),

        "2026_rank_1_all_scenarios":
            True,

        "relative_exposure_range":
            "[0,1]"

    }

}


with open(
    META_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=2
    )


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 72)
print("HYDROLOGICAL WATER-DEMAND EXPOSURE COMPLETE")
print("=" * 72)

print()
print(
    "Rows:",
    len(exposure)
)

print(
    "Scenarios:",
    exposure["scenario_id"].nunique()
)

print(
    "Years:",
    exposure["year"].nunique()
)

print()
print(
    "2026 LFSI:",
    float(
        exposure.loc[
            exposure["year"] == 2026,
            "LFSI"
        ].iloc[0]
    )
)

print()
print("SELECTED 2026 EXPOSURE SCENARIOS")
print("-" * 72)

selected = exposure[
    (exposure["year"] == 2026)
    &
    (
        exposure["it_load_mw"].isin(
            [50, 100, 200, 500]
        )
    )
    &
    (
        exposure["wue_l_per_kwh"].isin(
            [0.2, 0.5, 1.0]
        )
    )
    &
    (
        exposure["pue"] == 1.3
    )
].copy()

selected_columns = [
    "scenario_id",
    "it_load_mw",
    "wue_l_per_kwh",
    "pue",
    "annual_water_m3",
    "jan_jul_water_demand_m3",
    "LFSI",
    "HSWDE_m3_equivalent",
    "HSWDE_relative_0_1",
    "HSWDE_year_rank_within_scenario"
]

print(
    selected[
        selected_columns
    ].to_string(
        index=False
    )
)

print()
print(
    "Maximum HSWDE:",
    max_hswde,
    "m3-equivalent"
)

print(
    "Coupling max error:",
    coupling_error
)

print()
print(
    "Output  :",
    OUT_FILE
)

print(
    "Metadata:",
    META_FILE
)

print(
    "QC      : PASS"
)
