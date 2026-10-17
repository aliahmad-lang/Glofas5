import pandas as pd
import numpy as np
import json
from pathlib import Path

# ============================================================
# SCENARIO SENSITIVITY MATRIX v1
# ============================================================
#
# Purpose:
#   Quantify sensitivity of data-center water demand and
#   hydrological-stress-weighted exposure to:
#
#       1. IT load
#       2. WUE
#       3. Hydrological year / LFSI
#
# Outputs:
#       - Full scenario sensitivity dataset
#       - 2026 vs baseline hydrological comparison
#       - Elasticity checks
#       - Scenario rankings
#
# IMPORTANT:
#   HSWDE is a comparative exposure metric:
#
#       HSWDE = Jan-Jul water demand * LFSI
#
#   It is NOT a physical water deficit or water shortage.
# ============================================================

BASE = Path(r"D:\Scopus\AI_Scopus")

DERIVED = (
    BASE
    / r"04_DATA\GloFAS_Historical_v5_Johor\DERIVED"
)

EXPOSURE_FILE = (
    DERIVED
    / "hydrological_water_demand_exposure_v1.parquet"
)

OUT_FILE = (
    DERIVED
    / "scenario_sensitivity_matrix_v1.parquet"
)

META_FILE = (
    DERIVED
    / "scenario_sensitivity_matrix_v1_metadata.json"
)


# ============================================================
# READ INPUT
# ============================================================

print("=" * 72)
print("SCENARIO SENSITIVITY MATRIX v1")
print("=" * 72)

df = pd.read_parquet(
    EXPOSURE_FILE
)

print()
print(
    "Input rows:",
    len(df)
)


# ============================================================
# INPUT QC
# ============================================================

required = [
    "scenario_id",
    "year",
    "it_load_mw",
    "wue_l_per_kwh",
    "pue",
    "annual_water_m3",
    "jan_jul_water_demand_m3",
    "LFSI",
    "HSWDE_m3_equivalent"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:

    raise RuntimeError(
        f"Missing required columns: {missing}"
    )

if len(df) != 384:

    raise RuntimeError(
        f"Expected 384 rows, found {len(df)}"
    )

if df["scenario_id"].nunique() != 48:

    raise RuntimeError(
        "Expected 48 unique scenarios."
    )

if df["year"].nunique() != 8:

    raise RuntimeError(
        "Expected 8 years."
    )


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_YEARS = list(range(2019, 2027))

BASELINE_YEARS = list(range(2019, 2026))

IT_LOADS = sorted(
    df["it_load_mw"].unique()
)

WUES = sorted(
    df["wue_l_per_kwh"].unique()
)


# ============================================================
# BASELINE HYDROLOGICAL CONDITION
# ============================================================
#
# Baseline LFSI is the mean of annual LFSI for 2019-2025.
# This provides a relative historical hydrological reference.
# ============================================================

baseline_lfsi = float(
    df.loc[
        df["year"].isin(BASELINE_YEARS),
        "LFSI"
    ].mean()
)

lfsi_2026 = float(
    df.loc[
        df["year"] == 2026,
        "LFSI"
    ].iloc[0]
)


# ============================================================
# HYDROLOGICAL STRESS RATIO
# ============================================================

hydro_ratio_2026 = (
    lfsi_2026
    / baseline_lfsi
)


# ============================================================
# DEMAND SENSITIVITY
# ============================================================
#
# Water demand elasticity:
#
#   Elasticity = % change in demand / % change in parameter
#
# Because the model is multiplicative:
#
#   elasticity to IT load = 1
#   elasticity to WUE      = 1
#
# We verify this empirically.
# ============================================================

# ------------------------------------------------------------
# Reference scenario:
# 100 MW, WUE 0.5, PUE 1.3
# ------------------------------------------------------------

reference = df[
    (df["it_load_mw"] == 100)
    &
    (df["wue_l_per_kwh"] == 0.5)
    &
    (df["pue"] == 1.3)
].copy()

if len(reference) != 8:

    raise RuntimeError(
        "Reference scenario should have 8 yearly rows."
    )


# ============================================================
# 2026 SCENARIO TABLE
# ============================================================

df_2026 = df[
    df["year"] == 2026
].copy()

df_2026["demand_vs_reference_pct"] = (
    (
        df_2026["jan_jul_water_demand_m3"]
        /
        df_2026[
            "jan_jul_water_demand_m3"
        ].loc[
            df_2026["scenario_id"]
            ==
            df_2026.iloc[0]["scenario_id"]
        ].iloc[0]
    ) - 1.0
) * 100.0


# ============================================================
# BASELINE EXPOSURE BY SCENARIO
# ============================================================
#
# Since the same demand scenario is held constant across
# years, baseline exposure is:
#
#   demand * mean(LFSI 2019-2025)
#
# We calculate the historical annual exposure directly as
# well as the baseline mean.
# ============================================================

baseline_exposure = (
    df[
        df["year"].isin(BASELINE_YEARS)
    ]
    .groupby(
        "scenario_id",
        as_index=False
    )
    .agg(
        baseline_mean_lfsi=(
            "LFSI",
            "mean"
        ),
        baseline_mean_hswde=(
            "HSWDE_m3_equivalent",
            "mean"
        ),
        baseline_max_hswde=(
            "HSWDE_m3_equivalent",
            "max"
        ),
        baseline_min_hswde=(
            "HSWDE_m3_equivalent",
            "min"
        )
    )
)


# ============================================================
# 2026 EXPOSURE COMPARISON
# ============================================================

comparison = df[
    df["year"] == 2026
][
    [
        "scenario_id",
        "it_load_mw",
        "wue_l_per_kwh",
        "pue",
        "annual_water_m3",
        "jan_jul_water_demand_m3",
        "LFSI",
        "HSWDE_m3_equivalent"
    ]
].copy()

comparison = comparison.merge(
    baseline_exposure,
    on="scenario_id",
    how="left"
)

comparison["HSWDE_2026_vs_baseline_pct"] = (
    (
        comparison["HSWDE_m3_equivalent"]
        /
        comparison["baseline_mean_hswde"]
    ) - 1.0
) * 100.0


# ============================================================
# DEMAND RATIO
# ============================================================
#
# Same scenario means demand itself is unchanged between
# years. Only LFSI changes the exposure.
# ============================================================

comparison["demand_change_2026_vs_baseline_pct"] = 0.0

comparison["hydrological_exposure_multiplier_2026"] = (
    comparison["HSWDE_m3_equivalent"]
    /
    comparison["baseline_mean_hswde"]
)


# ============================================================
# SCENARIO RANKING
# ============================================================

comparison["HSWDE_2026_rank"] = (
    comparison[
        "HSWDE_m3_equivalent"
    ]
    .rank(
        method="min",
        ascending=False
    )
    .astype(int)
)


# ============================================================
# IT LOAD SENSITIVITY
# ============================================================

it_reference = 100

it_sensitivity = []

for load in IT_LOADS:

    subset = comparison[
        (comparison["it_load_mw"] == load)
        &
        (comparison["wue_l_per_kwh"] == 0.5)
        &
        (comparison["pue"] == 1.3)
    ]

    if len(subset) != 1:

        raise RuntimeError(
            f"Missing IT sensitivity case for {load} MW."
        )

    row = subset.iloc[0]

    demand_ratio = (
        row["jan_jul_water_demand_m3"]
        /
        comparison[
            (comparison["it_load_mw"] == it_reference)
            &
            (comparison["wue_l_per_kwh"] == 0.5)
            &
            (comparison["pue"] == 1.3)
        ]["jan_jul_water_demand_m3"].iloc[0]
    )

    exposure_ratio = (
        row["HSWDE_m3_equivalent"]
        /
        comparison[
            (comparison["it_load_mw"] == it_reference)
            &
            (comparison["wue_l_per_kwh"] == 0.5)
            &
            (comparison["pue"] == 1.3)
        ]["HSWDE_m3_equivalent"].iloc[0]
    )

    it_sensitivity.append({

        "parameter":
            "IT load",

        "value":
            float(load),

        "reference_value":
            float(it_reference),

        "demand_ratio_to_reference":
            float(demand_ratio),

        "HSWDE_ratio_to_reference":
            float(exposure_ratio),

        "expected_linear_ratio":
            float(load / it_reference)

    })


it_sensitivity = pd.DataFrame(
    it_sensitivity
)


# ============================================================
# WUE SENSITIVITY
# ============================================================

wue_reference = 0.5

wue_sensitivity = []

for wue in WUES:

    subset = comparison[
        (comparison["it_load_mw"] == 100)
        &
        (comparison["wue_l_per_kwh"] == wue)
        &
        (comparison["pue"] == 1.3)
    ]

    if len(subset) != 1:

        raise RuntimeError(
            f"Missing WUE sensitivity case for {wue}."
        )

    row = subset.iloc[0]

    ref = comparison[
        (comparison["it_load_mw"] == 100)
        &
        (comparison["wue_l_per_kwh"] == wue_reference)
        &
        (comparison["pue"] == 1.3)
    ].iloc[0]

    demand_ratio = (
        row["jan_jul_water_demand_m3"]
        /
        ref["jan_jul_water_demand_m3"]
    )

    exposure_ratio = (
        row["HSWDE_m3_equivalent"]
        /
        ref["HSWDE_m3_equivalent"]
    )

    wue_sensitivity.append({

        "parameter":
            "WUE",

        "value":
            float(wue),

        "reference_value":
            float(wue_reference),

        "demand_ratio_to_reference":
            float(demand_ratio),

        "HSWDE_ratio_to_reference":
            float(exposure_ratio),

        "expected_linear_ratio":
            float(wue / wue_reference)

    })


wue_sensitivity = pd.DataFrame(
    wue_sensitivity
)


# ============================================================
# ELASTICITY QC
# ============================================================

it_error = (
    it_sensitivity[
        "demand_ratio_to_reference"
    ]
    -
    it_sensitivity[
        "expected_linear_ratio"
    ]
).abs().max()

wue_error = (
    wue_sensitivity[
        "demand_ratio_to_reference"
    ]
    -
    wue_sensitivity[
        "expected_linear_ratio"
    ]
).abs().max()

if float(it_error) > 1e-10:

    raise RuntimeError(
        "IT-load linearity QC failed."
    )

if float(wue_error) > 1e-10:

    raise RuntimeError(
        "WUE linearity QC failed."
    )


# ============================================================
# ATTACH BASELINE COMPARISON TO FULL DATASET
# ============================================================

full = df.merge(
    baseline_exposure,
    on="scenario_id",
    how="left"
)

full["HSWDE_vs_baseline_pct"] = (
    (
        full["HSWDE_m3_equivalent"]
        /
        full["baseline_mean_hswde"]
    ) - 1.0
) * 100.0


# ============================================================
# OUTPUT DATASET
# ============================================================

# Store the complete annual scenario dataset with the
# baseline-comparison fields.
full.to_parquet(
    OUT_FILE,
    index=False
)


# ============================================================
# METADATA
# ============================================================

metadata = {

    "method":
        "Scenario Sensitivity Matrix v1",

    "source":
        str(EXPOSURE_FILE),

    "study_window":
        "January-July 2019-2026",

    "baseline_years":
        BASELINE_YEARS,

    "reference_scenario":
        {
            "it_load_mw": 100,
            "wue_l_per_kwh": 0.5,
            "pue": 1.3
        },

    "baseline_lfsi":
        baseline_lfsi,

    "LFSI_2026":
        lfsi_2026,

    "LFSI_2026_to_baseline_multiplier":
        hydro_ratio_2026,

    "sensitivity_parameters":
        [
            "IT load",
            "WUE",
            "PUE",
            "hydrological year"
        ],

    "expected_elasticity":
        {
            "IT_load": 1.0,
            "WUE": 1.0
        },

    "interpretation_limit":
        "HSWDE is a comparative hydrological-stress-weighted "
        "demand exposure metric. It is not a physical water "
        "deficit, water shortage, water availability estimate, "
        "or observed data-center water consumption.",

    "qc":
        {
            "status": "PASS",
            "input_rows": int(len(df)),
            "output_rows": int(len(full)),
            "scenario_count": int(
                full["scenario_id"].nunique()
            ),
            "year_count": int(
                full["year"].nunique()
            ),
            "it_load_linearity_max_error":
                float(it_error),
            "wue_linearity_max_error":
                float(wue_error)
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
print("SCENARIO SENSITIVITY MATRIX v1 COMPLETE")
print("=" * 72)

print()
print(
    "Baseline LFSI 2019-2025:",
    baseline_lfsi
)

print(
    "LFSI 2026:",
    lfsi_2026
)

print(
    "2026 / baseline LFSI:",
    hydro_ratio_2026
)

print()
print("2026 VS BASELINE EXPOSURE")
print("-" * 72)

print(
    comparison[
        [
            "scenario_id",
            "it_load_mw",
            "wue_l_per_kwh",
            "pue",
            "annual_water_m3",
            "HSWDE_m3_equivalent",
            "baseline_mean_hswde",
            "HSWDE_2026_vs_baseline_pct",
            "hydrological_exposure_multiplier_2026",
            "HSWDE_2026_rank"
        ]
    ]
    .sort_values(
        "HSWDE_m3_equivalent",
        ascending=False
    )
    .head(12)
    .to_string(
        index=False
    )
)

print()
print("IT LOAD SENSITIVITY")
print("-" * 72)

print(
    it_sensitivity.to_string(
        index=False
    )
)

print()
print("WUE SENSITIVITY")
print("-" * 72)

print(
    wue_sensitivity.to_string(
        index=False
    )
)

print()
print(
    "IT-load linearity max error:",
    it_error
)

print(
    "WUE linearity max error:",
    wue_error
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
