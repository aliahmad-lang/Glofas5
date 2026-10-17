import pandas as pd
import json
from pathlib import Path

BASE = Path(r"D:\Scopus\AI_Scopus")

SRC = (
    BASE
    / r"04_DATA\GloFAS_Historical_v5_Johor\DERIVED"
    / "same_period_jan_jul_annual_comparison.parquet"
)

OUT = (
    BASE
    / r"04_DATA\GloFAS_Historical_v5_Johor\DERIVED"
    / "low_flow_stress_index_annual.parquet"
)

META = (
    BASE
    / r"04_DATA\GloFAS_Historical_v5_Johor\DERIVED"
    / "low_flow_stress_index_metadata.json"
)


# ============================================================
# READ INPUT
# ============================================================

print("Reading same-period comparison dataset...")

df = (
    pd.read_parquet(SRC)
    .sort_values("year")
    .reset_index(drop=True)
)

print("Rows :", len(df))
print(
    "Years:",
    int(df["year"].min()),
    "to",
    int(df["year"].max())
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = [
    "year",
    "days",
    "mean_discharge_m3s",
    "low_flow_days_q05",
    "max_low_flow_run_days"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# EXPECTED PERIOD
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


# ============================================================
# INPUT QC
# ============================================================

print()
print("Running input QC...")

if df["year"].tolist() != expected_years:
    raise RuntimeError(
        "Expected years 2019-2026 exactly."
    )

if df["year"].duplicated().any():
    raise RuntimeError(
        "Duplicate year detected."
    )

for row in df.itertuples():

    expected = expected_days_by_year[
        int(row.year)
    ]

    if int(row.days) != expected:

        raise RuntimeError(
            f"Unexpected Jan-Jul day count "
            f"for {row.year}: "
            f"{row.days}, expected {expected}"
        )

total_days = int(df["days"].sum())

if total_days != 1698:
    raise RuntimeError(
        f"Expected 1698 observations, "
        f"found {total_days}"
    )


# ============================================================
# BASELINE
# ============================================================

baseline = df[
    df["year"].between(2019, 2025)
].copy()

baseline_mean = float(
    baseline["mean_discharge_m3s"].mean()
)

baseline_std = float(
    baseline["mean_discharge_m3s"].std(
        ddof=1
    )
)


# ============================================================
# COMPONENT A
# NEGATIVE MEAN DISCHARGE ANOMALY
# ============================================================

df["mean_anomaly_pct"] = (
    df["mean_discharge_m3s"]
    / baseline_mean
) - 1.0

negative_gap = (
    -df["mean_anomaly_pct"]
).clip(lower=0)

a_max = float(
    negative_gap.max()
)

if a_max > 0:

    df["anomaly_component"] = (
        negative_gap / a_max
    )

else:

    df["anomaly_component"] = 0.0


# ============================================================
# COMPONENT F
# LOW-FLOW FREQUENCY
# ============================================================

df["low_flow_fraction"] = (
    df["low_flow_days_q05"]
    / df["days"]
)

f_max = float(
    df["low_flow_fraction"].max()
)

if f_max > 0:

    df["frequency_component"] = (
        df["low_flow_fraction"]
        / f_max
    )

else:

    df["frequency_component"] = 0.0


# ============================================================
# COMPONENT P
# LOW-FLOW PERSISTENCE
# ============================================================

p_max = float(
    df["max_low_flow_run_days"].max()
)

if p_max > 0:

    df["persistence_component"] = (
        df["max_low_flow_run_days"]
        / p_max
    )

else:

    df["persistence_component"] = 0.0


# ============================================================
# LFSI v1
# EQUAL WEIGHT
# ============================================================

df["LFSI"] = (
    df["anomaly_component"]
    + df["frequency_component"]
    + df["persistence_component"]
) / 3.0

df["LFSI_rank"] = (
    df["LFSI"]
    .rank(
        method="min",
        ascending=False
    )
    .astype(int)
)


def classify_stress(value):

    if value >= 0.75:
        return "Very High"

    if value >= 0.50:
        return "High"

    if value >= 0.25:
        return "Moderate"

    return "Low"


df["stress_category"] = (
    df["LFSI"]
    .apply(classify_stress)
)


# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

weights = {

    "equal": (
        1 / 3,
        1 / 3,
        1 / 3
    ),

    "anomaly_heavy": (
        0.50,
        0.25,
        0.25
    ),

    "frequency_heavy": (
        0.25,
        0.50,
        0.25
    ),

    "persistence_heavy": (
        0.25,
        0.25,
        0.50
    )
}


for name, (
    weight_a,
    weight_f,
    weight_p
) in weights.items():

    df[f"LFSI_{name}"] = (

        weight_a
        * df["anomaly_component"]

        +

        weight_f
        * df["frequency_component"]

        +

        weight_p
        * df["persistence_component"]

    )

    df[
        f"LFSI_{name}_rank"
    ] = (

        df[f"LFSI_{name}"]
        .rank(
            method="min",
            ascending=False
        )
        .astype(int)

    )


# ============================================================
# OUTPUT QC
# ============================================================

print()
print("Running LFSI QC...")

component_columns = [

    "anomaly_component",
    "frequency_component",
    "persistence_component",
    "LFSI"

]

if (
    df[component_columns]
    .isna()
    .any()
    .any()
):

    raise RuntimeError(
        "NaN detected in LFSI components."
    )


for column in component_columns:

    if not df[column].between(
        0,
        1
    ).all():

        raise RuntimeError(
            f"{column} contains values outside [0,1]."
        )


# ============================================================
# 2026 RANK CHECK
# ============================================================

rank_2026 = int(
    df.loc[
        df["year"] == 2026,
        "LFSI_rank"
    ].iloc[0]
)

if rank_2026 != 1:

    raise RuntimeError(
        f"Expected 2026 LFSI rank #1, "
        f"found rank {rank_2026}."
    )


# ============================================================
# SENSITIVITY QC
# ============================================================

sensitivity_columns = [

    "LFSI_equal_rank",
    "LFSI_anomaly_heavy_rank",
    "LFSI_frequency_heavy_rank",
    "LFSI_persistence_heavy_rank"

]

sensitivity_results = {}

for column in sensitivity_columns:

    row_2026 = df.loc[
        df["year"] == 2026,
        column
    ]

    rank = int(
        row_2026.iloc[0]
    )

    sensitivity_results[column] = rank


# ============================================================
# SAVE PARQUET
# ============================================================

df.to_parquet(
    OUT,
    index=False
)


# ============================================================
# METADATA
# ============================================================

metadata = {

    "method":
        "Low-Flow Stress Index v1",

    "source":
        str(SRC),

    "study_window":
        "January-July 2019-2026",

    "baseline":
        "2019-2025",

    "comparison_year":
        2026,

    "baseline_mean_m3s":
        baseline_mean,

    "baseline_std_m3s":
        baseline_std,

    "q05_definition":
        "Global Q05 daily regional-mean discharge "
        "threshold calculated from the complete "
        "2019-01-01 to 2026-07-31 study period.",

    "components": {

        "anomaly_component":
            "Negative Jan-Jul mean discharge anomaly "
            "relative to the 2019-2025 baseline, "
            "normalized by the maximum negative anomaly "
            "observed during 2019-2026.",

        "frequency_component":
            "Number of Q05 low-flow days divided by "
            "the number of Jan-Jul observations.",

        "persistence_component":
            "Maximum consecutive Q05 low-flow run "
            "normalized by the maximum observed run "
            "during 2019-2026."

    },

    "weights_equal": {

        "anomaly": 1 / 3,
        "frequency": 1 / 3,
        "persistence": 1 / 3

    },

    "sensitivity_weights": {

        name: {
            "anomaly": values[0],
            "frequency": values[1],
            "persistence": values[2]
        }

        for name, values in weights.items()

    },

    "interpretation_limit":
        "Descriptive hydrological stress indicator. "
        "It is not a water availability index, "
        "water shortage index, or data-center "
        "water-demand index.",

    "qc": {

        "years": 8,

        "expected_years":
            "2019-2026",

        "total_jan_jul_observations":
            total_days,

        "expected_total_observations":
            1698,

        "components_range":
            "[0,1]",

        "LFSI_range":
            "[0,1]",

        "2026_equal_weight_rank":
            rank_2026,

        "sensitivity_2026_ranks":
            sensitivity_results,

        "status":
            "PASS"

    }

}


with open(
    META,
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
print("LOW-FLOW STRESS INDEX v1 COMPLETE")
print("=" * 72)

report_columns = [

    "year",
    "mean_discharge_m3s",
    "mean_anomaly_pct",
    "anomaly_component",
    "frequency_component",
    "persistence_component",
    "LFSI",
    "LFSI_rank",
    "stress_category"

]

print(
    df[
        report_columns
    ].to_string(
        index=False
    )
)

print()
print("SENSITIVITY RANKING")
print("-" * 72)

print(
    df[
        [
            "year",
            "LFSI_equal_rank",
            "LFSI_anomaly_heavy_rank",
            "LFSI_frequency_heavy_rank",
            "LFSI_persistence_heavy_rank"
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "Baseline mean :",
    baseline_mean
)

print(
    "Baseline std  :",
    baseline_std
)

print(
    "2026 LFSI rank:",
    rank_2026
)

print()
print(
    "Output  :",
    OUT
)

print(
    "Metadata:",
    META
)

print(
    "QC      : PASS"
)
