import pandas as pd
import numpy as np
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
    / "low_flow_stress_robustness_v1.parquet"
)

META = (
    BASE
    / r"04_DATA\GloFAS_Historical_v5_Johor\DERIVED"
    / "low_flow_stress_robustness_v1_metadata.json"
)


# ============================================================
# PURPOSE
# ============================================================
# Robustness analysis for LFSI v1.
#
# This script DOES NOT modify LFSI v1.
#
# It tests whether the identification of 2026 as the most
# stressed Jan-Jul period remains stable under:
#
#   R1. LFSI v1 reference formulation
#   R2. Standardized negative anomaly
#   R3. Alternative percentile normalization
#   R4. Alternative baseline: 2019-2024
#   R5. Leave-one-year-out baseline robustness
#
# All formulations remain descriptive hydrological indicators.
# They are NOT water availability or data-center water-demand
# indices.
# ============================================================


print("=" * 72)
print("LOW-FLOW STRESS ROBUSTNESS ANALYSIS v1")
print("=" * 72)

print()
print("Reading same-period comparison dataset...")

df = (
    pd.read_parquet(SRC)
    .sort_values("year")
    .reset_index(drop=True)
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
        f"Missing columns: {missing}"
    )

if df["year"].tolist() != expected_years:
    raise RuntimeError(
        "Expected years 2019-2026 exactly."
    )

if df["year"].duplicated().any():
    raise RuntimeError(
        "Duplicate year detected."
    )

for row in df.itertuples():

    expected_days = expected_days_by_year[
        int(row.year)
    ]

    if int(row.days) != expected_days:

        raise RuntimeError(
            f"Unexpected Jan-Jul day count "
            f"for {row.year}: "
            f"{row.days}; expected {expected_days}"
        )

if int(df["days"].sum()) != 1698:
    raise RuntimeError(
        "Expected 1698 total Jan-Jul observations."
    )


# ============================================================
# COMMON VARIABLES
# ============================================================

years = df["year"].to_numpy(dtype=int)

mean_flow = (
    df["mean_discharge_m3s"]
    .to_numpy(dtype=float)
)

low_flow_fraction = (
    df["low_flow_days_q05"]
    / df["days"]
).to_numpy(dtype=float)

persistence = (
    df["max_low_flow_run_days"]
    .to_numpy(dtype=float)
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def minmax_normalize_positive(values):

    values = np.asarray(
        values,
        dtype=float
    )

    maximum = np.nanmax(values)

    if maximum <= 0:
        return np.zeros_like(values)

    return values / maximum


def negative_anomaly_vs_baseline(
    values,
    baseline_mean
):

    anomaly = (
        values / baseline_mean
    ) - 1.0

    return np.clip(
        -anomaly,
        0,
        None
    )


def build_index(
    anomaly_raw,
    frequency_raw,
    persistence_raw
):

    a = minmax_normalize_positive(
        anomaly_raw
    )

    f = minmax_normalize_positive(
        frequency_raw
    )

    p = minmax_normalize_positive(
        persistence_raw
    )

    index = (
        a + f + p
    ) / 3.0

    return (
        a,
        f,
        p,
        index
    )


def rank_index(values):

    return (
        pd.Series(values)
        .rank(
            method="min",
            ascending=False
        )
        .astype(int)
        .to_numpy()
    )


# ============================================================
# R1 — REFERENCE LFSI v1
# ============================================================

baseline_2019_2025 = mean_flow[
    years <= 2025
].mean()

anomaly_r1 = negative_anomaly_vs_baseline(
    mean_flow,
    baseline_2019_2025
)

a1, f1, p1, index_r1 = build_index(
    anomaly_r1,
    low_flow_fraction,
    persistence
)

rank_r1 = rank_index(index_r1)


# ============================================================
# R2 — STANDARDIZED NEGATIVE ANOMALY
# ============================================================
# Uses baseline standard deviation rather than maximum
# negative anomaly as the severity scale.
#
# Frequency and persistence remain normalized by observed
# maxima. The purpose is to test sensitivity to anomaly
# scaling only.
# ============================================================

baseline_mask_2019_2025 = (
    years <= 2025
)

baseline_std_2019_2025 = (
    mean_flow[baseline_mask_2019_2025]
    .std(ddof=1)
)

negative_z = np.clip(
    (
        baseline_2019_2025
        - mean_flow
    ) / baseline_std_2019_2025,
    0,
    None
)

a2 = minmax_normalize_positive(
    negative_z
)

f2 = minmax_normalize_positive(
    low_flow_fraction
)

p2 = minmax_normalize_positive(
    persistence
)

index_r2 = (
    a2 + f2 + p2
) / 3.0

rank_r2 = rank_index(index_r2)


# ============================================================
# R3 — PERCENTILE-BASED ANOMALY
# ============================================================
# Uses the empirical 25th percentile of Jan-Jul annual mean
# discharge as a reference. This tests whether the result
# depends on the baseline mean.
# ============================================================

q25 = float(
    np.percentile(
        mean_flow,
        25
    )
)

negative_percentile_gap = np.clip(
    q25 - mean_flow,
    0,
    None
)

a3 = minmax_normalize_positive(
    negative_percentile_gap
)

f3 = minmax_normalize_positive(
    low_flow_fraction
)

p3 = minmax_normalize_positive(
    persistence
)

index_r3 = (
    a3 + f3 + p3
) / 3.0

rank_r3 = rank_index(index_r3)


# ============================================================
# R4 — ALTERNATIVE BASELINE 2019-2024
# ============================================================
# 2025 is removed from the baseline. This is useful because
# 2025 is the highest-flow year and could influence the
# 2019-2025 reference mean.
# ============================================================

baseline_2019_2024 = mean_flow[
    years <= 2024
].mean()

anomaly_r4 = negative_anomaly_vs_baseline(
    mean_flow,
    baseline_2019_2024
)

a4, f4, p4, index_r4 = build_index(
    anomaly_r4,
    low_flow_fraction,
    persistence
)

rank_r4 = rank_index(index_r4)


# ============================================================
# R5 — LEAVE-ONE-YEAR-OUT BASELINE ROBUSTNESS
# ============================================================
# For every year, compute its anomaly against a baseline
# excluding that same year. This avoids evaluating a year
# against a baseline containing itself.
#
# This is particularly useful for determining whether the
# 2026 result is robust to baseline construction.
# ============================================================

loo_baseline_mean = np.zeros(
    len(df),
    dtype=float
)

loo_anomaly = np.zeros(
    len(df),
    dtype=float
)

for i, year in enumerate(years):

    reference_mask = (
        (years >= 2019)
        & (years <= 2025)
        & (years != year)
    )

    reference_values = mean_flow[
        reference_mask
    ]

    if len(reference_values) == 0:

        raise RuntimeError(
            f"No LOO baseline available for {year}"
        )

    reference_mean = (
        reference_values.mean()
    )

    loo_baseline_mean[i] = (
        reference_mean
    )

    loo_anomaly[i] = max(
        (
            reference_mean
            - mean_flow[i]
        ) / reference_mean,
        0
    )

a5 = minmax_normalize_positive(
    loo_anomaly
)

f5 = minmax_normalize_positive(
    low_flow_fraction
)

p5 = minmax_normalize_positive(
    persistence
)

index_r5 = (
    a5 + f5 + p5
) / 3.0

rank_r5 = rank_index(index_r5)


# ============================================================
# STORE RESULTS
# ============================================================

df["robustness_R1_LFSI_v1"] = index_r1
df["robustness_R1_rank"] = rank_r1

df["robustness_R2_standardized_anomaly"] = index_r2
df["robustness_R2_rank"] = rank_r2

df["robustness_R3_percentile_anomaly"] = index_r3
df["robustness_R3_rank"] = rank_r3

df["robustness_R4_baseline_2019_2024"] = index_r4
df["robustness_R4_rank"] = rank_r4

df["robustness_R5_leave_one_year_out"] = index_r5
df["robustness_R5_rank"] = rank_r5

df["LOO_baseline_mean_m3s"] = (
    loo_baseline_mean
)

# ------------------------------------------------------------
# Robustness consensus
# ------------------------------------------------------------

rank_columns = [
    "robustness_R1_rank",
    "robustness_R2_rank",
    "robustness_R3_rank",
    "robustness_R4_rank",
    "robustness_R5_rank"
]

df["robustness_rank_mean"] = (
    df[rank_columns]
    .mean(axis=1)
)

df["robustness_rank_max"] = (
    df[rank_columns]
    .max(axis=1)
)

df["robustness_rank_min"] = (
    df[rank_columns]
    .min(axis=1)
)

df["robustness_rank_stable"] = (
    df[rank_columns]
    .nunique(axis=1)
    == 1
)


# ============================================================
# ROBUSTNESS QC
# ============================================================

print()
print("Running robustness QC...")

robustness_value_columns = [
    "robustness_R1_LFSI_v1",
    "robustness_R2_standardized_anomaly",
    "robustness_R3_percentile_anomaly",
    "robustness_R4_baseline_2019_2024",
    "robustness_R5_leave_one_year_out"
]

for column in robustness_value_columns:

    if df[column].isna().any():

        raise RuntimeError(
            f"NaN detected in {column}"
        )

    if not df[column].between(
        0,
        1
    ).all():

        raise RuntimeError(
            f"{column} outside [0,1]"
        )


# ============================================================
# 2026 ROBUSTNESS TEST
# ============================================================

row_2026 = df[
    df["year"] == 2026
].iloc[0]

ranks_2026 = {

    "R1_LFSI_v1":
        int(row_2026["robustness_R1_rank"]),

    "R2_standardized_anomaly":
        int(row_2026["robustness_R2_rank"]),

    "R3_percentile_anomaly":
        int(row_2026["robustness_R3_rank"]),

    "R4_baseline_2019_2024":
        int(row_2026["robustness_R4_rank"]),

    "R5_leave_one_year_out":
        int(row_2026["robustness_R5_rank"])

}

print()
print("2026 robustness ranks:")

for name, rank in ranks_2026.items():

    print(
        f"{name:35s}: {rank}"
    )


if not all(
    rank == 1
    for rank in ranks_2026.values()
):

    raise RuntimeError(
        "2026 is not rank #1 under all robustness formulations."
    )


# ============================================================
# RANK STABILITY
# ============================================================

rank_matrix = df[
    [
        "year"
    ] + rank_columns
].copy()

print()
print("ROBUSTNESS RANKING")
print("-" * 72)

print(
    rank_matrix.to_string(
        index=False
    )
)


# ============================================================
# METADATA
# ============================================================

metadata = {

    "method":
        "Low-Flow Stress Robustness Analysis v1",

    "source":
        str(SRC),

    "study_window":
        "January-July 2019-2026",

    "reference_baseline":
        "2019-2025",

    "alternative_baseline":
        "2019-2024",

    "comparison_year":
        2026,

    "formulations": {

        "R1":
            "Reference LFSI v1 using negative mean anomaly, "
            "low-flow frequency, and persistence.",

        "R2":
            "Negative standardized anomaly using the "
            "2019-2025 baseline standard deviation.",

        "R3":
            "Negative departure from the empirical 25th "
            "percentile of annual Jan-Jul mean discharge.",

        "R4":
            "Reference formulation using a 2019-2024 baseline.",

        "R5":
            "Leave-one-year-out baseline anomaly formulation."

    },

    "q25_annual_mean_m3s":
        q25,

    "baseline_2019_2025_mean_m3s":
        float(baseline_2019_2025),

    "baseline_2019_2025_std_m3s":
        float(baseline_std_2019_2025),

    "baseline_2019_2024_mean_m3s":
        float(baseline_2019_2024),

    "interpretation_limit":
        "Robustness indicators are descriptive hydrological "
        "stress measures. They do not represent actual "
        "water availability, water shortage, environmental "
        "flow compliance, or data-center water demand.",

    "qc": {

        "status":
            "PASS",

        "years":
            8,

        "total_jan_jul_observations":
            1698,

        "2026_rank_R1":
            ranks_2026["R1_LFSI_v1"],

        "2026_rank_R2":
            ranks_2026["R2_standardized_anomaly"],

        "2026_rank_R3":
            ranks_2026["R3_percentile_anomaly"],

        "2026_rank_R4":
            ranks_2026["R4_baseline_2019_2024"],

        "2026_rank_R5":
            ranks_2026["R5_leave_one_year_out"],

        "2026_rank_stable":
            all(
                rank == 1
                for rank in ranks_2026.values()
            )

    }

}


# ============================================================
# SAVE OUTPUT
# ============================================================

df.to_parquet(
    OUT,
    index=False
)

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
# FINAL REPORT
# ============================================================

print()
print("=" * 72)
print("LOW-FLOW STRESS ROBUSTNESS ANALYSIS COMPLETE")
print("=" * 72)

print()
print("Baseline 2019-2025 mean :",
      baseline_2019_2025)

print("Baseline 2019-2025 std  :",
      baseline_std_2019_2025)

print("Baseline 2019-2024 mean :",
      baseline_2019_2024)

print("Annual Jan-Jul Q25       :",
      q25)

print()
print("2026 rank under all five formulations:")

for name, rank in ranks_2026.items():

    print(
        f"  {name:33s} = {rank}"
    )

print()
print("2026 robust across all formulations:",
      all(rank == 1 for rank in ranks_2026.values()))

print()
print("Output  :", OUT)
print("Metadata:", META)
print("QC      : PASS")
