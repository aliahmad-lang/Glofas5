from pathlib import Path
import pandas as pd
import numpy as np
import json

BASE = Path(r"D:\Scopus\AI_Scopus")

DAILY = (
    BASE / "04_DATA" / "GloFAS_Historical_v5_Johor"
    / "DERIVED" / "extreme_hydrological_stress_daily.parquet"
)

OUTDIR = (
    BASE / "04_DATA" / "GloFAS_Historical_v5_Johor"
    / "DERIVED"
)

OUT = OUTDIR / "same_period_jan_jul_annual_comparison.parquet"
META = OUTDIR / "same_period_jan_jul_metadata.json"

print("Reading extreme hydrological dataset...")

df = pd.read_parquet(DAILY)

df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# =========================================================
# SAME-PERIOD WINDOW
# =========================================================

df = df[df["month"] <= 7].copy()

print("Period :", df["date"].min(), "to", df["date"].max())
print("Rows   :", len(df))

expected_years = list(range(2019, 2027))
expected_days = 212

# =========================================================
# YEARLY SAME-PERIOD SUMMARY
# =========================================================

g = df.groupby("year")

annual = pd.DataFrame({
    "days":
        g["date"].count(),

    "mean_discharge_m3s":
        g["discharge_mean_m3s"].mean(),

    "median_discharge_m3s":
        g["discharge_mean_m3s"].median(),

    "min_discharge_m3s":
        g["discharge_mean_m3s"].min(),

    "max_discharge_m3s":
        g["discharge_mean_m3s"].max(),

    "std_discharge_m3s":
        g["discharge_mean_m3s"].std(),

    "q05_discharge_m3s":
        g["discharge_mean_m3s"].quantile(0.05),

    "q95_discharge_m3s":
        g["discharge_mean_m3s"].quantile(0.95),

    "q99_discharge_m3s":
        g["discharge_mean_m3s"].quantile(0.99),

    "low_flow_days_q05":
        g["low_flow_q05"].sum(),

    "high_flow_days_q95":
        g["high_flow_q95"].sum(),

    "high_flow_days_q99":
        g["high_flow_q99"].sum(),

    "max_low_flow_run_days":
        g["low_flow_run_days"].max(),

    "max_q95_run_days":
        g["high_flow_q95_run_days"].max(),

    "max_q99_run_days":
        g["high_flow_q99_run_days"].max(),

    "max_cell_discharge_m3s":
        g["discharge_max_m3s"].max()
}).reset_index()

# =========================================================
# PEAK AND LOW DATES
# =========================================================

peak_idx = (
    df.groupby("year")["discharge_mean_m3s"]
    .idxmax()
)

peak = (
    df.loc[
        peak_idx,
        ["year", "date", "discharge_mean_m3s"]
    ]
    .rename(
        columns={
            "date": "peak_date",
            "discharge_mean_m3s":
                "peak_regional_mean_m3s"
        }
    )
)

annual = annual.merge(
    peak,
    on="year",
    how="left"
)

low_idx = (
    df.groupby("year")["discharge_mean_m3s"]
    .idxmin()
)

low = (
    df.loc[
        low_idx,
        ["year", "date", "discharge_mean_m3s"]
    ]
    .rename(
        columns={
            "date": "lowest_flow_date",
            "discharge_mean_m3s":
                "lowest_regional_mean_m3s"
        }
    )
)

annual = annual.merge(
    low,
    on="year",
    how="left"
)

# =========================================================
# BASELINE 2019-2025
# =========================================================

baseline = annual[
    annual["year"].between(2019, 2025)
].copy()

baseline_mean = baseline["mean_discharge_m3s"].mean()
baseline_std = baseline["mean_discharge_m3s"].std()

annual["baseline_2019_2025_mean_m3s"] = baseline_mean
annual["baseline_2019_2025_std_m3s"] = baseline_std

annual["same_period_mean_zscore"] = (
    annual["mean_discharge_m3s"]
    - baseline_mean
) / baseline_std

annual["mean_vs_baseline_pct"] = (
    (
        annual["mean_discharge_m3s"]
        - baseline_mean
    )
    / baseline_mean
) * 100.0

annual["low_flow_fraction"] = (
    annual["low_flow_days_q05"]
    / annual["days"]
)

annual["q95_high_flow_fraction"] = (
    annual["high_flow_days_q95"]
    / annual["days"]
)

annual["q99_high_flow_fraction"] = (
    annual["high_flow_days_q99"]
    / annual["days"]
)

# =========================================================
# RANKINGS
# =========================================================

annual["mean_discharge_rank"] = (
    annual["mean_discharge_m3s"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

annual["low_flow_exposure_rank"] = (
    annual["low_flow_fraction"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

annual["q99_extreme_rank"] = (
    annual["high_flow_q99_days"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

# =========================================================
# QC
# =========================================================

print()
print("Running QC...")

if len(annual) != 8:
    raise RuntimeError(
        f"Expected 8 years, got {len(annual)}"
    )

if sorted(annual["year"].tolist()) != expected_years:
    raise RuntimeError(
        "Unexpected year range"
    )

if annual["days"].nunique() != 1:
    raise RuntimeError(
        "Same-period day counts are not identical"
    )

if int(annual["days"].iloc[0]) != expected_days:
    raise RuntimeError(
        f"Expected {expected_days} days/year"
    )

if annual["year"].duplicated().sum() != 0:
    raise RuntimeError(
        "Duplicate years detected"
    )

if annual["mean_discharge_m3s"].isna().sum() != 0:
    raise RuntimeError(
        "NaN same-period mean detected"
    )

if annual["same_period_mean_zscore"].isna().sum() != 0:
    raise RuntimeError(
        "NaN baseline z-score detected"
    )

if baseline_std <= 0:
    raise RuntimeError(
        "Baseline standard deviation is invalid"
    )

# =========================================================
# SAVE
# =========================================================

annual.to_parquet(
    OUT,
    index=False
)

metadata = {
    "dataset":
        "GloFAS Historical V5 Johor same-period annual comparison",

    "source":
        str(DAILY),

    "comparison_window":
        "January through July",

    "days_per_year":
        expected_days,

    "years":
        expected_years,

    "baseline":
        "2019-2025",

    "baseline_mean_m3s":
        float(baseline_mean),

    "baseline_std_m3s":
        float(baseline_std),

    "interpretation_note":
        "2026 is compared with 2019-2025 using the identical January-July observation window. This avoids partial-year bias.",

    "qc_status":
        "PASS",

    "source_modified":
        False
}

META.write_text(
    json.dumps(
        metadata,
        indent=2
    ),
    encoding="utf-8"
)

# =========================================================
# REPORT
# =========================================================

lowest_year = int(
    annual.loc[
        annual["mean_discharge_m3s"].idxmin(),
        "year"
    ]
)

highest_year = int(
    annual.loc[
        annual["mean_discharge_m3s"].idxmax(),
        "year"
    ]
)

most_low_flow_year = int(
    annual.loc[
        annual["low_flow_days_q05"].idxmax(),
        "year"
    ]
)

most_q99_year = int(
    annual.loc[
        annual["high_flow_days_q99"].idxmax(),
        "year"
    ]
)

print()
print("==============================================")
print("SAME-PERIOD JAN-JUL COMPARISON COMPLETE")
print("==============================================")
print("Years                  :", len(annual))
print("Days per year          :", int(annual["days"].iloc[0]))
print("Baseline               : 2019-2025")
print("Baseline mean          :", baseline_mean)
print("Baseline std           :", baseline_std)
print("Lowest Jan-Jul year    :", lowest_year)
print("Highest Jan-Jul year   :", highest_year)
print("Most Q05 low-flow days :", most_low_flow_year)
print("Most Q99 extreme days  :", most_q99_year)
print("Output                 :", OUT)
print("Metadata               :", META)
print("QC                     : PASS")
