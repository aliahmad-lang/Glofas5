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

ANNUAL_OUT = OUTDIR / "annual_hydrological_profile.parquet"
SEASONAL_OUT = OUTDIR / "monthly_seasonality_profile.parquet"
META = OUTDIR / "annual_seasonality_profile_metadata.json"

print("Reading extreme hydrological dataset...")

df = pd.read_parquet(DAILY)

df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

print("Rows :", len(df))
print("Years:", df["year"].nunique())

# =========================================================
# GLOBAL THRESHOLDS
# =========================================================

q05 = df["discharge_mean_m3s"].quantile(0.05)
q95 = df["discharge_mean_m3s"].quantile(0.95)
q99 = df["discharge_mean_m3s"].quantile(0.99)

# =========================================================
# ANNUAL PROFILE
# =========================================================

g = df.groupby("year")

annual = pd.DataFrame({
    "days":
        g["date"].count(),

    "mean_discharge_m3s":
        g["discharge_mean_m3s"].mean(),

    "median_discharge_m3s":
        g["discharge_mean_m3s"].median(),

    "min_daily_discharge_m3s":
        g["discharge_mean_m3s"].min(),

    "max_daily_discharge_m3s":
        g["discharge_mean_m3s"].max(),

    "q05_discharge_m3s":
        g["discharge_mean_m3s"].quantile(0.05),

    "q95_discharge_m3s":
        g["discharge_mean_m3s"].quantile(0.95),

    "q99_discharge_m3s":
        g["discharge_mean_m3s"].quantile(0.99),

    "low_flow_days":
        g["low_flow_q05"].sum(),

    "high_flow_q95_days":
        g["high_flow_q95"].sum(),

    "high_flow_q99_days":
        g["high_flow_q99"].sum(),

    "max_low_flow_run_days":
        g["low_flow_run_days"].max(),

    "max_q95_run_days":
        g["high_flow_q95_run_days"].max(),

    "max_q99_run_days":
        g["high_flow_q99_run_days"].max(),

    "annual_max_cell_discharge_m3s":
        g["discharge_max_m3s"].max()
}).reset_index()

# Peak date for every year
peak_idx = (
    df.groupby("year")["discharge_mean_m3s"]
    .idxmax()
)

peaks = (
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
    peaks,
    on="year",
    how="left"
)

# Low-flow date for every year
low_idx = (
    df.groupby("year")["discharge_mean_m3s"]
    .idxmin()
)

lows = (
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
    lows,
    on="year",
    how="left"
)

# =========================================================
# STANDARDIZED ANNUAL INDICATORS
# =========================================================

annual_mean = annual["mean_discharge_m3s"].mean()
annual_std = annual["mean_discharge_m3s"].std()

annual["annual_mean_zscore"] = (
    annual["mean_discharge_m3s"] - annual_mean
) / annual_std

annual["low_flow_ratio"] = (
    annual["low_flow_days"]
    / annual["days"]
)

annual["q95_high_flow_ratio"] = (
    annual["high_flow_q95_days"]
    / annual["days"]
)

annual["q99_high_flow_ratio"] = (
    annual["high_flow_q99_days"]
    / annual["days"]
)

# =========================================================
# MONTHLY SEASONALITY PROFILE
# =========================================================

mg = df.groupby("month")

seasonal = pd.DataFrame({
    "days":
        mg["date"].count(),

    "mean_discharge_m3s":
        mg["discharge_mean_m3s"].mean(),

    "median_discharge_m3s":
        mg["discharge_mean_m3s"].median(),

    "std_discharge_m3s":
        mg["discharge_mean_m3s"].std(),

    "min_discharge_m3s":
        mg["discharge_mean_m3s"].min(),

    "max_discharge_m3s":
        mg["discharge_mean_m3s"].max(),

    "q05_discharge_m3s":
        mg["discharge_mean_m3s"].quantile(0.05),

    "q95_discharge_m3s":
        mg["discharge_mean_m3s"].quantile(0.95),

    "q99_discharge_m3s":
        mg["discharge_mean_m3s"].quantile(0.99),

    "low_flow_days":
        mg["low_flow_q05"].sum(),

    "high_flow_q95_days":
        mg["high_flow_q95"].sum(),

    "high_flow_q99_days":
        mg["high_flow_q99"].sum(),

    "max_low_flow_run_days":
        mg["low_flow_run_days"].max(),

    "max_q95_run_days":
        mg["high_flow_q95_run_days"].max(),

    "max_q99_run_days":
        mg["high_flow_q99_run_days"].max()
}).reset_index()

seasonal["month_name"] = pd.to_datetime(
    seasonal["month"],
    format="%m"
).dt.strftime("%b")

# Fraction of observations in each calendar month
seasonal["low_flow_fraction"] = (
    seasonal["low_flow_days"]
    / seasonal["days"]
)

seasonal["q95_high_flow_fraction"] = (
    seasonal["high_flow_q95_days"]
    / seasonal["days"]
)

seasonal["q99_high_flow_fraction"] = (
    seasonal["high_flow_q99_days"]
    / seasonal["days"]
)

# =========================================================
# MONTHLY RANKS
# =========================================================

seasonal["mean_discharge_rank"] = (
    seasonal["mean_discharge_m3s"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

seasonal["low_flow_risk_rank"] = (
    seasonal["low_flow_fraction"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

seasonal["high_flow_rank"] = (
    seasonal["q95_high_flow_fraction"]
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
        f"Expected 8 annual rows, got {len(annual)}"
    )

if len(seasonal) != 12:
    raise RuntimeError(
        f"Expected 12 calendar months, got {len(seasonal)}"
    )

if annual["year"].duplicated().sum() != 0:
    raise RuntimeError("Duplicate annual years detected")

if seasonal["month"].duplicated().sum() != 0:
    raise RuntimeError("Duplicate calendar months detected")

if annual["days"].sum() != 2769:
    raise RuntimeError(
        "Annual day totals do not equal 2769"
    )

if seasonal["days"].sum() != 2769:
    raise RuntimeError(
        "Seasonal day totals do not equal 2769"
    )

if annual["mean_discharge_m3s"].isna().sum() != 0:
    raise RuntimeError(
        "NaN annual mean detected"
    )

if seasonal["mean_discharge_m3s"].isna().sum() != 0:
    raise RuntimeError(
        "NaN seasonal mean detected"
    )

# =========================================================
# SAVE
# =========================================================

annual.to_parquet(
    ANNUAL_OUT,
    index=False
)

seasonal.to_parquet(
    SEASONAL_OUT,
    index=False
)

metadata = {
    "dataset":
        "GloFAS Historical V5 Johor annual and seasonality profile",

    "source":
        str(DAILY),

    "period_start":
        "2019-01-01",

    "period_end":
        "2026-07-31",

    "annual_years":
        int(len(annual)),

    "calendar_months":
        int(len(seasonal)),

    "global_thresholds_m3s": {
        "q05": float(q05),
        "q95": float(q95),
        "q99": float(q99)
    },

    "annual_output":
        str(ANNUAL_OUT),

    "seasonality_output":
        str(SEASONAL_OUT),

    "interpretation_note":
        "Seasonality profiles describe temporal discharge patterns and extreme-flow frequency. They do not by themselves represent water availability, water withdrawal capacity, or data-center water demand.",

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

print()
print("==============================================")
print("ANNUAL & SEASONALITY PROFILE COMPLETE")
print("==============================================")
print("Annual years          :", len(annual))
print("Calendar months       :", len(seasonal))
print("Period                :", df["date"].min(), "to", df["date"].max())

print()
print("Annual mean range     :",
      annual["mean_discharge_m3s"].min(),
      "-",
      annual["mean_discharge_m3s"].max())

print("Highest-flow year     :",
      int(
          annual.loc[
              annual["mean_discharge_m3s"].idxmax(),
              "year"
          ]
      ))

print("Lowest-flow year      :",
      int(
          annual.loc[
              annual["mean_discharge_m3s"].idxmin(),
              "year"
          ]
      ))

print()
print("Wettest calendar month:",
      seasonal.loc[
          seasonal["mean_discharge_m3s"].idxmax(),
          "month_name"
      ])

print("Lowest calendar month :",
      seasonal.loc[
          seasonal["mean_discharge_m3s"].idxmin(),
          "month_name"
      ])

print()
print("Annual output         :", ANNUAL_OUT)
print("Seasonality output    :", SEASONAL_OUT)
print("Metadata              :", META)
print("QC                    : PASS")
