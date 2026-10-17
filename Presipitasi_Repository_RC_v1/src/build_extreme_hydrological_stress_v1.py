from pathlib import Path
import pandas as pd
import numpy as np
import json

BASE = Path(r"D:\Scopus\AI_Scopus")

DAILY = (
    BASE / "04_DATA" / "GloFAS_Historical_v5_Johor"
    / "DERIVED" / "daily_hydrological_indicators.parquet"
)

OUTDIR = (
    BASE / "04_DATA" / "GloFAS_Historical_v5_Johor"
    / "DERIVED"
)

OUT = OUTDIR / "extreme_hydrological_stress_daily.parquet"
ANNUAL_OUT = OUTDIR / "extreme_hydrological_stress_annual.parquet"
META = OUTDIR / "extreme_hydrological_stress_metadata.json"

print("Reading daily indicators...")

df = pd.read_parquet(DAILY)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

required = {
    "date",
    "discharge_mean_m3s",
    "discharge_q05_m3s",
    "discharge_q95_m3s",
    "discharge_q99_m3s",
    "discharge_max_m3s"
}

missing = required - set(df.columns)
if missing:
    raise RuntimeError(f"Missing columns: {sorted(missing)}")

print("Rows :", len(df))
print("First:", df["date"].min())
print("Last :", df["date"].max())

# ---------------------------------------------------------
# GLOBAL THRESHOLDS
# ---------------------------------------------------------

q05 = df["discharge_mean_m3s"].quantile(0.05)
q95 = df["discharge_mean_m3s"].quantile(0.95)
q99 = df["discharge_mean_m3s"].quantile(0.99)

print()
print("Global thresholds")
print("Q05 :", q05)
print("Q95 :", q95)
print("Q99 :", q99)

# ---------------------------------------------------------
# DAILY EXTREME FLAGS
# ---------------------------------------------------------

df["low_flow_q05"] = df["discharge_mean_m3s"] <= q05
df["high_flow_q95"] = df["discharge_mean_m3s"] >= q95
df["high_flow_q99"] = df["discharge_mean_m3s"] >= q99

# ---------------------------------------------------------
# CONSECUTIVE-RUN FUNCTION
# ---------------------------------------------------------

def consecutive_run_lengths(flag):
    values = flag.to_numpy(dtype=bool)
    result = np.zeros(len(values), dtype=np.int32)

    run = 0

    for i, value in enumerate(values):
        if value:
            run += 1
            result[i] = run
        else:
            run = 0

    return result

df["low_flow_run_days"] = consecutive_run_lengths(
    df["low_flow_q05"]
)

df["high_flow_q95_run_days"] = consecutive_run_lengths(
    df["high_flow_q95"]
)

df["high_flow_q99_run_days"] = consecutive_run_lengths(
    df["high_flow_q99"]
)

# ---------------------------------------------------------
# DAILY STRESS INDICATOR
# ---------------------------------------------------------
#
# This is a normalized descriptive indicator only.
# It is NOT yet a water-shortage index and must not be
# interpreted as data-center water availability.
#
# Low-flow stress:
#   0 when >= Q05
#   approaches 1 as discharge approaches zero.
#
# ---------------------------------------------------------

df["low_flow_stress"] = np.clip(
    1.0 - (df["discharge_mean_m3s"] / q05),
    0.0,
    1.0
)

# ---------------------------------------------------------
# PEAK / EVENT IDENTIFICATION
# ---------------------------------------------------------

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["year_month"] = df["date"].dt.to_period("M").astype(str)

# ---------------------------------------------------------
# ANNUAL EXTREME SUMMARY
# ---------------------------------------------------------

g = df.groupby("year")

annual = pd.DataFrame({
    "days": g["date"].count(),

    "mean_discharge_m3s":
        g["discharge_mean_m3s"].mean(),

    "median_discharge_m3s":
        g["discharge_mean_m3s"].median(),

    "annual_min_discharge_m3s":
        g["discharge_mean_m3s"].min(),

    "annual_max_discharge_m3s":
        g["discharge_mean_m3s"].max(),

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

    "max_high_flow_q95_run_days":
        g["high_flow_q95_run_days"].max(),

    "max_high_flow_q99_run_days":
        g["high_flow_q99_run_days"].max(),

    "annual_max_cell_discharge_m3s":
        g["discharge_max_m3s"].max()
}).reset_index()

# ---------------------------------------------------------
# PEAK DATE PER YEAR
# ---------------------------------------------------------

idx = df.groupby("year")["discharge_mean_m3s"].idxmax()

annual_peak = (
    df.loc[
        idx,
        ["year", "date", "discharge_mean_m3s"]
    ]
    .rename(
        columns={
            "date": "peak_date",
            "discharge_mean_m3s": "peak_regional_mean_m3s"
        }
    )
)

annual = annual.merge(
    annual_peak,
    on="year",
    how="left"
)

# ---------------------------------------------------------
# QC
# ---------------------------------------------------------

print()
print("Running QC...")

if len(df) != 2769:
    raise RuntimeError(
        f"Expected 2769 daily rows, got {len(df)}"
    )

if df["date"].duplicated().sum() != 0:
    raise RuntimeError("Duplicate dates detected")

if len(annual) != 8:
    raise RuntimeError(
        f"Expected 8 calendar years, got {len(annual)}"
    )

if annual["days"].min() < 212:
    raise RuntimeError("Unexpected annual day count")

if (
    annual["low_flow_days_q05"].sum()
    <= 0
):
    raise RuntimeError("No low-flow days detected")

if (
    annual["high_flow_days_q95"].sum()
    <= 0
):
    raise RuntimeError("No Q95 high-flow days detected")

if (
    annual["high_flow_days_q99"].sum()
    <= 0
):
    raise RuntimeError("No Q99 high-flow days detected")

if df["low_flow_stress"].min() < 0:
    raise RuntimeError("Invalid low-flow stress below zero")

if df["low_flow_stress"].max() > 1:
    raise RuntimeError("Invalid low-flow stress above one")

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

df.to_parquet(
    OUT,
    index=False
)

annual.to_parquet(
    ANNUAL_OUT,
    index=False
)

metadata = {
    "dataset":
        "GloFAS Historical V5 Johor extreme-flow and hydrological stress indicators",

    "source_daily":
        str(DAILY),

    "period_start":
        "2019-01-01",

    "period_end":
        "2026-07-31",

    "daily_rows":
        int(len(df)),

    "annual_years":
        int(len(annual)),

    "global_thresholds_m3s": {
        "q05":
            float(q05),
        "q95":
            float(q95),
        "q99":
            float(q99)
    },

    "definitions": {
        "low_flow_q05":
            "Daily regional mean discharge <= global daily Q05.",

        "high_flow_q95":
            "Daily regional mean discharge >= global daily Q95.",

        "high_flow_q99":
            "Daily regional mean discharge >= global daily Q99.",

        "low_flow_stress":
            "Descriptive normalized low-flow indicator clipped to [0,1]. It is not a water availability or data-center water shortage index."
    },

    "outputs": {
        "daily": str(OUT),
        "annual": str(ANNUAL_OUT)
    },

    "qc_status":
        "PASS",

    "source_modified":
        False
}

META.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

print()
print("==============================================")
print("EXTREME HYDROLOGICAL STRESS COMPLETE")
print("==============================================")
print("Daily rows             :", len(df))
print("Annual years            :", len(annual))
print("Global Q05              :", q05)
print("Global Q95              :", q95)
print("Global Q99              :", q99)
print(
    "Low-flow days Q05       :",
    int(df["low_flow_q05"].sum())
)
print(
    "High-flow days Q95      :",
    int(df["high_flow_q95"].sum())
)
print(
    "High-flow days Q99      :",
    int(df["high_flow_q99"].sum())
)
print(
    "Maximum low-flow run   :",
    int(df["low_flow_run_days"].max())
)
print(
    "Maximum Q95 high run   :",
    int(df["high_flow_q95_run_days"].max())
)
print(
    "Maximum Q99 high run   :",
    int(df["high_flow_q99_run_days"].max())
)
print("Daily output            :", OUT)
print("Annual output           :", ANNUAL_OUT)
print("Metadata                :", META)
print("QC                      : PASS")
