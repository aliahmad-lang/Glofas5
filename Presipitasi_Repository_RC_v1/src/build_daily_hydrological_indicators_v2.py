from pathlib import Path
import pandas as pd
import numpy as np
import json

BASE = Path(r"D:\Scopus\AI_Scopus")
MASTER = BASE / "04_DATA" / "GloFAS_Historical_v5_Johor" / "MASTER" / "glofas_johor_daily.parquet"
OUTDIR = BASE / "04_DATA" / "GloFAS_Historical_v5_Johor" / "DERIVED"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / "daily_hydrological_indicators.parquet"
META = OUTDIR / "daily_hydrological_indicators_metadata.json"

df = pd.read_parquet(MASTER)
required = {"date", "river_cell_id", "latitude", "longitude", "discharge_m3s"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df["date"] = pd.to_datetime(df["date"])

# Use GroupBy.agg on the full DataFrame.
# This syntax is compatible with current pandas named aggregation.
g = df.groupby("date")["discharge_m3s"]

daily = g.agg(
    river_cells="count",
    discharge_mean_m3s="mean",
    discharge_median_m3s="median",
    discharge_min_m3s="min",
    discharge_max_m3s="max",
    discharge_std_m3s="std",
    discharge_q05_m3s=lambda x: x.quantile(0.05),
    discharge_q25_m3s=lambda x: x.quantile(0.25),
    discharge_q75_m3s=lambda x: x.quantile(0.75),
    discharge_q95_m3s=lambda x: x.quantile(0.95),
    discharge_q99_m3s=lambda x: x.quantile(0.99),
).reset_index()

daily["cv"] = np.where(
    daily["discharge_mean_m3s"] != 0,
    daily["discharge_std_m3s"] / daily["discharge_mean_m3s"],
    np.nan
)

daily["year"] = daily["date"].dt.year
daily["month"] = daily["date"].dt.month
daily["day"] = daily["date"].dt.day
daily["doy"] = daily["date"].dt.dayofyear
daily["month_name"] = daily["date"].dt.strftime("%b")

expected_days = 2769
expected_cells = 969

checks = {
    "rows": int(len(daily)),
    "unique_dates": int(daily["date"].nunique()),
    "min_daily_cell_count": int(daily["river_cells"].min()),
    "max_daily_cell_count": int(daily["river_cells"].max()),
    "duplicate_dates": int(daily["date"].duplicated().sum()),
    "nan_mean": int(daily["discharge_mean_m3s"].isna().sum()),
    "negative_mean": int((daily["discharge_mean_m3s"] < 0).sum()),
    "date_first": str(daily["date"].min().date()),
    "date_last": str(daily["date"].max().date()),
}

if checks["rows"] != expected_days:
    raise RuntimeError(f"Expected {expected_days} daily rows, got {checks['rows']}")
if checks["unique_dates"] != expected_days:
    raise RuntimeError("Daily date count mismatch")
if checks["min_daily_cell_count"] != expected_cells or checks["max_daily_cell_count"] != expected_cells:
    raise RuntimeError("Not all days contain exactly 969 river cells")
if checks["duplicate_dates"] != 0:
    raise RuntimeError("Duplicate daily dates detected")
if checks["nan_mean"] != 0:
    raise RuntimeError("NaN daily mean detected")
if checks["negative_mean"] != 0:
    raise RuntimeError("Negative daily mean detected")

daily.to_parquet(OUT, index=False)

metadata = {
    "dataset": "GloFAS Historical V5 Johor derived daily hydrological indicators",
    "source_master": str(MASTER),
    "period": {"start": "2019-01-01", "end": "2026-07-31"},
    "source_cells": 969,
    "source_days": 2769,
    "aggregation": "Cross-river-cell daily descriptive statistics across 969 valid GloFAS river cells",
    "columns": list(daily.columns),
    "qc": checks,
    "raw_data_modified": False,
    "master_modified": False,
}

META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print("DAILY HYDROLOGICAL INDICATORS COMPLETE")
print("Rows               :", len(daily))
print("River cells/day    :", daily["river_cells"].min(), "-", daily["river_cells"].max())
print("Date first         :", daily["date"].min())
print("Date last          :", daily["date"].max())
print("Mean daily regional:", daily["discharge_mean_m3s"].mean())
print("Max daily regional :", daily["discharge_mean_m3s"].max())
print("Max cell discharge :", daily["discharge_max_m3s"].max())
print("Parquet            :", OUT)
print("Metadata           :", META)
print("QC                 : PASS")
