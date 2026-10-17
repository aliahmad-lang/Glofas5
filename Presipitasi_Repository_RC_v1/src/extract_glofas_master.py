from __future__ import annotations

import csv
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "01_RAW" / "GloFAS_HISTORICAL_V5"
OUT_DIR = ROOT / "04_DATA" / "GloFAS_Historical_v5_Johor"
MASTER_DIR = OUT_DIR / "MASTER"
EXTRACT_DIR = OUT_DIR / "EXTRACTION"

MASTER_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PARQUET = MASTER_DIR / "glofas_johor_daily.parquet"
GRID_CSV = MASTER_DIR / "spatial_grid.csv"
META_JSON = MASTER_DIR / "glofas_johor_daily_metadata.json"
MANIFEST_CSV = EXTRACT_DIR / "extraction_manifest.csv"

MONTHS = []
y, m = 2019, 1
while (y, m) <= (2026, 7):
    MONTHS.append(f"{y:04d}-{m:02d}")
    m += 1
    if m == 13:
        y += 1
        m = 1

def expected_days(ym):
    y, m = map(int, ym.split("-"))
    return __import__("calendar").monthrange(y, m)[1]

def extract_grib(zip_path, out_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        if "data.grib" not in z.namelist():
            raise RuntimeError(f"data.grib not found: {zip_path.name}")
        with z.open("data.grib") as src, out_path.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)

def main():
    # Establish the canonical spatial mask from the first month.
    first_ym = MONTHS[0]
    first_zip = RAW_DIR / f"GloFAS_Johor_Historical_v5_{first_ym[:4]}_{first_ym[5:7]}.zip"

    with tempfile.TemporaryDirectory(prefix="glofas_master_") as tmp:
        tmp = Path(tmp)
        first_grib = tmp / "first.grib"
        extract_grib(first_zip, first_grib)

        ds0 = xr.open_dataset(first_grib, engine="cfgrib")
        try:
            v0 = ds0["avg_dis"]
            mask = v0.isel(time=0).notnull().values
            lat = ds0["latitude"].values
            lon = ds0["longitude"].values
        finally:
            ds0.close()

        iy, ix = np.where(mask)
        grid = pd.DataFrame({
            "river_cell_id": [f"R{n:04d}" for n in range(1, len(iy) + 1)],
            "latitude": lat[iy].astype(float),
            "longitude": lon[ix].astype(float),
        })
        grid.to_csv(GRID_CSV, index=False, encoding="utf-8-sig")

        cell_ids = np.full(mask.shape, "", dtype=object)
        for n, (a, b) in enumerate(zip(iy, ix), start=1):
            cell_ids[a, b] = f"R{n:04d}"

        rows = []
        manifest = []

        for idx, ym in enumerate(MONTHS, 1):
            print(f"[{idx:02d}/{len(MONTHS)}] {ym}", flush=True)
            zip_path = RAW_DIR / f"GloFAS_Johor_Historical_v5_{ym[:4]}_{ym[5:7]}.zip"
            grib_path = tmp / f"{ym.replace('-', '_')}.grib"

            rec = {
                "month": ym,
                "status": "ERROR",
                "expected_days": expected_days(ym),
                "actual_days": "",
                "valid_cells": len(grid),
                "rows_written": "",
                "error": "",
            }

            try:
                extract_grib(zip_path, grib_path)
                ds = xr.open_dataset(grib_path, engine="cfgrib")
                try:
                    v = ds["avg_dis"]
                    if int(ds.sizes["time"]) != expected_days(ym):
                        raise RuntimeError(
                            f"Expected {expected_days(ym)} days, got {ds.sizes['time']}"
                        )

                    current_mask = v.isel(time=0).notnull().values
                    if not np.array_equal(current_mask, mask):
                        raise RuntimeError("Spatial mask differs from canonical mask")

                    values = v.values[:, mask].astype(np.float32)
                    times = pd.to_datetime(ds["time"].values)

                    n_days = values.shape[0]
                    n_cells = values.shape[1]

                    # Long-format daily observations.
                    frame = pd.DataFrame(
                        values,
                        index=times,
                        columns=[f"R{n:04d}" for n in range(1, n_cells + 1)],
                    )
                    long = (
                        frame.stack()
                        .rename("discharge_m3s")
                        .reset_index()
                        .rename(columns={"level_0": "date", "level_1": "river_cell_id"})
                    )
                    long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")
                    rows.append(long)

                    rec.update({
                        "status": "PASS",
                        "actual_days": n_days,
                        "rows_written": int(long.shape[0]),
                    })
                    print(
                        f"    PASS: {n_days} days × {n_cells} cells = {long.shape[0]} rows",
                        flush=True,
                    )
                finally:
                    ds.close()

            except Exception as exc:
                rec["error"] = repr(exc)
                print(f"    ERROR: {exc}", flush=True)

            manifest.append(rec)

        # Combine all monthly chunks only after all months have been processed.
        if any(r["status"] != "PASS" for r in manifest):
            raise RuntimeError("One or more months failed. Master dataset not written.")

        master = pd.concat(rows, ignore_index=True)
        master["date"] = pd.to_datetime(master["date"])
        master["river_cell_id"] = master["river_cell_id"].astype("string")
        master["discharge_m3s"] = master["discharge_m3s"].astype("float32")

        # Attach coordinates.
        master = master.merge(grid, on="river_cell_id", how="left", validate="many_to_one")
        master = master[
            ["date", "river_cell_id", "latitude", "longitude", "discharge_m3s"]
        ].sort_values(["date", "river_cell_id"], ignore_index=True)

        master.to_parquet(MASTER_PARQUET, index=False)

        with MANIFEST_CSV.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
            writer.writeheader()
            writer.writerows(manifest)

        metadata = {
            "dataset": "GloFAS Historical V5",
            "product_type": "consolidated",
            "hydrological_model": "LISFLOOD",
            "variable": "avg_dis",
            "long_name": "Time-mean discharge from rivers or streams",
            "units": "m3 s-1",
            "period_start": "2019-01-01",
            "period_end": "2026-07-31",
            "months": len(MONTHS),
            "days": int(master["date"].nunique()),
            "river_cells": int(grid.shape[0]),
            "observations": int(master.shape[0]),
            "latitude_min": float(grid.latitude.min()),
            "latitude_max": float(grid.latitude.max()),
            "longitude_min": float(grid.longitude.min()),
            "longitude_max": float(grid.longitude.max()),
            "grid_resolution": "0.05 degree",
            "source_raw": str(RAW_DIR),
            "master_parquet": str(MASTER_PARQUET),
            "spatial_grid": str(GRID_CSV),
            "extraction_manifest": str(MANIFEST_CSV),
        }

        META_JSON.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print()
        print("MASTER EXTRACTION COMPLETE")
        print("Rows       :", len(master))
        print("Days       :", master["date"].nunique())
        print("Cells      :", grid.shape[0])
        print("Parquet    :", MASTER_PARQUET)
        print("Grid CSV   :", GRID_CSV)
        print("Manifest   :", MANIFEST_CSV)
        print("Metadata   :", META_JSON)

if __name__ == "__main__":
    main()
