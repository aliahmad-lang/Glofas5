from __future__ import annotations

import csv
import hashlib
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "01_RAW" / "GloFAS_HISTORICAL_V5"
OUT_DIR = ROOT / "04_DATA" / "GloFAS_Historical_v5_Johor" / "QC"
OUT_CSV = OUT_DIR / "spatial_mask_qc_2019-01_2026-07.csv"

START = "2019-01"
END = "2026-07"

def month_list():
    y, m = 2019, 1
    out = []
    while (y, m) <= (2026, 7):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y += 1
            m = 1
    return out

def expected_days(ym):
    y, m = map(int, ym.split("-"))
    if m == 2:
        return 29 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 28
    return [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def inspect_grib(grib_path, ym):
    ds = xr.open_dataset(grib_path, engine="cfgrib")
    try:
        if "avg_dis" not in ds.data_vars:
            raise RuntimeError(f"{ym}: avg_dis variable not found")

        v = ds["avg_dis"]

        if v.ndim != 3:
            raise RuntimeError(f"{ym}: unexpected dimensions {v.dims}")

        if "time" not in v.dims or "latitude" not in v.dims or "longitude" not in v.dims:
            raise RuntimeError(f"{ym}: expected time/latitude/longitude dimensions")

        days = int(ds.sizes["time"])
        lat_n = int(ds.sizes["latitude"])
        lon_n = int(ds.sizes["longitude"])
        total_cells = lat_n * lon_n

        valid_per_day = v.notnull().sum(dim=["latitude", "longitude"]).values.astype(int)
        unique_valid = sorted(set(valid_per_day.tolist()))

        mask_first = v.isel(time=0).notnull().values
        mask_consistent = True

        # Compare each day with the first day's spatial mask.
        for i in range(1, days):
            if not np.array_equal(mask_first, v.isel(time=i).notnull().values):
                mask_consistent = False
                break

        values = v.values
        finite = values[np.isfinite(values)]

        attrs = dict(v.attrs)
        units = attrs.get("units", "")
        long_name = attrs.get("long_name", "")

        expected = expected_days(ym)

        return {
            "month": ym,
            "status": "PASS" if days == expected and mask_consistent else "CHECK",
            "expected_days": expected,
            "actual_days": days,
            "latitude_count": lat_n,
            "longitude_count": lon_n,
            "total_grid_cells": total_cells,
            "valid_cells_min": int(valid_per_day.min()),
            "valid_cells_max": int(valid_per_day.max()),
            "unique_valid_cell_counts": len(unique_valid),
            "spatial_mask_consistent": mask_consistent,
            "nan_count": int(np.isnan(values).sum()),
            "total_values": int(values.size),
            "finite_count": int(np.isfinite(values).sum()),
            "min_discharge_m3s": float(finite.min()) if finite.size else "",
            "max_discharge_m3s": float(finite.max()) if finite.size else "",
            "mean_discharge_m3s": float(finite.mean()) if finite.size else "",
            "units": units,
            "long_name": long_name,
            "lat_min": float(ds.latitude.min()),
            "lat_max": float(ds.latitude.max()),
            "lon_min": float(ds.longitude.min()),
            "lon_max": float(ds.longitude.max()),
            "first_time": str(ds.time.values[0]),
            "last_time": str(ds.time.values[-1]),
            "grib_sha256": sha256_file(grib_path),
            "error": "",
        }
    finally:
        ds.close()

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    months = month_list()
    rows = []

    with tempfile.TemporaryDirectory(prefix="glofas_qc_") as tmp:
        tmpdir = Path(tmp)

        for idx, ym in enumerate(months, 1):
            zip_path = RAW_DIR / f"GloFAS_Johor_Historical_v5_{ym[:4]}_{ym[5:7]}.zip"
            print(f"[{idx:02d}/91] {ym}", flush=True)

            row = {
                "month": ym,
                "status": "ERROR",
                "expected_days": expected_days(ym),
                "actual_days": "",
                "latitude_count": "",
                "longitude_count": "",
                "total_grid_cells": "",
                "valid_cells_min": "",
                "valid_cells_max": "",
                "unique_valid_cell_counts": "",
                "spatial_mask_consistent": "",
                "nan_count": "",
                "total_values": "",
                "finite_count": "",
                "min_discharge_m3s": "",
                "max_discharge_m3s": "",
                "mean_discharge_m3s": "",
                "units": "",
                "long_name": "",
                "lat_min": "",
                "lat_max": "",
                "lon_min": "",
                "lon_max": "",
                "first_time": "",
                "last_time": "",
                "grib_sha256": "",
                "error": "",
            }

            try:
                if not zip_path.exists():
                    raise FileNotFoundError(f"Missing ZIP: {zip_path}")

                with zipfile.ZipFile(zip_path, "r") as z:
                    names = z.namelist()
                    if "data.grib" not in names:
                        raise RuntimeError("data.grib not found in ZIP")
                    grib_path = tmpdir / f"{ym.replace('-', '_')}.grib"
                    with z.open("data.grib") as src, grib_path.open("wb") as dst:
                        while True:
                            chunk = src.read(1024 * 1024)
                            if not chunk:
                                break
                            dst.write(chunk)

                row.update(inspect_grib(grib_path, ym))
                print(
                    f"    {row['status']}: days={row['actual_days']}, "
                    f"valid={row['valid_cells_min']}-{row['valid_cells_max']}, "
                    f"mask_consistent={row['spatial_mask_consistent']}",
                    flush=True,
                )

            except Exception as exc:
                row["error"] = repr(exc)
                print(f"    ERROR: {exc}", flush=True)

            rows.append(row)

    fields = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(r["status"] == "PASS" for r in rows)
    print()
    print("QC COMPLETE")
    print(f"PASS : {passed}/91")
    print(f"CSV  : {OUT_CSV}")

if __name__ == "__main__":
    main()
