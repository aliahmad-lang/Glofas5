import pandas as pd
import numpy as np
import json
from pathlib import Path

# ============================================================
# DATA CENTER WATER DEMAND SCENARIO MODEL v1
# ============================================================
#
# Purpose:
#   Create a transparent engineering-scenario estimate of
#   data-center operational water demand.
#
# IMPORTANT:
#   This is a SCENARIO MODEL, not observed Johor data.
#   Scenario assumptions must be replaced by documented
#   project/operator values when available.
#
# Hydrological data are NOT converted into water availability
# volume in this script.
#
# LFSI is retained separately and may be coupled later.
#
# Core relationship:
#
#   IT Power (MW)
#       -> Annual IT energy (MWh)
#       -> Water consumption via WUE (L/kWh)
#
#   Annual water consumption (m3)
#       = IT load (MW)
#       * operating hours
#       * 1000 kWh/MWh
#       * WUE L/kWh
#       / 1000 L/m3
#
# Therefore:
#
#   annual_water_m3
#       = IT_MW * hours * WUE
#
# because 1 MW * 1 h = 1 MWh = 1000 kWh,
# and 1000 L = 1 m3.
#
# PUE is retained to estimate total facility electricity,
# but WUE is applied to IT energy in this v1 formulation.
#
# This script does NOT claim:
#   - actual operator water consumption
#   - actual Johor water availability
#   - regulatory allocation
#   - water shortage
#   - causal impact of data centers on hydrology
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

OUT_FILE = (
    DERIVED
    / "data_center_water_demand_scenarios_v1.parquet"
)

META_FILE = (
    DERIVED
    / "data_center_water_demand_scenarios_v1_metadata.json"
)


# ============================================================
# SCENARIO DEFINITIONS
# ============================================================
#
# These are methodological scenario assumptions, NOT observed
# operator values.
#
# IT load:
#   50 MW, 100 MW, 200 MW, 500 MW
#
# WUE:
#   0.20, 0.30, 0.50, 1.00 L/kWh
#
# PUE:
#   1.20, 1.30, 1.40
#
# Availability:
#   8760 operating hours/year
#
# The full factorial scenario table is produced.
# ============================================================

IT_LOAD_MW = [
    50,
    100,
    200,
    500
]

WUE_L_PER_KWH = [
    0.20,
    0.30,
    0.50,
    1.00
]

PUE = [
    1.20,
    1.30,
    1.40
]

OPERATING_HOURS = 8760


# ============================================================
# VALIDATION
# ============================================================

if any(x <= 0 for x in IT_LOAD_MW):
    raise RuntimeError("IT load must be positive.")

if any(x <= 0 for x in WUE_L_PER_KWH):
    raise RuntimeError("WUE must be positive.")

if any(x < 1 for x in PUE):
    raise RuntimeError("PUE must be >= 1.")

if OPERATING_HOURS <= 0 or OPERATING_HOURS > 8784:
    raise RuntimeError("Invalid operating hours.")


# ============================================================
# SCENARIO GENERATION
# ============================================================

rows = []

for it_mw in IT_LOAD_MW:

    for wue in WUE_L_PER_KWH:

        for pue in PUE:

            annual_it_mwh = (
                it_mw
                * OPERATING_HOURS
            )

            annual_it_kwh = (
                annual_it_mwh
                * 1000
            )

            annual_facility_mwh = (
                annual_it_mwh
                * pue
            )

            annual_facility_kwh = (
                annual_facility_mwh
                * 1000
            )

            # WUE is L/kWh of IT energy.
            annual_water_litres = (
                annual_it_kwh
                * wue
            )

            annual_water_m3 = (
                annual_water_litres
                / 1000
            )

            monthly_water_m3 = (
                annual_water_m3
                / 12
            )

            daily_water_m3 = (
                annual_water_m3
                / 365
            )

            hourly_water_m3 = (
                annual_water_m3
                / OPERATING_HOURS
            )

            rows.append({

                "it_load_mw":
                    it_mw,

                "wue_l_per_kwh":
                    wue,

                "pue":
                    pue,

                "operating_hours_year":
                    OPERATING_HOURS,

                "annual_it_mwh":
                    annual_it_mwh,

                "annual_facility_mwh":
                    annual_facility_mwh,

                "annual_water_m3":
                    annual_water_m3,

                "monthly_water_m3":
                    monthly_water_m3,

                "daily_water_m3":
                    daily_water_m3,

                "hourly_water_m3":
                    hourly_water_m3

            })


scenarios = pd.DataFrame(rows)


# ============================================================
# SCENARIO ID
# ============================================================

scenarios.insert(
    0,
    "scenario_id",
    [
        f"S{i:03d}"
        for i in range(
            1,
            len(scenarios) + 1
        )
    ]
)


# ============================================================
# LOAD HYDROLOGICAL CONTEXT
# ============================================================
#
# This does NOT combine water demand with LFSI numerically.
# It simply attaches the observed LFSI study context so that
# the scenario table can be used later in a coupling analysis.
# ============================================================

if not LFSI_FILE.exists():

    raise FileNotFoundError(
        f"LFSI file not found: {LFSI_FILE}"
    )

lfsi = pd.read_parquet(
    LFSI_FILE
)

required_lfsi = [
    "year",
    "LFSI",
    "LFSI_rank",
    "stress_category"
]

missing_lfsi = [
    c
    for c in required_lfsi
    if c not in lfsi.columns
]

if missing_lfsi:

    raise RuntimeError(
        "Missing LFSI columns: "
        f"{missing_lfsi}"
    )


# ============================================================
# REPRESENTATIVE HYDROLOGICAL CONTEXT
# ============================================================
#
# For this v1 scenario model, 2026 is retained as the
# stress-reference year because it was rank #1 in all
# LFSI robustness formulations.
#
# This is NOT a conversion from LFSI into water volume.
# ============================================================

lfsi_2026 = lfsi[
    lfsi["year"] == 2026
].iloc[0]

hydro_context = {

    "hydrological_reference_year":
        2026,

    "LFSI_2026":
        float(lfsi_2026["LFSI"]),

    "LFSI_rank_2026":
        int(lfsi_2026["LFSI_rank"]),

    "stress_category_2026":
        str(lfsi_2026["stress_category"])

}

for key, value in hydro_context.items():

    scenarios[key] = value


# ============================================================
# SCENARIO NORMALIZATION
# ============================================================
#
# Relative water demand index is purely a scenario comparison
# tool. It is normalized to the maximum scenario demand.
# ============================================================

max_water = float(
    scenarios["annual_water_m3"].max()
)

scenarios["relative_water_demand_0_1"] = (
    scenarios["annual_water_m3"]
    / max_water
)


# ============================================================
# PUE CHECK
# ============================================================
#
# Water demand under the current WUE formulation does not
# change when PUE changes because WUE is defined against IT
# energy in this model.
#
# PUE affects facility energy and is therefore retained as an
# engineering context variable.
# ============================================================

water_by_it_wue = (
    scenarios
    .groupby(
        [
            "it_load_mw",
            "wue_l_per_kwh"
        ]
    )["annual_water_m3"]
    .nunique()
)

if not (water_by_it_wue == 1).all():

    raise RuntimeError(
        "Unexpected water-demand dependence on PUE."
    )


# ============================================================
# QC
# ============================================================

if len(scenarios) != (
    len(IT_LOAD_MW)
    * len(WUE_L_PER_KWH)
    * len(PUE)
):

    raise RuntimeError(
        "Unexpected number of scenario rows."
    )

if scenarios["scenario_id"].duplicated().any():

    raise RuntimeError(
        "Duplicate scenario ID."
    )

if scenarios["annual_water_m3"].isna().any():

    raise RuntimeError(
        "NaN annual water demand."
    )

if not (
    scenarios["annual_water_m3"] > 0
).all():

    raise RuntimeError(
        "Non-positive annual water demand."
    )

if not (
    scenarios["relative_water_demand_0_1"]
    .between(0, 1)
).all():

    raise RuntimeError(
        "Relative demand index outside [0,1]."
    )

if hydro_context["LFSI_rank_2026"] != 1:

    raise RuntimeError(
        "2026 LFSI rank is not #1."
    )


# ============================================================
# SAVE
# ============================================================

scenarios.to_parquet(
    OUT_FILE,
    index=False
)


metadata = {

    "method":
        "Data Center Water Demand Scenario Model v1",

    "purpose":
        "Transparent engineering scenario model for "
        "data-center operational water consumption.",

    "status":
        "Scenario analysis; not observed operator data.",

    "formula":
        "annual_water_m3 = IT_MW * operating_hours * WUE_L_per_kWh",

    "pue_role":
        "PUE estimates total facility electricity from IT energy. "
        "In this v1 formulation, WUE is applied to IT energy, "
        "so PUE does not directly alter water consumption.",

    "it_load_mw":
        IT_LOAD_MW,

    "wue_l_per_kwh":
        WUE_L_PER_KWH,

    "pue":
        PUE,

    "operating_hours":
        OPERATING_HOURS,

    "hydrological_context":
        hydro_context,

    "interpretation_limit":
        "Scenario water demand is not actual water withdrawal, "
        "actual consumption, water availability, regulatory "
        "allocation, or causal impact on Johor hydrology.",

    "qc": {

        "scenario_rows":
            int(len(scenarios)),

        "expected_rows":
            int(
                len(IT_LOAD_MW)
                * len(WUE_L_PER_KWH)
                * len(PUE)
            ),

        "annual_water_positive":
            True,

        "relative_index_range":
            "[0,1]",

        "LFSI_2026_rank":
            hydro_context["LFSI_rank_2026"],

        "status":
            "PASS"

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
print("DATA CENTER WATER DEMAND SCENARIO MODEL v1 COMPLETE")
print("=" * 72)

print()
print(
    "Scenario rows:",
    len(scenarios)
)

print(
    "IT load range:",
    min(IT_LOAD_MW),
    "-",
    max(IT_LOAD_MW),
    "MW"
)

print(
    "WUE range:",
    min(WUE_L_PER_KWH),
    "-",
    max(WUE_L_PER_KWH),
    "L/kWh"
)

print(
    "PUE range:",
    min(PUE),
    "-",
    max(PUE)
)

print()
print("2026 hydrological context:")
print(
    "  LFSI          :",
    hydro_context["LFSI_2026"]
)
print(
    "  LFSI rank     :",
    hydro_context["LFSI_rank_2026"]
)
print(
    "  Stress class  :",
    hydro_context["stress_category_2026"]
)

print()
print("SELECTED DEMAND SCENARIOS")
print("-" * 72)

display_columns = [
    "scenario_id",
    "it_load_mw",
    "wue_l_per_kwh",
    "pue",
    "annual_water_m3",
    "monthly_water_m3",
    "daily_water_m3"
]

print(
    scenarios[
        display_columns
    ].head(12).to_string(
        index=False
    )
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
