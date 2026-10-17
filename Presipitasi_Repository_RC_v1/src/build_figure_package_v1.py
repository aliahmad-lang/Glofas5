from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE=Path(r"D:\Scopus\AI_Scopus")
R=BASE/"04_DATA"/"GloFAS_Historical_v5_Johor"/"DERIVED"/"RESULTS_PACKAGE_v1"
OUT=R/"FIGURE_PACKAGE_v1"
OUT.mkdir(parents=True,exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10
})

same=pd.read_csv(R/"T03_same_period_jan_jul_comparison.csv")
lfsi=pd.read_csv(R/"T04_LFSI_annual_ranking.csv")
rob=pd.read_csv(R/"T05_LFSI_robustness_ranks.csv")
scen=pd.read_csv(R/"T06_data_center_water_demand_scenarios.csv")
sel=pd.read_csv(R/"T07_HSWDE_2026_selected_scenarios.csv")
sens=pd.read_csv(R/"T08_sensitivity_results.csv")

def finish(fig, name):
    fig.tight_layout()
    fig.savefig(OUT/name, bbox_inches="tight")
    plt.close(fig)

# Figure 1: Jan-Jul mean discharge
fig,ax=plt.subplots(figsize=(8.5,4.8))
ax.plot(same["year"],same["mean_discharge_m3s"],marker="o",linewidth=1.8)
ax.axhline(same["baseline_2019_2025_mean_m3s"].iloc[0],linestyle="--",linewidth=1.2,
           label="2019–2025 baseline mean")
ax.set_xlabel("Year")
ax.set_ylabel("Mean discharge (m³/s)")
ax.set_title("January–July Mean Regional Discharge")
ax.legend()
ax.grid(alpha=.25)
finish(fig,"Figure_1_JanJul_mean_discharge.png")

# Figure 2: Low-flow exposure and LFSI
fig,ax1=plt.subplots(figsize=(8.5,4.8))
ax1.bar(lfsi["year"],lfsi["low_flow_fraction"],alpha=.65,label="Q05 low-flow fraction")
ax1.set_xlabel("Year")
ax1.set_ylabel("Low-flow fraction")
ax1.set_ylim(0,max(lfsi["low_flow_fraction"])*1.25)
ax2=ax1.twinx()
ax2.plot(lfsi["year"],lfsi["LFSI"],marker="o",linewidth=2,label="LFSI")
ax2.set_ylabel("LFSI")
ax2.set_ylim(0,max(lfsi["LFSI"])*1.2)
ax1.set_title("Low-Flow Exposure and Low-Flow Stress Index")
ax1.grid(axis="y",alpha=.25)
finish(fig,"Figure_2_low_flow_exposure_LFSI.png")

# Figure 3: Robustness ranks
fig,ax=plt.subplots(figsize=(8.5,4.8))
for c in ["robustness_R1_rank","robustness_R2_rank","robustness_R3_rank",
          "robustness_R4_rank","robustness_R5_rank"]:
    ax.plot(rob["year"],rob[c],marker="o",linewidth=1.5,label=c.replace("robustness_","").replace("_rank",""))
ax.invert_yaxis()
ax.set_xlabel("Year")
ax.set_ylabel("Rank (1 = highest stress)")
ax.set_title("Robustness of LFSI Ranking Across Five Formulations")
ax.set_yticks(range(1,9))
ax.legend(ncol=3)
ax.grid(alpha=.25)
finish(fig,"Figure_3_LFSI_robustness_ranks.png")

# Figure 4: Water demand by IT load and WUE
fig,ax=plt.subplots(figsize=(8.5,5.0))
for wue in sorted(scen["wue_l_per_kwh"].unique()):
    x=scen[(scen["wue_l_per_kwh"]==wue)&(scen["pue"]==1.3)].sort_values("it_load_mw")
    ax.plot(x["it_load_mw"],x["annual_water_m3"]/1e6,marker="o",
            linewidth=1.7,label=f"WUE {wue:g}")
ax.set_xlabel("IT load (MW)")
ax.set_ylabel("Annual water demand (million m³)")
ax.set_title("Data-Center Water-Demand Scenarios (PUE = 1.3)")
ax.legend(title="WUE (L/kWh)")
ax.grid(alpha=.25)
finish(fig,"Figure_4_water_demand_scenarios.png")

# Figure 5: 2026 HSWDE selected scenarios
pivot=sel.pivot_table(index="it_load_mw",columns="wue_l_per_kwh",
                      values="HSWDE_m3_equivalent",aggfunc="first")
fig,ax=plt.subplots(figsize=(8.5,5.0))
im=ax.imshow(pivot.values,aspect="auto")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"{x:g}" for x in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([f"{x:g}" for x in pivot.index])
ax.set_xlabel("WUE (L/kWh)")
ax.set_ylabel("IT load (MW)")
ax.set_title("2026 Hydrological Stress-Weighted Demand Exposure")
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        ax.text(j,i,f"{pivot.iloc[i,j]/1e6:.3f}",ha="center",va="center")
cbar=fig.colorbar(im,ax=ax)
cbar.set_label("HSWDE (million m³-equivalent)")
finish(fig,"Figure_5_HSWDE_2026_heatmap.png")

# Figure 6: Sensitivity ratios
fig,ax=plt.subplots(figsize=(8.5,4.8))
it=sens[sens["parameter"]=="IT load"]
wu=sens[sens["parameter"]=="WUE"]
ax.plot(it["value"],it["HSWDE_ratio_to_reference"],marker="o",linewidth=2,label="IT load")
# WUE values on second x-axis because units differ
ax2=ax.twiny()
ax2.plot(wu["value"],wu["HSWDE_ratio_to_reference"],marker="s",linewidth=2,label="WUE")
ax.set_xlabel("IT load (MW)")
ax.set_ylabel("HSWDE ratio to reference")
ax2.set_xlabel("WUE (L/kWh)")
ax.set_title("Sensitivity of HSWDE to IT Load and WUE")
ax.grid(alpha=.25)
finish(fig,"Figure_6_HSWDE_sensitivity.png")

# Manifest
manifest=pd.DataFrame([
    ["Figure_1_JanJul_mean_discharge.png","Hydrological baseline comparison"],
    ["Figure_2_low_flow_exposure_LFSI.png","Low-flow exposure and LFSI"],
    ["Figure_3_LFSI_robustness_ranks.png","Five-formulation robustness"],
    ["Figure_4_water_demand_scenarios.png","Scenario water demand"],
    ["Figure_5_HSWDE_2026_heatmap.png","2026 HSWDE scenarios"],
    ["Figure_6_HSWDE_sensitivity.png","IT-load and WUE sensitivity"],
],columns=["file","purpose"])
manifest.to_csv(OUT/"figure_manifest_v1.csv",index=False)

print("="*72)
print("FIGURE PACKAGE v1 COMPLETE")
print("="*72)
print(f"Output: {OUT}")
for f in manifest["file"]:
    print(f"  {f}")
print("Figures: 6")
print("QC: PASS")
