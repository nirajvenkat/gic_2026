"""
Phase 3 - Step 4: Build the AI/data-center load overlay from the IM3+EPRI
TELL BA Load dataset (2025, TVA balancing authority).

Source: PNNL/EPRI "IM3 + EPRI Data Center Load Projections"
        DOI 10.57931/3007669, scenario rcp45hotter_ssp3, year 2025
        Files: TELL_BA_Loads_moderate_growth.csv, TELL_BA_Loads_high_growth.csv

IMPORTANT DATA CHARACTERISTIC (confirmed from the data itself, not assumed):
The added data-center load (Scaled_TELL_BA_Load_with_DC_MWh minus
Scaled_TELL_BA_Load_MWh) is FLAT across all 8760 hours for a given BA-year
(std ~0.001 MW) -- this is because the source EPRI methodology distributes
each state's annual DC energy evenly across all hours (see readme.pdf).
So the "hourly shape" that actually matters is the TOTAL load (grid baseline
+ DC), which inherits the real seasonal/diurnal shape from the baseline.
That's what we use for siting/sizing: peak TOTAL demand hours are the
worst-case stress conditions a BESS/microgrid would need to cover.

Output:
  data/tva_ai_load_overlay_2025.csv - hourly baseline/DC/total load for TVA,
                                       both growth scenarios, 2025
"""
import pandas as pd

SRC_DIR = "/sessions/gifted-relaxed-darwin/mnt/phase 3"
OUT_DIR = "/sessions/gifted-relaxed-darwin/mnt/outputs/data"
BA = "TVA"

scenarios = {
    "moderate": "TELL_BA_Loads_moderate_growth.csv",
    "high": "TELL_BA_Loads_high_growth.csv",
}

frames = []
for label, fname in scenarios.items():
    df = pd.read_csv(f"{SRC_DIR}/{fname}")
    df = df[df["BA_Code"] == BA].copy()
    df["growth_scenario"] = label
    df["baseline_load_mw"] = df["Scaled_TELL_BA_Load_MWh"]
    df["total_load_with_dc_mw"] = df["Scaled_TELL_BA_Load_with_DC_MWh"]
    df["dc_load_added_mw"] = df["total_load_with_dc_mw"] - df["baseline_load_mw"]
    df["Time_UTC"] = pd.to_datetime(df["Time_UTC"])
    frames.append(df[["Time_UTC", "growth_scenario", "baseline_load_mw",
                       "dc_load_added_mw", "total_load_with_dc_mw"]])

overlay = pd.concat(frames, ignore_index=True).sort_values(
    ["growth_scenario", "Time_UTC"]
)
overlay.to_csv(f"{OUT_DIR}/tva_ai_load_overlay_2025.csv", index=False)

# Report
for label in scenarios:
    sub = overlay[overlay["growth_scenario"] == label]
    peak_row = sub.loc[sub["total_load_with_dc_mw"].idxmax()]
    print(f"--- {BA}, {label} growth, 2025 ---")
    print(f"  DC load added (flat)      : {sub['dc_load_added_mw'].mean():.2f} MW "
          f"(std {sub['dc_load_added_mw'].std():.4f})")
    print(f"  Baseline load range       : {sub['baseline_load_mw'].min():.0f} - "
          f"{sub['baseline_load_mw'].max():.0f} MW")
    print(f"  Total load (baseline+DC)  : {sub['total_load_with_dc_mw'].min():.0f} - "
          f"{sub['total_load_with_dc_mw'].max():.0f} MW")
    print(f"  Peak total-load hour      : {peak_row['Time_UTC']} "
          f"({peak_row['total_load_with_dc_mw']:.0f} MW)")
    print()

print(f"Saved {len(overlay):,} rows -> {OUT_DIR}/tva_ai_load_overlay_2025.csv")
