"""
Downloads PNNL/OEDI Event-Correlated Outage Dataset (OEDI Submission 6458)
via curl and converts 10-year outage event records into an empirical contingency table.

Source: PNNL "Event-correlated Outage Dataset in America" (OEDI 6458)
        https://data.openei.org/files/6458/Outage_Dataset_R1.zip
"""
import os
import glob
import zipfile
import subprocess
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
RAW_DIR = os.path.join(SCRIPT_DIR, "data", "raw", "correlated_outage")
OUT_DIR = os.path.join(SCRIPT_DIR, "data")
OEDI_ZIP_URL = "https://data.openei.org/files/6458/Outage_Dataset_R1.zip"


def download_and_extract_oedi_dataset():
    """Curles Outage_Dataset_R1.zip from OEDI 6458 and extracts raw CSVs."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    zip_path = os.path.join(SCRIPT_DIR, "data", "raw", "Outage_Dataset_R1.zip")
    if not os.path.exists(zip_path):
        print(f"Curling OEDI 6458 dataset from {OEDI_ZIP_URL}...")
        cmd = ["curl", "-sSL", OEDI_ZIP_URL, "-o", zip_path]
        subprocess.run(cmd, check=True)
        print(f"  Downloaded dataset zip to {zip_path}")

    if os.path.exists(zip_path):
        print(f"Extracting {zip_path} into {RAW_DIR}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(RAW_DIR)
        print(f"  Extracted raw dataset files to {RAW_DIR}")


def normalise_event_type(raw):
    s = str(raw).lower()
    if "weather" in s or "natural disaster" in s or "hurricane" in s or "winter" in s:
        return "Weather"
    if "cyber" in s:
        return "Cyber"
    if any(k in s for k in ["vandalism", "suspicious", "physical attack", "sabotage", "theft"]):
        return "Physical/Security"
    if any(k in s for k in ["transmission", "distribution", "system operations", "generation inadequacy", "fuel supply", "equipment"]):
        return "Equipment/Operational"
    return "Other/Unknown"


def process_contingency_scenarios():
    """Fetch raw EAGLE-I/DOE-417 outage dataset via curl, unzip, and convert to contingency table."""
    out_table = os.path.join(OUT_DIR, "contingency_scenario_table.csv")
    if os.path.exists(out_table):
        print(f"Contingency scenario table ready at {out_table}")
        return pd.read_csv(out_table)

    files = sorted(glob.glob(os.path.join(RAW_DIR, "**", "eaglei_outages_with_events_*.csv"), recursive=True))
    if not files:
        download_and_extract_oedi_dataset()
        files = sorted(glob.glob(os.path.join(RAW_DIR, "**", "eaglei_outages_with_events_*.csv"), recursive=True))

    if not files:
        raise FileNotFoundError(f"Failed to find raw outage event files in {RAW_DIR} after downloading.")

    print(f"Processing {len(files)} annual outage event files...")
    master = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    master["Datetime Event Began"] = pd.to_datetime(master["Datetime Event Began"])
    master["Datetime Restoration"] = pd.to_datetime(master["Datetime Restoration"], errors="coerce")
    master["year"] = master["Datetime Event Began"].dt.year
    master["event_category"] = master["Event Type"].apply(normalise_event_type)

    by_event = master.groupby("event_id").agg(
        event_category=("event_category", "first"),
        began=("Datetime Event Began", "first"),
        n_counties_affected=("fips", "nunique"),
        mean_duration_hours=("duration", "mean"),
    ).reset_index()

    n_total_events = len(by_event)
    scenario_table = by_event.groupby("event_category").agg(
        n_events=("event_id", "count"),
        mean_counties_affected=("n_counties_affected", "mean"),
        mean_duration_hours=("mean_duration_hours", "mean"),
    ).reset_index()

    scenario_table["empirical_probability"] = scenario_table["n_events"] / n_total_events
    scenario_table = scenario_table.sort_values("empirical_probability", ascending=False)
    scenario_table.to_csv(out_table, index=False)
    print(f"Converted outage dataset -> {out_table} ({len(scenario_table)} categories)")
    return scenario_table


if __name__ == "__main__":
    process_contingency_scenarios()
