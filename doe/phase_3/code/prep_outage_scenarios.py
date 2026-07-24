"""
Phase 3 - Step 2: Consolidate EAGLE-I x DOE-417 correlated outage data (2014-2023)
into an empirical contingency/scenario-probability table.

Source: PNNL "Event-correlated Outage Dataset in America" (USECPO)
        https://data.openei.org/submissions/6458
        files: eaglei_outages_with_events_<YEAR>.csv  (2014-2023)

Output:
  data/outage_events_master.csv       - one row per (event, county) - full detail
  data/outage_events_by_event.csv     - one row per unique event_id (663 events)
  data/contingency_scenario_table.csv - frequency/severity by normalised event category
                                         (feeds the QUBO resilience/contingency term)
"""
import pandas as pd
import glob

SRC_DIR = "/sessions/gifted-relaxed-darwin/mnt/phase 3/6458/correlated_outage_readme/correlated_outage"
OUT_DIR = "/sessions/gifted-relaxed-darwin/mnt/outputs/data"

# 1. Load & concatenate all 10 years
files = sorted(glob.glob(f"{SRC_DIR}/eaglei_outages_with_events_*.csv"))
master = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
master["Datetime Event Began"] = pd.to_datetime(master["Datetime Event Began"])
master["Datetime Restoration"] = pd.to_datetime(master["Datetime Restoration"], errors="coerce")
master["year"] = master["Datetime Event Began"].dt.year

# 2. Normalise messy Event Type strings into a small set of categories
def normalise_event_type(raw):
    s = str(raw).lower()
    if "weather" in s or "natural disaster" in s or "hurricane" in s or "winter" in s:
        return "Weather"
    if "cyber" in s:
        return "Cyber"
    if any(k in s for k in ["vandalism", "suspicious", "physical attack", "sabotage", "theft"]):
        return "Physical/Security"
    if any(k in s for k in ["transmission", "distribution", "system operations",
                             "generation inadequacy", "fuel supply", "equipment"]):
        return "Equipment/Operational"
    return "Other/Unknown"

master["event_category"] = master["Event Type"].apply(normalise_event_type)
master.to_csv(f"{OUT_DIR}/outage_events_master.csv", index=False)

# 3. Collapse to one row per unique event.
# NOTE: source data is one row per 15-min outage *segment*, so a single
# (event, county) pair can have hundreds of rows during a multi-day event.
# Summing max_customers directly across all rows double/triple counts the
# same population. Fix: take the PEAK customers-out per (event, county)
# first, then sum those peaks across counties -> defensible event-level
# "total customers affected" (peak-impact proxy).
per_event_county = (
    master.groupby(["event_id", "fips"])
    .agg(peak_customers=("max_customers", "max"))
    .reset_index()
)
event_totals = (
    per_event_county.groupby("event_id")
    .agg(total_customers_affected=("peak_customers", "sum"),
         max_customers_single_county=("peak_customers", "max"))
    .reset_index()
)

by_event = (
    master.groupby("event_id")
    .agg(
        event_category=("event_category", "first"),
        event_type_raw=("Event Type", "first"),
        began=("Datetime Event Began", "first"),
        restored=("Datetime Restoration", "first"),
        year=("year", "first"),
        n_counties_affected=("fips", "nunique"),
        n_states_affected=("state", "nunique"),
        mean_duration_hours=("duration", "mean"),
    )
    .reset_index()
    .merge(event_totals, on="event_id", how="left")
)
by_event.to_csv(f"{OUT_DIR}/outage_events_by_event.csv", index=False)

# 4. Empirical contingency / scenario-probability table by category
n_total_events = len(by_event)
scenario_table = (
    by_event.groupby("event_category")
    .agg(
        n_events=("event_id", "count"),
        mean_counties_affected=("n_counties_affected", "mean"),
        mean_duration_hours=("mean_duration_hours", "mean"),
        mean_customers_affected=("total_customers_affected", "mean"),
        median_customers_affected=("total_customers_affected", "median"),
        max_customers_affected=("total_customers_affected", "max"),
    )
    .reset_index()
)
scenario_table["empirical_probability"] = scenario_table["n_events"] / n_total_events
scenario_table = scenario_table.sort_values("empirical_probability", ascending=False)
scenario_table.to_csv(f"{OUT_DIR}/contingency_scenario_table.csv", index=False)

# Report
print(f"Master rows (event x county): {len(master):,}")
print(f"Unique events (2014-2023):    {n_total_events:,}")
print()
print("Contingency scenario table (feeds QUBO resilience/contingency weighting):")
print(scenario_table.to_string(index=False))
