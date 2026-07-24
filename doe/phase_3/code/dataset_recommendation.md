# Phase 3 Dataset Recommendation

**Context:** Phase 2 (IEEE 39-bus, warm-start QAOA) used only synthetic inputs — pandapower's
built-in `case39`, an arbitrary flat +300 MW/+90 MVAr "AI shock," and hand-picked N-1 line
outages. Phase 3 judging weights "Data Modeling Strategy" and "Phase 3 Execution" heavily and
explicitly rewards public, reproducible, citable data. The goal here is to replace the synthetic
pieces with real datasets, without changing the core QAOA/BESS-siting method.

## Top recommendation

**Layer three real datasets onto the existing IEEE test-system + QAOA pipeline:**

| Layer | Dataset | Replaces (Phase 2) | Why it's the strongest choice |
|---|---|---|---|
| Large-load growth | **IM3 + EPRI Data Center Load Projections** (PNNL/OSTI, DOI 10.57931/3007669) | Flat +300 MW constant | Real hourly (8760/yr) BA-level load, 2022–2040, 16 scenarios (4 climate/socioeconomic × 4 EPRI AI-growth rates: Low 3.7%–Higher 15%/yr). Directly satisfies the PDF's explicit ask for a "50–500 MW... representative load growth overlay" with a citable, DOE-funded source instead of a made-up number. |
| Contingency/weather scenarios | **DOE OE-417 official historical archive** (oe.netl.doe.gov, 2000–present, Excel/PDF) | Arbitrary single N-1 line trips | Confirmed live and downloadable back to 2000. Gives real event-type and outage-duration frequencies (weather, vandalism, cyber, equipment failure) to build a probability-weighted contingency/scenario tree instead of "trip every line once." |
| Siting hazard exposure | **FEMA National Risk Index** (county/tract CSV or geodatabase, OpenFEMA) | Nothing — Phase 2 had no geographic hazard layer | Confirmed downloadable per-county/tract for all 50 states. Lets you weight candidate BESS/microgrid buses by real hazard scores (wildfire, hurricane, ice storm, flood) — a resilience-relevant refinement judges can independently verify. |

This combination is the best fit because it upgrades exactly the three places in the Phase 2
notebook that were admittedly synthetic (load shock, contingency set, siting rationale), each with
a dataset that is (a) confirmed publicly accessible right now, (b) explicitly named or clearly
implied by the challenge PDF's "Key Datasets" list, and (c) citable/reproducible by a third-party
judge — which is the single biggest scoring risk called out in the PDF ("formulations that cannot
be reproduced... will score poorly").

## Availability audit of every source listed in the challenge PDF

| Source (PDF §3) | Status | Notes |
|---|---|---|
| IEEE 9/14/30/34/39/57/68/118-bus, RTS-GMLC, RTS-96 | **Available now, no ORNL contact needed** | Already bundled in `pandapower.networks` / MATPOWER / PowerModels.jl. Your PM is right — contacting Grid-Q/ORNL for "additional datasets" is unnecessary for these standard test cases. |
| NREL ReEDS scenario outputs | Available | Published scenario outputs downloadable from NREL's ReEDS site; usable as-is for capacity-expansion framing. |
| NREL WIND Toolkit / NSRDB | **Partially available** | Aggregated/point queries work via the `hsds`/`rex` API (see NREL/hsds-examples). Full-resolution bulk WIND Toolkit data requires a special request — not a quick download. Lower priority unless you need wind/solar profiles specifically. |
| EIA Form 860 / 923 | Available | Full bulk download (zip) at eia.gov/electricity/data. Useful if you want a realistic generator fleet, but adds limited value over the existing IEEE case unless you're rebuilding the generation mix. |
| DOE OE-417 (official) | **Available, confirmed** | oe.netl.doe.gov archive goes back to 2000, Excel/PDF format. Recommended over the ORNL CSV (see below). |
| FEMA National Risk Index | **Available, confirmed** | CSV/shapefile/geodatabase at county and census-tract level, all US states, via OpenFEMA. |
| PJM / MISO / ERCOT / CAISO / SPP open data (load, LMP, interconnection queue) | Available, some friction | CAISO OASIS, SPP, ERCOT, MISO all expose queue/load data via public files or simple APIs (e.g. MISO `getprojects`, SPP `GenerateSummaryCSV`). No login needed. Good secondary source if you want real large-load interconnection-queue sizes/locations rather than picking buses arbitrarily. |
| EPRI / LBNL data center load profiles | **Available, confirmed — this is the pick** | See IM3+EPRI dataset above. EPRI's own "Data Center Load Shape Library: 2025 Edition" is a second, more granular option if you want sub-hourly shapes. |
| Aqora-hosted / down-selected curated case study | Not yet available | Only released to down-selected teams later; don't plan around it. |

## Assessment of the CSV your PM found (`oe-417-annual-summaries.csv`, ORNL Open Energy Hub)

- It's real and it does load cleanly: 341 events, 11 columns (date, area affected, NERC region,
  alert criteria, event type, demand loss MW, customers affected).
- **Limitation: single year only (2023).** All rows fall in 2023 — there's no multi-year trend to
  build a defensible frequency/probability model from, and the "annual summaries" framing on
  openenergyhub.ornl.gov suggests it's a curated one-year mirror, not the full series.
- No lat/lon — "Area Affected" is free-text (e.g., "Washington: King County;"), so it needs
  geocoding before it can be joined to bus locations or FEMA NRI.
- **Recommendation:** use the official DOE source (oe.netl.doe.gov/OE417_annual_summary.aspx)
  instead, or in addition — it has the same field structure back to 2000, giving 20+ years of
  events to build real outage-type/frequency statistics. Your PM's instinct that this general
  dataset (OE-417) is relevant is correct; just pull from the primary DOE archive rather than the
  single-year mirror for anything you want to defend statistically in front of judges.

## Fixing the two OE-417 CSV limitations

**Better fix for both at once: use PNNL's "Event-correlated Outage Dataset in America" (USECPO).**
This is a peer-reviewed (IEEE Xplore data descriptor), DOE-funded dataset that already merges
DOE-417 events with **EAGLE-I** — ORNL's county-level outage tracker (15-minute intervals,
2014–2024, FIPS-coded, 92%+ of US customers) — plus county population. It's downloadable now:

- OEDI: https://data.openei.org/submissions/6458
- Data.gov (zip + individual resources "DOE-417 Dataset" / "EAGLE-I Dataset"):
  https://catalog.data.gov/dataset/event-correlated-outage-dataset-in-america

This replaces the single-year ORNL mirror outright:
- **Multi-year fixed** — EAGLE-I/DOE-417 correlation spans 2014–2024 (9+ years), enough to build
  a real per-region, per-event-type outage frequency/severity distribution instead of one year's
  snapshot.
- **Geocoding fixed** — EAGLE-I is natively FIPS-coded at the county level, and the merge already
  joins each DOE-417 event to its county. That FIPS code is the same join key FEMA NRI uses, so
  OE-417 → EAGLE-I → FEMA NRI chains together with zero free-text parsing.

**Backup, if that merged dataset has access friction:** the CSV your PM found is still usable —
"Area Affected" follows a `State: County;` pattern in 292 of 341 rows (86%), which a regex parser
can turn into county FIPS via a Census Gazetteer centroid lookup. The remaining 49 rows are
either state-only or a utility name (e.g. "Austin Energy") with no county — those would need a
utility→service-territory lookup (EIA-861) or just get dropped to a coarser state-level join. For
the multi-year piece in this fallback path, pull the individual annual OE-417 Excel files from
oe.netl.doe.gov back to whatever year range you need and concatenate them yourself.

**One caveat either way:** IEEE 118-bus is a synthetic topology with no real geography, so
"joining" outage/hazard data to buses still means assuming a documented bus→region mapping (e.g.
assign clusters of buses to representative counties). That's expected and fine under the rubric —
the PDF explicitly rewards "transparency about candidate sets, scenarios, and assumptions" — just
state the mapping rule plainly in the write-up rather than presenting it as literal geography.

## How this maps to the four Phase 3 focus areas

- **Storage siting/sizing for resilience + AI load integration** and **Multi-scenario capacity
  expansion under contingency** are the two tracks this dataset stack most directly strengthens,
  building straight on Phase 2's Direction A (AI shock) and Direction B (N-1 resilience) work.
- **Co-optimization with large flexible loads** is also well supported if you add the ISO
  interconnection-queue data (real MW sizes/timing of pending large-load requests) as a stretch
  goal.

## Suggested next step

Pick one focus area to commit to (recommend: Storage siting/sizing for resilience + AI load
integration, since it's the most direct extension of the existing notebook), then I can pull the
IM3+EPRI load file and OE-417/FEMA NRI data into `phase 3/code` and wire them into a rebuilt
version of the Phase 2 pipeline on IEEE 118-bus (the PDF's recommended "tractable Phase 2/3"
system).
