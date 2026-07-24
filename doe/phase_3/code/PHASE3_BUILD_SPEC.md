# Phase 3 Build Spec — IEEE 118-Bus Warm-Start QAOA (BESS Siting/Sizing)

Handoff doc for implementing this locally (Claude Code + Phase 2's `.venv`, which already has
scipy/pandapower/qiskit installed — no need to reinstall).

## Objective

Rebuild the Phase 2 warm-start QAOA pipeline (`phase_2/code/ieee39_dataset.ipynb`) on **IEEE
118-bus** for the **Storage siting/sizing for resilience + AI load integration** focus area,
replacing every synthetic Phase 2 input with a real, citable dataset. Method (warm-start QAOA,
QUBO structure, LP relaxation, Ising conversion) stays the same — only the *inputs* change. This
directly targets the rubric's "Data Modeling Strategy" and "Phase 3 Execution" criteria.

## What Phase 2 did vs. what changes

| Component | Phase 2 (IEEE 39-bus) | Phase 3 (IEEE 118-bus) |
|---|---|---|
| Test system | `pandapower.networks.case39()` | `pandapower.networks.case118()` |
| AI load shock | Flat +300 MW / +90 MVAr, arbitrary | Real TVA added-DC-load from IM3+EPRI dataset (~349–383 MW), injected at the real 2025 peak-total-demand hour |
| N-1 / contingency weighting | Every line tripped once, uniform weight | Empirical probability/severity weights from 10 years (2014–2023) of real DOE-417/EAGLE-I events |
| Everything else (QUBO, warm-start LP, Ising, QAOA circuit, decode, greedy baseline) | as-is | reuse directly, just resized for the new candidate-bus count |

## Data inventory (already in `phase 3/code/`)

- `data/tva_ai_load_overlay_2025.csv` — 17,520 rows. Columns: `Time_UTC, growth_scenario
  (moderate/high), baseline_load_mw, dc_load_added_mw, total_load_with_dc_mw`. Use
  `dc_load_added_mw` (flat ~349.33 MW moderate / ~383.40 MW high) as the injected load magnitude.
  Peak-stress hour = 2025-01-21 02:00:00 UTC (total load ~41,935–41,969 MW at TVA scale — use this
  to pick which *season/condition* your synthetic bus-level injection represents, not the raw BA
  MW figure itself, since TVA is a whole-region BA and your candidate buses are a single test
  system).
- `data/contingency_scenario_table.csv` — 5 rows, one per event category (Weather 44.9%,
  Physical/Security 27.3%, Equipment/Operational 24.6%, Other/Unknown 1.7%, Cyber 1.5%), with
  `mean_duration_hours` and `mean_customers_affected` per category. Use `empirical_probability` as
  scenario weights.
- `data/outage_events_by_event.csv` — 663 individual real events (2014–2023) if you want to sample
  actual events rather than just category-level aggregates.
- `dataset_recommendation.md` — full sourcing rationale/citations for the write-up.

## Pipeline stages

1. **Load network.** `net = pn.case118()`, `pp.runpp(net)`. Identify candidate buses = all buses
   minus slack/gen buses (same logic as Phase 2 cell 6).
2. **Prune candidates before building the QUBO.** 118-bus will have on the order of 90+
   non-slack/gen buses — too many for statevector QAOA (Phase 2 pruned 39-bus's ~29 candidates
   down to 15 qubits for exactly this reason). Rank candidates by voltage sensitivity `V_n` (same
   method as Phase 2 cell 8) and take the top 15–20. State this pruning explicitly in the write-up
   as required by the rubric ("transparency about candidate sets").
3. **Voltage sensitivity `V_n` and investment cost `r_n`.** Reuse Phase 2 cells 8 and 10 unchanged
   — these are structural, not dataset-dependent.
4. **AI load injection (replaces Direction A).** Instead of flat +300 MW at 2 arbitrary buses,
   inject `dc_load_added_mw` from `tva_ai_load_overlay_2025.csv` (moderate growth: 349.33 MW,
   high growth: 383.40 MW — run both as separate scenarios) at 1–2 candidate buses. Cite the real
   BA and growth-scenario source in the write-up instead of presenting it as an assumption.
5. **Contingency/resilience term (replaces Direction B).** Phase 2's `C_resilience` averaged
   voltage improvement uniformly across every N-1 line trip. Instead, weight each contingency
   scenario by `empirical_probability` from `contingency_scenario_table.csv`. Simplest defensible
   mapping: assign each of the 118-bus's lines to one of the 5 event categories by a documented
   rule (e.g., transmission lines default to "Equipment/Operational" unless deliberately flagged
   as a "weather-exposed" or "physical-security-exposed" scenario for the write-up's hazard
   narrative), then compute `C_resilience = Σ_categories (empirical_probability × ΔV_n under that
   category's representative outage)`. This turns Phase 2's uniform N-1 sweep into a real
   probability-weighted expectation — a direct, defensible upgrade.
6. **QUBO → Ising → warm-start QAOA → decode → classical greedy baseline.** Reuse Phase 2 cells
   12–24 unchanged, just resized to the new (pruned) candidate count.
7. **Multi-scenario table.** Phase 2's "Multi-Scenario Load Uncertainty" section used made-up
   probability weights (50/30/20 split). Replace those probabilities with the real
   `empirical_probability` values from `contingency_scenario_table.csv` crossed with the two real
   AI-growth scenarios (moderate/high) — this gives a defensible scenario tree instead of an
   assumed one, satisfying the PDF's "(a) multi-scenario uncertainty" requirement with real data
   on both axes (load growth AND outage type).
8. **Validation, hybrid architecture diagram, stakeholder section, IBM hardware run.** Reuse Phase
   2's structure/cells, updated for 118-bus qubit counts and the new data citations.

## Qubit / resource budget

Phase 2's scalability analysis (notebook cell 41) already estimated 118-bus resource needs — pull
those numbers into the Phase 3 write-up's "Platform Use and Results" section rather than
re-deriving from scratch. With 15–20 pruned candidate qubits at QAOA depth p=2, this stays within
state-vector/GPU-simulator range per the PDF's own guidance ("tens of qubits ... suitable for
state-vector simulation").

## Reproducibility checklist (per PDF §Rules and §Phase 3 Submission Requirements)

- [ ] Code runs on qBraid without modification
- [ ] README.md: team name, setup, step-by-step run instructions, expected outputs, known
      limitations, "Launch on qBraid" button
- [ ] All dataset sources cited with DOI/URL (see `dataset_recommendation.md`)
- [ ] Classical baseline reported alongside every QAOA result
- [ ] Qubit count, circuit depth, shot budget, wall-clock runtime, and key metric values stated
      explicitly in the write-up (not just qualitative claims)
