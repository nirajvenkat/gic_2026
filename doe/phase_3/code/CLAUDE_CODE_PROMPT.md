Paste this into Claude Code, run from `phase 3/code/`.

---

I'm building a submission for the DOE Global Industry Challenge 2026, Phase 3: quantum-enhanced
siting/sizing of battery energy storage (BESS) on a power grid facing AI data-center load growth.
Phase 2 already built a working warm-start QAOA pipeline on the IEEE 39-bus test system. Phase 3
needs the same method rebuilt on IEEE 118-bus, with all the synthetic Phase 2 inputs replaced by
real datasets that are already prepared.

Before writing any code, read these two files in full:

1. `phase 3/code/PHASE3_BUILD_SPEC.md` — the exact build spec: what changes vs. Phase 2, which
   real dataset feeds which pipeline stage, the candidate-bus pruning rationale, the contingency
   weighting formula, and the reproducibility checklist.
2. `phase_2/code/ieee39_dataset.ipynb` — the reference implementation. Reuse its structure and
   QUBO/warm-start-QAOA/Ising/decode logic as directly as possible; only the network size,
   candidate pruning, and the two dataset-driven inputs (AI load injection, contingency weighting)
   should differ.

Datasets already sitting in `phase 3/code/data/` (do not regenerate these, just load them):
- `tva_ai_load_overlay_2025.csv` — real AI/data-center load overlay (TVA balancing authority,
  2025, moderate/high growth scenarios)
- `contingency_scenario_table.csv` — empirical outage probability/severity by event category,
  built from 10 years (2014–2023) of real DOE-417/EAGLE-I data
- `outage_events_by_event.csv` — the 663 underlying real events, if you want event-level detail
  rather than just category aggregates

Environment: use the existing Phase 2 `.venv` (it already has pandapower, scipy, qiskit,
qiskit-ibm-runtime installed — don't reinstall unless something is actually missing).

Build it as a new notebook `phase 3/code/ieee118_dataset.ipynb`, following the same cell-by-cell
structure as the Phase 2 notebook (load network → candidate buses → voltage sensitivity →
investment cost → QUBO → LP warm-start → Ising → QAOA circuit → decode → classical greedy
baseline → validation), then add the AI-load-injection section and the contingency-weighted
resilience section per the build spec.

Work through it stage by stage — run and print/verify the output of each stage (e.g., confirm the
network loads and power-flow converges, confirm candidate pruning gives a sane qubit count, etc.)
before moving to the next, rather than writing the whole notebook blind and debugging at the end.

When a stage's real-world/statistical assumption isn't fully pinned down by the spec (for example,
exactly how to map the 118-bus lines to the 5 contingency categories), stop and ask me rather than
guessing silently — that mapping needs to be defensible in the write-up.
