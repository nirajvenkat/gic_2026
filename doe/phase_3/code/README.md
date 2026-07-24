# DOE GIC 2026 — Phase 3 | Entangled Trio

[![Launch On qBraid](https://qbraid-static.s3.amazonaws.com/badges/launch_on_qbraid.svg)](https://account.qbraid.com/?gitHubUrl=https://github.com/nirajvenkat/gic_2026.git)

## Phase 3: IEEE 118-bus, Real-Data Warm-Start QAOA for BESS Siting/Sizing

**Focus area:** Storage siting/sizing for resilience + AI load integration

**What changed vs. Phase 2:** same method (warm-start QAOA, QUBO, LP relaxation, Ising conversion,
manual gate-by-gate circuit, classical greedy baseline) rebuilt on **IEEE 118-bus**, with every
synthetic Phase 2 input replaced by a real, citable dataset:

| Component | Phase 2 (IEEE 39-bus) | Phase 3 (IEEE 118-bus) |
|---|---|---|
| AI load shock | Flat +300 MW / +90 MVAr, arbitrary | Real TVA added-DC-load overlay (IM3+EPRI), 349.33 MW moderate / 383.40 MW high |
| N-1 contingency weighting | Every line tripped once, uniform weight | Empirical probability/severity from 10 years (2014–2023) of real DOE-417/EAGLE-I events |
| Everything else | — | reused directly, resized to the new pruned candidate count |

See `PHASE3_BUILD_SPEC.md` and `dataset_recommendation.md` for the full build rationale and
dataset citations.

---

## Setup

Requires Python 3.13 with `pandapower`, `qiskit`, `qiskit-ibm-runtime`, `scipy`, `pandas`,
`matplotlib`, `nbformat`/`nbclient` (for headless execution) installed. On qBraid, use the
**Environments** panel to select a Qiskit-enabled environment, or run:

```bash
pip install pandapower qiskit qiskit-ibm-runtime scipy pandas matplotlib nbformat nbclient
```

An existing Phase 2 `.venv` at `doe/phase_2/code/.venv` already has everything except
`nbformat`/`nbclient`, which are only needed if you execute the notebook headlessly rather than
interactively in Jupyter.

## How to run

1. Open `ieee118_dataset.ipynb` in Jupyter (or qBraid Lab) with the environment above selected as
   the kernel.
2. **Run all cells top to bottom.** The notebook is self-contained — it loads
   `pandapower.networks.case118()` directly (no external grid file) and reads the three CSVs in
   `data/` (paths are relative to `phase 3/code/`, so run from that directory).
3. Expect **~15–20 minutes** total wall-clock time for a full run. The dominant cost is the N-1
   contingency sweep (173 lines × 15 pruned buses × 2 probe signs ≈ 5,190 power flows, ~12
   minutes); everything else (voltage sensitivity, 4 separate QAOA warm-start runs, sizing,
   multi-scenario table, diagram) finishes in well under a minute combined.
4. The final section (IBM Quantum hardware run) is **not** run automatically by "Run All" in the
   sense that it requires a configured `QiskitRuntimeService()` account — see below.

## Expected outputs (from the reference run in this repo)

- **Candidate pruning:** 118 buses → 64 non-slack/gen candidates → pruned to **15 qubits** (top-15
  by voltage sensitivity V_n; the spec's stated 15–20 range — 15 was chosen because 20 qubits took
  ~31 minutes per COBYLA run in verification vs. ~21 seconds at 15 qubits, and this notebook needs
  4 separate QAOA runs).
- **Base-case siting:** QAOA selects buses `[50, 51, 52, 57, 63]`, 4 violations vs. greedy's
  `[50, 51, 52, 57, 117]` at 6 violations — QAOA wins on the combined α·C_VC − μ·C_R objective.
- **Exact classical baseline (brute-force):** with 15 qubits and K=5, the base-case QUBO has only
  C(15,5) = 3,003 feasible placements — small enough to enumerate exhaustively. QAOA's selected set
  `[50, 51, 52, 57, 63]` **is the unique global optimum** (rank 1/3,003, gap 0.0000), found in 38
  COBYLA evaluations vs. 3,003 for exhaustive search (79× fewer). Greedy ranks 17/3,003, 0.37% above
  optimum — not bad, but not provably optimal the way QAOA's result is.
- **AI-load surge (real TVA data):** injecting 349.33 MW (moderate) / 383.40 MW (high) at buses 74
  and 46 (the pruned candidates with the largest existing load) raises violations by 0–1 with no BESS
  in place. The more important finding is under **high growth**: applying the *base-case* placement
  blindly (i.e., siting BESS for normal conditions, then not re-optimizing once the AI campus comes
  online) actually makes things worse than doing nothing — **4 violations with no BESS at all vs. 6
  violations with the static base-case placement**. Surge-aware re-siting (buses
  `[50, 51, 52, 57, 117]`) fixes this to **3 violations** — better than either alternative. This is
  the strongest evidence in this run for why re-optimizing under load growth matters: a static
  placement isn't just suboptimal here, it's actively counterproductive once real AI load lands. The
  moderate-growth case is a smaller, more typical version of the same effect (+0.0017 pu voltage
  margin from re-siting rather than a violation-count swing).
- **Contingency-weighted resilience (real DOE-417/EAGLE-I data):** resilience-aware QAOA swaps bus
  63 → 62 relative to the base placement. All 173 N-1 contingencies converged.
- **Multi-scenario table:** 10 real joint scenarios (2 growth × 5 outage categories), expected
  violations 5.0 → 4.174 with BESS placed.
- **BESS sizing:** MW-native LP, bounded by the DOE's own 50–500 MW/site target, 750 MW fleet-wide
  budget. Result: buses 50/57/63 at the 50 MW floor, bus 51 at 100 MW, bus 52 (highest
  voltage-sensitivity impact) at the 500 MW ceiling — **all 5 sites land within the 50–500 MW/site
  target**. Total fleet 750 MW / 3,000 MWh, ~$1,012.5M estimated cost (power + energy, excl.
  balance-of-system) — consistent with real utility-scale BESS pricing (~$337/kWh installed).

## Full run metrics (all four QAOA runs, from the reference run in this repo)

Every run uses the same 15 pruned candidate buses, P_LAYERS=2, |J|=105 ZZ terms, circuit depth 131
(excl. measure), 420 two-qubit gates, and 1024/4096 shots (optimization/decoding) — only the QUBO
weighting and resulting angles differ per run.

| Run | COBYLA evals | Wall-clock | Final ⟨H⟩ | QUBO energy | Selected buses | Key result |
|---|---|---|---|---|---|---|
| Base (α·C_VC + μ·C_R) | 38 | 2.2s | −21.5906 | −51.2191 | [50,51,52,57,63] | 4 violations vs. greedy's 6; **certified global optimum** (brute-force verified, rank 1/3,003) |
| AI-surge, moderate (+349.33 MW) | 43 | ~22.6s¹ | −21.6426 | −52.4251 | [50,51,52,57,117] | surge-aware re-siting gives +0.0017 pu margin over blind base placement |
| AI-surge, high (+383.40 MW) | 34 | ~17.8s¹ | −21.6692 | −52.4562 | [50,51,52,57,117] | blind base placement backfires (6 viol. vs. 4 with no BESS); surge-aware fixes to 3 |
| Resilience-weighted (+ν·C_resilience) | 36 | 43.7s² | −22.3106 | −52.9108 | [50,51,52,57,62] | swaps bus 63→62; ΣC_resilience 2.1146 (beats greedy's 2.2621, ~ties base QAOA's 2.1090) |

¹ Moderate/high surge runs share one notebook cell (40.4s combined); split shown is proportional to
COBYLA eval count, not independently timed.
² Includes the per-category resilience computation (power-flow probes across 173 lines × 5
contingency categories) bundled in the same cell as the QAOA optimization — not a QAOA-only time.

**Total notebook wall-clock (this reference run, excluding the live IBM cell): 19m 16s** —
dominated by the N-1 contingency sweep (~12 min for all 173 lines × 15 buses × 2 probe signs); the
four QAOA runs above total under 90 seconds combined.

## Scaling estimate for real deployment

Extrapolated from this project's own two measured data points (15 qubits: 0.058 s/eval, this
notebook; 20 qubits: 18.6 s/eval, pre-pruning verification benchmark) — an empirical cost exponent
of ~1.67 per qubit-doubling-adjacent step, steeper than the theoretical 2ⁿ state-space growth alone
because gate-by-gate circuit application in plain Python adds overhead that compounds with depth.
Projected cost of a 100-eval COBYLA run: **20 qubits ≈ 31 min, 25 qubits ≈ 166 hr, 30 qubits ≈
6+ years** on this plain-Python `StatevectorSampler` approach.

Practical implication: this project's method is tractable up to ~15–20 qubits, matching the DOE
PDF's own guidance ("tens of qubits... suitable for state-vector simulation"), but a real
ISO-scale deployment (hundreds of candidate substations, not 15–20) would need either (a)
Aer/cuQuantum GPU-accelerated simulation — available out-of-the-box on qBraid — likely 1–2 orders
of magnitude faster at the same qubit count, or (b) a different encoding entirely: mapping the
larger candidate-selection graph onto QuEra Aquila as a Maximum Independent Set problem, per the
PDF's own suggested approach for "larger combinatorial siting graphs," rather than scaling
gate-based QAOA past ~20–25 qubits at all.

## Known limitations

- **Resilience-aware QAOA does not dominate on its own ΣC_resilience metric** — it scores 2.1146
  vs. base QAOA's 2.1090 and greedy's 2.2621. This is expected: the resilience-aware run optimizes
  a *combined* three-objective QUBO (voltage + cost + resilience), not resilience in isolation, so
  it doesn't necessarily win on any single sub-objective. Reported as-is rather than tuned to look
  better.
- **Line → contingency-category mapping is a documented modeling assumption, not real geography.**
  IEEE 118-bus has no physical location data, so categories are assigned via computed electrical
  properties (line loading percentile, bus connectivity degree) rather than invented geographic
  exposure. See the "Contingency-Weighted Resilience" section in the notebook for the exact rule
  and the reasoning behind it.
- **AI-growth scenario weights (moderate/high) are assumed equal (p=0.5 each).** The source EPRI
  data names two growth tracks but documents no relative likelihood between them; equal weighting
  is the honest default absent a source-backed alternative.
- **StatevectorSampler is noiseless.** The IBM hardware run (below) is the only place real
  hardware noise enters the results.

## IBM Quantum hardware run

The last section of the notebook submits the base-case 15-qubit warm-start QAOA circuit (bound at
its COBYLA-optimized angles) to real IBM Quantum hardware via `qiskit-ibm-runtime`. This requires:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="<your IBM Quantum API token>", overwrite=True)
```

(Note: the legacy `"ibm_quantum"` channel is deprecated by IBM — use `"ibm_quantum_platform"`.)

run once per environment before executing that cell. It submits a live job to shared hardware
(queue wait + usage against your IBM Quantum plan), so it is deliberately kept as the last,
separately-run cell rather than part of "Run All."

---

## Dataset citations

- `data/tva_ai_load_overlay_2025.csv` — PNNL/EPRI "IM3 + EPRI Data Center Load Projections,"
  DOI [10.57931/3007669](https://doi.org/10.57931/3007669), scenario `rcp45hotter_ssp3`, TVA
  balancing authority, 2025.
- `data/contingency_scenario_table.csv`, `data/outage_events_by_event.csv` — derived from DOE-417 /
  EAGLE-I merged outage event data (2014–2023). See `dataset_recommendation.md` for full sourcing
  detail and the primary-source archive URLs.

## Reference implementation

`doe/phase_2/code/ieee39_dataset.ipynb` — the Phase 2 IEEE 39-bus notebook this pipeline was
rebuilt from. `PHASE3_BUILD_SPEC.md` documents exactly what changed and why.
