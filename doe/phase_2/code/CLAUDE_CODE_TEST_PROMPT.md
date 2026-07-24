Paste this into Claude Code, run from `phase_2/code/`.

---

I need you to execution-test `grid_opt.py` in this folder before it goes into a PR. I made two
edits to it (added an IEEE-118 grid option, and replaced a topological "distance from slack bus"
reliability proxy with real contingency-weighted resilience data loaded from
`contingency_scenario_table.csv`, which sits alongside the script in this same folder). I could not
run it myself (no pandapower/qiskit/CPLEX in my environment), so this needs to be verified in your
local `.venv`.

Please:

1. Activate the existing `.venv` in this folder (it already has pandapower, scipy, qiskit,
   qiskit-ibm-runtime, docplex/CPLEX, and the `qamoo` package installed).
2. First run it with `GRID_CASE = "IEEE-33"` (the current default) to confirm nothing broke for the
   existing case — this should reproduce whatever the PM's original script produced, since only the
   IEEE-118 branch and the `dist_n` computation changed.
3. Then edit `GRID_CASE` to `"IEEE-118"` and re-run. Confirm:
   - `pandapower.networks.case118()` loads and the base power flow converges.
   - The candidate-bus pruning produces a sane qubit count.
   - The new resilience block runs cleanly: it should print the loaded contingency categories, the
     representative line picked per category, and the final `dist_n` range. Confirm it does NOT
     raise `FileNotFoundError` (it looks for `contingency_scenario_table.csv` directly in this
     folder, not in a `data/` subfolder).
   - The QAMOO/Benders workflow downstream (QUBO build, QAOA training, execution, hypervolume,
     CPLEX/DPA benchmarks) all run without errors on the IEEE-118 case.
4. If anything errors, tell me the exact error and which stage it's from — don't try to silently
   patch around it, since the resilience-block logic needs to stay defensible for the write-up.
5. If everything runs clean, report the key output numbers (qubit count, selected buses, hypervolume
   values, Pareto front sizes) so I can sanity-check them against the `ieee118_dataset.ipynb`
   reference notebook.

This is a verification pass only — don't change the method or add new features, just confirm it
runs and report exactly what happens.
