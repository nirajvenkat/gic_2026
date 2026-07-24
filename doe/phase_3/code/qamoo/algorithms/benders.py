"""
Classical Benders prototype for the microgrid siting problem.

This module provides a light-weight Benders loop that can be used to
validate decomposition and generate feasibility cuts. It is intentionally
simple so it can be extended later (dual-based cut generation, cut
management, aggregation, etc.).

Notes:
- Requires `docplex` for MIP master solves (you confirmed CPLEX availability).
- Subproblem solver is left as a placeholder: integrate pandapower or an LP
  solver to extract duals for proper Benders cuts.
"""
from typing import List, Tuple
import os
import json
import numpy as np
import networkx as nx

from qamoo.utils.data_structures import ProblemGraphBuilder

try:
    from docplex.mp.model import Model
except Exception:
    Model = None


def _load_problem_graphs(num_qubits: int, problem_dir: str, num_objectives: int) -> List[nx.Graph]:
    pg = ProblemGraphBuilder(num_qubits)
    graphs = []
    for idx in range(num_objectives):
        graphs.append(pg.deserialize(os.path.join(problem_dir, f'problem_graph_{idx}.json')))
    return graphs


def _evaluate_subproblem_pandapower(x: List[int], net=None, candidate_buses: List[int] = None, selected_vm_pu_limits=(0.95, 1.05)) -> Tuple[bool, dict]:
    """Evaluate AC feasibility with pandapower and return diagnostic cuts.

    - `x` is a binary list indicating placement of microgrid/storage at buses.
    - `net` is an optional pandapower network instance to use; if not
      provided, this function will not run a powerflow and will return a
      conservative infeasible diagnostic.
    - `candidate_buses` maps decision variable index to actual bus number.
      If None, variable index is used directly as the bus number.

    Returns (feasible: bool, diagnostics: dict).
    If infeasible, diagnostics contains a list of `cuts`, each a dict with keys:
      - `coeff`: dict[node_index -> coeff] (linear cut coefficients)
      - `rhs`: float (right-hand side)
      - `dual`: float (severity estimate / pseudo-dual)

    Notes:
    - This function performs an AC powerflow check using `pandapower.runpp`.
    - True dual multipliers from an OPF are not computed here; instead we
      generate simple, interpretable feasibility cuts based on overloaded
      elements (lines) and voltage violations. Cuts are conservative and
      intended to prune infeasible binary placements.
    """
    try:
        import pandapower as pp
    except Exception:
        return False, {"reason": "pandapower_not_available"}

    if net is None:
        return False, {"reason": "no_network_provided"}

    # Create a copy of the network to avoid mutating the caller's net
    net_copy = net.deepcopy()

    # Apply placements: here we model a placed microgrid/storage as a simple
    # controllable generation at that bus with small capacity. This is a
    # heuristic mapping for feasibility checks and should be adapted to your
    # detailed modeling assumptions.
    gen_p_mw = 1.0  # per-placement nominal injection (tunable)
    created_gens = []
    for i, bit in enumerate(x):
        if bit:
            # Map variable index to actual bus number
            bus = candidate_buses[i] if candidate_buses is not None else i
            try:
                pp.create_gen(net_copy, bus=bus, p_mw=gen_p_mw, vm_pu=1.0, name=f"benders_gen_{i}")
                created_gens.append(i)
            except Exception:
                # bus may not exist or create_gen not available for this net
                pass

    # Run AC powerflow
    try:
        pp.runpp(net_copy, init='auto')
    except Exception as e:
        # Powerflow failed — produce conservative cuts based on placements
        cuts = []
        # For each created generator, forbid placing too many nearby: simple cut
        if created_gens:
            for idx in created_gens:
                cuts.append({
                    "coeff": {idx: 1.0},
                    "rhs": 0.0,
                    "dual": 1.0,
                    "reason": "pp_failed"
                })
        return False, {"reason": "runpp_failed", "error": str(e), "cuts": cuts}

    # Check line loading and voltages
    cuts = []
    overloaded_lines = []
    if "loading_percent" in net_copy.line.columns:
        for li in net_copy.line.index:
            loading = net_copy.res_line.loc[li, "loading_percent"]
            if loading > 100.0:
                overloaded_lines.append((li, loading))

    voltage_violations = []
    vm_min, vm_max = selected_vm_pu_limits
    if "vm_pu" in net_copy.res_bus.columns:
        for ib in net_copy.bus.index:
            v = net_copy.res_bus.loc[ib, "vm_pu"]
            if v < vm_min or v > vm_max:
                voltage_violations.append((ib, v))

    # If no violations, feasible
    if not overloaded_lines and not voltage_violations:
        return True, {"reason": "feasible"}

    # Construct simple cuts from overloaded lines: forbid placing at both end buses
    for li, loading in overloaded_lines:
        from_bus = int(net_copy.line.loc[li, "from_bus"])
        to_bus = int(net_copy.line.loc[li, "to_bus"])
        coeff = {from_bus: 1.0, to_bus: 1.0}
        rhs = 1.0  # cannot select both endpoints simultaneously
        dual = max(0.0, (loading - 100.0) / 100.0)
        cuts.append({"coeff": coeff, "rhs": rhs, "dual": float(dual), "reason": "line_overload", "line_idx": int(li)})

    # Construct cuts to address voltage violations: limit placements near violated buses
    for ib, v in voltage_violations:
        # forbid placing at the violated bus (simple conservative cut)
        coeff = {int(ib): 1.0}
        rhs = 0.0
        dual = float(abs(v - (vm_min if v < vm_min else vm_max)))
        cuts.append({"coeff": coeff, "rhs": rhs, "dual": dual, "reason": "voltage_violation", "bus_idx": int(ib)})

    return False, {"reason": "violations", "cuts": cuts}


def _forbid_exact_solution_cut(mdl: "Model", x_vars: List, sol: List[int]):
    # Add a constraint that forbids the exact binary vector `sol`.
    # sum_{i|sol[i]==1} x_i + sum_{i|sol[i]==0} (1 - x_i) <= n-1
    n = len(sol)
    lhs = 0
    for i, v in enumerate(sol):
        if v == 1:
            lhs += x_vars[i]
        else:
            lhs += (1 - x_vars[i])
    mdl.add_constraint(lhs <= n - 1)


def _apply_cuts_to_master(mdl: "Model", x_vars: List, cuts: List[dict]):
    """Apply linear cuts (as constraints) to the docplex master model.

    Each cut should be a dict with `coeff` mapping var index to coefficient and
    `rhs` value. This function mutates `mdl` in-place.
    """
    for i, cut in enumerate(cuts):
        coeff = cut.get("coeff", {})
        rhs = cut.get("rhs", 0.0)
        expr = 0
        for idx, c in coeff.items():
            expr += c * x_vars[int(idx)]
        mdl.add_constraint(expr <= rhs, ctname=f"benders_cut_{i}")


def classical_benders(num_qubits: int, problem_dir: str, num_objectives: int, max_iters: int = 20):
    """A simple classical Benders loop.

    - master: binary siting variables with real objectives and budget constraint
    - subproblem: AC feasibility check via pandapower (when available)

    Returns: dict with keys `cuts` (list), `last_solution` (list)
    """
    if Model is None:
        raise RuntimeError("docplex not available — install docplex to run classical Benders.")

    graphs = _load_problem_graphs(num_qubits, problem_dir, num_objectives)

    # Load QP objectives for problem-aware master
    import pickle
    qps = []
    for obj_idx in range(num_objectives):
        qp_path = os.path.join(problem_dir, f'problem_qp_{obj_idx}.pkl')
        if os.path.exists(qp_path):
            with open(qp_path, 'rb') as f:
                qps.append(pickle.load(f))

    mdl = Model(name="benders_master")
    x = mdl.binary_var_list(num_qubits, name="x")

    # Add budget constraint from the first QP (all QPs share the same constraint)
    if qps:
        for cstr in qps[0].linear_constraints:
            expr = mdl.sum(coef * x[int(idx)] for idx, coef in cstr.linear.to_dict().items())
            if cstr.sense.name == "LE":
                mdl.add_constraint(expr <= cstr.rhs)
            elif cstr.sense.name == "GE":
                mdl.add_constraint(expr >= cstr.rhs)
            elif cstr.sense.name == "EQ":
                mdl.add_constraint(expr == cstr.rhs)

    # Objective: equal-weighted scalarized sum of all QP objectives (minimization)
    if qps:
        obj_expr = 0
        for qp in qps:
            for idx, val in qp.objective.linear.to_dict().items():
                obj_expr += (val / num_objectives) * x[int(idx)]
        mdl.minimize(obj_expr)
    else:
        # Fallback to random weights if no QPs available
        rng = np.random.default_rng(42)
        weights = rng.random(num_qubits)
        mdl.maximize(mdl.sum(weights[i] * x[i] for i in range(num_qubits)))

    cuts = []

    # Attempt to load a pandapower network for feasibility checks if available
    try:
        import pandapower as pp
        # The caller should supply a network consistent with the problem; here
        # we try to load a net file if one exists at problem_dir/net.pickle (optional)
        netfile = os.path.join(problem_dir, "net.pickle")
        if os.path.exists(netfile):
            net = pp.from_pickle(netfile)
        else:
            net = None
    except Exception:
        net = None

    for it in range(max_iters):
        sol = mdl.solve()
        if sol is None:
            print(f"[Benders] Master infeasible or solver failed at iteration {it}")
            break

        sol_vec = [int(sol.get_value(x[i])) for i in range(num_qubits)]
        feasible, diag = _evaluate_subproblem_pandapower(sol_vec, net=net)

        print(f"[Benders] Iter {it}: selected={sum(sol_vec)} feasible={feasible} diag={diag}")

        if feasible:
            return {"cuts": cuts, "last_solution": sol_vec}

        # If diagnostics contain cuts, apply them to the master model
        new_cuts = []
        if isinstance(diag, dict) and "cuts" in diag:
            for c in diag["cuts"]:
                new_cuts.append(c)
        else:
            # fallback: forbid the exact assignment
            new_cuts.append({"coeff": {i: 1.0 for i, v in enumerate(sol_vec) if v == 1}, "rhs": max(0, sum(sol_vec) - 1), "dual": 1.0, "reason": "forbid_exact"})

        _apply_cuts_to_master(mdl, x, new_cuts)
        cuts.extend(new_cuts)

    return {"cuts": cuts, "last_solution": None}


def cut_to_qubo_penalty(cut: dict, num_qubits: int, penalty_weight: float = 10.0) -> np.ndarray:
    """Convert a Benders feasibility cut into a linear penalty vector for QUBO.

    Handles all cut types produced by `classical_benders()`:
      - Cuts with `coeff` dict: apply penalty_weight * coeff[i] to variable i
      - Legacy `forbid_vector` cuts: penalise matching bits

    Returns a vector `p` of length `num_qubits`. Adding p[i] to the QUBO
    diagonal discourages the sampler from selecting the penalised variables.
    """
    p = np.zeros(num_qubits, dtype=float)

    # Handle cuts with explicit coefficient dictionaries (all standard cut types)
    coeff = cut.get("coeff", {})
    if coeff:
        for idx_str, c in coeff.items():
            idx = int(idx_str)
            if 0 <= idx < num_qubits:
                p[idx] += penalty_weight * c
        return p

    # Legacy: forbid_vector type
    if cut.get("type") == "forbid_vector":
        v = np.array(cut.get("vec", [0] * num_qubits), dtype=int)
        p = penalty_weight * v.astype(float)

    return p
