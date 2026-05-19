# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: kernelspec,jupytext,language_info
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   language_info:
#     name: python
# ---

# %% [markdown]
# # Quantum-Enhanced Strategic Siting of Energy Storage and Microgrids
# **Team Name:** Entangled Trio
#
# This notebook demonstrates the QAMOO-based multi-objective optimization for strategic siting of energy storage and microgrids. We evaluate this over a Pandapower grid (e.g., IEEE-14) considering 3 objectives:
# 1. **Resilience / Reliability ($C_R$)**
# 2. **Capital Investment ($C_I$)**
# 3. **Voltage-Control Quality ($C_{VC}$)**
#
# Because QAMOO natively uses Qiskit's `Maxcut` under the hood, we map our objective functions as purely quadratic (edge weights on a graph) for this demonstration. In a fully rigorous application, linear terms can be absorbed using an auxiliary fixed qubit or by replacing the `Maxcut` call with direct `QuadraticProgram` translation.

# %%
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# %%
import os
import sys
sys.path.append(os.path.abspath('..'))
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pandapower.networks as nw
import pandapower.topology as top

from qamoo.configs.configs import ProblemSpecification, QAOAConfig
from qamoo.utils.utils import compute_hypervolume_progress
from qamoo.algorithms.qaoa import *
from qamoo.utils.data_structures import ProblemGraphBuilder

from qiskit_aer import AerSimulator

# %% [markdown]
# ## 1. Grid Loading and Objective Graph Serialization

# %%
# Load IEEE-14 system (can be replaced with case33bw() for IEEE-33)
net = nw.case33bw()
num_buses = len(net.bus)
print(f"Loaded grid with {num_buses} buses.")

# Extract topology to NetworkX
mg = top.create_nxgraph(net)
nx_graph = nx.Graph(mg)

# Relabel nodes to 0..N-1
mapping = {node: i for i, node in enumerate(nx_graph.nodes())}
nx_graph = nx.relabel_nodes(nx_graph, mapping)
edges = list(nx_graph.edges())

# Generate edge weights representing our three QUBO objectives:
# Obj 1: Resilience/Reliability Penalty
obj1_weights = [np.random.uniform(0.5, 2.0) for _ in edges]
# Obj 2: Capital Investment
obj2_weights = [np.random.uniform(1.0, 5.0) for _ in edges]
# Obj 3: Voltage-Control Quality
obj3_weights = [np.random.uniform(0.1, 1.0) for _ in edges]

# Target directory for QAMOO problem
problem_dir = f'./data/problems/{num_buses}q/problem_set_{num_buses}q_0s_3o_0/'
os.makedirs(problem_dir, exist_ok=True)

# Save the objective graphs using ProblemGraphBuilder
for idx, weights in enumerate([obj1_weights, obj2_weights, obj3_weights]):
    builder = ProblemGraphBuilder(num_buses)
    builder.build(edges)
    builder.assign_weights(weights)
    builder.serialize(f'problem_graph_{idx}', problem_dir)

print(f"Serialized {num_buses}-qubit problem graphs to {problem_dir}")

# %% [markdown]
# ## 2. Generate Configuration Placeholders & Train QAOA
# QAMOO requires valid `optimal_parameters.json` and objective weights distributions for Multi-Objective optimization. We use COBYLA to optimize a scalarized representation of the 3 objectives.

# %%
num_samples = 10
num_objectives = 3
p_layers = 3

# Lower and Upper bounds for the 3 objectives
with open(os.path.join(problem_dir, 'lower_bounds.json'), 'w') as f:
    json.dump([0.0, 0.0, 0.0], f)
with open(os.path.join(problem_dir, 'upper_bounds.json'), 'w') as f:
    json.dump([100.0, 100.0, 100.0], f)

# QAOA Parameter Optimization using COBYLA
print(f"Training p={p_layers} parameters using COBYLA (this may take a minute)...")
from qiskit_aer.primitives import Estimator as AerEstimator
from scipy.optimize import minimize
from qiskit_optimization.applications import Maxcut
from qiskit.circuit.library import QAOAAnsatz

combined_edges = []
for u, v in edges:
    w1 = obj1_weights[edges.index((u,v))]
    w2 = obj2_weights[edges.index((u,v))]
    w3 = obj3_weights[edges.index((u,v))]
    combined_edges.append((u, v, (w1+w2+w3)/3.0))
    
combined_graph = nx.Graph()
combined_graph.add_weighted_edges_from(combined_edges)
ising, _ = Maxcut(combined_graph).to_quadratic_program().to_ising()

ansatz = QAOAAnsatz(ising, reps=p_layers)
estimator = AerEstimator(backend_options={"method": "matrix_product_state", "matrix_product_state_max_bond_dimension": 20}, run_options={"shots": 1024})

def cost_func(params):
    bound_ansatz = ansatz.assign_parameters(params)
    job = estimator.run([bound_ansatz], [ising])
    return -job.result().values[0]  # Minimize negative energy = Maximize Cut

np.random.seed(42)
init_guess = np.random.uniform(-np.pi, np.pi, 2 * p_layers)
res = minimize(cost_func, init_guess, method="COBYLA", options={'maxiter': 60})
trained_params = res.x.tolist()
print("Optimal parameters found:", trained_params)

optimal_params = {str(p_layers): trained_params}
with open(os.path.join(problem_dir, 'optimal_parameters.json'), 'w') as f:
    json.dump(optimal_params, f)

# Generate objective weight distributions
weights_dir = './data/objective_weights/'
os.makedirs(weights_dir, exist_ok=True)
obj_weights = np.random.dirichlet((1, 1, 1), num_samples).tolist()
weights_file = os.path.join(weights_dir, f'objective_weights_{num_objectives}o_0.json')
with open(weights_file, 'w') as f:
    json.dump(obj_weights, f)

print("Generated parameter weights.")

# %% [markdown]
# ## 3. QAOA Setup and Execution

# %%
# Set up Qiskit Backend
backend = AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension=20, max_parallel_threads=1)
backend.options.use_fractional_gates = False

# QAMOO Problem Specification
problem = ProblemSpecification()
problem.data_folder = './data/'
problem.num_qubits = num_buses
problem.num_objectives = num_objectives
problem.num_swap_layers = 0
problem.problem_id = 0

# QAMOO QAOA Configuration
config = QAOAConfig()
config.p = p_layers
config.num_samples = num_samples
config.shots = 4096
config.objective_weights_id = 0
config.backend_name = backend.name
config.initial_layout = None
config.run_id = 'siting_run'
config.rep_delay = 0.0001
config.problem = problem

print("Preparing parameterized QAOA circuits...")
prepare_qaoa_circuits(config, backend, overwrite_results=True)

print("Transpiling QAOA circuits for target backend...")
transpile_qaoa_circuits_parametrized(config, backend)

print("Executing sampled circuits (this might take a moment)...")
batch_execute_qaoa_circuits_parametrized([config], backend)

# %% [markdown]
# ## 4. Hypervolume Analysis
# Plot the progression of the Hypervolume metric over the simulated sample runs.

# %%
step_size = max(1, config.total_num_samples // 10)
steps = range(0, config.total_num_samples + 1, step_size)

try:
    compute_hypervolume_progress(problem.problem_folder, config.results_folder, steps)
    x, y = config.progress_x_y()
    
    print(f"Max Hypervolume: {max(y):.2f}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o')
    plt.title(f'Hypervolume Progress - Siting Optimization ({net.name})')
    plt.xlabel('Samples Evaluated')
    plt.ylabel('Hypervolume')
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Evaluation encountered an issue: {e}")

# %% [markdown]
# ## 5. Exact Benchmark (DPA vs CPLEX)
# Here we execute a native Python scalarized multi-objective loop using IBM CPLEX (via the `docplex` library) as well as the DPA algorithm. DPA is an external C++ binary that natively solves MOIP (Multi-Objective Integer Programming). We build the QUBO constraint mapping and search the Pareto space using both a weighted-sum scalarization (CPLEX) and epsilon-constraint (DPA) to compare the results.

# %%
from docplex.mp.model import Model
from qamoo.utils.utils import pareto_front

def solve_classical_mo_cplex(num_qubits, problem_dir, num_objectives, num_evaluations=50):
    # Load problem graphs
    pg_builder = ProblemGraphBuilder(num_qubits)
    problem_graphs = []
    for obj in range(num_objectives):
        pg = pg_builder.deserialize(f"{problem_dir}/problem_graph_{obj}.json")
        problem_graphs.append(pg)
    
    graph_edges = list(problem_graphs[0].edges())
    
    # Generate random weights for weighted-sum scalarization
    np.random.seed(42)
    weights = np.random.dirichlet(np.ones(num_objectives), num_evaluations)
    
    classical_pareto_points = []
    
    for w in weights:
        mdl = Model(name='microgrid_mo')
        x = mdl.binary_var_list(num_qubits, name="x")
        
        obj_expr = 0
        for idx in range(num_objectives):
            pg = problem_graphs[idx]
            sub_obj = 0
            # Objective: Maximize sum w_ij * (x_i + x_j - 2x_i x_j)
            for u, v in graph_edges:
                weight = pg[u][v]["weight"]
                sub_obj += weight * (x[u] + x[v] - 2 * x[u] * x[v])
            obj_expr += w[idx] * sub_obj
            
        mdl.maximize(obj_expr)
        mdl.set_time_limit(5) # 5s per sample
        mdl.context.solver.log_output = False
        
        sol = mdl.solve()
        if sol:
            # Extract un-weighted objective values for the exact point
            vals = [sol.get_value(x[i]) for i in range(num_qubits)]
            f_vals = []
            for idx in range(num_objectives):
                pg = problem_graphs[idx]
                val = 0
                for u, v in graph_edges:
                    val += pg[u][v]["weight"] * (vals[u] + vals[v] - 2 * vals[u] * vals[v])
                f_vals.append(val)
            classical_pareto_points.append(f_vals)
            
    return np.array(classical_pareto_points)

print("Running Classical Exact Benchmark (CPLEX Weighted-Sum)...")
try:
    classical_points = solve_classical_mo_cplex(num_buses, problem_dir, num_objectives, 30)
    nd_idx = pareto_front(classical_points)
    classical_nd_points = classical_points[nd_idx]
except Exception as e:
    print(f"CPLEX Python API not fully configured. {e}")
    classical_nd_points = []

# %%
import subprocess
import os
import re

def solve_classical_mo_dpa(num_qubits, problem_dir, num_objectives):
    from docplex.mp.model import Model
    
    # Load problem graphs
    pg_builder = ProblemGraphBuilder(num_qubits)
    problem_graphs = []
    for obj in range(num_objectives):
        pg = pg_builder.deserialize(f"{problem_dir}/problem_graph_{obj}.json")
        problem_graphs.append(pg)
    
    graph_edges = list(problem_graphs[0].edges())
    
    mdl = Model(name='microgrid_mo_dpa')
    x = mdl.binary_var_list(num_qubits, name="x")
    
    # Linearize the quadratic terms: z[u,v] = x[u] * x[v]
    z = mdl.binary_var_dict(graph_edges, name="z")
    for u, v in graph_edges:
        mdl.add_constraint(z[u, v] <= x[u])
        mdl.add_constraint(z[u, v] <= x[v])
        mdl.add_constraint(z[u, v] >= x[u] + x[v] - 1)
        
    # We need to add the objective functions as constraints at the very end
    # since DPA expects them there. For maximization, DPA multiplies constraint coeffs by -1.
    for idx in range(num_objectives):
        pg = problem_graphs[idx]
        obj_expr = 0
        for u, v in graph_edges:
            weight = pg[u][v]["weight"]
            obj_expr += weight * (x[u] + x[v] - 2 * z[u, v])
            
        if idx == num_objectives - 1:
            # the last constraint's LB encodes the number of objectives in DPA
            mdl.add_constraint(-obj_expr >= num_objectives, ctname=f"obj_{idx}")
        else:
            mdl.add_constraint(-obj_expr >= 0, ctname=f"obj_{idx}")
            
    mdl.maximize(0)
    lp_path = f"{problem_dir}/dpa_problem.lp"
    mdl.export_as_lp(lp_path)
    
    print("Running DPA Exact Benchmark...")
    dpa_bin = "./dpa-main"
    out_file = f"{problem_dir}/dpa_out"
    # Pass 'empty.txt' for warmstart to avoid segmentation fault in DPA
    cmd = [dpa_bin, lp_path, "-a", out_file, "3600", "empty.txt"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"DPA failed (code {proc.returncode}): {proc.stderr[-500:]}")
    
    # Parse the pareto points from the output file
    sol_file = out_file + ".sol"
    if not os.path.exists(sol_file):
        raise FileNotFoundError(f"DPA solution file not found: {sol_file}")

    num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    pareto_points = []
    with open(sol_file, 'r') as f:
        for line in f:
            if line.startswith("---"):
                break
            vals = [float(v) for v in num_re.findall(line)]
            if len(vals) >= num_objectives:
                # DPA is fed with -obj_expr constraints; flip signs back to original maximization values.
                raw = vals[-num_objectives:]
                pareto_points.append([-v for v in raw])
    
    pts = np.array(pareto_points, dtype=float)
    print(f"DPA parsed points: {pts.shape}")
    if pts.size:
        print("Sample DPA point:", pts[0].tolist())
    return pts

try:
    dpa_points = solve_classical_mo_dpa(num_buses, problem_dir, num_objectives)
    dpa_nd_idx = pareto_front(dpa_points) if len(dpa_points) > 0 else []
    dpa_nd_points = dpa_points[dpa_nd_idx] if len(dpa_points) > 0 else []
except Exception as e:
    print(f"DPA failed. {e}")
    dpa_nd_points = []


# %% [markdown]
# ## 6. Discrepancy Table
# Visualizing the Hypervolume differences between Quantum and Classical methods.

# %%
try:
    q_hv = max(y)
except Exception:
    q_hv = 0.0

# Compute Exact Classical HV
from pymoo.indicators.hv import HV
from IPython.display import display

def _as_points(a, m):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return np.empty((0, m), dtype=float)
    return a.reshape(-1, m)

cplex_pts = _as_points(classical_nd_points, num_objectives)
dpa_pts = _as_points(dpa_nd_points, num_objectives)

# Convert max-objective points to minimization space for pymoo HV
cplex_min = -cplex_pts if len(cplex_pts) else cplex_pts
dpa_min = -dpa_pts if len(dpa_pts) else dpa_pts

if len(cplex_min) or len(dpa_min):
    all_min = np.vstack([p for p in [cplex_min, dpa_min] if len(p)])
    ref_point = np.max(all_min, axis=0) + 1e-6
else:
    ref_point = np.ones(num_objectives)

ind = HV(ref_point=ref_point)
cplex_hv = float(ind(cplex_min)) if len(cplex_min) else 0.0
dpa_hv = float(ind(dpa_min)) if len(dpa_min) else 0.0

table_data = {
    "Method": ["QAMOO (Quantum QAOA)", "CPLEX (Classical Exact)", "DPA (Classical Exact)"],
    "Hypervolume": [q_hv, cplex_hv, dpa_hv],
    "Discrepancy (vs Exact)": [f"{((dpa_hv - q_hv) / dpa_hv) * 100:.2f}%" if dpa_hv > 0 else "N/A", f"{((dpa_hv - cplex_hv) / dpa_hv) * 100:.2f}%" if dpa_hv > 0 else "N/A", "0.00%"]
}

df_comparison = pd.DataFrame(table_data)
print("\n=== Quantum vs Classical Comparison ===")
print(f"DPA ND points count: {len(dpa_pts)}")

# Create a readable notebook table view
df_view = df_comparison.copy()
df_view["Hypervolume"] = df_view["Hypervolume"].map(lambda v: f"{v:,.4f}")

styled = (
    df_view.style
    .hide(axis="index")
    .set_properties(**{"text-align": "center", "padding": "8px", "font-size": "12pt"})
    .set_table_styles([
        {"selector": "th", "props": "text-align:center; font-size:12pt; font-weight:bold; background-color:#f2f2f2;"},
        {"selector": "td", "props": "border:1px solid #dddddd;"},
        {"selector": "table", "props": "border-collapse:collapse; width:100%;"},
    ])
    .set_caption("Quantum vs Classical Hypervolume Summary")
)

display(styled)

# %% [markdown]
# ## 7. Configuration Visualization
# Visualizing the physical grid mapping for a selected optimal microgrid siting solution from the quantum Pareto front using `rustworkx`.

# %%
import rustworkx as rx
from rustworkx.visualization import mpl_draw
import matplotlib.lines as mlines

try:
    # Load the quantum non-dominated solutions
    nd_positions = np.load(config.results_folder + 'non_dominated_positions.npy')
    all_samples = np.load(config.results_folder + 'samples.npy')
    
    if len(nd_positions) > 0:
        # Pick a balanced solution from the Pareto front (e.g., the middle one)
        best_config = all_samples[nd_positions[len(nd_positions)//2]]
        
        # Create rustworkx graph from our edges
        rx_graph = rx.PyGraph()
        rx_graph.add_nodes_from(range(num_buses))
        rx_graph.add_edges_from_no_data([(u, v) for u, v in edges])
        
        # Define colors: Microgrid (Red), Standard Bus (Lightblue)
        node_colors = ['#FF6B6B' if bit == 1 else '#4ECDC4' for bit in best_config]
        
        plt.figure(figsize=(10, 8))
        plt.title('Microgrid & Energy Storage Siting Plan (Sample Solution from Pareto Front)', fontsize=14, pad=20)
        mpl_draw(rx_graph, node_color=node_colors, with_labels=True, node_size=600, font_size=10, font_color='black')
        
        # Add custom legend
        mg_legend = mlines.Line2D([], [], color='#FF6B6B', marker='o', linestyle='None', markersize=10, label='Microgrid/Storage Placed (1)')
        bus_legend = mlines.Line2D([], [], color='#4ECDC4', marker='o', linestyle='None', markersize=10, label='Standard Bus (0)')
        plt.legend(handles=[mg_legend, bus_legend], loc='upper right')
        plt.show()
    else:
        print("No non-dominated solutions found to visualize.")
except Exception as e:
    print(f"Could not render visualization. Error: {e}")

# %% [markdown]
# ## 8. Multi-Metric Pareto Quality Analysis
# The single discrepancy percentage is limited, so we add standard multi-objective indicators from optimization literature.
#
# - **Hypervolume (HV)**: volume dominated by a Pareto set w.r.t. a reference point (higher is better in our maximization setting after conversion).
# - **IGD** (Inverted Generational Distance): distance from reference Pareto set to an approximate set (lower is better).
# - **Additive Epsilon Indicator** ($\\epsilon^+$): minimum additive shift for one front to weakly dominate another in minimization form (lower is better; $\\le 0$ is very strong).
# - **Coverage** $C(A,B)$: fraction of points in $B$ dominated by points in $A$ (higher is better).
# - **Spacing**: spread/uniformity of points along a front (lower is better).
#
# We treat CPLEX and DPA outputs as the **exact reference union** when available.

# %%
# Additional Pareto-quality metrics for clearer method comparison
from pymoo.indicators.hv import HV
from IPython.display import display
import numpy as np
import pandas as pd
import os

def _as_points(a, m):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return np.empty((0, m), dtype=float)
    return a.reshape(-1, m)

def _to_min(points_max):
    # Our objectives are maximized; many indicators are defined for minimization.
    return -points_max if len(points_max) else points_max

def _dominates_min(a, b):
    return np.all(a <= b) and np.any(a < b)

def coverage_c_metric(A_min, B_min):
    # Fraction of B dominated by at least one point in A (minimization).
    if len(B_min) == 0:
        return np.nan
    if len(A_min) == 0:
        return 0.0
    dominated = 0
    for b in B_min:
        if any(_dominates_min(a, b) or np.allclose(a, b) for a in A_min):
            dominated += 1
    return dominated / len(B_min)

def additive_epsilon_indicator(A_min, B_min):
    # epsilon^+(A,B) = max_{b in B} min_{a in A} max_i (a_i - b_i)
    if len(A_min) == 0 or len(B_min) == 0:
        return np.nan
    eps_vals = []
    for b in B_min:
        eps_b = min(np.max(a - b) for a in A_min)
        eps_vals.append(eps_b)
    return float(np.max(eps_vals))

def spacing_metric(S_min):
    # Deb spacing-like metric with L1 nearest-neighbor distances.
    if len(S_min) <= 1:
        return np.nan
    d = []
    for i in range(len(S_min)):
        dist = np.sum(np.abs(S_min - S_min[i]), axis=1)
        dist[i] = np.inf
        d.append(np.min(dist))
    d = np.asarray(d, dtype=float)
    return float(np.sqrt(np.mean((d - np.mean(d)) ** 2)))

def igd_metric(approx_min, ref_min):
    # Mean distance from each reference point to nearest approximate point.
    if len(ref_min) == 0 or len(approx_min) == 0:
        return np.nan
    d = []
    for r in ref_min:
        d.append(np.min(np.linalg.norm(approx_min - r, axis=1)))
    return float(np.mean(d))

def _try_load_quantum_points(results_folder, m):
    # Best-effort loader to support different QAMOO output file names.
    candidates = [
        "non_dominated_points.npy",
        "non_dominated_objective_values.npy",
        "pareto_points.npy",
    ]
    for fn in candidates:
        p = os.path.join(results_folder, fn)
        if os.path.exists(p):
            return _as_points(np.load(p), m)

    # Fallback: objective values + non-dominated indices
    vals_p = os.path.join(results_folder, "objective_values.npy")
    nd_p = os.path.join(results_folder, "non_dominated_positions.npy")
    if os.path.exists(vals_p) and os.path.exists(nd_p):
        vals = _as_points(np.load(vals_p), m)
        idx = np.asarray(np.load(nd_p), dtype=int).ravel()
        idx = idx[(idx >= 0) & (idx < len(vals))]
        if len(idx):
            return vals[idx]
    return np.empty((0, m), dtype=float)

# Existing fronts from previous sections
cplex_pts = _as_points(classical_nd_points, num_objectives)
dpa_pts = _as_points(dpa_nd_points, num_objectives)
q_pts = _try_load_quantum_points(config.results_folder, num_objectives)

# Convert to minimization space for standard indicators
cplex_min = _to_min(cplex_pts)
dpa_min = _to_min(dpa_pts)
q_min = _to_min(q_pts)

# Exact reference: union of exact methods when available
exact_union_min = np.vstack([p for p in [cplex_min, dpa_min] if len(p)]) if (len(cplex_min) or len(dpa_min)) else np.empty((0, num_objectives))

# HV reference point from all available methods to keep scale comparable
all_min = np.vstack([p for p in [q_min, cplex_min, dpa_min] if len(p)]) if (len(q_min) or len(cplex_min) or len(dpa_min)) else np.empty((0, num_objectives))
ref_point = np.max(all_min, axis=0) + 1e-6 if len(all_min) else np.ones(num_objectives)
hv = HV(ref_point=ref_point)

methods = [
    ("QAMOO (Quantum)", q_min),
    ("CPLEX (Exact WS)", cplex_min),
    ("DPA (Exact)", dpa_min),
]

rows = []
for name, pts in methods:
    rows.append({
        "Method": name,
        "|PF|": int(len(pts)),
        "HV (higher better)": float(hv(pts)) if len(pts) else 0.0,
        "IGD to exact union (lower better)": igd_metric(pts, exact_union_min),
        "Spacing (lower better)": spacing_metric(pts),
        "C(method, exact)": coverage_c_metric(pts, exact_union_min),
        "C(exact, method)": coverage_c_metric(exact_union_min, pts),
        "eps+(method, exact)": additive_epsilon_indicator(pts, exact_union_min),
    })

df_metrics = pd.DataFrame(rows)

# Pretty display
df_view = df_metrics.copy()
for c in ["HV (higher better)", "IGD to exact union (lower better)", "Spacing (lower better)", "C(method, exact)", "C(exact, method)", "eps+(method, exact)"]:
    df_view[c] = df_view[c].map(lambda v: "N/A" if pd.isna(v) else f"{v:,.4f}")

styled = (
    df_view.style
    .hide(axis="index")
    .set_properties(**{"text-align": "center", "padding": "8px", "font-size": "11pt"})
    .set_table_styles([
        {"selector": "th", "props": "text-align:center; font-size:11pt; font-weight:bold; background-color:#eef4fb;"},
        {"selector": "td", "props": "border:1px solid #dddddd;"},
        {"selector": "table", "props": "border-collapse:collapse; width:100%;"},
    ])
    .set_caption("Pareto Quality Metrics (with exact-union reference)")
)

print("\n=== Multi-Metric Pareto Quality Summary ===")
print(f"QAMOO points: {len(q_pts)}, CPLEX points: {len(cplex_pts)}, DPA points: {len(dpa_pts)}")
if len(dpa_pts) == 0:
    print("Note: DPA points are empty, so DPA metrics will be degenerate (HV=0 and many N/A).")
display(styled)


