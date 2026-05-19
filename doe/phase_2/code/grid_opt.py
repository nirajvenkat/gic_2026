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
#   kernelspec:
#     display_name: qi
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.13.9
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
import shutil
import pandapower.networks as nw
import pandapower.topology as top

from qamoo.configs.configs import ProblemSpecification, QAOAConfig
from qamoo.utils.utils import compute_hypervolume_progress
from qamoo.algorithms.qaoa import *
from qamoo.utils.data_structures import ProblemGraphBuilder

from qiskit_aer import AerSimulator


def _problem_set_dir(data_root: str, num_buses: int, num_objectives: int, num_swap_layers: int = 0, problem_id: int = 0):
    return os.path.join(
        data_root,
        'problems',
        f'{num_buses}q',
        f'problem_set_{num_buses}q_{num_swap_layers}s_{num_objectives}o_{problem_id}',
    )


def _prepare_benders_data_root(source_problem_dir: str, source_data_root: str, target_data_root: str, num_buses: int, num_objectives: int, objective_weights_file: str):
    target_problem_dir = _problem_set_dir(target_data_root, num_buses, num_objectives)
    os.makedirs(os.path.dirname(target_problem_dir), exist_ok=True)
    shutil.copytree(source_problem_dir, target_problem_dir, dirs_exist_ok=True)

    # Mirror support files required by QAMOO.
    os.makedirs(os.path.join(target_data_root, 'objective_weights'), exist_ok=True)
    if os.path.exists(objective_weights_file):
        shutil.copy2(objective_weights_file, os.path.join(target_data_root, 'objective_weights', os.path.basename(objective_weights_file)))

    # Use the same lower/upper bounds as the native run.
    for bounds_name, bounds_value in [('lower_bounds.json', [0.0, 0.0, 0.0]), ('upper_bounds.json', [100.0, 100.0, 100.0])]:
        with open(os.path.join(target_problem_dir, bounds_name), 'w') as f:
            json.dump(bounds_value, f)

    return target_problem_dir


def _run_benders_qamoo_preprocessor(num_qubits: int, problem_dir: str, num_objectives: int, max_iters: int = 10):
    from qamoo.algorithms.benders import classical_benders, cut_to_qubo_penalty

    print("Running classical Benders prototype to collect initial cuts...")
    res = classical_benders(num_qubits, problem_dir, num_objectives, max_iters=max_iters)
    cuts = res.get('cuts', [])
    print(f"Collected {len(cuts)} initial cuts")

    pg = ProblemGraphBuilder(num_qubits)
    base_graph = pg.deserialize(os.path.join(problem_dir, 'problem_graph_0.json'))

    for cut in cuts:
        pvec = cut_to_qubo_penalty(cut, num_qubits, penalty_weight=10.0)
        for i, w in enumerate(pvec.tolist()):
            if w == 0:
                continue
            if base_graph.has_edge(i, i):
                base_graph[i][i]['weight'] += float(w)
            else:
                base_graph.add_edge(i, i, weight=float(w))

    target_dir = os.path.join(problem_dir, 'benders_qamoo')
    os.makedirs(target_dir, exist_ok=True)

    pg.num_nodes = num_qubits
    pg.graph = base_graph
    pg.serialize('problem_graph_0', target_dir)

    for idx in range(1, num_objectives):
        src = os.path.join(problem_dir, f'problem_graph_{idx}.json')
        dst = os.path.join(target_dir, f'problem_graph_{idx}.json')
        if os.path.exists(src):
            with open(src, 'r') as fsrc, open(dst, 'w') as fdst:
                fdst.write(fsrc.read())

    for bounds_name, bounds_value in [('lower_bounds.json', [0.0, 0.0, 0.0]), ('upper_bounds.json', [100.0, 100.0, 100.0])]:
        src = os.path.join(problem_dir, bounds_name)
        dst = os.path.join(target_dir, bounds_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            with open(dst, 'w') as f:
                json.dump(bounds_value, f)

    print(f"Wrote modified problem set with penalties to {target_dir}")
    return target_dir


def _train_qaoa_parameters(problem_folder: str, p_layers: int, num_objectives: int):
    objective_graphs = [
        ProblemGraphBuilder.deserialize(os.path.join(problem_folder, f'problem_graph_{idx}.json'))
        for idx in range(num_objectives)
    ]
    graph_edges = list(objective_graphs[0].edges())

    combined_edges = []
    for u, v in graph_edges:
        combined_weight = sum(pg[u][v]['weight'] for pg in objective_graphs) / float(num_objectives)
        combined_edges.append((u, v, combined_weight))

    combined_graph = nx.Graph()
    combined_graph.add_weighted_edges_from(combined_edges)
    ising, _ = Maxcut(combined_graph).to_quadratic_program().to_ising()

    ansatz = QAOAAnsatz(ising, reps=p_layers)
    estimator = AerEstimator(
        backend_options={"method": "matrix_product_state", "matrix_product_state_max_bond_dimension": 20},
        run_options={"shots": 1024},
    )

    def cost_func(params):
        bound_ansatz = ansatz.assign_parameters(params)
        job = estimator.run([bound_ansatz], [ising])
        return -job.result().values[0]

    np.random.seed(42)
    init_guess = np.random.uniform(-np.pi, np.pi, 2 * p_layers)
    res = minimize(cost_func, init_guess, method='COBYLA', options={'maxiter': 60})
    trained_params = res.x.tolist()

    with open(os.path.join(problem_folder, 'optimal_parameters.json'), 'w') as f:
        json.dump({str(p_layers): trained_params}, f)

    return trained_params


def _run_qamoo_workflow(data_root: str, run_id: str, num_buses: int, num_objectives: int, num_samples: int, p_layers: int):
    backend = AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension=20, max_parallel_threads=1)
    backend.options.use_fractional_gates = False

    problem = ProblemSpecification()
    problem.data_folder = data_root
    problem.num_qubits = num_buses
    problem.num_objectives = num_objectives
    problem.num_swap_layers = 0
    problem.problem_id = 0

    _train_qaoa_parameters(problem.problem_folder, p_layers, num_objectives)

    config = QAOAConfig()
    config.p = p_layers
    config.num_samples = num_samples
    config.shots = 4096
    config.objective_weights_id = 0
    config.backend_name = backend.name
    config.initial_layout = None
    config.run_id = run_id
    config.rep_delay = 0.0001
    config.problem = problem

    print(f"Preparing parameterized QAOA circuits for {run_id}...")
    prepare_qaoa_circuits(config, backend, overwrite_results=True)

    print(f"Transpiling QAOA circuits for {run_id}...")
    transpile_qaoa_circuits_parametrized(config, backend)

    print(f"Executing sampled circuits for {run_id}...")
    batch_execute_qaoa_circuits_parametrized([config], backend)

    return config

# Feature toggles
# If True, attempt to use IBM Quantum runtime for QAOA execution. Defaults
# to False (use AerSimulator). Enabling requires IBM account/runtime setup.
USE_QPU = False

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

# Shared experiment settings for both the native and Benders-QAMOO runs.
num_samples = 10
num_objectives = 3
p_layers = 3

# Run the Benders-QAMOO preprocessor which writes a modified problem set
# with simple Benders cuts converted to QUBO penalties. This is kept inline
# so the notebook remains self-contained.
try:
    _run_benders_qamoo_preprocessor(num_buses, problem_dir, num_objectives, max_iters=5)
    print("Benders-QAMOO preprocessing complete — modified problem set written under <original>/benders_qamoo/")
except Exception as e:
    print(f"Benders-QAMOO preprocessing failed: {e}")

# %% [markdown]
# ## 2. Generate Configuration Placeholders & Train QAOA
# QAMOO requires valid `optimal_parameters.json` and objective weights distributions for Multi-Objective optimization. We use COBYLA to optimize a scalarized representation of the 3 objectives.

# %%
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

# Mirror the Benders-modified problem set and support files into a second data root
# so the same QAMOO workflow can be executed on the Benders-QAMOO instance.
benders_data_root = './data_benders/'
benders_source_problem_dir = os.path.join(problem_dir, 'benders_qamoo')
benders_problem_dir = _prepare_benders_data_root(
    benders_source_problem_dir,
    './data/',
    benders_data_root,
    num_buses,
    num_objectives,
    weights_file,
)

print(f"Prepared Benders-QAMOO data root at {benders_problem_dir}")

# %% [markdown]
# ## 3. QAOA Setup and Execution

# %%
native_data_root = './data/'
benders_data_root = './data_benders/'

print("Running native QAMOO workflow...")
config = _run_qamoo_workflow(
    native_data_root,
    'siting_run_native',
    num_buses,
    num_objectives,
    num_samples,
    p_layers,
)
problem = config.problem

print("Running Benders-QAMOO workflow...")
benders_config = _run_qamoo_workflow(
    benders_data_root,
    'siting_run_benders',
    num_buses,
    num_objectives,
    num_samples,
    p_layers,
)
benders_problem = benders_config.problem

# %% [markdown]
# ## 4. Hypervolume Analysis
# Plot the progression of the Hypervolume metric over the simulated sample runs.

# %%
step_size = max(1, config.total_num_samples // 10)
steps = range(0, config.total_num_samples + 1, step_size)

try:
    compute_hypervolume_progress(problem.problem_folder, config.results_folder, steps)
    compute_hypervolume_progress(benders_problem.problem_folder, benders_config.results_folder, steps)

    native_x, native_y = config.progress_x_y()
    benders_x, benders_y = benders_config.progress_x_y()

    print(f"Max Hypervolume (native QAMOO): {max(native_y):.2f}")
    print(f"Max Hypervolume (Benders-QAMOO): {max(benders_y):.2f}")

    plt.figure(figsize=(8, 5))
    plt.plot(native_x, native_y, marker='o', label='Native QAMOO')
    plt.plot(benders_x, benders_y, marker='s', label='Benders-QAMOO')
    plt.title(f'Hypervolume Progress - Siting Optimization ({net.name})')
    plt.xlabel('Samples Evaluated')
    plt.ylabel('Hypervolume')
    plt.grid(True)
    plt.legend()
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
# ## 6. Configuration Visualization
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
# ## 7. Multi-Metric Pareto Quality Analysis
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
benders_q_pts = _try_load_quantum_points(benders_config.results_folder, num_objectives)

# Convert to minimization space for standard indicators
cplex_min = _to_min(cplex_pts)
dpa_min = _to_min(dpa_pts)
q_min = _to_min(q_pts)
benders_q_min = _to_min(benders_q_pts)

# Exact reference: union of exact methods when available
exact_union_min = np.vstack([p for p in [cplex_min, dpa_min] if len(p)]) if (len(cplex_min) or len(dpa_min)) else np.empty((0, num_objectives))

# HV reference point from all available methods to keep scale comparable
all_min = np.vstack([p for p in [q_min, benders_q_min, cplex_min, dpa_min] if len(p)]) if (len(q_min) or len(benders_q_min) or len(cplex_min) or len(dpa_min)) else np.empty((0, num_objectives))
ref_point = np.max(all_min, axis=0) + 1e-6 if len(all_min) else np.ones(num_objectives)
hv = HV(ref_point=ref_point)

methods = [
    ("QAMOO (Quantum)", q_min),
    ("Benders-QAMOO (Quantum)", benders_q_min),
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
print(f"QAMOO points: {len(q_pts)}, Benders-QAMOO points: {len(benders_q_pts)}, CPLEX points: {len(cplex_pts)}, DPA points: {len(dpa_pts)}")
if len(dpa_pts) == 0:
    print("Note: DPA points are empty, so DPA metrics will be degenerate (HV=0 and many N/A).")
display(styled)


