# ---
# jupyter:
#   jupytext:
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
# DOE GIC 2026 — Phase 2 
#
#         IEEE 14 --> 39 bus 
#          Warm-Start QAOA
# Objectives:
#
# C_VC  (Voltage Control,    Multiverse paper §II-B).
#
# C_R   (Investment Cost,     Multiverse paper §II-C)
#             
# Combined: α·C_VC + μ·C_R  (scalarised multi-objective)
#
# Algorithm : Warm-start QAOA (Egger et al. 2021) via Qiskit
# Sampler   : StatevectorSampler  (Samplomatic proxy, noiseless)

# %% [markdown]
# Pipeline
#   1. Load IEEE 14-bus in pandapower; identify candidate buses
#   2. Compute voltage-sensitivity V_n for each candidate bus
#   2b. Compute investment cost r_n for each candidate bus (C_R)
#   3. Build QUBO  Q  (α·C_VC + μ·C_R diagonals + budget coupling)
#   4. LP relaxation  →  warm-start angles  via arcsin(√x*)
#   5. Convert QUBO → Ising Hamiltonian (SparsePauliOp)
#   6. Run warm-start QAOA (QAOAAnsatz + COBYLA) with StatevectorSampler
#   7. Decode best bitstring; validate with pandapower
#   8. Classical greedy baseline for comparison

# %% [markdown]
# Libraries:

# %%
import warnings, logging, copy
warnings.filterwarnings("ignore")
logging.getLogger("pandapower").setLevel(logging.ERROR)

import numpy as np
import pandapower as pp
import pandapower.networks as pn
from scipy.optimize import minimize as sp_minimize, linprog

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.primitives import StatevectorSampler

# %%
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
K_BUDGET   = 5        # number of BESS units to place
LAMBDA     = 2.0      # budget-constraint soft penalty weight
ALPHA      = 1.0      # weight for C_VC objective (voltage control)
MU         = 0.5      # weight for C_R  objective (investment cost)
P_LAYERS   = 2        # QAOA circuit depth (reps)
SHOTS      = 1024     # shots per sampler call
DELTA_MVAR = 5.0      # MVAr injection for voltage-sensitivity probe
MAX_ITER   = 100      # COBYLA max iterations
SEED       = 42
PRUNE_TOP_N = 15      # keep only top-N candidates by V_n before QUBO
                      # 2^20 = 16 MB statevector → fits in RAM
                      # set to None to disable pruning (9-qubit IEEE 14-bus is fine)

np.random.seed(SEED)

print("=" * 60)
print("  DOE GIC 2026 | IEEE 14-bus | Warm-Start QAOA (C_VC + C_R)")
print("=" * 60)

# %% [markdown]
#  STEP 1: Load network & identify candidates

# %%
print("\n[1/8] Loading IEEE 14-bus network...")

net = pn.case39()
pp.runpp(net, algorithm="nr", calculate_voltage_angles=True)

slack_buses = net.ext_grid["bus"].tolist()
gen_buses   = net.gen["bus"].tolist()
excluded    = set(slack_buses + gen_buses)

candidate_buses = [b for b in net.bus.index.tolist() if b not in excluded]
n = len(candidate_buses)

vm_pu = net.res_bus["vm_pu"]
violations_base = ((vm_pu[candidate_buses] < 0.95).sum() +
                   (vm_pu[candidate_buses] > 1.05).sum())

print(f"      Total buses       : {len(net.bus)}")
print(f"      Candidate buses   : {n}  →  {n} qubits")
print(f"      Candidate bus IDs : {candidate_buses}")
print(f"      Base-case violations (0.95–1.05 pu): {violations_base}")
print(f"      Budget K          : {K_BUDGET} BESS units")

# %% [markdown]
# STEP 2: Voltage sensitivity V_n
#
# Inject DELTA_MVAR at each candidate bus.
#
# V_n  = reduction in total violation score.
# Normalise to [0, 1].

# %%
print("\n[2/8] Computing voltage sensitivities V_n...")

def violation_score(network):
    vm = network.res_bus["vm_pu"]
    return float(np.maximum(0, vm - 1.05).sum() +
                 np.maximum(0, 0.95 - vm).sum())

base_score = violation_score(net)
V_n = np.zeros(n)

for idx, bus in enumerate(candidate_buses):
    # Try both reactive injection (+) and absorption (-).
    # For overvoltage buses (> 1.05 pu) absorption helps; for undervoltage (+) helps.
    # V_n = best (maximum) improvement achievable at this bus.
    best_improvement = 0.0
    for q_sign in [1.0, -1.0]:
        trial = copy.deepcopy(net)
        pp.create_sgen(trial, bus=bus, p_mw=0.0, q_mvar=q_sign * DELTA_MVAR)
        try:
            pp.runpp(trial, algorithm="nr", calculate_voltage_angles=True)
            score_after = violation_score(trial)
            improvement = max(0.0, base_score - score_after)
            best_improvement = max(best_improvement, improvement)
        except Exception:
            pass
    V_n[idx] = best_improvement

# Normalise
v_max = V_n.max()
if v_max > 0:
    V_n = V_n / v_max

print(f"      Raw V_n (normalised, higher = better location):")
for idx, bus in enumerate(candidate_buses):
    print(f"        Bus {bus:2d}  →  V_n = {V_n[idx]:.4f}")

# ─────────────────────────────────────────────
# PRUNING: keep top-PRUNE_TOP_N by V_n
#   2^29 = 8 GiB statevector → OOM for IEEE 39-bus without pruning
#   2^20 = 16 MB → trivially fits
#   Physically justified: low-V_n buses contribute negligibly to C_VC
#   and will never appear in the optimal K-subset.
# ─────────────────────────────────────────────
if PRUNE_TOP_N is not None and n > PRUNE_TOP_N:
    top_idx      = np.argsort(V_n)[::-1][:PRUNE_TOP_N]
    top_idx_sort = np.sort(top_idx)          # preserve bus ordering
    candidate_buses = [candidate_buses[i] for i in top_idx_sort]
    V_n             = V_n[top_idx_sort]
    n               = len(candidate_buses)
    print(f"\n      [Pruned to top {PRUNE_TOP_N} by V_n → {n} qubits, "
          f"2^{n} = {2**n:,} amplitudes = {2**n * 16 / 1e6:.1f} MB statevector]")
    print(f"      Pruned candidate buses: {candidate_buses}")

# %% [markdown]
# STEP 2b: Investment cost r_n  (C_R objective)
#
# r_n = normalised active-load demand at each candidate bus.
#
# Higher load → more complex site → higher BESS installation cost.
#
# Buses with no connected load receive a small baseline cost (10 % of mean).
#
# Normalise to [0, 1] so C_R and C_VC are on comparable scales.

# %%
print("\n[2b/8] Computing investment costs r_n (C_R objective)...")

bus_loads: dict = {}
for _, row in net.load.iterrows():
    b = int(row["bus"])
    bus_loads[b] = bus_loads.get(b, 0.0) + float(row["p_mw"])

r_n = np.array([bus_loads.get(b, 0.0) for b in candidate_buses])

# Give zero-load buses a small non-zero baseline so they still carry cost
mean_load = r_n[r_n > 0].mean() if (r_n > 0).any() else 1.0
r_n = np.where(r_n == 0.0, mean_load * 0.10, r_n)

r_max = r_n.max()
if r_max > 0:
    r_n = r_n / r_max

print(f"      Investment costs r_n (normalised, higher = more expensive):")
for idx, bus in enumerate(candidate_buses):
    print(f"        Bus {bus:2d}  →  r_n = {r_n[idx]:.4f}")
print(f"      ALPHA (C_VC weight) = {ALPHA}   MU (C_R weight) = {MU}")

# %% [markdown]
# STEP 3: Build QUBO matrix Q  (multi-objective)
#
# C_VC diagonal   : Q[i,i] -= α.V_n[i]     (maximise voltage improvement)
#
# C_R  diagonal   : Q[i,i] += μ · r_n[i]     (minimise investment cost)
#
# Budget coupling : Q[i,i] += λ(1 - 2K)
#                    Q[i,j]  = 2λ   for i < j
#
#    Full QUBO: minimise x^T Q x
#              = α·C_VC_penalty + μ·C_R_cost + λ·budget_violation

# %%
print("\n[3/8] Building QUBO matrix (α·C_VC + μ·C_R + λ·budget)...")

Q = np.zeros((n, n))

# C_VC: diagonal negative V_n (minimising negative = maximising V_n)
for i in range(n):
    Q[i, i] -= ALPHA * V_n[i]

# C_R: diagonal positive r_n (minimising cost)
for i in range(n):
    Q[i, i] += MU * r_n[i]

# Budget constraint: λ(Σ x_i - K)² expanded
for i in range(n):
    Q[i, i] += LAMBDA * (1 - 2 * K_BUDGET)
for i in range(n):
    for j in range(i + 1, n):
        Q[i, j] += 2 * LAMBDA

print(f"      QUBO shape : {Q.shape}")
print(f"      Diagonal   : {np.diag(Q).round(4)}")
print(f"      Off-diag coupling (λ penalty): {LAMBDA * 2:.2f} per pair")
print(f"      C_VC contribution to diag: {(-ALPHA * V_n).round(4)}")
print(f"      C_R  contribution to diag: {(MU    * r_n).round(4)}")

# %% [markdown]
# STEP 4: LP relaxation → warm-start x*
#
# Relax x ∈ {0,1} → x ∈ [0,1]
#
# Objective: minimise (-α·V_n + μ·r_n)^T x   (combined scalarised cost;
#
#  LP ignores quadratic budget term but respects the linear tradeoff)
#
#  Constraint: Σ x_i = K

# %%
print("\n[4/8] LP relaxation for warm-start initialisation (α·C_VC + μ·C_R)...")

lp_c = -ALPHA * V_n + MU * r_n   # minimise this linear combination

lp_result = linprog(
    c=lp_c,                                     # combined objective
    A_eq=np.ones((1, n)),
    b_eq=[K_BUDGET],
    bounds=[(0.0, 1.0)] * n,
    method="highs"
)

if lp_result.success:
    x_star = np.clip(lp_result.x, 1e-6, 1 - 1e-6)   # avoid arcsin(0/1) edge cases
    print(f"      LP converged. x* = {x_star.round(4)}")
else:
    # Fallback: uniform
    x_star = np.full(n, K_BUDGET / n)
    print(f"      LP did not converge — using uniform warm start x* = {K_BUDGET}/{n}")

# Build warm-start circuit: Ry(2·arcsin(√x*_i))|0⟩ per qubit
ws_circ = QuantumCircuit(n)
for i in range(n):
    theta = 2.0 * np.arcsin(np.sqrt(x_star[i]))
    ws_circ.ry(theta, i)

print(f"      Warm-start Ry angles (rad): {[round(2*np.arcsin(np.sqrt(xi)), 4) for xi in x_star]}")

# %% [markdown]
# STEP 5: QUBO → Ising Hamiltonian
#
#  1. x_i = (1 - z_i) / 2
#  2. h[i] = -Q[i,i]/2 - Σ_{j≠i} Q[min,max]/4
#  3. J[i,j] = Q[i,j]/4   (upper triangle)
#  4. offset  = Σ Q[i,i]/2 + Σ_{i<j} Q[i,j]/4

# %%
print("\n[5/8] Converting QUBO → Ising Hamiltonian...")

h = np.zeros(n)
J = {}
offset = 0.0

for i in range(n):
    h[i]    -= Q[i, i] / 2.0
    offset  += Q[i, i] / 2.0

for i in range(n):
    for j in range(i + 1, n):
        if Q[i, j] != 0.0:
            J[(i, j)]  = Q[i, j] / 4.0
            h[i]      -= Q[i, j] / 4.0
            h[j]      -= Q[i, j] / 4.0
            offset    += Q[i, j] / 4.0

print(f"      h (linear bias) : {h.round(4)}")
print(f"      J pairs         : {len(J)}  ZZ terms")
print(f"      QUBO offset     : {offset:.4f}")

# %% [markdown]
# STEP 6: Manual QAOA circuit (gate-by-gate)
#
# WHY NOT QAOAAnsatz:
#
# -> QAOAAnsatz uses PauliEvolution internally, which builds the full 2^n × 2^n matrix of exp(-iγH) before applying it. 
#
# -> For n≥12 this exceeds available RAM (16 GiB at n=15). Building gates individually:
#
# — Rz for Z terms, CX-Rz-CX for ZZ terms — never requires a large matrix and scales to any qubit count within statevector RAM.
#
# 1. Cost layer:
#  
#  Z_i term  → Rz(2γ h_i, i)
#  
#  ZZ_{ij}   → CX(i,j); Rz(2γ J_{ij}, j); CX(i,j)
#
# 2. Mixer layer:
#
#  X_i term  → Rx(2β, i)   (transverse-field  mixer)

# %%
print(f"\n[6/8] Building QAOA circuit manually (p={P_LAYERS} layer(s))...")

gamma_vec = ParameterVector('γ', P_LAYERS)
beta_vec  = ParameterVector('β', P_LAYERS)

qaoa = QuantumCircuit(n)
qaoa.compose(ws_circ, inplace=True)   # warm-start initial state

for layer in range(P_LAYERS):
    g = gamma_vec[layer]
    b = beta_vec[layer]

    # — Cost unitary: exp(-i γ H_ising) —
    for i in range(n):                          # linear Z terms
        if abs(h[i]) > 1e-10:
            qaoa.rz(2.0 * g * h[i], i)
    for (i, j), jval in J.items():              # quadratic ZZ terms
        if abs(jval) > 1e-10:
            qaoa.cx(i, j)
            qaoa.rz(2.0 * g * jval, j)
            qaoa.cx(i, j)

    # — Mixer unitary: exp(-i β Σ X_i) —
    for i in range(n):
        qaoa.rx(2.0 * b, i)

qaoa.measure_all()

# Collect parameters: ParameterVector sorts β[0]<β[1]<...<γ[0]<γ[1]<...
gamma_params = list(gamma_vec)   # γ[0], γ[1], ...
beta_params  = list(beta_vec)    # β[0], β[1], ...

n_beta  = len(beta_params)
n_gamma = len(gamma_params)

print(f"      Circuit depth   : {qaoa.depth() - 1}  (excl. measure)")
print(f"      β params        : {[str(p) for p in beta_params]}")
print(f"      γ params        : {[str(p) for p in gamma_params]}")
print(f"      Two-qubit gates : {sum(1 for inst in qaoa.data if inst.operation.num_qubits == 2 and inst.operation.name != 'measure')}")

sampler = StatevectorSampler(seed=SEED)

def energy_from_counts(pub_result):
    """Compute ⟨H_ising⟩ from sampler bitstring counts."""
    counts = pub_result.data.meas.get_counts()
    total  = sum(counts.values())
    energy = 0.0
    for bitstring, cnt in counts.items():
        bits = [int(b) for b in reversed(bitstring)]
        z    = [1 - 2 * b for b in bits[:n]]
        e    = sum(h[i] * z[i] for i in range(n))
        e   += sum(jval * z[i] * z[j] for (i, j), jval in J.items())
        energy += (cnt / total) * e
    return energy

eval_count = [0]

def objective(theta_vec):
    """COBYLA objective: ⟨H_ising⟩ at given angles."""
    # theta_vec layout: [β[0], β[1], ..., γ[0], γ[1], ...]
    param_dict = {}
    for k, p in enumerate(beta_params):
        param_dict[p] = float(theta_vec[k])
    for k, p in enumerate(gamma_params):
        param_dict[p] = float(theta_vec[n_beta + k])

    bound_circ = qaoa.assign_parameters(param_dict)
    result = sampler.run([bound_circ], shots=SHOTS).result()
    e = energy_from_counts(result[0])
    eval_count[0] += 1
    if eval_count[0] % 20 == 0:
        print(f"        iter {eval_count[0]:4d}  ⟨H⟩ = {e:.6f}")
    return e

# Initial angles: β=π/4 (halfway through mixer), γ=0.1 (small phase kick)
x0 = np.array([np.pi / 4] * n_beta + [0.1] * n_gamma)

print(f"\n      Starting COBYLA optimisation (max_iter={MAX_ITER})...")
print(f"      Initial angles: β={x0[:n_beta].tolist()}, γ={x0[n_beta:].tolist()}")

opt_result = sp_minimize(
    objective,
    x0,
    method="COBYLA",
    options={"maxiter": MAX_ITER, "rhobeg": 0.5, "disp": False}
)

print(f"\n      Optimisation complete.")
print(f"      Final ⟨H_ising⟩ = {opt_result.fun:.6f}  (after {eval_count[0]} evals)")
print(f"      Optimal β = {opt_result.x[:n_beta].round(4).tolist()}")
print(f"      Optimal γ = {opt_result.x[n_beta:].round(4).tolist()}")

# %% [markdown]
#  STEP 7: Decode best bitstring → BESS placement

# %%
print("\n[7/8] Decoding optimal BESS placement...")

# Sample with optimal parameters
opt_param_dict = {}
for k, p in enumerate(beta_params):
    opt_param_dict[p] = opt_result.x[k]
for k, p in enumerate(gamma_params):
    opt_param_dict[p] = opt_result.x[n_beta + k]

bound_final = qaoa.assign_parameters(opt_param_dict)
final_result = sampler.run([bound_final], shots=SHOTS * 4).result()  # more shots for decoding
final_counts = final_result[0].data.meas.get_counts()

# Find bitstring with minimum QUBO energy (x^T Q x)
best_bs     = None
best_energy = np.inf

for bitstring, cnt in final_counts.items():
    bits = np.array([int(b) for b in reversed(bitstring)])[:n]
    e    = float(bits @ Q @ bits)
    if e < best_energy:
        best_energy = e
        best_bs     = bits.copy()

selected_indices = np.where(best_bs == 1)[0].tolist()
selected_buses   = [candidate_buses[i] for i in selected_indices]

print(f"      Best bitstring (qubit→bus): {dict(zip(candidate_buses, best_bs.tolist()))}")
print(f"      QUBO energy of solution  : {best_energy:.4f}")
print(f"      Selected buses           : {selected_buses}")
print(f"      Number of BESS placed    : {int(best_bs.sum())}  (target K={K_BUDGET})")

# %% [markdown]
# STEP 8: Classical greedy baseline

# %%
print("\n[8/8] Classical greedy baseline (top-K by V_n)...")

greedy_indices = np.argsort(V_n)[::-1][:K_BUDGET].tolist()
greedy_buses   = [candidate_buses[i] for i in greedy_indices]
print(f"      Greedy selected buses: {greedy_buses}")

# %% [markdown]
# VALIDATION: pandapower power flow

# %%
print("\n" + "─" * 60)
print("  VALIDATION — Power Flow with BESS Placement")
print("─" * 60)

def validate_placement(buses, label):
    net_v = copy.deepcopy(net)
    for bus in buses:
        # Negative MVAr = reactive absorption → corrects overvoltage
        pp.create_sgen(net_v, bus=bus, p_mw=0.0, q_mvar=-DELTA_MVAR)
    try:
        pp.runpp(net_v, algorithm="nr", calculate_voltage_angles=True)
        vm   = net_v.res_bus["vm_pu"]
        viol = int((vm < 0.95).sum() + (vm > 1.05).sum())
        v_min = float(vm.min())
        v_max = float(vm.max())
        idxs = [candidate_buses.index(b) for b in buses if b in candidate_buses]
        cvc  = float(sum(V_n[i] for i in idxs))   # higher = better
        cr   = float(sum(r_n[i] for i in idxs))   # lower  = better (cheaper)
        combined = ALPHA * cvc - MU * cr           # scalarised benefit (higher = better)
        print(f"  {label}")
        print(f"    Buses placed        : {buses}")
        print(f"    Violations          : {viol}  (base: {violations_base})")
        print(f"    V range             : [{v_min:.4f}, {v_max:.4f}] pu")
        print(f"    ΣV_n  (C_VC)       : {cvc:.4f}  (higher = better)")
        print(f"    Σr_n  (C_R)        : {cr:.4f}  (lower  = cheaper)")
        print(f"    α·ΣV_n − μ·Σr_n   : {combined:.4f}  (combined benefit)")
        return viol, cvc, cr
    except Exception as ex:
        print(f"  {label}  →  Power flow FAILED: {ex}")
        return None, None, None

v_base_all = int((net.res_bus["vm_pu"] < 0.95).sum() +
                 (net.res_bus["vm_pu"] > 1.05).sum())
print(f"  Base case violations: {v_base_all}")
print()

viol_qaoa,   cvc_qaoa,   cr_qaoa   = validate_placement(selected_buses, "QAOA (warm-start)")
print()
viol_greedy, cvc_greedy, cr_greedy = validate_placement(greedy_buses,   "Classical Greedy")

# %% [markdown]
# SUMMARY

# %%
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Qubits used    : {n}  (IEEE 14-bus candidate buses)")
print(f"  QAOA depth p   : {P_LAYERS}")
print(f"  Optimiser      : COBYLA  ({eval_count[0]} function evals)")
print(f"  Sampler        : StatevectorSampler  (Samplomatic proxy)")
print(f"  Shots          : {SHOTS} (optimisation)  /  {SHOTS*4} (decoding)")
print(f"  Objectives     : α·C_VC + μ·C_R  (α={ALPHA}, μ={MU})")
print()
print(f"  ┌──────────────────────┬──────────┬──────────┬──────────┐")
print(f"  │ Method               │Violations│ ΣV_n(↑)  │ Σr_n(↓)  │")
print(f"  ├──────────────────────┼──────────┼──────────┼──────────┤")
print(f"  │ Base case            │  {v_base_all:6d}  │    —     │    —     │")
if viol_qaoa is not None:
    print(f"  │ QAOA warm-start      │  {viol_qaoa:6d}  │  {cvc_qaoa:6.4f}  │  {cr_qaoa:6.4f}  │")
if viol_greedy is not None:
    print(f"  │ Classical greedy     │  {viol_greedy:6d}  │  {cvc_greedy:6.4f}  │  {cr_greedy:6.4f}  │")
print(f"  └──────────────────────┴──────────┴──────────┴──────────┘")
print()
print("  Next steps:")
print("   • Increase p (QAOA layers) to p=3,4 and compare convergence")
print("   • Tune α and μ weights via Pareto-front sweep")
print("   • Scale to IEEE 39-bus (primary DOE candidate, 29 qubits)")
print("   • Replace StatevectorSampler with Samplomatic for error mitigation")
print("=" * 60)


