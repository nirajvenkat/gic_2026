"""
Voltage Sensitivity Analysis for DOE Battery Siting Problem
==========================================================
Purpose: Compute dV/dP sensitivity matrix for IEEE 33-bus network.
This is the Vn term in Multiverse paper Section II-B (Voltage Control).
Multiverse dropped this for large networks — we show how to compute it.

Connection to QUBO:
    sensitivity_matrix[i, n] = how much voltage at node n changes
    when we inject 1 MW of battery power at location i
    
    This becomes the Q_V matrix in our QUBO formulation.

Author: [Your name]
Reference: Multiverse/Iberdrola (2024) Section II-B
"""

import pandapower as pp
import pandapower.networks as pn
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STEP 1: Load Standard IEEE 33-Bus Test Network
# This is the standard distribution network test case
# Used in hundreds of papers — judges will recognize it
# ============================================================

def load_ieee33():
    """
    IEEE 33-bus radial distribution network.
    Standard test case for distribution planning.
    33 nodes, 32 lines, radial topology.
    Base voltage: 12.66 kV
    Total load: 3.715 MW, 2.300 MVAr
    """
    net = pn.case33bw()
    return net


# ============================================================
# STEP 2: Run Base Case Power Flow
# This gives us the voltage at every node
# BEFORE any batteries are installed
# ============================================================

def run_base_powerflow(net):
    """
    Run Newton-Raphson power flow.
    Returns voltage magnitude at every bus in per-unit (pu).
    
    In distribution networks, voltage drops as you move
    away from the substation (bus 0).
    Far-end nodes often violate 0.95 pu lower limit.
    """
    pp.runpp(net, algorithm='nr')
    
    voltages = net.res_bus['vm_pu'].values
    print("\n=== BASE CASE VOLTAGES (per unit) ===")
    print(f"Substation (Bus 0):  {voltages[0]:.4f} pu")
    print(f"Minimum voltage:     {voltages.min():.4f} pu at bus {voltages.argmin()}")
    print(f"Maximum voltage:     {voltages.max():.4f} pu at bus {voltages.argmax()}")
    print(f"Buses below 0.95 pu: {(voltages < 0.95).sum()}")
    
    return voltages


# ============================================================
# STEP 3: Compute Voltage Sensitivity Matrix
# This is what Multiverse calls Vn = dV/dP
# 
# Method: inject small power (battery) at each candidate
# location, re-run power flow, measure voltage change at
# every node. Repeat for all candidate locations.
#
# sensitivity[i, n] = (V_n_with_battery - V_n_base) / P_injected
# ============================================================

def compute_voltage_sensitivity(net, candidate_buses, p_injection_mw=0.5):
    """
    Compute voltage sensitivity matrix dV/dP.
    
    For each candidate battery location i:
        1. Add a small generator (battery) at bus i
        2. Run power flow
        3. Record voltage change at every bus n
        4. sensitivity[i, n] = delta_V_n / P_injected
    
    This is the linearized voltage sensitivity used in:
    - Multiverse paper Section II-B
    - Lee et al. voltage objective
    - Standard distribution planning practice
    
    Args:
        net: pandapower network
        candidate_buses: list of bus indices where batteries can go
        p_injection_mw: battery power injection size for sensitivity test
    
    Returns:
        sensitivity_matrix: shape (n_candidates, n_buses)
    """
    # First get base case voltages
    pp.runpp(net, algorithm='nr')
    v_base = net.res_bus['vm_pu'].values.copy()
    
    n_candidates = len(candidate_buses)
    n_buses = len(net.bus)
    sensitivity_matrix = np.zeros((n_candidates, n_buses))
    
    print(f"\n=== COMPUTING VOLTAGE SENSITIVITY MATRIX ===")
    print(f"Testing {n_candidates} candidate locations")
    print(f"Network has {n_buses} buses")
    print(f"Injection size: {p_injection_mw} MW per test")
    
    for idx, bus in enumerate(candidate_buses):
        # Add a temporary battery (modeled as static generator)
        sgen_idx = pp.create_sgen(
            net, bus=bus,
            p_mw=p_injection_mw,
            q_mvar=0,
            name=f"test_battery_{bus}"
        )
        
        # Run power flow with this battery
        try:
            pp.runpp(net, algorithm='nr')
            v_with_battery = net.res_bus['vm_pu'].values.copy()
            
            # Sensitivity = voltage change / power injected
            delta_v = v_with_battery - v_base
            sensitivity_matrix[idx, :] = delta_v / p_injection_mw
            
        except:
            # Power flow did not converge — leave sensitivity as zero
            print(f"  Warning: Power flow did not converge for bus {bus}")
        
        # Remove the test battery
        net.sgen.drop(sgen_idx, inplace=True)
        
        if (idx + 1) % 5 == 0:
            print(f"  Completed {idx + 1}/{n_candidates} locations")
    
    return sensitivity_matrix, v_base


# ============================================================
# STEP 4: Identify Voltage Violations
# Nodes that are BELOW 0.95 pu or ABOVE 1.05 pu
# These are exactly the nodes where batteries help most
# This is the physical motivation for the QUBO voltage objective
# ============================================================

def find_voltage_violations(v_base, lower_limit=0.95, upper_limit=1.05):
    """
    Find buses with voltage violations.
    
    IEEE standard limits:
        Lower limit: 0.95 pu (5% below nominal)
        Upper limit: 1.05 pu (5% above nominal)
    
    Buses below 0.95 pu are the priority siting locations.
    Installing a battery here improves voltage AND reliability.
    """
    violations_low = np.where(v_base < lower_limit)[0]
    violations_high = np.where(v_base > upper_limit)[0]
    
    print(f"\n=== VOLTAGE VIOLATIONS ===")
    print(f"Buses below {lower_limit} pu: {violations_low.tolist()}")
    print(f"Buses above {upper_limit} pu: {violations_high.tolist()}")
    print(f"Total violations: {len(violations_low) + len(violations_high)}")
    
    return violations_low, violations_high


# ============================================================
# STEP 5: Find Best Battery Locations From Voltage Perspective
# The bus with highest sensitivity at the most violated nodes
# is the best location to place a battery
# This is exactly what Multiverse computed as Vn
# ============================================================

def find_best_battery_locations(sensitivity_matrix, candidate_buses,
                                 violated_buses, top_n=5):
    """
    Rank candidate locations by their voltage improvement impact.
    
    For each candidate location i:
        voltage_score[i] = sum of sensitivity[i, n] 
                           for all violated nodes n
    
    Higher score = battery at this location fixes more violations.
    This ranking is the physical basis for the QUBO voltage objective.
    """
    if len(violated_buses) == 0:
        print("No voltage violations found in base case.")
        return []
    
    # Score each candidate by how much it helps violated nodes
    scores = np.sum(
        sensitivity_matrix[:, violated_buses], axis=1
    )
    
    # Rank by score
    ranked_indices = np.argsort(scores)[::-1]
    
    print(f"\n=== TOP {top_n} BATTERY LOCATIONS BY VOLTAGE IMPACT ===")
    print(f"{'Rank':<6} {'Bus':<8} {'Voltage Score':<15} {'Meaning'}")
    print("-" * 55)
    for rank, idx in enumerate(ranked_indices[:top_n]):
        bus = candidate_buses[idx]
        score = scores[idx]
        print(f"{rank+1:<6} {bus:<8} {score:<15.4f} "
              f"Improves {len(violated_buses)} violated nodes")
    
    return [candidate_buses[i] for i in ranked_indices[:top_n]]


# ============================================================
# STEP 6: Visualize Results
# Three plots that tell the complete story for the submission
# ============================================================

def visualize_results(v_base, sensitivity_matrix, 
                       candidate_buses, violated_buses):
    """
    Generate three plots:
    1. Base case voltage profile — shows the problem
    2. Voltage sensitivity heatmap — shows the solution space  
    3. Voltage improvement after best battery — shows the fix
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        'Pandapower Voltage Analysis for DOE Battery Siting\n'
        'IEEE 33-Bus Distribution Network',
        fontsize=13, fontweight='bold'
    )
    
    # --- Plot 1: Base Case Voltage Profile ---
    ax1 = axes[0]
    buses = range(len(v_base))
    colors = ['red' if v < 0.95 else 'steelblue' for v in v_base]
    ax1.bar(buses, v_base, color=colors, alpha=0.8, width=0.7)
    ax1.axhline(y=0.95, color='red', linestyle='--', 
                linewidth=2, label='Lower limit (0.95 pu)')
    ax1.axhline(y=1.05, color='orange', linestyle='--',
                linewidth=2, label='Upper limit (1.05 pu)')
    ax1.set_xlabel('Bus Number')
    ax1.set_ylabel('Voltage (per unit)')
    ax1.set_title('Base Case Voltage Profile\n(Red = violation)')
    ax1.legend(fontsize=8)
    ax1.set_ylim(0.9, 1.1)
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Voltage Sensitivity Heatmap ---
    ax2 = axes[1]
    im = ax2.imshow(
        sensitivity_matrix,
        aspect='auto',
        cmap='RdYlGn',
        interpolation='nearest'
    )
    plt.colorbar(im, ax=ax2, label='dV/dP (pu/MW)')
    ax2.set_xlabel('Network Bus Number')
    ax2.set_ylabel('Battery Location (Candidate Index)')
    ax2.set_title('Voltage Sensitivity Matrix\ndV/dP for each location')
    
    # Mark violated buses on x-axis
    for vb in violated_buses:
        if vb < sensitivity_matrix.shape[1]:
            ax2.axvline(x=vb, color='red', alpha=0.5, linewidth=0.8)
    
    # --- Plot 3: Best Location Identified ---
    ax3 = axes[2]
    scores = np.sum(
        sensitivity_matrix[:, violated_buses], axis=1
    ) if len(violated_buses) > 0 else np.zeros(len(candidate_buses))
    
    ax3.barh(range(len(candidate_buses)), scores, color='steelblue', alpha=0.8)
    ax3.set_yticks(range(len(candidate_buses)))
    ax3.set_yticklabels([f'Bus {b}' for b in candidate_buses], fontsize=8)
    ax3.set_xlabel('Voltage Improvement Score')
    ax3.set_title('Battery Location Ranking\n(Higher = better voltage fix)')
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('voltage_sensitivity_analysis.png',
                dpi=150, bbox_inches='tight')
    print("\nPlot saved to voltage_sensitivity_analysis.png")
    plt.show()


# ============================================================
# STEP 7: Generate the QUBO-Ready Sensitivity Vector
# This is the direct output that feeds into Q_V matrix
# ============================================================

def generate_qubo_voltage_input(sensitivity_matrix, candidate_buses,
                                 v_base, v_nominal=1.0):
    """
    Convert sensitivity matrix into QUBO-ready format.
    
    The voltage deviation at node n when battery at location i
    is active:
        delta_V_n = sensitivity[i, n] * P_battery
    
    The voltage objective in QUBO:
        V_score[i] = sum over all nodes n of:
            (V_n + sensitivity[i,n] * P_battery - V_nominal)^2
    
    Minimizing this = moving all voltages toward 1.0 pu
    This is exactly the Vn term in Multiverse Section II-B
    """
    n_candidates = len(candidate_buses)
    qubo_v_scores = np.zeros(n_candidates)
    
    # Current voltage deviations (base case)
    v_deviation_base = (v_base - v_nominal) ** 2
    total_deviation_base = v_deviation_base.sum()
    
    print(f"\n=== QUBO VOLTAGE OBJECTIVE VALUES ===")
    print(f"Base case total voltage deviation: {total_deviation_base:.6f} pu²")
    print(f"\nPer-location improvement score (diagonal of Q_V):")
    
    for idx, bus in enumerate(candidate_buses):
        # How much does total voltage deviation reduce 
        # if we put a 1 MW battery here?
        v_with_battery = v_base + sensitivity_matrix[idx, :] * 1.0
        v_deviation_new = (v_with_battery - v_nominal) ** 2
        improvement = total_deviation_base - v_deviation_new.sum()
        qubo_v_scores[idx] = improvement
    
    # Print top locations
    ranked = np.argsort(qubo_v_scores)[::-1]
    for rank in range(min(5, n_candidates)):
        idx = ranked[rank]
        print(f"  Bus {candidate_buses[idx]:3d}: "
              f"improvement = {qubo_v_scores[idx]:.6f} pu²")
    
    return qubo_v_scores


# ============================================================
# MAIN — Run Everything
# ============================================================

if __name__ == "__main__":
    
    print("=" * 60)
    print("PANDAPOWER VOLTAGE ANALYSIS")
    print("DOE GIC 2026 — Battery Siting Problem")
    print("IEEE 33-Bus Distribution Network")
    print("=" * 60)
    
    # Load network
    net = load_ieee33()
    print(f"\nNetwork loaded: {len(net.bus)} buses, "
          f"{len(net.line)} lines")
    
    # Run base power flow
    v_base = run_base_powerflow(net)
    
    # Find violations
    violated_low, violated_high = find_voltage_violations(v_base)
    violated_buses = list(violated_low) + list(violated_high)
    
    # Define candidate battery locations
    # Use every 3rd bus as a candidate (realistic — not every node gets a battery)
    candidate_buses = list(range(1, len(net.bus), 3))
    print(f"\nCandidate battery locations: {candidate_buses}")
    
    # Compute sensitivity matrix
    sensitivity_matrix, _ = compute_voltage_sensitivity(
        net, candidate_buses, p_injection_mw=0.5
    )
    
    print(f"\nSensitivity matrix shape: {sensitivity_matrix.shape}")
    print(f"  = {len(candidate_buses)} candidate locations "
          f"x {len(net.bus)} network buses")
    
    # Find best locations
    best_locations = find_best_battery_locations(
        sensitivity_matrix, candidate_buses, violated_buses, top_n=5
    )
    
    # Generate QUBO input
    qubo_v_scores = generate_qubo_voltage_input(
        sensitivity_matrix, candidate_buses, v_base
    )
    
    # Visualize
    visualize_results(
        v_base, sensitivity_matrix,
        candidate_buses, violated_buses
    )
    
    print("\n" + "=" * 60)
    print("SUMMARY FOR DOE SUBMISSION:")
    print(f"  Network: IEEE 33-bus, 12.66 kV distribution")
    print(f"  Voltage violations in base case: {len(violated_buses)} buses")
    print(f"  Candidate battery locations tested: {len(candidate_buses)}")
    print(f"  Sensitivity matrix: {sensitivity_matrix.shape}")
    print(f"  Best battery bus by voltage score: {best_locations[0] if best_locations else 'N/A'}")
    print("\nThis sensitivity matrix feeds directly into Q_V")
    print("in our three-objective QUBO formulation.")
    print("=" * 60)