# Requirements of DoE:

1. Multi-Objective Optimization: Optimize across resilience, investment cost, and voltage performance simultaneously — not just one goal.
2. QUBO Formulation Suitable for Quantum Computing: Develop mappings to QUBO or variational optimization frameworks.
3. Handle Constraints Without Breaking the QUBO: Account for transmission constraints, contingency requirements, and voltage stability.
4. Scale to Realistic Grid Sizes: Evaluate thousands of potential infrastructure configurations across diverse operating conditions.
5. Benchmark Quantum Against Classical: Benchmark hybrid quantum approaches against established classical planning solvers.



### Most Important Issues regarding DoE:

1. **AI data centers have exploded electricity demand:** The US power grid was designed for slow, predictable demand growth of about 1% per year. AI data centers now need 10-20x that growth rate in specific regions.
2. **Major grid failures proved the old planning approach is broken:** The Texas 2021 freeze, California wildfires, and Hurricane Ian showed that centralized grid planning without resilience optimization costs lives and billions of dollars.
3. **Quantum hardware crossed a practical threshold.**

## 

## Most Useful things extracted from each paper:


1.  IBM/Kotil et al. — Quantum Approximate Multi-Objective Optimization
---

We adopt the multi-objective QUBO formulation of Lee et al. (2025), which encodes three simultaneous objectives — cost, reliability, and constraint satisfaction — as separate QUBO matrices and combines them via weighted scalarization. We extend their normalized unbalanced penalty method to encode voltage-deviation constraints and adapt their α-Expansion decomposition algorithm to handle the siting decision space of the DOE problem. We port this framework from D-Wave to IBM Qiskit using QAOA as the subproblem solver, enabling execution on gate-based quantum hardware and direct comparison with the Multiverse/Iberdrola pilot results.



### 2\. Quantum Annealing-Infused Microgrids Formation

Only paper with real quantum hardware solving a microgrid formation problem directly comparable to DOE's siting question.

Gap: Reactive (after disaster) not proactive (where to build before disaster). No voltage, no cost, no multi-year horizon.

### 3\. Lee et al. — Multi-Objective Quantum Power System Redispatch

Most technically rigorous MOO-QUBO formulation for power systems. The normalized penalty method and α-Expansion are directly portable to the DOE siting problem.

Gap: Wrong problem type (scheduling not siting), no voltage, no multi-year planning, uses D-Wave not Qiskit.

### 4\. Multiverse Computing + Iberdrola Pilot - Industry Report, July 2024

Only paper covering all three DOE objectives simultaneously. Real industrial validation. 

Gap: Proprietary platform (Singularity), not open-source. No Qiskit. No multi-year uncertainty. No standard IEEE test case. Results not fully reproducible. No Pareto front analysis.

### 5\. Blenninger et al. — Q-GRID: Quantum Optimization for the Future Energy Grid

Shows exactly where quantum wins — classical becomes impractical beyond 30 nodes. Uses IEEE test cases. Open source.

Gap: No cost, no voltage, single objective, no investment decisions.

### 6\. Morstyn \& Wang — Opportunities for Quantum Computing within Net-Zero Power System Optimization

Most comprehensive strategic review. Maps every quantum algorithm to every power system problem. Shows QA can solve PV placement/sizing with voltage constraints.

Gap: Review paper, no new experiments. Identifies opportunities but doesn't implement them.

### 7\. PNNL — Review of Quantum Computing Technologies in Power System Optimization

Most rigorous benchmark methodology.

**Sections to read:** Table 2 (QC algorithms for unit commitment) + Table 3 (QC for optimal power flow) + Section on "Intelligent Switching and Network Optimization"











|**DOE Requirement**|**Best Existing Paper**|**What's Missing**|
|-|-|-|
|Where to site storage|Nikmehr 2025 (post-disaster formation)|Pre-disaster strategic siting. Multi-year investment horizon|
|How to size storage|Morstyn 2024 (conceptual only, 3 PV sites)| PV sites)Full QUBO for discrete sizing decisions with real capacity options|
|Minimize investment cost|Multiverse/Iberdrola (proprietary, not reproducible)|Open Qiskit implementation with transparent formulation|
|Maximize reliability/resilience|Lee et al. 2026 (line overloads only)|Full outage modeling, islanding capability, N-1 contingency|
|Maintain voltage performance|Morstyn 2024 (voltage limits in QA, 3 nodes only)|Nodal voltage as a QUBO objective at distribution network scale|
|**Multi-year planning under uncertainty**|**Nobody**|**This is completely open. Zero papers address it with quantum**|
|Gate-based IBM Qiskit + energy|QAMOO (abstract objectives only)|Apply to concrete energy siting objectives|
|Honest benchmark vs MILP at grid scale|Q-GRID (up to 30 nodes only)|IEEE 24/57-bus standard test cases|
|All three DOE objectives simultaneously|Multiverse pilot (proprietary)|Open, reproducible, Qiskit-based implementation|





### Our Proposed Contribution — The Research Gap We Fill



No existing paper simultaneously addresses:

1. Pre-disaster strategic siting (WHERE to build)
2. Discrete sizing decisions (HOW BIG to build)  
3. All three DOE objectives: cost + reliability + voltage
4. Multi-year planning under AI-driven load uncertainty
5. Open Qiskit implementation on IBM gate-based hardware



### Our Approach (Three-Layer Architecture)

#### Layer 1 — Problem Formulation:

1. &#x20; Adapt Lee et al. MOO-QUBO structure for siting decisions
2. &#x20; Add voltage deviation as fourth QUBO matrix Q\_V
3. &#x20; Use discrete battery size options as one-hot encoded variables



#### Layer 2 — Quantum Solver:

1. &#x20; Port α-Expansion from D-Wave to Qiskit QAOA subproblem solver
2. &#x20; Apply QAMOO parameter transfer to eliminate training overhead



#### Layer 3 — Benchmark:

1. &#x20; Test on IEEE 24-bus and 57-bus standard systems
2. &#x20; Compare vs MILP (Gurobi) and PSO classical baselines
3. &#x20; Report Pareto hypervolume, solution diversity, runtime



#### Unique Claim

**First open-source, Qiskit-based, three-objective QUBO framework for strategic energy storage siting and sizing that handles voltage, reliability, and investment cost simultaneously, validated on IEEE standard test cases with multi-scenario AI load growth uncertainty.**



