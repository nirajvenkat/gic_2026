============================================================

#### **PANDAPOWER VOLTAGE ANALYSIS**

#### **DOE GIC 2026 — Battery Siting Problem**

#### **IEEE 33-Bus Distribution Network**

============================================================



Network loaded: 33 buses, 37 lines



**=== BASE CASE VOLTAGES (per unit) ===**

Substation (Bus 0):  1.0000 pu

Minimum voltage:     0.9131 pu at bus 17

Maximum voltage:     1.0000 pu at bus 0

Buses below 0.95 pu: 21



**=== VOLTAGE VIOLATIONS ===**

Buses below 0.95 pu: \[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29, 30, 31, 32]

Buses above 1.05 pu: \[]

Total violations: 21



Candidate battery locations: \[1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31]



**=== COMPUTING VOLTAGE SENSITIVITY MATRIX ===**

Testing 11 candidate locations

Network has 33 buses

Injection size: 0.5 MW per test

&#x20; Completed 5/11 locations

&#x20; Completed 10/11 locations



Sensitivity matrix shape: (11, 33)

&#x20; = 11 candidate locations x 33 network buses



**=== TOP 5 BATTERY LOCATIONS BY VOLTAGE IMPACT ===**

Rank   Bus      Voltage Score   Meaning

\-------------------------------------------------------

1      16       0.6847          Improves 21 violated nodes

2      13       0.6398          Improves 21 violated nodes

3      10       0.5399          Improves 21 violated nodes

4      31       0.4614          Improves 21 violated nodes

5      28       0.4209          Improves 21 violated nodes



**=== QUBO VOLTAGE OBJECTIVE VALUES ===**

Base case total voltage deviation: 0.117094 pu²



Per-location improvement score (diagonal of Q\_V):

&#x20; Bus  16: improvement = 0.075994 pu²

&#x20; Bus  13: improvement = 0.073814 pu²

&#x20; Bus  10: improvement = 0.066113 pu²

&#x20; Bus  31: improvement = 0.057807 pu²

&#x20; Bus  28: improvement = 0.054092 pu²





============================================================

##### **SUMMARY FOR DOE SUBMISSION:**

* &#x20; Network: IEEE 33-bus, 12.66 kV distribution
* &#x20; Voltage violations in base case: 21 buses
* &#x20; Candidate battery locations tested: 11
* &#x20; Sensitivity matrix: (11, 33)
* &#x20; Best battery bus by voltage score: 16



**This sensitivity matrix feeds directly into Q\_V in our three-objective QUBO formulation.**

