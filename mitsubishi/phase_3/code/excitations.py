"""Helpers for generating excitation/occupation configurations.

The functions here are used by the qscEOM implementation.
"""

from __future__ import annotations
import pennylane as qml

def inite(ref_occ: list[int] | int, orb: int) -> list[list[int]]:
    """Generate a list of occupation configurations.

    Parameters
    ----------
    ref_occ
        Occupied indices of the reference state, or number of active electrons.
    orb
        Number of spin-orbitals/qubits.

    Returns
    -------
    list[list[int]]
        A list of configurations, each represented as a list of occupied indices.
    """
    if isinstance(ref_occ, int):
        ref_occ = list(range(ref_occ))
        
    ref_occ_set = set(ref_occ)
    unoccupied = [i for i in range(orb) if i not in ref_occ_set]
    
    list1: list[list[int]] = []
    
    # Singles
    for p in ref_occ:
        for q in unoccupied:
            c = ref_occ_set.copy()
            c.remove(p)
            c.add(q)
            list1.append(sorted(list(c)))
            
    # Doubles
    for i, p in enumerate(ref_occ):
        for j in range(i + 1, len(ref_occ)):
            q = ref_occ[j]
            for a_idx, r in enumerate(unoccupied):
                for b_idx in range(a_idx + 1, len(unoccupied)):
                    s = unoccupied[b_idx]
                    c = ref_occ_set.copy()
                    c.remove(p)
                    c.remove(q)
                    c.add(r)
                    c.add(s)
                    list1.append(sorted(list(c)))

    return list1
