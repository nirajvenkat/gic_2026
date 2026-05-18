"""Helpers for generating excitation/occupation configurations.

The functions here are used by the qscEOM implementation.
"""

from __future__ import annotations
import pennylane as qml

def inite(elec: int, orb: int) -> list[list[int]]:
    """Generate a list of occupation configurations.

    Parameters
    ----------
    elec
        Number of active electrons.
    orb
        Number of spin-orbitals/qubits.

    Returns
    -------
    list[list[int]]
        A list of configurations, each represented as a list of occupied indices.
    """
    singles, doubles = qml.qchem.excitations(elec, orb)
    base_config = set(range(elec))
    
    list1: list[list[int]] = []
    
    for s in singles:
        c = base_config.copy()
        c.remove(s[0])
        c.add(s[1])
        list1.append(sorted(list(c)))
        
    for d in doubles:
        c = base_config.copy()
        c.remove(d[0])
        c.remove(d[1])
        c.add(d[2])
        c.add(d[3])
        list1.append(sorted(list(c)))

    return list1
