"""qscEOM implementation.

This module was previously stored under ``QCANT/tests`` as an experiment/script.
It has been promoted into the package so it can be imported and documented.

Notes
-----
This code depends on optional scientific Python packages (e.g. PennyLane).
Imports are intentionally performed inside functions so that importing QCANT
does not require these optional dependencies.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache
import multiprocessing as mp
from typing import Any, Mapping, Optional, Sequence, Tuple

from excitations import inite


_FERMIONIC_ANSATZ_ALIASES = {"fermionic", "fermionic_sd", "sd"}
_QUBIT_ANSATZ_ALIASES = {"qubit_excitation", "qe", "qubit"}


def _normalize_ansatz_type(ansatz_type: Optional[str]) -> str:
    """Normalize ansatz-type aliases into a canonical value."""
    if ansatz_type is None:
        return "fermionic"
    normalized = str(ansatz_type).strip().lower()
    if normalized in _FERMIONIC_ANSATZ_ALIASES:
        return "fermionic"
    if normalized in _QUBIT_ANSATZ_ALIASES:
        return "qubit_excitation"
    raise ValueError(
        "ansatz_type must be one of {'fermionic', 'fermionic_sd', 'sd', "
        "'qubit_excitation', 'qe', 'qubit'}"
    )


def _normalize_projector_backend(projector_backend: Optional[str]) -> str:
    """Normalize analytic projector backend selection."""
    normalized = str(projector_backend or "auto").strip().lower()
    if normalized in {"auto", "dense", "sparse_number_preserving"}:
        return normalized
    raise ValueError(
        "projector_backend must be one of {'auto', 'dense', 'sparse_number_preserving'}"
    )


def _apply_excitation_gate(qml, excitation, weight, ansatz_type: str) -> None:
    """Apply one ansatz excitation gate in the selected operator family."""
    if ansatz_type == "fermionic":
        if len(excitation) == 4:
            qml.FermionicDoubleExcitation(
                weight=weight,
                wires1=list(range(excitation[0], excitation[1] + 1)),
                wires2=list(range(excitation[2], excitation[3] + 1)),
            )
            return
        if len(excitation) == 2:
            qml.FermionicSingleExcitation(
                weight=weight,
                wires=list(range(excitation[0], excitation[1] + 1)),
            )
            return
    elif ansatz_type == "qubit_excitation":
        if len(excitation) == 4:
            qml.DoubleExcitation(
                weight,
                wires=[int(excitation[0]), int(excitation[1]), int(excitation[2]), int(excitation[3])],
            )
            return
        if len(excitation) == 2:
            qml.SingleExcitation(
                weight,
                wires=[int(excitation[0]), int(excitation[1])],
            )
            return

    raise ValueError(
        "Each excitation must have length 2 (single) or 4 (double); "
        f"received {excitation!r} for ansatz_type='{ansatz_type}'."
    )


def _build_pyscf_molecular_integrals(
    *,
    symbols: Sequence[str],
    geometry,
    basis: str,
    charge: int,
    active_electrons: int,
    active_orbitals: int,
    use_ecp_avas: bool = False,
):
    """Build molecular integrals matching PennyLane's ``method='pyscf'`` path."""
    import numpy as np

    symbols_key = tuple(str(symbol) for symbol in symbols)
    geometry_key = tuple(float(value) for value in np.asarray(geometry, dtype=float).reshape(-1))
    core_constant, one_mo, two_mo = _build_pyscf_molecular_integrals_cached(
        symbols_key=symbols_key,
        geometry_key=geometry_key,
        basis=str(basis),
        charge=int(charge),
        active_electrons=int(active_electrons),
        active_orbitals=int(active_orbitals),
        use_ecp_avas=use_ecp_avas,
    )
    return (
        np.array(core_constant, dtype=float, copy=True),
        np.array(one_mo, dtype=float, copy=True),
        np.array(two_mo, dtype=float, copy=True),
    )


def _build_pyscf_molecular_integrals_ecp(
    *,
    symbols_key: tuple[str, ...],
    geometry_key: tuple[float, ...],
    charge: int,
    active_electrons: int,
    active_orbitals: int,
):
    """Classical pre-reduction using PySCF + ECP to bypass the N^4 all-electron bottleneck."""
    import numpy as np
    from pyscf import gto, scf, ao2mo

    coordinates = np.asarray(geometry_key, dtype=float).reshape(-1, 3)

    # 1. Build the molecule classically with an ECP for Iodine, and STO-3G for other light elements
    mol = gto.Mole()
    mol.atom = list(zip(symbols_key, coordinates))
    
    basis_map = {}
    ecp_map = {}
    for s in symbols_key:
        if s == "I":
            basis_map[s] = "lanl2dz"
            ecp_map[s] = "lanl2dz"
        else:
            basis_map[s] = "sto-3g"
            
    mol.basis = basis_map
    mol.ecp = ecp_map
    mol.charge = charge
    mol.spin = 0
    mol.build(verbose=0)

    # 2. Run Hartree-Fock
    mf = scf.RHF(mol)
    mf.kernel()

    # 3. Extract molecular orbital coefficients and raw integrals
    n_orbitals = mf.mo_coeff.shape[1]
    h_core = mf.get_hcore()
    T = mf.mo_coeff.T @ h_core @ mf.mo_coeff
    
    two_body_ao = mol.intor('int2e')
    V = ao2mo.kernel(two_body_ao, mf.mo_coeff)
    V = ao2mo.restore(1, V, n_orbitals) # chemist notation (N, N, N, N)

    # 4. Perform active space selection
    n_electrons = mol.nelectron
    n_core = (n_electrons // 2) - (active_electrons // 2)
    core_indices = list(range(n_core))
    active_indices = list(range(n_core, n_core + active_orbitals))

    # Core energy constant (nuclear repulsion + core orbital energies)
    core_constant = float(mf.energy_nuc())
    for i in core_indices:
        core_constant += 2.0 * T[i, i]
        for j in core_indices:
            core_constant += 2.0 * V[i, i, j, j] - V[i, j, j, i]

    # Effective one-body integrals
    n_active = len(active_indices)
    one_mo = np.zeros((n_active, n_active))
    for u_idx, u in enumerate(active_indices):
        for v_idx, v in enumerate(active_indices):
            val = T[u, v]
            for i in core_indices:
                val += 2.0 * V[u, v, i, i] - V[u, i, i, v]
            one_mo[u_idx, v_idx] = val

    # Effective two-body integrals in OpenFermion transpose format
    V_active = V[active_indices][:, active_indices][:, :, active_indices][:, :, :, active_indices]
    two_mo = V_active.transpose(0, 2, 3, 1)

    return core_constant, one_mo, two_mo


@lru_cache(maxsize=32)
def _build_pyscf_molecular_integrals_cached(
    *,
    symbols_key: tuple[str, ...],
    geometry_key: tuple[float, ...],
    basis: str,
    charge: int,
    active_electrons: int,
    active_orbitals: int,
    use_ecp_avas: bool = False,
):
    """Cached PySCF integral builder for repeated qscEOM runs on the same molecule."""
    import numpy as np

    if use_ecp_avas and "I" in symbols_key:
        core_constant, one_mo, two_mo = _build_pyscf_molecular_integrals_ecp(
            symbols_key=symbols_key,
            geometry_key=geometry_key,
            charge=charge,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
        )
    else:
        from pennylane.qchem import openfermion_pyscf as qchem_ofp
        coordinates = np.asarray(geometry_key, dtype=float)
        core_constant, one_mo, two_mo = qchem_ofp._pyscf_integrals(
            list(symbols_key),
            coordinates,
            charge=charge,
            mult=1,
            basis=basis,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
        )
    return (
        np.asarray(core_constant, dtype=float),
        np.asarray(one_mo, dtype=float),
        np.asarray(two_mo, dtype=float),
    )


def _expand_spatial_integrals_to_spin_orbital(one_mo, two_mo):
    """Expand spatial-orbital integrals into the spin-orbital convention used by qchem."""
    import numpy as np

    one_mo = np.asarray(one_mo, dtype=float)
    two_mo = np.asarray(two_mo, dtype=float)

    n_orbitals = int(one_mo.shape[0])
    n_spin_orbitals = 2 * n_orbitals

    one_spin = np.zeros((n_spin_orbitals, n_spin_orbitals), dtype=float)
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            one_spin[2 * p, 2 * q] = float(one_mo[p, q])
            one_spin[2 * p + 1, 2 * q + 1] = float(one_mo[p, q])

    # Match PennyLane's fermionic_observable spin expansion exactly.
    two_spin = np.zeros((n_spin_orbitals,) * 4, dtype=float)
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            for r in range(n_orbitals):
                for s in range(n_orbitals):
                    coeff = float(two_mo[p, q, r, s]) / 2.0
                    if coeff == 0.0:
                        continue
                    two_spin[2 * p, 2 * q, 2 * r, 2 * s] = coeff
                    two_spin[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = coeff
                    two_spin[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = coeff
                    two_spin[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = coeff

    return one_spin, two_spin


def _build_exact_fermion_operator(
    *,
    symbols: Sequence[str],
    geometry,
    basis: str,
    charge: int,
    active_electrons: int,
    active_orbitals: int,
    use_ecp_avas: bool = False,
):
    """Build the exact molecular Hamiltonian as an OpenFermion FermionOperator."""
    import numpy as np
    try:
        from openfermion import InteractionOperator, get_fermion_operator
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The sparse qscEOM projector requires the optional dependency `openfermion`."
        ) from exc

    core_constant, one_mo, two_mo = _build_pyscf_molecular_integrals(
        symbols=symbols,
        geometry=geometry,
        basis=basis,
        charge=charge,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        use_ecp_avas=use_ecp_avas,
    )
    one_spin, two_spin = _expand_spatial_integrals_to_spin_orbital(one_mo, two_mo)
    interaction = InteractionOperator(
        float(np.asarray(core_constant, dtype=float).reshape(-1)[0]),
        one_spin,
        two_spin,
    )
    return get_fermion_operator(interaction)


def _build_brg_fermion_operator(
    *,
    symbols: Sequence[str],
    geometry,
    basis: str,
    charge: int,
    active_electrons: int,
    active_orbitals: int,
    brg_tolerance: float,
    use_ecp_avas: bool = False,
):
    """Build a BRG-truncated molecular Hamiltonian as an OpenFermion FermionOperator."""
    import numpy as np
    try:
        from openfermion import (
            FermionOperator,
            low_rank_two_body_decomposition,
            normal_ordered,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError("brg_tolerance requires the optional dependency `openfermion`.") from exc

    core_constant, one_mo, two_mo = _build_pyscf_molecular_integrals(
        symbols=symbols,
        geometry=geometry,
        basis=basis,
        charge=charge,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        use_ecp_avas=use_ecp_avas,
    )
    one_spin, two_spin = _expand_spatial_integrals_to_spin_orbital(one_mo, two_mo)
    eigenvalues, one_body_squares, one_body_correction, truncation_value = low_rank_two_body_decomposition(
        two_spin,
        truncation_threshold=float(brg_tolerance),
        spin_basis=True,
    )

    n_spin_orbitals = int(one_spin.shape[0])
    fermion_op = FermionOperator("", float(np.asarray(core_constant, dtype=float).reshape(-1)[0]))

    corrected_one_body = np.asarray(one_spin + one_body_correction, dtype=complex)
    for p in range(n_spin_orbitals):
        for q in range(n_spin_orbitals):
            coeff = complex(corrected_one_body[p, q])
            if abs(coeff) <= 1e-15:
                continue
            fermion_op += FermionOperator(((p, 1), (q, 0)), coeff)

    for lam, g_mat in zip(np.asarray(eigenvalues, dtype=float), np.asarray(one_body_squares, dtype=complex)):
        generator = FermionOperator()
        for p in range(n_spin_orbitals):
            for q in range(n_spin_orbitals):
                coeff = complex(g_mat[p, q])
                if abs(coeff) <= 1e-15:
                    continue
                generator += FermionOperator(((p, 1), (q, 0)), coeff)
        fermion_op += float(lam) * normal_ordered(generator * generator)

    details = {
        "brg_applied": True,
        "brg_tolerance": float(brg_tolerance),
        "brg_rank": int(len(eigenvalues)),
        "brg_truncation_value": float(truncation_value),
    }
    return fermion_op, details


def _build_brg_hamiltonian_dense(
    *,
    symbols: Sequence[str],
    geometry,
    basis: str,
    charge: int,
    active_electrons: int,
    active_orbitals: int,
    brg_tolerance: float,
    use_ecp_avas: bool = False,
):
    """Build a BRG-truncated dense molecular Hamiltonian matrix."""
    import numpy as np
    try:
        from openfermion import get_sparse_operator
    except ImportError as exc:  # pragma: no cover
        raise ImportError("brg_tolerance requires the optional dependency `openfermion`.") from exc

    fermion_op, details = _build_brg_fermion_operator(
        symbols=symbols,
        geometry=geometry,
        basis=basis,
        charge=charge,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        brg_tolerance=brg_tolerance,
        use_ecp_avas=use_ecp_avas,
    )
    n_spin_orbitals = int(2 * int(active_orbitals))
    dense = np.asarray(get_sparse_operator(fermion_op, n_qubits=n_spin_orbitals).toarray(), dtype=complex)
    return dense, details


def _build_number_preserving_sparse_hamiltonian(
    *,
    fermion_operator,
    qubits: int,
    active_electrons: int,
):
    """Build a number-preserving sparse Hamiltonian for the fixed-electron sector."""
    try:
        from openfermion import get_number_preserving_sparse_operator
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The sparse qscEOM projector requires the optional dependency `openfermion`."
        ) from exc

    sparse_operator = get_number_preserving_sparse_operator(
        fermion_operator,
        num_qubits=int(qubits),
        num_electrons=int(active_electrons),
        spin_preserving=False,
    ).tocsr()
    details = {
        "sector_dimension": int(sparse_operator.shape[0]),
        "hamiltonian_nnz": int(sparse_operator.nnz),
    }
    return sparse_operator, details


def _jw_number_sector_indices(n_electrons: int, n_qubits: int):
    """Return sector basis indices in the ordering used by OpenFermion's sparse converter."""
    from itertools import combinations

    n_electrons = int(n_electrons)
    n_qubits = int(n_qubits)
    reference_occ = tuple(range(n_electrons))
    virtual_occ = tuple(range(n_electrons, n_qubits))
    indices = []

    for excitation_rank in range(n_electrons + 1):
        for removed in combinations(reference_occ, excitation_rank):
            removed_set = set(removed)
            kept = [orb for orb in reference_occ if orb not in removed_set]
            for added in combinations(virtual_occ, excitation_rank):
                occupation = sorted(kept + list(added))
                index = 0
                for orb in occupation:
                    index |= 1 << (n_qubits - 1 - int(orb))
                indices.append(int(index))

    return indices


def _restrict_basis_states_to_number_sector(basis_states, *, active_electrons: int, qubits: int):
    """Restrict full-space statevectors to the fixed-electron sparse-operator sector."""
    import numpy as np

    select_indices = np.asarray(_jw_number_sector_indices(int(active_electrons), int(qubits)), dtype=int)
    restricted = np.asarray(basis_states[select_indices, :], dtype=complex)
    return restricted, select_indices


def _resolve_worker_count(max_workers: Optional[int]) -> int:
    """Resolve worker thread count for optional parallel sections."""
    if max_workers is not None:
        return int(max_workers)
    detected = os.cpu_count()
    return int(detected if detected is not None else 1)


def _resolve_chunk_size(
    *,
    total_items: int,
    worker_count: int,
    user_chunk_size: Optional[int],
) -> int:
    """Resolve a chunk size for batched worker submission."""
    if total_items <= 0:
        return 1
    if user_chunk_size is not None:
        return int(user_chunk_size)
    return max(1, (int(total_items) + int(worker_count) - 1) // int(worker_count))


def _iter_chunks(items, chunk_size: int):
    """Yield fixed-size chunks from a sequence-like container."""
    for start in range(0, len(items), int(chunk_size)):
        yield items[start : start + int(chunk_size)]


def _resolve_parallel_backend(parallel_backend: str) -> str:
    """Resolve optional parallel backend selection."""
    if parallel_backend == "auto":
        if os.name == "nt":
            return "thread"
        return "process"
    return parallel_backend


_QSCEOM_WORKER_STATE = {}


def _qsceom_worker_init(payload):
    """Initialize persistent worker-local state for qscEOM matrix evaluation."""
    import numpy as np
    import pennylane as qml

    device_kwargs = dict(payload.get("device_kwargs") or {})

    def _make_device(
        name: Optional[str],
        wires: int,
        shots: int,
        extra_kwargs: Optional[dict[str, Any]] = None,
    ):
        kwargs = dict(extra_kwargs or {})
        kwargs.pop("wires", None)
        kwargs.pop("shots", None)
        if shots > 0:
            kwargs["shots"] = shots
        if name is not None:
            return qml.device(name, wires=wires, **kwargs)
        try:
            return qml.device("lightning.qubit", wires=wires, **kwargs)
        except Exception:
            return qml.device("default.qubit", wires=wires, **kwargs)

    qubits = int(payload["qubits"])
    shots = int(payload["shots"])
    H = payload["H"]
    params = payload["params"]
    ash_excitation = payload["ash_excitation"]
    list1 = payload["list1"]
    null_state = payload["null_state"]
    ansatz_type = str(payload["ansatz_type"])
    device_name = payload["device_name"]
    use_taper = payload.get("use_taper", False)
    tapered_ansatz = payload.get("tapered_ansatz", None)

    dev = _make_device(device_name, qubits, shots, device_kwargs)

    def _apply_ansatz_local(params_local, ash_local):
        if use_taper and tapered_ansatz is not None:
            for t_ops in tapered_ansatz:
                for op in t_ops:
                    qml.apply(op)
        else:
            for i, excitations in enumerate(ash_local):
                _apply_excitation_gate(qml, excitations, params_local[i], ansatz_type)

    @qml.qnode(dev)
    def _diag_by_index(idx):
        occ = list1[int(idx)]
        qml.BasisState(null_state, wires=range(qubits))
        for w in occ:
            qml.X(wires=w)
        _apply_ansatz_local(params, ash_excitation)
        return qml.expval(H)

    @qml.qnode(dev)
    def _offdiag_by_pair(i, j):
        occ1 = list1[int(i)]
        occ2 = list1[int(j)]
        qml.BasisState(null_state, wires=range(qubits))
        for w in occ1:
            qml.X(wires=w)
        first = -1
        for v in occ2:
            if v not in occ1:
                if first == -1:
                    first = v
                    qml.Hadamard(wires=v)
                else:
                    qml.CNOT(wires=[first, v])
        for v in occ1:
            if v not in occ2:
                if first == -1:
                    first = v
                    qml.Hadamard(wires=v)
                else:
                    qml.CNOT(wires=[first, v])
        _apply_ansatz_local(params, ash_excitation)
        return qml.expval(H)

    _QSCEOM_WORKER_STATE.clear()
    _QSCEOM_WORKER_STATE.update(
        {
            "np": np,
            "diag_fn": _diag_by_index,
            "offdiag_fn": _offdiag_by_pair,
        }
    )


def _qsceom_worker_diagonal(chunk_indices):
    """Evaluate diagonal matrix entries for a chunk of state indices."""
    np = _QSCEOM_WORKER_STATE["np"]
    diag_fn = _QSCEOM_WORKER_STATE["diag_fn"]
    out = {}
    for idx in chunk_indices:
        value = diag_fn(idx)
        out[idx] = float(np.real(np.asarray(value).item()))
    return out


def _qsceom_worker_offdiagonal(chunk_pairs):
    """Evaluate raw off-diagonal expectation values for a chunk of pairs."""
    np = _QSCEOM_WORKER_STATE["np"]
    offdiag_fn = _QSCEOM_WORKER_STATE["offdiag_fn"]
    out = {}
    for i, j in chunk_pairs:
        value = offdiag_fn(i, j)
        out[(i, j)] = float(np.real(np.asarray(value).item()))
    return out


def _qsceom_state_worker_init(payload):
    """Initialize worker-local qnode for analytic statevector generation."""
    import numpy as np
    import pennylane as qml

    device_kwargs = dict(payload.get("device_kwargs") or {})

    def _make_device(
        name: Optional[str],
        wires: int,
        extra_kwargs: Optional[dict[str, Any]] = None,
    ):
        kwargs = dict(extra_kwargs or {})
        kwargs.pop("wires", None)
        kwargs.pop("shots", None)
        if name is not None:
            return qml.device(name, wires=wires, **kwargs)
        try:
            return qml.device("lightning.qubit", wires=wires, **kwargs)
        except Exception:
            return qml.device("default.qubit", wires=wires, **kwargs)

    qubits = int(payload["qubits"])
    params = payload["params"]
    ash_excitation = payload["ash_excitation"]
    list1 = payload["list1"]
    null_state = payload["null_state"]
    ansatz_type = str(payload["ansatz_type"])
    device_name = payload["device_name"]
    use_taper = payload.get("use_taper", False)
    tapered_ansatz = payload.get("tapered_ansatz", None)

    dev = _make_device(device_name, qubits, device_kwargs)

    def _apply_ansatz_local(params_local, ash_local):
        if use_taper and tapered_ansatz is not None:
            for t_ops in tapered_ansatz:
                for op in t_ops:
                    qml.apply(op)
        else:
            for i, excitations in enumerate(ash_local):
                _apply_excitation_gate(qml, excitations, params_local[i], ansatz_type)

    @qml.qnode(dev)
    def _state_by_index(idx):
        occ = list1[int(idx)]
        qml.BasisState(null_state, wires=range(qubits))
        for w in occ:
            qml.X(wires=w)
        _apply_ansatz_local(params, ash_excitation)
        return qml.state()

    _QSCEOM_WORKER_STATE.clear()
    _QSCEOM_WORKER_STATE.update(
        {
            "np": np,
            "state_fn": _state_by_index,
        }
    )


def _qsceom_state_worker_chunk(chunk_indices):
    """Generate analytic ansatz statevectors for a chunk of basis indices."""
    np = _QSCEOM_WORKER_STATE["np"]
    state_fn = _QSCEOM_WORKER_STATE["state_fn"]
    out = {}
    for idx in chunk_indices:
        out[idx] = np.asarray(state_fn(idx), dtype=complex)
    return out


def qscEOM(
    symbols: Sequence[str],
    geometry,
    active_electrons: int,
    active_orbitals: int,
    charge: int,
    mult: int = 1,
    params=None,
    ash_excitation=None,
    *,
    ansatz: Optional[Tuple[Any, Any, Any]] = None,
    ansatz_type: Optional[str] = None,
    basis: str = "sto-3g",
    method: str = "pyscf",
    shots: int = 0,
    device_name: Optional[str] = None,
    include_identity: bool = True,
    max_states: Optional[int] = None,
    state_seed: Optional[int] = None,
    symmetric: bool = True,
    parallel_matrix: bool = False,
    parallel_backend: str = "auto",
    max_workers: Optional[int] = None,
    matrix_chunk_size: Optional[int] = None,
    pauli_grouping: bool = False,
    grouping_type: str = "qwc",
    device_kwargs: Optional[Mapping[str, Any]] = None,
    brg_tolerance: Optional[float] = None,
    projector_backend: str = "auto",
    return_details: bool = False,
    outpath: str = ".",
    use_ecp_avas: bool = False,
    use_taper: bool = False,
):
    """Compute qscEOM eigenvalues from an ansatz state.

    Parameters
    ----------
    symbols
        Atomic symbols.
    geometry
        Nuclear coordinates (as an array-like object).
    active_electrons
        Number of active electrons.
    active_orbitals
        Number of active orbitals.
    charge
        Total molecular charge.
    params
        Ansatz parameters.
    ash_excitation
        Excitation list describing the ansatz.
    ansatz_type
        Ansatz gate family used to replay ``ash_excitation``:
        ``"fermionic"`` (default) or ``"qubit_excitation"`` (alias ``"qe"``).
        This controls only ansatz replay, not the qscEOM basis construction.
    shots
        If 0, run in analytic mode; otherwise use shot-based estimation.
    device_name
        Optional PennyLane device name (e.g. ``"lightning.qubit"``).
    include_identity
        If True (default), include the HF/reference state ``I`` in the qscEOM
        projected basis in addition to standard singles+doubles configurations.
    max_states
        Deprecated compatibility argument. qscEOM now always uses the full
        selected basis and rejects any non-``None`` value.
    state_seed
        Deprecated compatibility argument kept only to avoid breaking older
        call sites. Ignored when ``max_states=None``.
    symmetric
        If True, compute only the upper-triangular off-diagonal elements and
        mirror them to reduce circuit evaluations.
    parallel_matrix
        If True, evaluate independent qscEOM matrix elements concurrently.
    parallel_backend
        Parallel backend used when ``parallel_matrix=True``:
        ``"process"``, ``"thread"``, or ``"auto"`` (default). ``"auto"``
        selects processes on POSIX and threads on Windows.
    max_workers
        Maximum number of worker threads used when ``parallel_matrix=True``.
        If omitted, ``os.cpu_count()`` is used.
    matrix_chunk_size
        Number of matrix entries per submitted worker task. If omitted, a
        balanced chunk size based on ``max_workers`` is used.
    pauli_grouping
        If True, pre-compute Pauli grouping metadata (e.g. QWC) for the
        Hamiltonian before shot-based measurements.
    grouping_type
        Grouping strategy passed to ``compute_grouping`` when
        ``pauli_grouping=True``.
    device_kwargs
        Optional keyword arguments forwarded to ``qml.device``. This is useful
        when selecting hardware/noise-model specific backends that require
        extra constructor parameters.
    brg_tolerance
        If provided, apply basis-rotation grouping (BRG) to the molecular
        Hamiltonian before the analytic projected-matrix construction. This is
        supported only when ``shots=0`` and ``method='pyscf'``.
    projector_backend
        Analytic projected-matrix backend. ``"dense"`` preserves the historical
        dense-Hamiltonian path. ``"sparse_number_preserving"`` restricts the
        molecular Hamiltonian to the fixed-electron Jordan-Wigner sector using
        OpenFermion sparse operators. ``"auto"`` selects the sparse backend for
        analytic ``method='pyscf'`` runs when OpenFermion is available.
    return_details
        If True, also return a details dictionary containing projected-matrix
        eigenvectors and basis metadata. Default False preserves historical
        return type.

    Returns
    -------
    list
        Sorted eigenvalues for the constructed effective matrix.
    tuple
        When ``return_details=True``, returns ``(values, details)`` where
        ``details`` contains eigenvectors and matrix metadata.
    """

    inferred_ansatz_type = None
    if ansatz is not None:
        try:
            params_from_adapt, ash_excitation_from_adapt, _energies = ansatz
        except Exception as exc:
            raise ValueError(
                "ansatz must be a 3-tuple like (params, ash_excitation, energies) "
                "as returned by QCANT.adapt_vqe"
            ) from exc

        params = params_from_adapt
        ash_excitation = ash_excitation_from_adapt
        inferred_ansatz_type = getattr(ash_excitation_from_adapt, "ansatz_type", None)
        if inferred_ansatz_type is None:
            inferred_ansatz_type = getattr(_energies, "ansatz_type", None)

    if ansatz_type is None:
        ansatz_type = inferred_ansatz_type
    ansatz_type = _normalize_ansatz_type(ansatz_type)
    projector_backend = _normalize_projector_backend(projector_backend)
    method_normalized = str(method).strip().lower()

    if params is None or ash_excitation is None:
        raise TypeError(
            "qscEOM requires either (params, ash_excitation) or ansatz=(params, ash_excitation, energies)."
        )
    if max_states is not None:
        raise ValueError(
            "max_states-based truncation has been removed; qscEOM now always uses the full basis. "
            "Pass max_states=None."
        )
    if max_workers is not None and max_workers <= 0:
        raise ValueError("max_workers must be > 0")
    if matrix_chunk_size is not None and matrix_chunk_size <= 0:
        raise ValueError("matrix_chunk_size must be > 0")
    if brg_tolerance is not None and brg_tolerance <= 0:
        raise ValueError("brg_tolerance must be > 0")
    if parallel_backend not in {"auto", "thread", "process"}:
        raise ValueError("parallel_backend must be one of {'auto', 'thread', 'process'}")
    if pauli_grouping and grouping_type not in {"qwc", "commuting", "anticommuting"}:
        raise ValueError(
            "grouping_type must be one of {'qwc', 'commuting', 'anticommuting'} "
            "when pauli_grouping=True"
        )
    if brg_tolerance is not None and shots != 0:
        raise ValueError("brg_tolerance requires shots=0")
    if brg_tolerance is not None and method_normalized != "pyscf":
        raise ValueError("brg_tolerance requires method='pyscf'")
    if projector_backend == "sparse_number_preserving" and shots != 0:
        raise ValueError("projector_backend='sparse_number_preserving' requires shots=0")
    if projector_backend == "sparse_number_preserving" and method_normalized != "pyscf":
        raise ValueError("projector_backend='sparse_number_preserving' requires method='pyscf'")

    try:
        if len(params) != len(ash_excitation):
            raise ValueError
    except Exception as exc:
        raise ValueError("params and ash_excitation must have the same length") from exc

    try:
        import numpy as np
        import pennylane as qml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "qscEOM requires dependencies. Install at least: "
            "`pip install numpy pennylane`."
        ) from exc

    resolved_projector_backend = projector_backend
    resolved_projector_backend = projector_backend
    if use_taper:
        resolved_projector_backend = "dense"
    elif projector_backend == "auto":
        if shots == 0 and method_normalized == "pyscf":
            try:
                import openfermion  # noqa: F401
            except ImportError:
                resolved_projector_backend = "dense"
            else:
                resolved_projector_backend = "sparse_number_preserving"
        else:
            resolved_projector_backend = "dense"

    H = None
    qubits = 2 * int(active_orbitals)
    if shots != 0 or resolved_projector_backend == "dense":
        if use_ecp_avas and "I" in symbols:
            # Construct H from ECP integrals classically
            from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner
            core_constant, one_mo, two_mo = _build_pyscf_molecular_integrals(
                symbols=symbols,
                geometry=geometry,
                basis=basis,
                charge=charge,
                active_electrons=active_electrons,
                active_orbitals=active_orbitals,
                use_ecp_avas=True,
            )
            one_spin, two_spin = _expand_spatial_integrals_to_spin_orbital(one_mo, two_mo)
            interaction = InteractionOperator(
                float(np.asarray(core_constant, dtype=float).reshape(-1)[0]),
                one_spin,
                two_spin,
            )
            ferm_op = get_fermion_operator(interaction)
            qubit_op = jordan_wigner(ferm_op)
            H = qml.from_openfermion(qubit_op)
            qubits = 2 * int(active_orbitals)
        else:
            H, qubits = qml.qchem.molecular_hamiltonian(
                symbols,
                geometry,
                basis=basis,
                method=method,
                active_electrons=active_electrons,
                active_orbitals=active_orbitals,
                charge=charge,
                mult=mult,
                outpath=outpath,
            )
        if pauli_grouping and H is not None and hasattr(H, "compute_grouping"):
            H.compute_grouping(grouping_type=grouping_type)

    hf_state = qml.qchem.hf_state(active_electrons, qubits)
    
    generators = []
    tapered_ansatz = None
    if use_taper and H is not None:
        generators = qml.symmetry_generators(H)
        if len(generators) > 0:
            paulixops = qml.paulix_ops(generators, qubits)
            paulix_sector = qml.qchem.optimal_sector(H, generators, active_electrons)
            
            # Pre-taper the ansatz operations
            tapered_ansatz = []
            wire_order = list(range(qubits))
            for i, exc in enumerate(ash_excitation or []):
                theta = params[i]
                if len(exc) == 2:
                    op = qml.SingleExcitation(theta, wires=exc)
                elif len(exc) == 4:
                    op = qml.DoubleExcitation(theta, wires=exc)
                else:
                    tapered_ansatz.append([])
                    continue
                try:
                    t_ops = qml.qchem.taper_operation(op, generators, paulixops, paulix_sector, wire_order)
                    tapered_ansatz.append(t_ops)
                except Exception:
                    tapered_ansatz.append([])

            H = qml.taper(H, generators, paulixops, paulix_sector)
            hf_state = qml.qchem.taper_hf(generators, paulixops, paulix_sector, active_electrons, qubits)
            
            qubits = qubits - len(generators)
            list1 = [list(int(v) for v in occ) for occ in inite(active_electrons, qubits)]
        else:
            list1 = [list(int(v) for v in occ) for occ in inite(active_electrons, qubits)]
    else:
        list1 = [list(int(v) for v in occ) for occ in inite(active_electrons, qubits)]

    singles, doubles = qml.qchem.excitations(active_electrons, qubits + (len(generators) if (use_taper and len(generators) > 0) else 0))
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    wires = range(qubits)

    null_state = np.zeros(qubits, int)
    ref_occ = [int(i) for i, bit in enumerate(np.asarray(hf_state, dtype=int)) if int(bit) == 1]
    if include_identity:
        basis_occ = [ref_occ] + [occ for occ in list1 if occ != ref_occ]
    else:
        basis_occ = [occ for occ in list1 if occ != ref_occ]

    seen_occ = set()
    list1 = []
    for occ in basis_occ:
        key = tuple(int(v) for v in occ)
        if key in seen_occ:
            continue
        seen_occ.add(key)
        list1.append(list(key))

    if len(list1) == 0:
        raise ValueError("qscEOM basis is empty after include_identity filtering.")

    values = []

    device_kwargs_local = dict(device_kwargs or {})

    def _make_device(name: Optional[str], wires: int):
        kwargs = dict(device_kwargs_local)
        kwargs.pop("wires", None)
        kwargs.pop("shots", None)
        if shots > 0:
            kwargs["shots"] = shots
        if name is not None:
            return qml.device(name, wires=wires, **kwargs)
        try:
            return qml.device("lightning.qubit", wires=wires, **kwargs)
        except Exception:
            return qml.device("default.qubit", wires=wires, **kwargs)

    def _to_real_scalar(value):
        """Convert PennyLane/numpy scalar-like values to python float."""
        arr = np.asarray(value)
        return float(np.real(arr).item())

    def _build_circuit_state(dev):
        @qml.qnode(dev)
        def circuit_state_local(params_local, occ, hf_state_local, ash_local, tapered_ansatz_local=None):
            qml.BasisState(hf_state_local, wires=range(qubits))
            for w in occ:
                qml.X(wires=w)
            if use_taper and tapered_ansatz_local is not None:
                for t_ops in tapered_ansatz_local:
                    for op in t_ops:
                        qml.apply(op)
            else:
                for i, excitations in enumerate(ash_local):
                    _apply_excitation_gate(qml, excitations, params_local[i], ansatz_type)
            return qml.state()

        return circuit_state_local

    def _build_circuit_d(dev):
        @qml.qnode(dev)
        def circuit_d_local(params_local, occ, wires, s_wires, d_wires, hf_state_local, ash_local, tapered_ansatz_local=None):
            qml.BasisState(hf_state_local, wires=range(qubits))
            for w in occ:
                qml.X(wires=w)
            if use_taper and tapered_ansatz_local is not None:
                for t_ops in tapered_ansatz_local:
                    for op in t_ops:
                        qml.apply(op)
            else:
                for i, excitations in enumerate(ash_local):
                    _apply_excitation_gate(qml, excitations, params_local[i], ansatz_type)
            return qml.expval(H)

        return circuit_d_local

    def _build_circuit_od(dev):
        @qml.qnode(dev)
        def circuit_od_local(params_local, occ1, occ2, wires, s_wires, d_wires, hf_state_local, ash_local, tapered_ansatz_local=None):
            qml.BasisState(hf_state_local, wires=range(qubits))
            for w in occ1:
                qml.X(wires=w)
            first = -1
            for v in occ2:
                if v not in occ1:
                    if first == -1:
                        first = v
                        qml.Hadamard(wires=v)
                    else:
                        qml.CNOT(wires=[first, v])
            for v in occ1:
                if v not in occ2:
                    if first == -1:
                        first = v
                        qml.Hadamard(wires=v)
                    else:
                        qml.CNOT(wires=[first, v])
            if use_taper and tapered_ansatz_local is not None:
                for t_ops in tapered_ansatz_local:
                    for op in t_ops:
                        qml.apply(op)
            else:
                for i, excitations in enumerate(ash_local):
                    _apply_excitation_gate(qml, excitations, params_local[i], ansatz_type)
            return qml.expval(H)

        return circuit_od_local

    n_states = len(list1)
    worker_count = _resolve_worker_count(max_workers)
    backend = _resolve_parallel_backend(parallel_backend)
    brg_details = {
        "brg_applied": False,
        "brg_tolerance": None,
        "brg_rank": None,
        "brg_truncation_value": None,
        "projector_backend": resolved_projector_backend if shots == 0 else "shot_based",
        "sector_dimension": None,
        "hamiltonian_nnz": None,
    }

    # Analytic-mode path: build the exact complex projected matrix M_ij = <psi_i|H|psi_j>.
    # This preserves Rayleigh-Ritz variational behavior (with include_identity=True, the
    # ADAPT state is included in the subspace).
    if shots == 0:
        state_map = {}

        def _state_chunk(chunk_indices):
            local_dev = _make_device(device_name, qubits)
            circuit_state_local = _build_circuit_state(local_dev)
            out = {}
            for idx in chunk_indices:
                out[idx] = np.asarray(
                    circuit_state_local(
                        params,
                        list1[idx],
                        null_state,
                        ash_excitation,
                        tapered_ansatz,
                    ),
                    dtype=complex,
                )
            return out

        if parallel_matrix and worker_count > 1 and n_states > 1:
            chunk_size = _resolve_chunk_size(
                total_items=n_states,
                worker_count=worker_count,
                user_chunk_size=matrix_chunk_size,
            )
            chunked_indices = list(_iter_chunks(list(range(n_states)), chunk_size))
            state_executor = None
            try:
                if backend == "process":
                    payload = {
                        "qubits": int(qubits),
                        "params": np.asarray(params),
                        "ash_excitation": tuple(tuple(int(v) for v in exc) for exc in ash_excitation),
                        "list1": tuple(tuple(int(v) for v in occ) for occ in list1),
                        "null_state": np.asarray(null_state),
                        "ansatz_type": ansatz_type,
                        "device_name": device_name,
                        "device_kwargs": device_kwargs_local,
                        "use_taper": use_taper,
                        "tapered_ansatz": tapered_ansatz,
                    }
                    try:
                        mp_context = mp.get_context("fork")
                    except ValueError:
                        mp_context = mp.get_context()
                    try:
                        state_executor = ProcessPoolExecutor(
                            max_workers=worker_count,
                            mp_context=mp_context,
                            initializer=_qsceom_state_worker_init,
                            initargs=(payload,),
                        )
                    except (PermissionError, OSError, NotImplementedError):
                        state_executor = None
                elif backend == "thread":
                    state_executor = ThreadPoolExecutor(max_workers=worker_count)

                if state_executor is not None:
                    if backend == "process":
                        futures = [
                            state_executor.submit(_qsceom_state_worker_chunk, chunk)
                            for chunk in chunked_indices
                        ]
                    else:
                        futures = [state_executor.submit(_state_chunk, chunk) for chunk in chunked_indices]
                    for future in as_completed(futures):
                        state_map.update(future.result())
                else:
                    state_map.update(_state_chunk(list(range(n_states))))
            finally:
                if state_executor is not None:
                    state_executor.shutdown(wait=True)
        else:
            state_map.update(_state_chunk(list(range(n_states))))

        basis_states = np.column_stack([state_map[i] for i in range(n_states)])
        if resolved_projector_backend == "sparse_number_preserving":
            if brg_tolerance is not None:
                fermion_operator, sparse_brg_details = _build_brg_fermion_operator(
                    symbols=symbols,
                    geometry=geometry,
                    basis=basis,
                    charge=charge,
                    active_electrons=active_electrons,
                    active_orbitals=active_orbitals,
                    brg_tolerance=float(brg_tolerance),
                )
                brg_details.update(sparse_brg_details)
            else:
                fermion_operator = _build_exact_fermion_operator(
                    symbols=symbols,
                    geometry=geometry,
                    basis=basis,
                    charge=charge,
                    active_electrons=active_electrons,
                    active_orbitals=active_orbitals,
                )

            sparse_hamiltonian, sparse_details = _build_number_preserving_sparse_hamiltonian(
                fermion_operator=fermion_operator,
                qubits=qubits,
                active_electrons=active_electrons,
            )
            restricted_basis_states, _sector_indices = _restrict_basis_states_to_number_sector(
                basis_states,
                active_electrons=active_electrons,
                qubits=qubits,
            )
            brg_details.update(sparse_details)
            M_exact = restricted_basis_states.conj().T @ (sparse_hamiltonian @ restricted_basis_states)
        else:
            if brg_tolerance is not None:
                H_dense, dense_brg_details = _build_brg_hamiltonian_dense(
                    symbols=symbols,
                    geometry=geometry,
                    basis=basis,
                    charge=charge,
                    active_electrons=active_electrons,
                    active_orbitals=active_orbitals,
                    brg_tolerance=float(brg_tolerance),
                )
                brg_details.update(dense_brg_details)
            else:
                H_dense = np.asarray(qml.matrix(H, wire_order=range(qubits)), dtype=complex)
            M_exact = basis_states.conj().T @ (H_dense @ basis_states)
        M_exact = 0.5 * (M_exact + M_exact.conj().T)
        eigvals, eigvecs = np.linalg.eigh(M_exact)
        order = np.argsort(np.real(eigvals))
        eigvals_sorted = np.asarray(eigvals[order], dtype=float)
        eigvecs_sorted = np.asarray(eigvecs[:, order], dtype=complex)
        values.append(eigvals_sorted)
        if not return_details:
            return values
        details = {
            "projected_matrix": np.asarray(M_exact, dtype=complex),
            "basis_states": np.asarray(basis_states, dtype=complex),
            "eigenvectors": eigvecs_sorted,
            "basis_occupations": [list(int(v) for v in occ) for occ in list1],
        }
        details.update(brg_details)
        return values, details

    M = np.zeros((n_states, n_states), dtype=float)
    executor = None
    if parallel_matrix and worker_count > 1:
        if backend == "process":
            payload = {
                "qubits": int(qubits),
                "shots": int(shots),
                "H": H,
                "params": np.asarray(params),
                "ash_excitation": tuple(tuple(int(v) for v in exc) for exc in ash_excitation),
                "list1": tuple(tuple(int(v) for v in occ) for occ in list1),
                "null_state": np.asarray(null_state),
                "ansatz_type": ansatz_type,
                "device_name": device_name,
                "device_kwargs": device_kwargs_local,
                "use_taper": use_taper,
                "tapered_ansatz": tapered_ansatz,
            }
            try:
                mp_context = mp.get_context("fork")
            except ValueError:
                mp_context = mp.get_context()
            try:
                executor = ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=mp_context,
                    initializer=_qsceom_worker_init,
                    initargs=(payload,),
                )
            except (PermissionError, OSError, NotImplementedError):
                backend = "thread"
                executor = ThreadPoolExecutor(max_workers=worker_count)
        else:
            executor = ThreadPoolExecutor(max_workers=worker_count)

    def _diagonal_chunk(chunk_indices):
        local_dev = _make_device(device_name, qubits)
        circuit_d_local = _build_circuit_d(local_dev)
        out = {}
        for idx in chunk_indices:
            value = circuit_d_local(params, list1[idx], wires, s_wires, d_wires, null_state, ash_excitation, tapered_ansatz)
            out[idx] = _to_real_scalar(value)
        return out

    def _off_diagonal_chunk(chunk_pairs, diagonal_values):
        local_dev = _make_device(device_name, qubits)
        circuit_od_local = _build_circuit_od(local_dev)
        out = {}
        for i, j in chunk_pairs:
            mtmp = circuit_od_local(
                params,
                list1[i],
                list1[j],
                wires,
                s_wires,
                d_wires,
                null_state,
                ash_excitation,
                tapered_ansatz,
            )
            value = _to_real_scalar(mtmp) - diagonal_values[i] / 2.0 - diagonal_values[j] / 2.0
            out[(i, j)] = float(value)
        return out

    try:
        diagonal_indices = list(range(n_states))
        if executor is not None and n_states > 1:
            diag_chunk_size = _resolve_chunk_size(
                total_items=n_states,
                worker_count=worker_count,
                user_chunk_size=matrix_chunk_size,
            )
            diagonal_map = {}
            if backend == "process":
                futures = [
                    executor.submit(_qsceom_worker_diagonal, chunk)
                    for chunk in _iter_chunks(diagonal_indices, diag_chunk_size)
                ]
            else:
                futures = [
                    executor.submit(_diagonal_chunk, chunk)
                    for chunk in _iter_chunks(diagonal_indices, diag_chunk_size)
                ]
            for future in as_completed(futures):
                diagonal_map.update(future.result())
            for i in diagonal_indices:
                M[i, i] = diagonal_map[i]
        else:
            dev = _make_device(device_name, qubits)
            circuit_d = _build_circuit_d(dev)
            for i in diagonal_indices:
                M[i, i] = _to_real_scalar(
                    circuit_d(params, list1[i], wires, s_wires, d_wires, null_state, ash_excitation, tapered_ansatz)
                )

        if symmetric:
            pair_indices = [(i, j) for i in range(n_states) for j in range(i + 1, n_states)]
        else:
            pair_indices = [(i, j) for i in range(n_states) for j in range(n_states) if i != j]

        if executor is not None and len(pair_indices) > 1:
            pair_chunk_size = _resolve_chunk_size(
                total_items=len(pair_indices),
                worker_count=worker_count,
                user_chunk_size=matrix_chunk_size,
            )
            pair_map = {}
            if backend == "process":
                futures = [
                    executor.submit(_qsceom_worker_offdiagonal, chunk)
                    for chunk in _iter_chunks(pair_indices, pair_chunk_size)
                ]
            else:
                futures = [
                    executor.submit(_off_diagonal_chunk, chunk, M.diagonal().copy())
                    for chunk in _iter_chunks(pair_indices, pair_chunk_size)
                ]
            for future in as_completed(futures):
                pair_map.update(future.result())

            for i, j in pair_indices:
                if backend == "process":
                    value = pair_map[(i, j)] - M[i, i] / 2.0 - M[j, j] / 2.0
                else:
                    value = pair_map[(i, j)]
                M[i, j] = value
                if symmetric:
                    M[j, i] = value
        else:
            dev = _make_device(device_name, qubits)
            circuit_od = _build_circuit_od(dev)
            if symmetric:
                for i in range(n_states):
                    for j in range(i + 1, n_states):
                        mtmp = circuit_od(
                            params,
                            list1[i],
                            list1[j],
                            wires,
                            s_wires,
                            d_wires,
                            null_state,
                            ash_excitation,
                            tapered_ansatz,
                        )
                        value = _to_real_scalar(mtmp) - M[i, i] / 2.0 - M[j, j] / 2.0
                        M[i, j] = value
                        M[j, i] = value
            else:
                for i in range(n_states):
                    for j in range(n_states):
                        if i != j:
                            mtmp = circuit_od(
                                params,
                                list1[i],
                                list1[j],
                                wires,
                                s_wires,
                                d_wires,
                                null_state,
                                ash_excitation,
                                tapered_ansatz,
                            )
                            M[i, j] = _to_real_scalar(mtmp) - M[i, i] / 2.0 - M[j, j] / 2.0
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    eigvals, eigvecs = np.linalg.eig(M)
    order = np.argsort(np.real(eigvals))
    eigvals_sorted = np.asarray(eigvals[order])
    eigvecs_sorted = np.asarray(eigvecs[:, order], dtype=complex)
    values.append(eigvals_sorted)
    if not return_details:
        return values
    details = {
        "projected_matrix": np.asarray(M, dtype=complex),
        "eigenvectors": eigvecs_sorted,
        "basis_occupations": [list(int(v) for v in occ) for occ in list1],
    }
    details.update(brg_details)
    return values, details