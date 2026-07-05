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
# # Quantum Chemistry for Spectroscopy in EUV Lithography
# **Team Name:** Entangled Trio
#
# This notebook demonstrates the calculation of $\rm H_2O$ spectra:
# - **Photoabsorption Spectrum**: Using the time-domain Green's function propagated via a second-order Trotter product formula on a Compressed Double-Factorized (CDF) Hamiltonian, based on the work by [Kharazi et al.](https://arxiv.org/abs/2602.20234)
# - **Auger Electron Spectrum**: Using the Generative Quantum Eigensolver (GQE) combined with the Quantum Self-Consistent Equation-of-Motion (q-sc-EOM) and One-Center Approximation (OCA), based on the work by [Keithley et al.](https://arxiv.org/abs/2603.12859)
#
# Below is an image taken from both papers and illustrates how we combined both approaches.
#


# %% [markdown]
# ![pipeline](../../img/pipeline.png)

# %% [markdown]
# ## Setup

# %%
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import os
import sys
import pickle
import matplotlib.pyplot as plt

# Resolve absolute paths
def get_current_file_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        if os.path.exists("mitsubishi/phase_3/code"):
            return os.path.abspath("mitsubishi/phase_3/code")
        return os.getcwd()

current_file_dir = get_current_file_dir()

# %% [markdown]
# ## Parameter Settings
#
# Configure the execution parameters for the simulation:
#
# * **`USE_CUDA`**: If `True`, enable cuQuantum using PennyLane device "lightning.gpu" and enable CUDA kernels in PyTorch through the "cuda" device.
# * **`USE_DIT`**: If `True`, enable Diffusion Transformer (DIT) for GQE training as an alternative to the auto-regressive GPTnano.
# * **`USE_ECP_AVAS`**: If `True`, apply Effective Core Potentials (ECP) with active space selection (AVAS) for heavy atoms (like Iodine) to reduce the molecular active space size.
# * **`USE_TAPER`**: If `True`, apply $Z_2$ symmetry tapering classically to aggressively reduce active qubits.
# * **`USE_CDF_*`**: Enable Compressed Double Factorization (CDF) to calculate a low-rank Hamiltonian approximation with linear scaling.
#     * **`USE_CDF_EMISSION`**: If `True`, apply CDF to the Auger emission Hamiltonian, yielding a low-rank two-body approximation.
#     * **`USE_CDF_ABSORPTION`**: If `True`, apply CDF to the absorption spectrum Hamiltonian, using factorized Trotter evolution instead of the full molecular Hamiltonian.
# * **`ANALYTIC_QUANTUM_ABSORPTION`**: If `True`, use exact (analytic) statevector expectation values for the Hadamard-test measurements in the absorption simulation, removing shot noise. Set to `False` to simulate finite-shot sampling with an exponentially-decaying kernel allocation.
# * **`SAVE_PLOTS`**: If `True`, export matplotlib rasterized copies of the Auger and absorption spectra to PNG files alongside the interactive hvplot output.

# %%
target_molecule = "H2O"
USE_CUDA = False
USE_DIT = True
USE_ECP_AVAS = True
USE_TAPER = True
USE_CDF_EMISSION = False
USE_CDF_ABSORPTION = True
ANALYTIC_QUANTUM_ABSORPTION = True
SAVE_PLOTS = False

cache_dir = os.path.join(current_file_dir, "data", "qsceom")
setting_suffix = ""
if USE_ECP_AVAS:
    setting_suffix += "_ecpavas"
if USE_TAPER:
    setting_suffix += "_taper"

cache_path = os.path.join(cache_dir, f"qsceom_cache{setting_suffix}.pkl")
has_cache = os.path.exists(cache_path)
openmolcas_filepath = os.path.abspath(os.path.join(current_file_dir, "data", "openmolcas", f"{target_molecule.lower()}_aes.gqe_integrals.h5"))

seq_len = 4
trial_name = f"trial_{target_molecule.lower()}{setting_suffix}"
save_dir = os.path.abspath(os.path.join(current_file_dir, "data", f"seq_len={seq_len}/{trial_name}"))


# %% [markdown]
# ## Compressed Double Factorization (CDF) Helper
#
# We define a helper function (`compute_cdf_hamiltonian`) to perform the Compressed Double Factorization (CDF) tensor reduction pipeline:
#
# 1. **Chemist Notation & BLISS Conversion**:
#    * Converts molecular one-body and two-body integrals into chemist notation ($T_{pq}$ and $V_{pqrs}$).
#    * Applies the **Block-Invariant Symmetry Shift (BLISS)** to shift eigenvalues and minimize the Hamiltonian's one-norm.
# 2. **Compressed Factorization**:
#    * Decomposes the shifted two-body tensor into low-rank core and leaf tensor fragments using a compressed numerical fitting routine with **L2 regularization** via `optax` and `jax`.
# 3. **One-Body Correction & Error Metrics**:
#    * Computes a corrected one-body tensor to absorb the two-body Z-gate reduction artifacts.
#    * Calculates the Frobenius norm reconstruction error ($||V_{\text{shift}} - V'_{\text{shift}}||_F$) to monitor approximation fidelity.

# %%
import numpy as np
import pennylane as qml
import pubchempy as pcp

def compute_cdf_hamiltonian(molecule, hamiltonian, use_ecp_avas, active_electrons_val, active_orbitals_val, molecule_name, use_bliss=True):
    """Computes the Compressed Double Factorization (CDF) representation of the Hamiltonian."""
    print(f"Constructing CDF Hamiltonian for {molecule_name}...")
    import optax
    import jax
    
    if use_ecp_avas and "I" in molecule.symbols:
        from qsceom import _build_pyscf_molecular_integrals
        nuc_core, one_body, two_mo_of = _build_pyscf_molecular_integrals(
            symbols=molecule.symbols,
            geometry=molecule.coordinates,
            basis="sto-3g",
            charge=molecule.charge,
            active_electrons=active_electrons_val,
            active_orbitals=active_orbitals_val,
            use_ecp_avas=True,
        )
        V_active = two_mo_of.transpose(0, 3, 1, 2)
        two_body = V_active.transpose(0, 3, 2, 1)
    elif use_ecp_avas:
        core_list, active_list = qml.qchem.active_space(
            molecule.n_electrons, molecule.n_orbitals, 
            active_electrons=active_electrons_val, active_orbitals=active_orbitals_val
        )
        nuc_core, one_body, two_body = qml.qchem.electron_integrals(
            molecule, core=core_list, active=active_list
        )()
    else:
        nuc_core, one_body, two_body = qml.qchem.electron_integrals(molecule)()

    # Convert to chemist notation
    two_chem = qml.math.swapaxes(two_body, 1, 3)  # V_pqrs
    one_chem = one_body - 0.5 * qml.math.einsum("pqss", two_body)  # T_pq

    if use_bliss:
        # Apply Block-Invariant Symmetry Shift (BLISS)
        core_shift, one_shift, two_shift = qml.qchem.symmetry_shift(
            nuc_core, one_chem, two_chem, n_elec=active_electrons_val
        )
    else:
        core_shift = [nuc_core]
        one_shift = one_chem
        two_shift = two_chem

    # Compressed Double Factorization (CDF)
    factors, two_body_cores, two_body_leaves = qml.qchem.factorize(
        two_shift, tol_factor=1e-2, cholesky=True, compressed=True, regularization="L2"
    )

    # One-body correction
    two_core_prime = (qml.math.eye(active_orbitals_val) * two_body_cores.sum(axis=-1)[:, None, :])
    one_body_extra = qml.math.einsum('tpk,tkk,tqk->pq', two_body_leaves, two_core_prime, two_body_leaves)

    # Factorize corrected one-body tensor
    one_body_eigvals, one_body_eigvecs = np.linalg.eigh(one_shift + one_body_extra)
    one_body_cores = np.expand_dims(np.diag(one_body_eigvals), axis=0)
    one_body_leaves = np.expand_dims(one_body_eigvecs, axis=0)

    # Reconstruction error: ||two_shift - approx_two_shift||_F
    approx_two_shift = qml.math.einsum(
        "tpk,tqk,tkl,trl,tsl->pqrs",
        two_body_leaves, two_body_leaves, two_body_cores, two_body_leaves, two_body_leaves
    )
    reconstruction_error = float(qml.math.norm(two_shift - approx_two_shift))

    return {
        "nuc_constant": float(core_shift[0]),
        "core_tensors": qml.math.concatenate((one_body_cores, two_body_cores), axis=0),
        "leaf_tensors": qml.math.concatenate((one_body_leaves, two_body_leaves), axis=0),
        "reconstruction_error": reconstruction_error,
        "standard_hamiltonian": hamiltonian,
        "n_orbitals": active_orbitals_val,
    }

# %% [markdown]
# ## Molecular Data Generation (Main Helper)
#
# We define a helper function (`generate_molecule_data`) to construct the molecule Hamiltonian, operator pool, Hartree-Fock state, and expected ground state energy. This function supports several execution paths:
#
# 1. **Data Source Selection**:
#    * **`qchem` path**: Loads precomputed molecular coordinates and settings directly from PennyLane's curated chemistry datasets.
#    * **`pubchem` path**: Dynamically queries the PubChem 3D database (using compound names or CIDs) to fetch geometries for custom target molecules like IMePh.
# 2. **Effective Core Potential & Active Space Selection (ECP + AVAS)**:
#    * For Iodine-containing systems (like IMePh), replaces the 46 core electrons with a pseudopotential (LANL2DZ ECP) and extracts a classically constructed CAS(24e, 18o) active space to ensure classical tractability.
# 3. **Z2 Symmetry Tapering**:
#    * If `USE_TAPER=True`, finds the Z2 generators of the molecular Hamiltonian and projects the Hamiltonian, HF state, and the operator pool into the optimal sector, reducing the active qubits count by the number of symmetries found.
# 4. **CDF Integration**:
#    * If `use_cdf=True`, delegates the integral transformation and compressed factorization to `compute_cdf_hamiltonian`.

# %%
def generate_molecule_data(molecule_name="H2", source="qchem", local_dataset_path=None, use_ecp_avas=False, use_cdf=False):
    if local_dataset_path is None:
        local_dataset_path = os.path.join(get_current_file_dir(), "data")
    # Get the time set T
    op_times = np.sort(np.array([-2**k for k in range(1, 5)] + [2**k for k in range(1, 5)]) / 160)

    # Build operator set P for each molecule
    molecule_data = dict()
    
    if source == "pubchem":
        import pickle
        cache_suffix = "_ecpavas" if use_ecp_avas else ""
        cache_file = os.path.join(local_dataset_path, f"{molecule_name}_pubchem{cache_suffix}.pkl")
        if os.path.exists(cache_file):
            print(f"Loading {molecule_name} (pubchem) from local cache: {cache_file}")
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                return {molecule_name: cached_data}
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

    if source == "qchem":
        import h5py
        from pathlib import Path

        try:
            # Try loading locally first
            datasets = qml.data.load("qchem", molname=molecule_name, folder_path=local_dataset_path)
        except Exception:
            # Clean up corrupted/truncated HDF5 files in the local path and current directory
            for path in [Path(local_dataset_path), Path(".")]:
                if path.exists():
                    for p in path.rglob("*.h5"):
                        try:
                            with h5py.File(p, "r") as f:
                                _ = list(f.keys())
                        except Exception as e:
                            print(f"Removing corrupted dataset file {p} due to: {e}")
                            p.unlink(missing_ok=True)
            try:
                datasets = qml.data.load("qchem", molname=molecule_name, folder_path=local_dataset_path)
            except Exception:
                datasets = qml.data.load("qchem", molname=molecule_name)
            
        dataset = datasets[0]
        molecule = dataset.molecule
        symbols = molecule.symbols
        coords = molecule.coordinates
        
        num_electrons, num_qubits = molecule.n_electrons, 2 * molecule.n_orbitals
        hf_state = dataset.hf_state
        hamiltonian = dataset.hamiltonian
        expected_ground_state_E = dataset.fci_energy
        
    elif source == "pubchem":
        # Fetch from PubChem
        # Map specific molecules to their CID for robust 3D structure queries
        cid_map = {
            "4-iodo-2-methylphenol": 143713,
            "IMePh": 143713
        }
        
        if molecule_name in cid_map:
            compounds = pcp.get_compounds(cid_map[molecule_name], 'cid', record_type='3d')
            if not compounds or not hasattr(compounds[0].atoms[0], 'x') or compounds[0].atoms[0].x is None:
                compounds = pcp.get_compounds(cid_map[molecule_name], 'cid')
        elif isinstance(molecule_name, int) or (isinstance(molecule_name, str) and molecule_name.isdigit()):
            compounds = pcp.get_compounds(int(molecule_name), 'cid', record_type='3d')
            if not compounds or not hasattr(compounds[0].atoms[0], 'x') or compounds[0].atoms[0].x is None:
                compounds = pcp.get_compounds(int(molecule_name), 'cid')
        else:
            compounds = pcp.get_compounds(molecule_name, 'name', record_type='3d')
            if not compounds or not hasattr(compounds[0].atoms[0], 'x') or compounds[0].atoms[0].x is None:
                compounds = pcp.get_compounds(molecule_name, 'name')
            
        c = compounds[0]
        symbols = [atom.element for atom in c.atoms]
        # Extract coordinates, defaulting to 0.0 if not present
        coords = np.array([[getattr(atom, 'x', 0.0) or 0.0, 
                            getattr(atom, 'y', 0.0) or 0.0, 
                            getattr(atom, 'z', 0.0) or 0.0] for atom in c.atoms])
        
    # Build Hamiltonian using ECP + AVAS if enabled
    if use_ecp_avas and "I" in symbols:
        # Set active space parameters for IMePh (default: 24 electrons, 18 orbitals)
        active_electrons = 24
        active_orbitals = 18
        
        # Print progress
        print(f"\n--- {molecule_name} Qubit Reduction Progress ---")
        
        # 1. All-electron qubits
        try:
            from pyscf import gto
            mol_all = gto.Mole()
            mol_all.atom = list(zip(symbols, coords))
            mol_all.basis = "sto-3g"
            mol_all.build(verbose=0)
            print(f"[Step 1] All-Electron Qubits: {2 * mol_all.nao} (sto-3g)")
        except Exception:
            pass
        
        # 2. ECP Reduced Qubits
        try:
            mol_ecp = gto.Mole()
            mol_ecp.atom = list(zip(symbols, coords))
            basis_map = {}
            ecp_map = {}
            for s in symbols:
                if s == "I":
                    basis_map[s] = "lanl2dz"
                    ecp_map[s] = "lanl2dz"
                else:
                    basis_map[s] = "sto-3g"
            mol_ecp.basis = basis_map
            mol_ecp.ecp = ecp_map
            mol_ecp.build(verbose=0)
            print(f"[Step 2] ECP Reduced Qubits:  {2 * mol_ecp.nao} (lanl2dz ECP for I)")
        except Exception:
            pass
            
        # 3. ECP+AVAS Active Qubits
        print(f"[Step 3] ECP+AVAS Active Qubits: {2 * active_orbitals} (CAS({active_electrons}e, {active_orbitals}o))")

        # Construct H from ECP integrals classically
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from qsceom import _build_pyscf_molecular_integrals, _expand_spatial_integrals_to_spin_orbital
        from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner
        
        core_constant, one_mo, two_mo = _build_pyscf_molecular_integrals(
            symbols=symbols,
            geometry=coords,
            basis="sto-3g",
            charge=0,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            use_ecp_avas=True,
        )
        one_spin, two_spin = _expand_spatial_integrals_to_spin_orbital(one_mo, two_mo)
        interaction = InteractionOperator(
            float(core_constant),
            one_spin,
            two_spin,
        )
        ferm_op = get_fermion_operator(interaction)
        qubit_op = jordan_wigner(ferm_op)
        hamiltonian = qml.from_openfermion(qubit_op)
        
        num_qubits = 2 * active_orbitals
        num_electrons = active_electrons
        hf_state = qml.qchem.hf_state(num_electrons, num_qubits)
        expected_ground_state_E = None
    else:
        # Build all-electron molecular Hamiltonian if not already defined (i.e. if pubchem source)
        if source == "pubchem":
            hamiltonian, num_qubits = qml.qchem.molecular_hamiltonian(symbols, coords, load_data=True)
            
            # Simple estimation of electrons (assuming neutral molecule)
            electron_map = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'I': 53}
            num_electrons = sum([electron_map.get(s, 0) for s in symbols]) 
            
            hf_state = qml.qchem.hf_state(num_electrons, num_qubits)
            expected_ground_state_E = None
        
        # Print progress for water or small molecules
        print(f"\n--- {molecule_name} Qubit Reduction Progress ---")
        try:
            print(f"[Step 1] All-Electron Qubits: {num_qubits} (sto-3g)")
            print(f"[Step 2] ECP Reduced Qubits:  {num_qubits} (No ECP applied)")
            print(f"[Step 3] ECP+AVAS Active Qubits: {num_qubits} (All-Electron)")
        except Exception:
            pass

    active_electrons_save = num_electrons
    active_orbitals_save = num_qubits // 2

    singles, doubles = qml.qchem.excitations(num_electrons, num_qubits)
    double_excs = [qml.DoubleExcitation(time, wires=double) for double in doubles for time in op_times]
    single_excs = [qml.SingleExcitation(time, wires=single) for single in singles for time in op_times]
    identity_ops = [qml.PhaseShift(0.0, wires=0) for time in op_times] # For Identity
    operator_pool = double_excs + single_excs + identity_ops

    # 4. Z2 Tapering
    if USE_TAPER:
        generators = qml.symmetry_generators(hamiltonian)
        if len(generators) > 0:
            paulixops = qml.paulix_ops(generators, num_qubits)
            paulix_sector = qml.qchem.optimal_sector(hamiltonian, generators, num_electrons)
            
            # Taper the operator pool
            tapered_pool = []
            wire_order = list(range(num_qubits))
            for op in operator_pool:
                try:
                    t_ops = qml.qchem.taper_operation(op, generators, paulixops, paulix_sector, wire_order)
                    if len(t_ops) > 0:
                        if len(t_ops) == 1:
                            tapered_pool.append(t_ops[0])
                        else:
                            tapered_pool.append(qml.prod(*t_ops))
                except Exception:
                    pass
            
            hamiltonian = qml.taper(hamiltonian, generators, paulixops, paulix_sector)
            hf_state = qml.qchem.taper_hf(generators, paulixops, paulix_sector, num_electrons, num_qubits)
            operator_pool = tapered_pool
            
            qubits_tapered = num_qubits - len(generators)
            print(f"[Step 4] Z2 Tapered Qubits:    {qubits_tapered} ({len(generators)} symmetry generators found)")
            num_qubits = qubits_tapered
        else:
            print(f"[Step 4] Z2 Tapered Qubits:    {num_qubits} (No symmetry generators found)")
    else:
        print(f"[Step 4] Z2 Tapered Qubits:    {num_qubits} (Tapering disabled)")
        
    print("--------------------------------------\n")

    if use_cdf:
        if source == "qchem":
            molecule = dataset.molecule
        else:
            molecule = qml.qchem.Molecule(symbols, coords)

        if use_ecp_avas:
            active_electrons_val = active_electrons_save
            active_orbitals_val = active_orbitals_save
        else:
            active_electrons_val = molecule.n_electrons
            active_orbitals_val = molecule.n_orbitals

        hamiltonian = compute_cdf_hamiltonian(
            molecule, hamiltonian, use_ecp_avas, active_electrons_val, active_orbitals_val, molecule_name
        )
    
    molecule_data[molecule_name] = {
        "op_pool": np.array(operator_pool, dtype=object), 
        "num_qubits": num_qubits,
        "hf_state": hf_state,
        "hamiltonian": hamiltonian,
        "expected_ground_state_E": expected_ground_state_E,
        "symbols": symbols,
        "geometry": coords,
        "active_electrons": active_electrons_save,
        "active_orbitals": active_orbitals_save,
    }
    
    if source == "pubchem":
        try:
            import pickle
            cache_suffix = "_ecpavas" if use_ecp_avas else ""
            cache_file = os.path.join(local_dataset_path, f"{molecule_name}_pubchem{cache_suffix}.pkl")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            print(f"Caching computed {molecule_name} (pubchem) to: {cache_file}")
            with open(cache_file, "wb") as f:
                pickle.dump(molecule_data[molecule_name], f)
        except Exception as e:
            print(f"Failed to write cache: {e}")
            
    return molecule_data


# %% [markdown]
# ## Loading and Configuring Molecular Parameters
#
# For comparison, we load:
# 1. $\rm H_2O$ via PL datasets (qchem)
# 2. $\rm H_2O$ from PubChem
#
# This allows us to extract and compare coordinates, Hamiltonian terms, Hartree-Fock reference states, and the full operator pools of single and double excitations. Note that we will proceed with choice 1 ($\rm H_2O$ via qchem) for the rest of the notebook.

# %%
# target_molecule configured at the top of parameters block

# Helper to safely get the length of hamiltonian terms across PL versions
def get_hamiltonian_terms_len(h):
    if isinstance(h, dict) and "core_tensors" in h:
        return len(h["core_tensors"])  # Returns the number of fragments L (including one-body)
    return len(getattr(h, "ops", getattr(h, "operands", [])))

# 1. Load via qchem
qchem_molecule_data = generate_molecule_data(target_molecule, source="qchem", use_ecp_avas=USE_ECP_AVAS, use_cdf=USE_CDF_EMISSION)
qchem_data = qchem_molecule_data[target_molecule]

print(f"--- {target_molecule} (qchem) ---")
print(f"Number of Qubits: {qchem_data['num_qubits']}")
print(f"Operator Pool Size: {len(qchem_data['op_pool'])}")
print(f"HF State: {qchem_data['hf_state']}")
print(f"FCI Energy: {qchem_data['expected_ground_state_E']}")
print(f"Hamiltonian terms: {get_hamiltonian_terms_len(qchem_data['hamiltonian'])}")

# 2. Load via PubChem
pubchem_molecule_data = generate_molecule_data(target_molecule, source="pubchem", use_ecp_avas=USE_ECP_AVAS, use_cdf=USE_CDF_EMISSION)
pubchem_data = pubchem_molecule_data[target_molecule]

print(f"\n--- {target_molecule} (PubChem) ---")
print(f"Number of Qubits: {pubchem_data['num_qubits']}")
print(f"Operator Pool Size: {len(pubchem_data['op_pool'])}")
print(f"HF State: {pubchem_data['hf_state']}")
print(f"FCI Energy: {pubchem_data['expected_ground_state_E']}")
print(f"Hamiltonian terms: {get_hamiltonian_terms_len(pubchem_data['hamiltonian'])}")

op_pool = qchem_data["op_pool"]
num_qubits = qchem_data["num_qubits"]
init_state = qchem_data["hf_state"]
hamiltonian = qchem_data["hamiltonian"]
grd_E = qchem_data["expected_ground_state_E"]
op_pool_size = len(op_pool)

# Identify and print the emitter atom
symbols = qchem_data["symbols"]
if "I" in symbols:
    emitter_atom_idx_info = symbols.index("I")
else:
    non_h_indices_info = [i for i, sym in enumerate(symbols) if sym != "H"]
    emitter_atom_idx_info = non_h_indices_info[0] if non_h_indices_info else 0
print(f"Detected Emitter Atom for OCA: {symbols[emitter_atom_idx_info]} (Index {emitter_atom_idx_info})")

# %% [markdown]
# # Step 1: Absorption Spectroscopy Simulation
#
# In this section, we implement the time-domain absorption spectroscopy simulation for the $\rm H_2O$ active space in the 10-100 eV range.
# Since $\rm H_2O$ has only valence excitations (no core-level transitions) below 100 eV in the minimal STO-3G basis, we expect the classical baseline to show peaks in the 10-45 eV region and be flat in the 45-100 eV range (encompassing the 92 eV EUV regime).
#
# We also implement caching to avoid re-running the expensive time-domain simulation and classical transition dipole calculations on subsequent runs.

# %% [markdown]
# ## Step 1.1: Molecular Setup and PySCF CASCI Ground State
# We initialize the PySCF `Mole` and run Hartree-Fock to get molecular orbital coefficients. 
# Then, we compute the ground state wavefunction under the complete active space configuration interaction (CASCI) method and convert it to a PennyLane-compatible statevector.

# %%
from pyscf import gto, scf, mcscf
from scipy.sparse import coo_matrix
from pyscf.fci.cistring import addrs2str
from pennylane.qchem.convert import _sign_chem_to_phys, _wfdict_to_statevector
from jax import config
from itertools import product
import pandas as pd
import hvplot.pandas
import pickle

# Ensure JAX uses float64 for factorization alignment
config.update("jax_enable_x64", True)

def get_active_space_dipole_operators(symbols, geometry, active_electrons, active_orbitals, use_ecp_avas=False):
    import numpy as np
    import pennylane as qml
    from pyscf import gto, scf
    from openfermion import FermionOperator, jordan_wigner
    
    mol = gto.Mole()
    mol.atom = list(zip(symbols, geometry))
    if use_ecp_avas and "I" in symbols:
        basis_map = {}
        ecp_map = {}
        for s in symbols:
            if s == "I":
                basis_map[s] = "lanl2dz"
                ecp_map[s] = "lanl2dz"
            else:
                basis_map[s] = "sto-3g"
        mol.basis = basis_map
        mol.ecp = ecp_map
    else:
        mol.basis = "sto-3g"
    mol.charge = 0
    mol.spin = 0
    mol.unit = "bohr"
    mol.build(verbose=0)
    
    mf = scf.RHF(mol)
    mf.kernel(verbose=0)
    
    r_ao = mol.intor('int1e_r')
    C = mf.mo_coeff
    r_mo = np.einsum('pi,kpq,qj->kij', C, r_ao, C)
    
    n_electrons = mol.nelectron
    n_core = (n_electrons // 2) - (active_electrons // 2)
    active_indices = list(range(n_core, n_core + active_orbitals))
    
    core_dipole = np.zeros(3)
    for i in range(n_core):
        core_dipole += 2.0 * r_mo[:, i, i]
        
    dipole_ops = []
    for rho in range(3):
        op = FermionOperator('', core_dipole[rho])
        for u_idx, u in enumerate(active_indices):
            for v_idx, v in enumerate(active_indices):
                coeff = r_mo[rho, u, v]
                if abs(coeff) < 1e-8:
                    continue
                op += FermionOperator(f'{2*u_idx}^ {2*v_idx}', coeff)
                op += FermionOperator(f'{2*u_idx+1}^ {2*v_idx+1}', coeff)
                
        qubit_op = jordan_wigner(op)
        pl_op = qml.from_openfermion(qubit_op)
        dipole_ops.append(pl_op)
        
    return dipole_ops

# Define absorption caching directory and path
absorption_cache_dir = os.path.join(current_file_dir, "data", "absorption")
absorption_cache_path = os.path.join(
    absorption_cache_dir, 
    f"absorption_cache{'_cdf' if USE_CDF_ABSORPTION else '_nocdf'}.pkl"
)
use_cache_absorption = os.path.exists(absorption_cache_path)

if use_cache_absorption:
    print(f"Pre-computed absorption results found at {absorption_cache_path}. Loading cache...")
    with open(absorption_cache_path, "rb") as f_abs:
        abs_cache_data = pickle.load(f_abs)
    expvals = abs_cache_data["expvals"]
    dipole_norm = abs_cache_data["dipole_norm"]
    wgrid_ev = abs_cache_data["wgrid_ev"]
    spectrum = abs_cache_data["spectrum"]
    spectrum_classical = abs_cache_data["spectrum_classical"]
    print("Successfully loaded pre-computed absorption simulation data.")
else:
    print("No absorption cache found. Computing simulation...")

# Retrieve symbols and geometry from qchem_data
symbols = qchem_data["symbols"]
geometry = qchem_data["geometry"]

# Build PySCF Mole object and run SCF
if USE_ECP_AVAS and "I" in symbols:
    mol = gto.Mole()
    mol.atom = list(zip(symbols, geometry))
    basis_map = {}
    ecp_map = {}
    for s in symbols:
        if s == "I":
            basis_map[s] = "lanl2dz"
            ecp_map[s] = "lanl2dz"
        else:
            basis_map[s] = "sto-3g"
    mol.basis = basis_map
    mol.ecp = ecp_map
    mol.charge = 0
    mol.spin = 0
    mol.unit = "bohr"
    mol.build(verbose=0)
else:
    mol = gto.Mole(atom=list(zip(symbols, geometry)), basis="sto-3g", symmetry=None, unit="bohr")
    mol.build(verbose=0)

hf = scf.RHF(mol)
hf.run(verbose=0)

# Build PennyLane Molecule object and match MO coefficients
if not (USE_ECP_AVAS and "I" in symbols):
    mole = qml.qchem.Molecule(symbols, geometry, basis_name="sto-3g", unit="bohr")
    _, coeffs, _, _, _ = qml.qchem.scf(mole)()
    hf.mo_coeff = coeffs

# Setup active space
n_cas = qchem_data["active_orbitals"]
n_electron_cas = qchem_data["active_electrons"]
n_core = (mol.nelectron - n_electron_cas) // 2

if not use_cache_absorption:
    # Solve ground state to convert to PennyLane statevector
    mycasci = mcscf.CASCI(hf, ncas=n_cas, nelecas=n_electron_cas)
    mycasci.run(verbose=0)
    casci_state = mycasci.ci
    casci_state[abs(casci_state) < 1e-6] = 0
    E_i = mycasci.e_tot

    # Convert CASCI ground state vector to dictionary format and adjust spin ordering sign
    sparse_cascimatr = coo_matrix(casci_state, shape=np.shape(mycasci.ci), dtype=float)
    row, col, dat = sparse_cascimatr.row, sparse_cascimatr.col, sparse_cascimatr.data
    strs_row = addrs2str(n_cas, n_electron_cas // 2, row)
    strs_col = addrs2str(n_cas, n_electron_cas // 2, col)
    wf_casci_dict = dict(zip(list(zip(strs_row, strs_col)), dat))
    wf_casci_dict = _sign_chem_to_phys(wf_casci_dict, n_cas)
    wf_casci = _wfdict_to_statevector(wf_casci_dict, n_cas)

# %% [markdown]
# ## Step 1.2: Transition Dipole Moments Construction
# We generate the dipole moment operators in the active space and project the ground state wavefunction to obtain the initial states $\hat{m}_\rho |I\rangle$ along the $x, y, z$ Cartesian coordinates.

# %%
if not use_cache_absorption:
    # Dipole moment operator in molecular orbital basis
    m_rho = get_active_space_dipole_operators(
        symbols, geometry, n_electron_cas, n_cas, use_ecp_avas=USE_ECP_AVAS
    )
    rhos = range(len(m_rho))

    wf_dipole = []
    dipole_norm = []

    # Project initial state using the dipole moment operators
    for rho in rhos:
        dipole_matrix_rho = qml.matrix(m_rho[rho], wire_order=range(2 * n_cas))
        wf = dipole_matrix_rho.dot(wf_casci)
        if np.allclose(wf, np.zeros_like(wf)):
            wf_dipole.append(wf)
            dipole_norm.append(0.0)
        else:
            norm_val = np.linalg.norm(wf)
            dipole_norm.append(norm_val)
            wf_dipole.append(wf / norm_val)

# %% [markdown]
# ## Step 1.3: Compressed Double Factorization (CDF) of the Hamiltonian
# We use the pre-defined helper method `compute_cdf_hamiltonian` to perform Compressed Double Factorization on the active space Hamiltonian.
# This yields the low-rank two-body fragments, the corrected one-body term, and the constant energy shift (including Block-Invariant Symmetry Shift, BLISS, for Trotter error minimization).

# %%
if not use_cache_absorption:
    if USE_CDF_ABSORPTION:
        # Call the helper method to compute the CDF Hamiltonian representation
        cdf_res = compute_cdf_hamiltonian(
            molecule=mole,
            hamiltonian=None,
            use_ecp_avas=USE_ECP_AVAS,
            active_electrons_val=n_electron_cas,
            active_orbitals_val=n_cas,
            molecule_name="H2O",
            use_bliss=False
        )

        core_constant = cdf_res["nuc_constant"]
        _Z = cdf_res["core_tensors"]
        _U = cdf_res["leaf_tensors"]
    else:
        # Generate the standard molecular Hamiltonian on the active space
        h_exact, _ = qml.qchem.molecular_hamiltonian(
            symbols, geometry, active_electrons=n_electron_cas, active_orbitals=n_cas, unit="bohr"
        )
        # Map wires of h_exact to [1, ..., 2 * n_cas] to avoid conflict with the control qubit on wire 0
        h_mapped = qml.map_wires(h_exact, {i: i + 1 for i in range(2 * n_cas)})

# %% [markdown]
# ## Step 1.4: Quantum Circuit Setup (Hadamard Test & Trotter Evolution)
# We define the QNodes and auxiliary functions implementing the first- and second-order Trotter product formula step.

# %%
if not use_cache_absorption:
    # Setup simulated device and QNodes
    device_type = "lightning.gpu" if USE_CUDA else "lightning.qubit"
    dev_prop = qml.device(device_type, wires=int(2*n_cas) + 1)

    # Simulation parameters
    eta = 0.05
    H_norm = np.pi
    tau = np.pi / (2 * H_norm)

    @qml.qnode(dev_prop)
    def initial_circuit(wf):
        qml.StatePrep(wf, wires=dev_prop.wires.tolist()[1:])
        qml.Hadamard(wires=0)
        return qml.state()

    if USE_CDF_ABSORPTION:
        def U_rotations(U, control_wires):
            U_spin = qml.math.kron(U, qml.math.eye(2))
            qml.BasisRotation(
                unitary_matrix=U_spin, wires=[int(i + control_wires) for i in range(2 * n_cas)]
            )

        def Z_rotations(Z, step, is_one_electron_term, control_wires):
            if is_one_electron_term:
                for sigma in range(2):
                    for i in range(n_cas):
                        qml.ctrl(
                            qml.X(wires=int(2*i + sigma + control_wires)),
                            control=range(control_wires),
                            control_values=0,
                        )
                        qml.RZ(-Z[i, i] * step / 2, wires=int(2*i + sigma + control_wires))
                        qml.ctrl(
                            qml.X(wires=int(2*i + sigma + control_wires)),
                            control=range(control_wires),
                            control_values=0,
                        )
                globalphase = np.sum(Z) * step
            else:
                for sigma, tau_spin in product(range(2), repeat=2):
                    for i, k in product(range(n_cas), repeat=2):
                        if i != k or sigma != tau_spin:
                            qml.ctrl(qml.X(wires=int(2*i + sigma + control_wires)),
                                     control=range(control_wires), control_values=0)
                            qml.MultiRZ(Z[i, k] / 8.0 * step,
                                wires=[int(2*i + sigma + control_wires),
                                       int(2*k + tau_spin + control_wires)])
                            qml.ctrl(qml.X(wires=int(2 * i + sigma + control_wires)),
                                control=range(control_wires), control_values=0)
                globalphase = np.trace(Z)/4.0*step - np.sum(Z)*step/2.0
            qml.PhaseShift(-globalphase, wires=0)

        def first_order_trotter(step, prior_U, final_rotation, reverse=False):
            num_two_electron_fragments = _U.shape[0] - 1
            is_one_body = np.array([True] + [False] * num_two_electron_fragments)
            order = list(range(len(_Z)))

            if reverse:
                order = order[::-1]

            for fragment in order:
                U_rotations(prior_U @ _U[fragment], 1)
                Z_rotations(_Z[fragment], step, is_one_body[fragment], 1)
                prior_U = _U[fragment].T

            if final_rotation:
                U_rotations(prior_U, 1)

            qml.PhaseShift(-core_constant * step, wires=0)
            return prior_U

    # Define compiled step and measurement QNodes to avoid recompiling overhead
    @qml.qnode(dev_prop)
    def trotter_step_circuit(state_in):
        qml.StatePrep(state_in, wires=dev_prop.wires.tolist())
        if USE_CDF_ABSORPTION:
            prior_U = np.eye(n_cas)
            prior_U = first_order_trotter(tau / 2, prior_U=prior_U, 
                                          final_rotation=False, reverse=False)
            prior_U = first_order_trotter(tau / 2, prior_U=prior_U, 
                                          final_rotation=True, reverse=True)
        else:
            qml.ctrl(qml.TrotterProduct(h_mapped, time=tau, order=2), control=0)
        return qml.state()

    @qml.qnode(dev_prop)
    def measurement_circuit(state_in):
        qml.StatePrep(state_in, wires=dev_prop.wires.tolist())
        return [qml.expval(op) for op in [qml.PauliX(wires=0), qml.PauliY(wires=0)]]


# %% [markdown]
# ## Step 1.5: Time-Domain Quantum Simulation Run
# We execute the time-domain quantum propagation over 200 time-steps using a kernel-aware sampling distribution that allocates shots exponentially decaying with time.

# %%
if not use_cache_absorption:
    jmax = 100
    total_shots = 500 * 2 * jmax
    jrange = np.arange(1, 2 * int(jmax) + 1, 1)
    time_interval = tau * jrange

    def L_j(t_j):
        return np.exp(-eta * t_j)

    alpha = 1.1
    A = np.sum([L_j(alpha * t_j) for t_j in time_interval])
    shots_list = [int(round(total_shots * L_j(alpha * t_j) / A)) for t_j in time_interval]

    expvals = np.zeros((2, len(time_interval)))

    # Execute time-domain quantum simulation
    print("\n--- Running Absorption Spectrum Time-Domain Simulation ---")
    for rho in rhos:
        if dipole_norm[rho] == 0:
            continue
        state = initial_circuit(wf_dipole[rho])
        for i in range(0, len(time_interval)):
            state = trotter_step_circuit(state)
            if ANALYTIC_QUANTUM_ABSORPTION:
                measurement = measurement_circuit(state)
            else:
                shots = shots_list[i]
                measurement = qml.set_shots(measurement_circuit, shots)(state)
            expvals[:, i] += dipole_norm[rho]**2 * np.array(measurement).real

# %% [markdown]
# ## Step 1.6: Fourier Transform Post-Processing & Classical Validation Plot
# We Fourier-transform the quantum time-domain signal to the frequency domain (10-100 eV range).
# For validation, we solve for 250 CASCI roots using PySCF's FCI solver to obtain the exact classical reference spectrum, and plot them overlayed.

# %%
if not use_cache_absorption:
    # Discrete Fourier transform post-processing
    L_js = L_j(time_interval)
    f_domain_Greens_func = (
        lambda w: tau/(2*np.pi) * (np.sum(np.array(dipole_norm)**2) 
                + 2*np.sum(L_js * (expvals[0, :] * np.cos(time_interval * w)
                - expvals[1, :] * np.sin(time_interval * w)))))

    # Define range in eV (10 eV to 100 eV)
    wgrid_ev = np.linspace(10.0, 100.0, 10000)
    wgrid = E_i + wgrid_ev / 27.211386
    w_min, w_step = wgrid[0], wgrid[1] - wgrid[0]

    spectrum = np.array([f_domain_Greens_func(w) for w in wgrid])

    # Compute Classical spectrum reference (FCI / CASCI)
    print("Solving for classical transition dipole moments...")
    mycasci.fcisolver.nroots = 250
    mycasci.run(verbose=0)
    energies = mycasci.e_tot
    E_i_val = mycasci.e_tot[0]

    # Determine the dipole integrals using atomic orbitals and convert to molecular orbital basis
    dip_ints_ao = hf.mol.intor("int1e_r_cart", comp=3)
    mo_coeffs = coeffs[:, n_core : n_core + n_cas]
    dip_ints_mo = qml.math.einsum("ik,xkl,lj->xij", mo_coeffs.T, dip_ints_ao, mo_coeffs)

    def final_state_overlap(ci_id):
        t_dm1 = mycasci.fcisolver.trans_rdm1(
            mycasci.ci[0], mycasci.ci[ci_id], n_cas, n_electron_cas
        )
        return qml.math.einsum("xij,ji->x", dip_ints_mo, t_dm1)

    F_m_Is = np.array([final_state_overlap(i) for i in range(len(energies))])
    spectrum_classical_func = lambda E: (1 / np.pi) * np.sum(
                    [np.sum(np.abs(F_m_I)**2) * eta / ((E - e)**2 + eta**2)
                        for (F_m_I, e) in zip(F_m_Is, energies)])

    spectrum_classical = np.array([spectrum_classical_func(w) for w in wgrid])

    # Save calculated absorption results to cache
    os.makedirs(absorption_cache_dir, exist_ok=True)
    abs_cache_data = {
        "expvals": expvals,
        "dipole_norm": dipole_norm,
        "wgrid_ev": wgrid_ev,
        "spectrum": spectrum,
        "spectrum_classical": spectrum_classical,
    }
    with open(absorption_cache_path, "wb") as f_abs:
        pickle.dump(abs_cache_data, f_abs)
    print(f"Cached absorption spectrum simulation results to: {absorption_cache_path}")

# Plotting using hvplot
df_abs_dict = {
    "Energy (eV)": wgrid_ev,
    "Quantum (FT)": spectrum,
    "Classical (Exact)": spectrum_classical,
}
df_abs = pd.DataFrame(df_abs_dict)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(df_abs["Energy (eV)"], df_abs["Quantum (FT)"], label="Quantum (FT)", color="darkcyan")
ax.plot(df_abs["Energy (eV)"], df_abs["Classical (Exact)"], "--", label="Classical (Exact)", color="magenta")
ax.set_title("Simulated Absorption Spectrum for H2O Active Space")
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Absorption (arb.)")
ax.legend(loc="upper right")
fig.tight_layout()

if SAVE_PLOTS:
    plot_path = os.path.join(current_file_dir, "absorption_spectrum.png")
    fig.savefig(plot_path, dpi=300)
    print(f"Saved absorption spectrum plot to {plot_path}")

display(fig)
plt.close(fig)

# Done
# %% [markdown]
# # Step 2: Emission Spectroscopy Simulation
#
# In this section, we simulate the Auger emission spectrum of $\rm H_2O$ using:
# 1. Generative Quantum Eigensolver (GQE) to train an optimal reference state.
# 2. Quantum Self-Consistent Equation-of-Motion (q-sc-EOM) to calculate core-ionized and double-ionization energy levels (IP and DIP spaces).
# 3. Minimal basis projection and One-Center Approximation (OCA) to compute transition intensities.
# 4. Lorentzian broadening and visualization.

# %% [markdown]
# ## Compressed Double Factorization (CDF) Statistics
#
# If CDF is enabled, we display details of the factorization fragments, reconstruction error, and compression ratio.

# %%
if USE_CDF_EMISSION and isinstance(hamiltonian, dict) and "core_tensors" in hamiltonian:
    n_orbitals = hamiltonian["n_orbitals"]
    df_upper_bound = n_orbitals ** 2
    num_two_body_factors = len(hamiltonian["core_tensors"]) - 1
    reduction_pct = (1.0 - num_two_body_factors / df_upper_bound) * 100.0
    
    print("--- Compressed Double Factorization (CDF) Summary ---")
    print(f"Active orbitals (N): {n_orbitals}")
    print(f"Double Factorization terms upper bound (N^2): {df_upper_bound}")
    print(f"Compressed two-body fragments (L): {num_two_body_factors}")
    print(f"Compression reduction percentage: {reduction_pct:.2f}%")
    print(f"CDF Hamiltonian reconstruction error (Frobenius norm): {hamiltonian['reconstruction_error']:.2e}")

# %% [markdown]
# ## Subsequence Energy Evaluation
#
# We define standard PennyLane QNode and subsequence collation function to compute intermediate energies efficiently in a single simulator run using snapshots.
# If `USE_CUDA=True`, we use the cuQuantum-backed GPU-accelerated simulator `lightning.gpu`, otherwise we default to the standard CPU-backed `lightning.qubit`.

# %%
import pennylane as qml
import numpy as np

# Select PennyLane device based on USE_CUDA configuration
if USE_CUDA:
    dev = qml.device("lightning.gpu", wires=num_qubits)
else:
    dev = qml.device("lightning.qubit", wires=num_qubits)

# Resolve standard Hamiltonian if using CDF representation for stats
meas_hamiltonian = hamiltonian["standard_hamiltonian"] if isinstance(hamiltonian, dict) and "standard_hamiltonian" in hamiltonian else hamiltonian

@qml.qnode(dev)
def energy_circuit(gqe_ops):
    # Computes Eq. 1 from Nakaji et al. based on the selected unitary operators
    qml.BasisState(init_state, wires=range(num_qubits)) # Initial state <-- Hartree Fock state
    for op in gqe_ops:
        qml.Snapshot(measurement=qml.expval(meas_hamiltonian))
        qml.apply(op) # Applies each of the unitary operators
    return qml.expval(meas_hamiltonian)

energy_circuit = qml.snapshots(energy_circuit)

def get_subsequence_energies(op_seq):
    # Collates the energies of each subsequence for a batch of sequences
    energies = []
    for ops in op_seq:
        es = energy_circuit(ops)
        energies.append(
            [float(es[k]) for k in list(range(1, len(ops))) + ["execution_results"]]
        )
    return np.array(energies)

# Verify with a tiny sequence
print(get_subsequence_energies([[op_pool[0], op_pool[1]]]))

# %% [markdown]
# ## Dataset Generation
#
# Generate the training set consisting of random operator index sequences, their corresponding token sequences (pre-padded with starting/special tokens), and evaluate their subsequence energies.

# %%
import os
import pickle

# Resolve absolute paths
current_file_dir = get_current_file_dir()

cache_dir = os.path.join(current_file_dir, "data", "qsceom")
use_ecp_avas_val = globals().get("USE_ECP_AVAS", False)
use_taper_val = globals().get("USE_TAPER", False)
setting_suffix = ""
if use_ecp_avas_val:
    setting_suffix += "_ecpavas"
if use_taper_val:
    setting_suffix += "_taper"

cache_path = os.path.join(cache_dir, f"qsceom_cache{setting_suffix}.pkl")
has_cache = os.path.exists(cache_path)

seq_len = 4
trial_name = f"trial_h2o{setting_suffix}"
save_dir = os.path.abspath(os.path.join(current_file_dir, "data", f"seq_len={seq_len}/{trial_name}"))

if not has_cache:
    dataset_cache_file = os.path.join(save_dir, "gqe_dataset.pkl")
    if os.path.exists(dataset_cache_file):
        print(f"Loading GQE training dataset from cache: {dataset_cache_file}")
        with open(dataset_cache_file, "rb") as f:
            ds_cache = pickle.load(f)
        train_op_pool_inds = ds_cache["train_op_pool_inds"]
        train_op_seq = ds_cache["train_op_seq"]
        train_token_seq = ds_cache["train_token_seq"]
        train_sub_seq_en = ds_cache["train_sub_seq_en"]
        train_size = len(train_op_seq)
    else:
        # Generate sequence of indices of operators in vocab
        train_size = 1024
        os.makedirs(save_dir, exist_ok=True)

        train_op_pool_inds = np.random.randint(op_pool_size, size=(train_size, seq_len))

        # Corresponding sequence of operators
        train_op_seq = op_pool[train_op_pool_inds]

        # Corresponding tokens with special starting tokens
        train_token_seq = np.concatenate([
            np.zeros(shape=(train_size, 1), dtype=int), # starting token is 0
            train_op_pool_inds + 1 # shift operator inds by one
        ], axis=1)

        # Calculate the energies for each subsequence in the training set
        train_sub_seq_en = get_subsequence_energies(train_op_seq)

        # Save to cache
        print(f"Saving GQE training dataset to cache: {dataset_cache_file}")
        with open(dataset_cache_file, "wb") as f:
            pickle.dump({
                "train_op_pool_inds": train_op_pool_inds,
                "train_op_seq": train_op_seq,
                "train_token_seq": train_token_seq,
                "train_sub_seq_en": train_sub_seq_en,
            }, f)

# %% [markdown]
# ## GPTQE and DiTQE Model Architectures
#
# We define two generative models for GQE:
# 1. **GPTQE**: A causal language model (based on nanoGPT) that learns to auto-regressively generate circuit excitation sequences conditioned on minimizing energy.
# 2. **DITQE**: A Diffusion-Transformer-style architecture that modulates token embeddings using the final target energies as continuous conditions, based on [*6.S184/6.S975: Generative AI with Stochastic Differential Equations*](https://diffusion.csail.mit.edu).

# %%
# # !curl -O https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py
from model import GPT, GPTConfig
import torch
import math
from torch.nn import functional as F
import torch.nn as nn
import math

class GPTQE(GPT):
    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    def calculate_loss(self, tokens, energies, grd_E=None):
        current_tokens, next_tokens = tokens[:, :-1], tokens[:, 1:]
        # calculate the logits for the next possible tokens in the sequence
        logits = self(current_tokens)
        # get the logit for the actual next token in the sequence
        next_token_mask = torch.nn.functional.one_hot(
            next_tokens, num_classes=self.config.vocab_size
        )
        next_token_logits = (logits * next_token_mask).sum(axis=2)
        # calculate the cumulative logits for each subsequence
        cumsum_logits = torch.cumsum(next_token_logits, dim=1)
        # match cumulative logits to subsequence energies
        loss = torch.mean(torch.square(cumsum_logits - energies))
        if grd_E is not None:
            physical_penalty = torch.mean(torch.relu(grd_E - cumsum_logits)**2)
            loss = loss + 10.0 * physical_penalty
        return loss
    
    @torch.no_grad()
    def generate(self, n_sequences, max_new_tokens, temperature=1.0, device="cpu"):
        idx = torch.zeros(size=(n_sequences, 1), dtype=int, device=device)
        total_logits = torch.zeros(size=(n_sequences, 1), device=device)
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step 
            logits = logits[:, -1, :] 
            
            # Since logits predict changes in energy, they might have tiny variance, causing F.softmax(-logits/temperature) 
            # to be completely flat. We standardize the valid logits so the temperature is meaningful and sampling works.
            std, mean = torch.std_mean(logits[:, 1:], dim=-1, keepdim=True)
            logits_scaled = (logits - mean) / (std + 1e-8)
            
            # set the logit of the first token so that its probability will be zero
            logits_scaled[:, 0] = 1e7
            # apply softmax to convert logits to (normalized) probabilities and scale by desired temperature
            probs = F.softmax(-logits_scaled / temperature, dim=-1)
            # Guard against MPS minor negative float inaccuracies or NaNs if diverged
            probs = torch.nan_to_num(probs, nan=1e-10)
            probs = torch.clamp(probs, min=1e-10)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Accumulate true unscaled logits for actual energy changes computation
            total_logits += torch.gather(logits, index=idx_next, dim=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx, total_logits


class MHA(nn.Module):
    """Multi-headed self-attention with causal masking"""
    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        # Compute queries, keys, and values
        qkv = self.qkv(x).chunk(3, dim=-1) # 3 of b n d
        q, k, v = [t.view(b, n, self.heads, d // self.heads).transpose(1, 2) for t in qkv] # b h n d_h

        # Compute attention
        qk = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
        
        # Apply causal mask: mask out future tokens (key position > query position)
        mask = torch.tril(torch.ones(n, n, device=qk.device)).view(1, 1, n, n)
        qk = qk.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(qk, dim=-1)

        # Combine with values
        out = torch.einsum('bhal,bhld->bhad', attn, v) # b h n d_h
        out = out.transpose(1, 2).reshape(b, n, d)
        
        return self.out(out)


def modulate(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + bias


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features)
        )
    def forward(self, x):
        return self.net(x)


class DiffusionTransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        # Normalization
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        self.ada_ln = nn.Sequential(
            nn.RMSNorm(dim, elementwise_affine=False),
            nn.Linear(dim, dim * 6)
        )
        # Initialize conditioning to zero
        nn.init.zeros_(self.ada_ln[1].weight)
        nn.init.zeros_(self.ada_ln[1].bias)
        # Attention
        self.attn = MHA(dim, heads)
        # Feedforward
        self.ff = MLP(dim, 4 * dim, dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c = self.ada_ln(c).unsqueeze(1) # b 1 d*6
        attn_scale, attn_bias, attn_gate, ff_scale, ff_bias, ff_gate = c.chunk(6, dim=-1)
        x = x + attn_gate * self.attn(modulate(self.norm1(x), attn_scale, attn_bias))
        x = x + ff_gate * self.ff(modulate(self.norm2(x), ff_scale, ff_bias))
        return x


class DITQE(GPT):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # DiT specific layers
        self.energy_emb = nn.Sequential(
            nn.Linear(1, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd)
        )
        
        # Override the transformer layers with DiT layers
        self.transformer.h = nn.ModuleList([
            DiffusionTransformerLayer(dim=config.n_embd, heads=config.n_head)
            for _ in range(config.n_layer)
        ])
        
    def forward(self, idx, energies):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # Forward embeddings
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Conditioning 
        c = self.energy_emb(energies.unsqueeze(-1)) # b n_embd
        
        # Forward DiT blocks
        for block in self.transformer.h:
            x = block(x, c)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    def calculate_loss(self, tokens, energies, grd_E=None):
        current_tokens, next_tokens = tokens[:, :-1], tokens[:, 1:]
        # pass in the final target energies as conditioning to the model
        logits = self(current_tokens, energies[:, -1])
        next_token_mask = torch.nn.functional.one_hot(
            next_tokens, num_classes=self.config.vocab_size
        )
        next_token_logits = (logits * next_token_mask).sum(axis=2)
        cumsum_logits = torch.cumsum(next_token_logits, dim=1)
        loss = torch.mean(torch.square(cumsum_logits - energies))
        if grd_E is not None:
            physical_penalty = torch.mean(torch.relu(grd_E - cumsum_logits)**2)
            loss = loss + 10.0 * physical_penalty
        return loss
    
    @torch.no_grad()
    def generate(self, n_sequences, max_new_tokens, energies, temperature=1.0, device="cpu"):
        idx = torch.zeros(size=(n_sequences, 1), dtype=int, device=device)
        total_logits = torch.zeros(size=(n_sequences, 1), device=device)
        
        # Ensure energies has proper shape for batched generation
        if not isinstance(energies, torch.Tensor):
            energies = torch.tensor(energies, dtype=torch.float32, device=device).expand(n_sequences)
            
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond, energies)
            logits = logits[:, -1, :] 
            
            # standardize the valid logits to prevent flat softmax
            std, mean = torch.std_mean(logits[:, 1:], dim=-1, keepdim=True)
            logits_scaled = (logits - mean) / (std + 1e-8)
            
            # set the logit of the first token so that its probability will be close to zero
            logits_scaled[:, 0] = 1e7
            
            probs = F.softmax(-logits_scaled / temperature, dim=-1)
            # Guard against MPS minor negative float inaccuracies or NaNs if diverged
            probs = torch.nan_to_num(probs, nan=1e-10)
            probs = torch.clamp(probs, min=1e-10)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            # Accumulate true unscaled logits
            total_logits += torch.gather(logits, index=idx_next, dim=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx, total_logits


# %% [markdown]
# ## Model Training and Optimization
#
# We configure the model hyperparameters, initialize the optimizer, and train the Generative Quantum Eigensolver (GQE) network. Every 500 epochs, we sample candidate sequences and evaluate their energies to monitor convergence towards the physical ground state.

# %%
# Select best available device: prefer MPS, then CUDA, then CPU
import torch
import pandas as pd
from tqdm.auto import tqdm

if USE_CUDA and torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
# Some optimizer helpers expect either 'cuda' or 'cpu' — map MPS to cpu for those APIs
device_type_opt = "cuda" if device == "cuda" else "cpu"
print(f"Using device: {device} (optimizer device type: {device_type_opt})")

if not has_cache:
    import pandas as pd
    from tqdm.auto import tqdm

    # Move training tensors and model to the selected device
    tokens = torch.from_numpy(train_token_seq).to(device)
    energies = torch.from_numpy(train_sub_seq_en).float().to(device)

    config = GPTConfig(
        vocab_size=op_pool_size + 1,
        block_size=seq_len,
        dropout=0.2,
        bias=False,
    )

    if USE_DIT:
        gpt = DITQE(config).to(device)
    else:
        gpt = GPTQE(config).to(device)

    opt = gpt.configure_optimizers(
        weight_decay=0.01, learning_rate=5e-4, betas=(0.9, 0.999), device_type=device_type_opt
    )

    n_batches = 8
    train_inds = np.arange(train_size)

    losses = []
    pred_Es_t = []
    true_Es_t = []
    eval_iterations = []
    current_mae = 10000

    # Check if a pre-trained model is already available
    model_path = os.path.join(save_dir, "gqe.pt")
    if os.path.exists(model_path):
        print(f"Pre-trained GQE model found at {model_path}. Loading model and skipping training loop...")
        gpt = torch.load(model_path, map_location=device, weights_only=False)
    else:
        gpt.train()
        for i in tqdm(range(10000), desc="Training"):
            # Shuffle batches of the training set
            np.random.shuffle(train_inds)
            token_batches = torch.tensor_split(tokens[train_inds], n_batches)
            energy_batches = torch.tensor_split(energies[train_inds], n_batches)
            
            # SGD on random minibatches
            loss_record = 0
            for token_batch, energy_batch in zip(token_batches, energy_batches):
                opt.zero_grad()
                loss = gpt.calculate_loss(token_batch, energy_batch, grd_E)
                loss.backward()
                opt.step()
                loss_record += loss.item() / n_batches
            losses.append(loss_record)

            if (i+1) % 500 == 0:
                # For GPT evaluation
                gpt.eval()
                gen_kwargs = {
                    "n_sequences": 100, 
                    "max_new_tokens": seq_len, 
                    "temperature": 0.001, 
                    "device": device
                }
                if USE_DIT:
                    gen_kwargs["energies"] = grd_E
                    
                gen_token_seq, pred_Es = gpt.generate(**gen_kwargs)
                pred_Es = pred_Es.cpu().numpy()

                gen_inds = (gen_token_seq[:, 1:] - 1).cpu().numpy()
                gen_op_seq = op_pool[gen_inds]
                true_Es = get_subsequence_energies(gen_op_seq)[:, -1].reshape(-1, 1)

                mae = np.mean(np.abs(pred_Es - true_Es))
                ave_E = np.mean(true_Es)
                
                eval_iterations.append(i + 1)
                
                pred_Es_t.append(pred_Es)
                true_Es_t.append(true_Es)
                
                # tqdm.write is better than print when using tqdm in notebooks/terminal to avoid visual bugs
                tqdm.write(f"Iteration: {i+1}, Loss: {losses[-1]:.8f}, MAE: {mae:.4f}, Ave E: {ave_E:.4f}")
                
                if mae < current_mae:
                    current_mae = mae
                    torch.save(gpt, f"{save_dir}/gqe.pt")
                    tqdm.write("Saved model!")
                    
                gpt.train()
                
        pred_Es_t = np.concatenate(pred_Es_t, axis=1)
        true_Es_t = np.concatenate(true_Es_t, axis=1)

        # Persist training outputs for later analysis cells
        pd.DataFrame(losses).to_csv(f"{save_dir}/losses.csv")
        pd.DataFrame(true_Es_t, columns=eval_iterations).to_csv(f"{save_dir}/true_Es_t.csv")
        pd.DataFrame(pred_Es_t, columns=eval_iterations).to_csv(f"{save_dir}/pred_Es_t.csv")

# %% [markdown]
# ## Training Performance Visualization
#
# We visualize the training loss progress over the epochs to check optimization stability.

# %%
import holoviews as hv
import hvplot.pandas
import pandas as pd

hvplot.extension('matplotlib')

try:
    losses_path = f"{save_dir}/losses.csv"
    if os.path.exists(losses_path):
        losses = pd.read_csv(losses_path)["0"]
        loss_fig = losses.hvplot(
            title="Training loss progress", ylabel="loss", xlabel="Training epochs", logy=True
        ).opts(fig_size=600, fontscale=2, aspect=1.2)
        from IPython.display import display
        display(loss_fig)
except Exception as e:
    print(f"Skipping training loss plot: {e}")

# %% [markdown]
# ## GQE Energy Predictions vs. True Energies
#
# We plot the mean and range of the GQE-predicted energy sequences against their true quantum simulator energy values, demonstrating how the generator converges towards the ground-state energy.

# %%
try:
    true_path = f"{save_dir}/true_Es_t.csv"
    pred_path = f"{save_dir}/pred_Es_t.csv"
    if os.path.exists(true_path) and os.path.exists(pred_path):
        df_true = pd.read_csv(true_path).iloc[:, 1:]
        df_pred = pd.read_csv(pred_path).iloc[:, 1:]

        df_true.columns = df_true.columns.astype(int)
        df_pred.columns = df_pred.columns.astype(int)

        df_trues_stats = pd.concat([df_true.mean(axis=0), df_true.min(axis=0), df_true.max(axis=0)], axis=1).reset_index()
        df_trues_stats.columns = ["Training Iterations", "Ave True E", "Min True E", "Max True E"]

        df_preds_stats = pd.concat([df_pred.mean(axis=0), df_pred.min(axis=0), df_pred.max(axis=0)], axis=1).reset_index()
        df_preds_stats.columns = ["Training Iterations", "Ave Pred E", "Min Pred E", "Max Pred E"]

        fig = (
            df_trues_stats.hvplot.scatter(x="Training Iterations", y="Ave True E", label="Mean True Energies") * 
            df_trues_stats.hvplot.line(x="Training Iterations", y="Ave True E", alpha=0.5, linewidth=1) * 
            df_trues_stats.hvplot.area(x="Training Iterations", y="Min True E", y2="Max True E", alpha=0.1)
        ) * (
            df_preds_stats.hvplot.scatter(x="Training Iterations", y="Ave Pred E", label="Mean Predicted Energies") * 
            df_preds_stats.hvplot.line(x="Training Iterations", y="Ave Pred E", alpha=0.5, linewidth=1) * 
            df_preds_stats.hvplot.area(x="Training Iterations", y="Min Pred E", y2="Max Pred E", alpha=0.1)
        )
        fig = fig * hv.Curve([[0, grd_E], [10000, grd_E]], label="Ground State Energy").opts(color="k", alpha=0.4, linestyle="dashed")
        fig = fig.opts(ylabel="Sequence Energies", title="GQE Evaluations", fig_size=600, fontscale=2)
        from IPython.display import display
        display(fig)
except Exception as e:
    print(f"Skipping evaluation stats plot: {e}")

# %% [markdown]
# ## Evaluation Summary
#
# We compare the statistics (average, minimum, and maximum energy) of random sequences against the outputs of the latest trained model and the best-performing model checkpoint.

# %%
df_compare_Es = None
csv_path = f"{save_dir}/compare_Es.csv"
if os.path.exists(csv_path):
    try:
        df_compare_Es = pd.read_csv(csv_path)
    except Exception:
        pass

if df_compare_Es is None:
    try:
        if 'train_sub_seq_en' not in globals():
            # Regenerate a small random sample to compute the "Random" baseline
            train_size_sample = 128
            sample_op_pool_inds = np.random.randint(op_pool_size, size=(train_size_sample, seq_len))
            sample_op_seq = op_pool[sample_op_pool_inds]
            train_sub_seq_en_val = get_subsequence_energies(sample_op_seq)
        else:
            train_sub_seq_en_val = train_sub_seq_en

        if 'gpt' not in globals() or gpt is None:
            gpt = torch.load(f"{save_dir}/gqe.pt", map_location=device, weights_only=False)

        # Latest model
        gen_kwargs = {
            "n_sequences": 128, 
            "max_new_tokens": seq_len, 
            "temperature": 0.001, 
            "device": device
        }
        if USE_DIT:
            gen_kwargs["energies"] = grd_E

        gen_token_seq_, _ = gpt.generate(**gen_kwargs)
        gen_inds_ = (gen_token_seq_[:, 1:] - 1).cpu().numpy()
        gen_op_seq_ = op_pool[gen_inds_]
        true_Es_ = get_subsequence_energies(gen_op_seq_)[:, -1].reshape(-1, 1)

        # Best model
        loaded = torch.load(f"{save_dir}/gqe.pt", map_location=device, weights_only=False)
        loaded_token_seq_, _ = loaded.generate(**gen_kwargs)
        loaded_inds_ = (loaded_token_seq_[:, 1:] - 1).cpu().numpy()
        loaded_op_seq_ = op_pool[loaded_inds_]
        loaded_true_Es_ = get_subsequence_energies(loaded_op_seq_)[:, -1].reshape(-1, 1)

        # Summary table
        df_compare_Es = pd.DataFrame({
            "Source": ["Random", "Latest Model", "Best Model"], 
            "Aves": [train_sub_seq_en_val[:, -1].mean(), true_Es_.mean(), loaded_true_Es_.mean()],
            "Mins": [train_sub_seq_en_val[:, -1].min(), true_Es_.min(), loaded_true_Es_.min()],
            "Maxs": [train_sub_seq_en_val[:, -1].max(), true_Es_.max(), loaded_true_Es_.max()],
            "Mins_error": [
                abs(train_sub_seq_en_val[:, -1].min() - grd_E),
                abs(true_Es_.min() - grd_E),
                abs(loaded_true_Es_.min() - grd_E),
            ],
        })
        df_compare_Es.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Skipping summary table generation: {e}")

if df_compare_Es is not None:
    from IPython.display import display
    display(df_compare_Es)

# %% [markdown]
# ## Step 2.1: Quantum Self-Consistent Equation-of-Motion (q-sc-EOM)
#
# We identify the best ansatz sequence generated by GQE and use it as a reference state. We then execute the q-sc-EOM algorithm to calculate:
# 1. Core-ionized energy levels (IP space, $N-1$ electrons).
# 2. Doubly ionized energy levels (DIP space, $N-2$ electrons).

# %%
if 'has_cache' not in globals():
    import os
    current_file_dir = get_current_file_dir()
    cache_dir = os.path.join(current_file_dir, "data", "qsceom")
    use_ecp_avas_val = globals().get("USE_ECP_AVAS", False)
    use_taper_val = globals().get("USE_TAPER", False)
    setting_suffix = ""
    if use_ecp_avas_val:
        setting_suffix += "_ecpavas"
    if use_taper_val:
        setting_suffix += "_taper"
    cache_path = os.path.join(cache_dir, f"qsceom_cache{setting_suffix}.pkl")
    has_cache = os.path.exists(cache_path)
    seq_len = 4
    trial_name = f"trial_h2o{setting_suffix}"
    save_dir = os.path.abspath(os.path.join(current_file_dir, "data", f"seq_len={seq_len}/{trial_name}"))

if not has_cache:
    from qsceom import qscEOM

    # Ensure loaded_true_Es_ and loaded_op_seq_ are defined in memory (even if summary table was loaded from cache)
    if 'loaded_true_Es_' not in globals() or 'loaded_op_seq_' not in globals():
        print("Re-evaluating GQE models to extract best ansatz sequence...")
        loaded = torch.load(f"{save_dir}/gqe.pt", map_location=device, weights_only=False)
        gen_kwargs = {
            "n_sequences": 128, 
            "max_new_tokens": seq_len, 
            "temperature": 0.001, 
            "device": device
        }
        if USE_DIT:
            gen_kwargs["energies"] = grd_E
        loaded_token_seq_, _ = loaded.generate(**gen_kwargs)
        loaded_inds_ = (loaded_token_seq_[:, 1:] - 1).cpu().numpy()
        loaded_op_seq_ = op_pool[loaded_inds_]
        loaded_true_Es_ = get_subsequence_energies(loaded_op_seq_)[:, -1].reshape(-1, 1)

    # 1. Identify the best sequence of operations from the trained Generative Quantum Eigensolver (GQE)
    best_seq_idx = np.argmin(loaded_true_Es_)
    best_op_seq = loaded_op_seq_[best_seq_idx]

    # 2. Extract parameters and excitations for qscEOM compatibility
    params = []
    ash_excitation = []

    for op in best_op_seq:
        # Skip Identity operators which have no effect on state preparation in this context
        if isinstance(op, qml.Identity) or getattr(op, "name", "") == "Identity" or (hasattr(op, "base") and isinstance(op.base, qml.Identity)):
            continue
        
        # Extract the parameter (time/angle) and the wires (excitation indices)
        # This maps the GQE state preparation to standard qscEOM parameterized inputs
        if len(op.parameters) > 0:
            val = op.parameters[0]
            params.append(float(val.imag) if np.iscomplexobj(val) else float(val))
            wires = op.wires.tolist()
            ash_excitation.append(tuple(wires))

    # 3. Retrieve molecule details directly from the in-memory dictionary to prevent redundant downloads/DNS errors!
    symbols = qchem_data["symbols"]
    geometry = qchem_data["geometry"]
    active_electrons = qchem_data["active_electrons"]
    active_orbitals = qchem_data["active_orbitals"]
    charge = 0 # Default calculation assumes a neutral molecule (e.g. H2)

    print(f"Preparing q-sc-EOM for {target_molecule}...")
    print(f"Number of excitations in reference ansatz from GQE: {len(params)}")

    # Resolve absolute path for pyscf datasets directory
    import os
    current_file_dir = get_current_file_dir()
    datasets_pyscf_dir = os.path.abspath(os.path.join(current_file_dir, "./data/pyscf"))
    os.makedirs(datasets_pyscf_dir, exist_ok=True)

    # 4. Run qscEOM to compute the energies (IP and DIP spaces)
    # Run 1: IP space (Core-hole state, N-1 electrons)
    qsceom_ip_res = qscEOM(
        symbols=symbols,
        geometry=geometry,
        active_electrons=active_electrons - 1,
        active_orbitals=active_orbitals,
        charge=charge + 1,
        mult=2, # Open shell N-1 -> Doublet
        params=params,
        ash_excitation=ash_excitation,
        ansatz_type="qubit_excitation",
        basis="sto-3g",
        method="openfermion", # Needed to support open shell
        shots=0, 
        return_details=True,
        outpath=datasets_pyscf_dir,
        use_ecp_avas=USE_ECP_AVAS,
        use_taper=USE_TAPER,
    )
    eigs_ip, details_ip = qsceom_ip_res
    print("\nq-sc-EOM IP (N-1) state energies:")
    print(eigs_ip)

    # Run 2: DIP space (Double-hole state, N-2 electrons)
    qsceom_dip_res = qscEOM(
        symbols=symbols,
        geometry=geometry,
        active_electrons=active_electrons - 2,
        active_orbitals=active_orbitals,
        charge=charge + 2,
        mult=1, # N-2 -> Singlet (or Triplet depending on states, Singlet used for basis setup)
        params=params,
        ash_excitation=ash_excitation,
        ansatz_type="qubit_excitation",
        basis="sto-3g",
        method="openfermion", # Consistency
        shots=0, 
        return_details=True,
        outpath=datasets_pyscf_dir,
        use_ecp_avas=USE_ECP_AVAS,
        use_taper=USE_TAPER,
    )
    eigs_dip, details_dip = qsceom_dip_res
    print("\nq-sc-EOM DIP (N-2) state energies:")
    print(eigs_dip)

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    cache_data = {
        "eigs_ip": eigs_ip,
        "details_ip": details_ip,
        "eigs_dip": eigs_dip,
        "details_dip": details_dip,
        "symbols": symbols,
        "geometry": geometry,
        "active_electrons": active_electrons,
        "active_orbitals": active_orbitals,
        "charge": charge,
        "init_state": init_state,
        "params": params,
        "ash_excitation": ash_excitation,
        "USE_DIT": USE_DIT if 'USE_DIT' in globals() else False,
        "USE_ECP_AVAS": USE_ECP_AVAS if 'USE_ECP_AVAS' in globals() else False,
        "grd_E": grd_E if 'grd_E' in globals() else None,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)
    print(f"Saved q-sc-EOM results and molecular parameters to cache: {cache_path}")
else:
    print(f"Loading q-sc-EOM results and molecular parameters from cache: {cache_path}")
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)
    eigs_ip = cache_data["eigs_ip"]
    details_ip = cache_data["details_ip"]
    eigs_dip = cache_data["eigs_dip"]
    details_dip = cache_data["details_dip"]
    symbols = cache_data["symbols"]
    geometry = cache_data["geometry"]
    active_electrons = cache_data["active_electrons"]
    active_orbitals = cache_data["active_orbitals"]
    charge = cache_data["charge"]
    init_state = cache_data["init_state"]
    params = cache_data["params"]
    ash_excitation = cache_data["ash_excitation"]
    USE_DIT = cache_data.get("USE_DIT", False)
    if cache_data.get("grd_E") is not None:
        grd_E = cache_data["grd_E"]

# %% [markdown]
# ## Step 2.2: Minimal Basis Projection of Atomic Integrals
#
# The One-Center Approximation (OCA) requires transition matrix elements to be restricted to the emitter atom's center. We reconstruct the system in PySCF and project molecular orbital coefficients onto the minimal basis set (MBS) of the emitter atom (Oxygen, index 0).

# %%
# Paper Eq. 18-19: OCA uses atomic basis integrals, not molecular orbital integrals
# ⟨χ_Elm^A χ_c^A | χ_ν^A χ_ρ^A⟩ where {χ^A} are minimal basis set (MBS) atomic orbitals on emitter atom
from pyscf import gto, scf, ao2mo
import torch
import numpy as np
import pickle
import os

if 'symbols' not in globals():
    current_file_dir = get_current_file_dir()
    cache_dir = os.path.join(current_file_dir, "data", "qsceom")
    use_ecp_avas_val = globals().get("USE_ECP_AVAS", False)
    use_taper_val = globals().get("USE_TAPER", False)
    setting_suffix = ""
    if use_ecp_avas_val:
        setting_suffix += "_ecpavas"
    if use_taper_val:
        setting_suffix += "_taper"
    cache_path = os.path.join(cache_dir, f"qsceom_cache{setting_suffix}.pkl")
    if os.path.exists(cache_path):
        print(f"Loading variables from cache for Step 2.2: {cache_path}")
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)
        eigs_ip = cache_data["eigs_ip"]
        details_ip = cache_data["details_ip"]
        eigs_dip = cache_data["eigs_dip"]
        details_dip = cache_data["details_dip"]
        symbols = cache_data["symbols"]
        geometry = cache_data["geometry"]
        active_electrons = cache_data["active_electrons"]
        active_orbitals = cache_data["active_orbitals"]
        charge = cache_data["charge"]
        init_state = cache_data["init_state"]
        params = cache_data["params"]
        ash_excitation = cache_data["ash_excitation"]
        USE_DIT = cache_data.get("USE_DIT", False)
        if cache_data.get("grd_E") is not None:
            grd_E = cache_data["grd_E"]

device = torch.device("cuda" if (USE_CUDA and torch.cuda.is_available()) else ("mps" if torch.backends.mps.is_available() else "cpu"))

print("Computing integral data via PySCF with minimal basis projection...")
print("(Paper Eq. 19: D = T^{-1} U C, where T=MBS overlap, U=MBS-CGTO overlap, C=MO coefficients)")

# 1. Reconstruct molecule and compute RHF
mol_str = "; ".join([f"{s} {g[0]} {g[1]} {g[2]}" for s, g in zip(symbols, geometry)])

if USE_ECP_AVAS and "I" in symbols:
    mol = gto.Mole()
    mol.atom = mol_str
    mol.unit = "Bohr"
    basis_map = {}
    ecp_map = {}
    for s in symbols:
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
else:
    mol = gto.M(atom=mol_str, basis="sto-3g", charge=charge, unit="Bohr", symmetry=False)
mf = scf.RHF(mol)
mf.kernel(verbose=0)

# 2. Identify emitter atom and its basis functions
# Look for Iodine (I) first as the heavy emitter, otherwise fall back to the first non-Hydrogen atom
if "I" in symbols:
    emitter_atom_idx = symbols.index("I")
else:
    non_h_indices = [i for i, sym in enumerate(symbols) if sym != "H"]
    emitter_atom_idx = non_h_indices[0] if non_h_indices else 0

ao_labels = mol.ao_labels(fmt=False)  # List of (atom_idx, angular_momentum, component_idx)
mbs_indices = [i for i, (atom_idx, *_) in enumerate(ao_labels) if atom_idx == emitter_atom_idx]

print(f"Emitter atom: {symbols[emitter_atom_idx]} at index {emitter_atom_idx}")
print(f"Minimal basis set (STO-3G) indices on emitter atom: {mbs_indices}")

# 3. Compute overlap matrices for projection (Eq. 19)
# T_μν = ⟨χ_μ | χ_ν⟩ (MBS overlap)
# U_μκ = ⟨χ_μ | f_κ⟩ (MBS-CGTO overlap)
# C_κr = MO coefficient in full basis
overlap_full = mol.intor("int1e_ovlp")  # Full basis overlap
T = overlap_full[np.ix_(mbs_indices, mbs_indices)]  # Extract MBS block
U = overlap_full[np.ix_(mbs_indices, range(len(overlap_full)))]  # MBS to full basis

# 4. Compute molecular orbital 2-electron integrals
eri_mo = ao2mo.kernel(mol, mf.mo_coeff)
eri_mo_matrix = ao2mo.restore(1, eri_mo, mol.nao_nr())

# 5. Apply projection: D_μr = (T^{-1} U C)_μr
# This projects MO coefficients to the minimal basis
C_full = mf.mo_coeff  # Full basis MO coefficients (nao × norb)
C_mbs = np.linalg.inv(T) @ U @ C_full  # MBS coefficients (n_mbs × norb)

print(f"Projected MO coefficients to MBS: shape {C_mbs.shape}")

# 6. Extract atomic integrals in the AO basis and project using C_mbs
n_core_auger = (mol.nelectron - active_electrons) // 2
active_indices = list(range(n_core_auger, n_core_auger + active_orbitals))

import h5py

file_needs_generation = True
if os.path.exists(openmolcas_filepath):
    try:
        with h5py.File(openmolcas_filepath, "r") as f_test:
            if "energies" in f_test:
                file_needs_generation = False
    except Exception:
        pass
        
if file_needs_generation:
    print(f"Generating GQE-compatible OpenMolcas integrals at: {openmolcas_filepath}")
    print("Attempting fallback to generate integrals from installed 'auger_oca' database...")
    try:
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(get_current_file_dir(), "../../external/code/AugerOca")))
        from auger_oca.oca_integrals import elmij
        
        # Setup directories
        os.makedirs(os.path.dirname(openmolcas_filepath), exist_ok=True)
        
        # Define channels (standard L, M quantum numbers)
        channels = np.array([
            [0, 0],
            [1, -1], [1, 0], [1, 1],
            [2, -2], [2, -1], [2, 0], [2, 1], [2, 2]
        ])
        
        # Dynamically determine emitter symbol and valence information
        emitter_element = symbols[emitter_atom_idx]
        core_orbital_name = f"{emitter_element} 1s"
        
        # Extract valence AOs for STO-3G dynamically from PySCF labels
        ao_labels = mol.ao_labels(fmt=False)
        mbs_labels = [label for label in ao_labels if label[0] == emitter_atom_idx]
        
        valence_indices = []
        valence_names = {}
        for local_idx, label in enumerate(mbs_labels):
            # label format: (atom_idx, atom_symbol, orbital_name, component)
            orb_name = label[2]
            component = label[3]
            full_orb_name = orb_name + component # e.g. "2s", "2px"
            if "1s" not in full_orb_name:
                abs_idx = mbs_indices[local_idx]
                valence_indices.append(abs_idx)
                valence_names[abs_idx] = full_orb_name
                
        valence_indices = np.array(valence_indices)
        energies = np.array([450.0, 470.0, 490.0, 510.0, 530.0])
        
        integrals = np.zeros((len(energies), len(channels), len(valence_indices), len(valence_indices)))
        for e_idx in range(len(energies)):
            for c_idx, (L, M) in enumerate(channels):
                for i_idx, mu_ao in enumerate(valence_indices):
                    name_mu = valence_names[mu_ao]
                    for j_idx, nu_ao in enumerate(valence_indices):
                        name_nu = valence_names[nu_ao]
                        val = elmij(emitter_element, core_orbital_name, core_orbital_name, name_mu, name_nu, int(L), int(M))
                        integrals[e_idx, c_idx, i_idx, j_idx] = val
                        
        with h5py.File(openmolcas_filepath, "w") as f_w:
            f_w.create_dataset("energies", data=energies)
            f_w.create_dataset("integrals", data=integrals)
            f_w.create_dataset("channels", data=channels)
            f_w.create_dataset("valence_indices", data=valence_indices)
            f_w.create_dataset("core_index", data=0)
        print(f"Successfully generated database fallback integrals file at: {openmolcas_filepath}")
    except ImportError as e:
        print("ImportError: 'auger_oca' is not installed and no precomputed HDF5 file exists.")
        raise e
        
print(f"Loading OpenMolcas integrals from: {openmolcas_filepath}")
with h5py.File(openmolcas_filepath, "r") as f_h5:
    molcas_energies = np.array(f_h5["energies"])
    molcas_integrals = np.array(f_h5["integrals"])
    molcas_channels = np.array(f_h5["channels"])
    molcas_valence_indices = np.array(f_h5["valence_indices"])
    
# Interpolate at the dominant/average kinetic energy (e.g. 490.0 eV for water)
target_rep_energy_ev = 490.0

if len(molcas_energies) == 1:
    ints_interpolated = molcas_integrals[0]
else:
    idx = np.searchsorted(molcas_energies, target_rep_energy_ev)
    if idx == 0:
        ints_interpolated = molcas_integrals[0]
    elif idx == len(molcas_energies):
        ints_interpolated = molcas_integrals[-1]
    else:
        e0, e1 = molcas_energies[idx-1], molcas_energies[idx]
        w1 = (target_rep_energy_ev - e0) / (e1 - e0)
        w0 = 1.0 - w1
        ints_interpolated = w0 * molcas_integrals[idx-1] + w1 * molcas_integrals[idx]
        
n_channels = ints_interpolated.shape[0]
f_orbitals = list(range(n_channels))

V_pq_direct_channels = []
V_pq_exchange_channels = []
ao_to_mbs_row = {ao: idx for idx, ao in enumerate(mbs_indices)}

for c_idx in range(n_channels):
    V_pq_projected_direct = np.zeros((active_orbitals, active_orbitals))
    V_pq_projected_exchange = np.zeros((active_orbitals, active_orbitals))
    for p_idx, p in enumerate(active_indices):
        for q_idx, q in enumerate(active_indices):
            val_dir = 0.0
            val_exc = 0.0
            for i_val, mu_ao in enumerate(molcas_valence_indices):
                row_mu = ao_to_mbs_row[mu_ao]
                for j_val, nu_ao in enumerate(molcas_valence_indices):
                    row_nu = ao_to_mbs_row[nu_ao]
                    
                    val_direct = ints_interpolated[c_idx, i_val, j_val]
                    val_exchange = ints_interpolated[c_idx, j_val, i_val]
                    val_dir += C_mbs[row_mu, p] * C_mbs[row_nu, q] * val_direct
                    val_exc += C_mbs[row_mu, p] * C_mbs[row_nu, q] * val_exchange
            V_pq_projected_direct[p_idx, q_idx] = val_dir
            V_pq_projected_exchange[p_idx, q_idx] = val_exc
    V_pq_direct_channels.append(V_pq_projected_direct)
    V_pq_exchange_channels.append(V_pq_projected_exchange)
    
V_pq_direct = torch.tensor(np.array(V_pq_direct_channels), device=device, dtype=torch.float32)
V_pq_exchange = torch.tensor(np.array(V_pq_exchange_channels), device=device, dtype=torch.float32)
print(f"Constructed projected V_pq_direct and V_pq_exchange tensors using OpenMolcas integrals: shape {V_pq_direct.shape}")

# %% [markdown]
# ## Step 2.3: One-Center Approximation (OCA) & Auger Transition Intensities
#
# We calculate the transition reduced density matrix (RDM) $\gamma_{pq}$ between the core-ionized state and double-ionization states, and contract it with the projected atomic integrals to compute Auger transition intensities under Fermi's Golden Rule.

# %%
import torch
import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas
hv.extension('matplotlib')

HARTREE_TO_EV = 27.211386245988
device = torch.device("cuda" if (USE_CUDA and torch.cuda.is_available()) else ("mps" if torch.backends.mps.is_available() else "cpu"))

def compute_gamma_pq(C_IP, C_DIP, list_IP, list_DIP, core_idx, n_spin):
    """
    Classical mapping of the Transition RDM: < Phi_DIP | a^dagger_c a_p a_q | Phi_IP >
    Returns a tensor of shape (N_DIP_states, N_IP_states, n_spin, n_spin)
    Note: this is actually called R_{KI;csr} in the paper.
    """
    num_ip = C_IP.shape[0]
    num_dip = C_DIP.shape[0]

    # Pre-map the N-2 bitstrings for O(1) matching
    dip_dict = {tuple(sorted(occ)): idx for idx, occ in enumerate(list_DIP)}

    gamma_pq = np.zeros((num_dip, num_ip, n_spin, n_spin), dtype=complex)

    # Left, Middle, and Right Matrix Contraction (Parity-Aware)
    for ip_idx, occ_ip in enumerate(list_IP):
        for p in occ_ip:
            if p == core_idx:
                continue
            for q in occ_ip:
                if q == p or q == core_idx:
                    continue

                # Apply annihilation operators a_p, a_q
                state_minus_pq = [x for x in occ_ip if x != p and x != q]

                # Apply creation operator a^dagger_c
                if core_idx in state_minus_pq:
                    continue
                new_state = sorted(state_minus_pq + [core_idx])

                dip_idx = dip_dict.get(tuple(new_state), -1)
                if dip_idx != -1:
                    # 1. Fermionic parity for a_q
                    pos_q = occ_ip.index(q)
                    parity = (-1) ** pos_q

                    # 2. Fermionic parity for a_p
                    state_minus_q = occ_ip[:pos_q] + occ_ip[pos_q + 1 :]
                    pos_p = state_minus_q.index(p)
                    parity *= (-1) ** pos_p

                    # 3. Fermionic parity for a^dagger_c
                    c_pos = sum(1 for x in state_minus_pq if x < core_idx)
                    parity *= (-1) ** c_pos

                    # Vectorized tensor contraction for K states (DIPs) and I states (IPs)
                    bridge_val = np.conj(C_DIP[:, dip_idx]) * parity
                    gamma_pq[:, :, p, q] += np.outer(bridge_val, C_IP[:, ip_idx])

    return torch.tensor(gamma_pq, dtype=torch.complex128)

# 1. Extract eigenvectors and lists
C_IP = np.array(details_ip["eigenvectors"]).T
C_DIP = np.array(details_dip["eigenvectors"]).T
list_IP = details_ip["basis_occupations"]
list_DIP = details_dip["basis_occupations"]
num_states = C_DIP.shape[0]

# 2. Keep OCA contraction on CPU for stable float64/complex128 support
if not isinstance(V_pq_direct, torch.Tensor):
    V_pq_direct = torch.tensor(V_pq_direct, dtype=torch.float64)
if not isinstance(V_pq_exchange, torch.Tensor):
    V_pq_exchange = torch.tensor(V_pq_exchange, dtype=torch.float64)
V_pq_dir_cpu = V_pq_direct.detach().cpu().to(torch.float64)
V_pq_exc_cpu = V_pq_exchange.detach().cpu().to(torch.float64)

n_spin_full = active_orbitals * 2
print(f"Constructing {n_spin_full}x{n_spin_full} spin RDM tensors based on active orbitals.")

# 3. Generate the Transition RDMs (gamma_pq) with robust core-spin selection
print("Computing transition RDM (gamma_pq)...")
full_init_occ = list(range(active_electrons))
core_spin_candidates = [0, 1]

core_channel_scan = []
best_choice = None

for core_spin_idx_try in core_spin_candidates:
    gamma_try = compute_gamma_pq(
        C_IP, C_DIP, list_IP, list_DIP, core_idx=core_spin_idx_try, n_spin=n_spin_full
    )

    core_hole_occ_try = [i for i in full_init_occ if i != core_spin_idx_try]
    if core_hole_occ_try in list_IP:
        core_hole_basis_idx_try = list_IP.index(core_hole_occ_try)
        overlap_vec_try = np.abs(np.asarray(details_ip["eigenvectors"])[core_hole_basis_idx_try, :]) ** 2
        target_ip_state_try = int(np.argmax(overlap_vec_try))
        max_overlap_try = float(np.max(overlap_vec_try))
    else:
        # If tapering or active space reduction is active, the core-ionized state
        # is the highest energy state (maximum eigenvalue) in the doublet space.
        target_ip_state_try = int(np.argmax(eigs_ip))
        max_overlap_try = 1.0

    gamma_slice_try = gamma_try[:, target_ip_state_try, :, :]
    gamma_norm_try = float(torch.linalg.vector_norm(gamma_slice_try).item())

    core_channel_scan.append(
        {
            "core_spin_idx": int(core_spin_idx_try),
            "target_ip_state": int(target_ip_state_try),
            "core_overlap": max_overlap_try,
            "gamma_norm": gamma_norm_try,
        }
    )

    if best_choice is None or gamma_norm_try > best_choice["gamma_norm"]:
        best_choice = {
            "core_spin_idx": int(core_spin_idx_try),
            "target_ip_state": int(target_ip_state_try),
            "gamma_norm": gamma_norm_try,
            "gamma_tensor": gamma_try,
        }

core_spin_idx = best_choice["core_spin_idx"]
target_IP_state = best_choice["target_ip_state"]
gamma_pq_tensor = best_choice["gamma_tensor"]
gamma_pq = gamma_pq_tensor[:, target_IP_state, :, :].cpu()

# 4. Spin-orbital expansion for classical integrals (V_pq) with correct spin selection rules
# V_pq_dir_cpu and V_pq_exc_cpu have shape (n_f, active_orbitals, active_orbitals)
V_lmpq_spin = torch.zeros((len(f_orbitals), 2, n_spin_full, n_spin_full), dtype=torch.float64)

# Populate direct and exchange terms with proper spin-selection conservation
# Direct: spin of p (s1) must match continuum spin (se) and spin of q (s2) must match core-hole spin (sc)
# Exchange: spin of q (s2) must match continuum spin (se) and spin of p (s1) must match core-hole spin (sc)
for f_idx in range(len(f_orbitals)):
    for i, act_i in enumerate(active_indices):
        for j, act_j in enumerate(active_indices):
            val_dir = V_pq_dir_cpu[f_idx, i, j].item()
            val_exc = V_pq_exc_cpu[f_idx, i, j].item()
            for se in (0, 1):  # Emitted continuum electron spin channel (0 = up, 1 = down)
                for s1 in (0, 1):  # Spin of orbital p
                    for s2 in (0, 1):  # Spin of orbital q
                        term = 0.0
                        if s1 == se and s2 == core_spin_idx:
                            term += val_dir
                        if s2 == se and s1 == core_spin_idx:
                            term -= val_exc
                        V_lmpq_spin[f_idx, se, i * 2 + s1, j * 2 + s2] = term

# 5. Tensor contraction (OCA) in complex128 for numerical stability
# amplitudes will have shape (num_dip, len(f_orbitals), 2)
amplitudes = torch.einsum(
    "kpq,lmpq->klm",
    gamma_pq.to(torch.complex128),
    V_lmpq_spin.to(torch.complex128),
)

# Apply Fermi's Golden Rule
# Sum over both orbital channels (l) and continuum electron spins (m)
Gamma_k = 2 * torch.pi * torch.sum(torch.abs(amplitudes) ** 2, dim=(1, 2))
Gamma_k_vals = Gamma_k.cpu().numpy()
print(f"\nComputed Auger Transition Intensities (Gamma_k) for {num_states} states.")

# %% [markdown]
# ## Step 2.4: Classical OpenMolcas Input Generation
#
# We compute the Auger electron kinetic energies ($E_{\text{kin}} = E_{\text{IP}} - E_{\text{DIP}}$) and apply Lorentzian broadening with a typical experimental line-width parameter ($\Gamma = 1.5\text{ eV}$) to simulate the final Auger electron spectrum.
#
# ### Classical OpenMolcas Reference Setup
#
# To plot the classical reference spectrum overlaid with the GQE quantum spectrum, you must copy the pre-computed outputs of the classical CASSCF/RASSI calculation from OpenMolcas into this project.
#
# #### 1. Running OpenMolcas
# Run the calculation in the OpenMolcas build directory, ensuring that you set the `MOLCAS_WORKDIR` variable to `.` to write the scratch transition density files locally:
# ```bash
# MOLCAS_WORKDIR=. /Users/nvenkat/anaconda3/envs/qi/bin/python3 ./pymolcas <molecule>_aes.input
# ```
#
# #### 2. Required Files
# Locate the following outputs generated by the run:
# - **`{molecule}_aes.rassi.h5`**: The main RASSI output database containing state energies.
# - **`r2TM_*` files** (e.g., `r2TM_SDA_002_001` through `r2TM_SDA_011_001`): The transition density matrix files containing the spin matrix elements.
#
# #### 3. File Destination
# Copy all of these files into the following directory in this project:
# `mitsubishi/phase_3/code/data/openmolcas/`
#
# Once copied, the parser in Step 2.5 will automatically detect them, run the real-time intensity calculations under the One-Center Approximation (OCA), and display the classical reference alongside your quantum spectrum!

# %%
def generate_openmolcas_files(target_molecule, symbols, geometry, n_core, n_cas, n_electron_cas, emitter_sym, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(current_file_dir, "data", "openmolcas")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate XYZ coordinate file (Bohr units)
    xyz_path = os.path.join(output_dir, f"{target_molecule}.xyz")
    xyz_lines = [f"{len(symbols)}", f"Coord for {target_molecule} (Bohr)"]
    for s, (x, y, z) in zip(symbols, geometry):
        xyz_lines.append(f"{s} {x:.8f} {y:.8f} {z:.8f}")
    with open(xyz_path, "w") as f:
        f.write("\n".join(xyz_lines))
    print(f"Generated OpenMolcas coordinate file at: {xyz_path}")
    
    # 2. Generate openmolcas input file (.input)
    input_path = os.path.join(output_dir, f"{target_molecule.lower()}_aes.input")
    
    # Map basis library (STO-3G by default, LANL2DZ for heavy Iodine)
    basis_section = ""
    for s in set(symbols):
        if s == "I":
            basis_section += f"  I.ECP.lanl2dz.0s.0s.0p.0d.\n"
        else:
            basis_section += f"  {s} = STO-3G\n"
            
    # Input template using Symmetry = 1 (C1 group, 1 irrep)
    input_content = f"""# OpenMolcas input file generated automatically for {target_molecule}
# Active Space CAS({n_electron_cas}e, {n_cas}o)
>>> EXPORT MOLCAS_PRINT = 2

&GATEWAY
  Title = {target_molecule} Normal Auger Spectrum
  Coord = {target_molecule}.xyz
  Basis = STO-3G
  Unit = Bohr
  Symmetry = 1

&SEWARD

&SCF
  Title = Reference RHF ground state

* --- Step 1: Initial Core-Ionized State (Doublet, N-1 electrons) ---
&RASSCF
  Title = Core-ionized initial state
  Spin = 2
  Symmetry = 1
  Inactive = {n_core - 1 if n_core > 0 else 0}
  RAS1 = 1
  RAS2 = {n_cas}
  nActEl = {n_electron_cas + 1} 0 0
  HEXS
  1
  1

>>> COPY $Project.JobIph JOB001

* --- Step 2: Final Doubly-Ionized States (Singlet, N-2 electrons) ---
&RASSCF
  Title = Double-hole final states
  Spin = 1
  Symmetry = 1
  Inactive = {n_core}
  RAS1 = 0
  RAS2 = {n_cas}
  nActEl = {n_electron_cas} 0 0
  CIROOT
  10 10 1

>>> COPY $Project.JobIph JOB002

* --- Step 3: RASSI Transition Density Matrix Computation ---
&RASSI
  NrofJobIphs = 2 all
  Dyson
  TDYS
  1
  {emitter_sym} 1s
"""
    with open(input_path, "w") as f:
        f.write(input_content)
    print(f"Generated OpenMolcas input file at: {input_path}")

# Run generator to build inputs for target_molecule
try:
    emitter_sym = symbols[emitter_atom_idx]
    generate_openmolcas_files(target_molecule, symbols, geometry, n_core, n_cas, n_electron_cas, emitter_sym)
except Exception as e:
    print(f"Skipping automated OpenMolcas file generation: {e}")


# %% [markdown]
# ## Step 2.5: Auger Spectrum Broadening and Overlay Plotting
#
# We compute the Auger electron kinetic energies ($E_{\text{kin}} = E_{\text{IP}} - E_{\text{DIP}}$) and apply Lorentzian broadening with a typical experimental line-width parameter ($\Gamma = 1.5\text{ eV}$) to simulate the final Auger electron spectrum.
# We then plot the computed GQE quantum spectrum overlaid with the classical reference spectrum (if RASSI outputs are available).

# %%
# qscEOM eigenvalues are used to compute Auger electron kinetic energy
# Paper Eq. 14: E_kin^Auger = E_IP(I) - E_DIP(K)
EIP = float(np.asarray(eigs_ip).flatten()[target_IP_state])
EDIPs = np.asarray(eigs_dip).flatten()

# Auger kinetic energy: emitted electron's KE = difference in ionization levels
# (single ionization costs more energy than double ionization; difference is carried away by electron)
E_kinetics_ev = (EIP - EDIPs) * HARTREE_TO_EV

# Keep only physically allowed channels
valid_ke_mask = E_kinetics_ev > 0
E_ke_valid = E_kinetics_ev[valid_ke_mask]
Gamma_ke_valid = np.asarray(Gamma_k_vals)[valid_ke_mask]

# Main-peak diagnostics (no shift applied)
dominant_raw_ke_ev = np.nan
if E_ke_valid.size > 0 and Gamma_ke_valid.size > 0:
    dominant_idx = int(np.argmax(Gamma_ke_valid))
    dominant_raw_ke_ev = float(E_ke_valid[dominant_idx])

# Build unshifted spectrum over its natural KE support
gamma_width_ev = 1.5
if E_ke_valid.size == 0:
    x_energies = np.linspace(0.0, 1.0, 1000)
    spectrum = np.zeros_like(x_energies, dtype=float)
else:
    x_min = max(0.0, float(E_ke_valid.min()) - 15.0)
    x_max = float(E_ke_valid.max()) + 15.0
    x_energies = np.linspace(x_min, x_max, 1500)
    spectrum = np.zeros_like(x_energies, dtype=float)

    for E_k, intensity in zip(E_ke_valid, Gamma_ke_valid):
        spectrum += float(intensity) * (gamma_width_ev / ((x_energies - E_k) ** 2 + gamma_width_ev ** 2))

spectrum_norm = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum

# Construct classical reference spectrum for overlay
classical_peaks = []

spec_out_path = os.path.join(current_file_dir, "data", "openmolcas", "auger.spectrum.out")
if not os.path.exists(spec_out_path):
    spec_out_path = "auger.spectrum.out"

# 1. Attempt to load classical peaks from cached spectrum file
if os.path.exists(spec_out_path) and os.path.getsize(spec_out_path) > 0:
    try:
        loaded_peaks = []
        with open(spec_out_path, "r") as f_spec:
            for line in f_spec:
                if line.strip() and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        loaded_peaks.append((float(parts[0]), float(parts[1])))
        if loaded_peaks:
            classical_peaks = loaded_peaks
            print(f"Loaded {len(classical_peaks)} classical peaks from cache: {spec_out_path}")
    except Exception as e:
        print(f"Could not parse classical spectrum file: {e}")

# 2. If cache is empty or missing, compute from RASSI and r2TM_ files in real-time
if not classical_peaks:
    openmolcas_dir = os.path.join(current_file_dir, "data", "openmolcas")
    r2tm_dir = openmolcas_dir if os.path.exists(openmolcas_dir) else "."
    r2tm_files = sorted([f for f in os.listdir(r2tm_dir) if f.startswith("r2TM_")])
    raw_rassi_path = os.path.join(openmolcas_dir, f"{target_molecule.lower()}_aes.rassi.h5")

    if r2tm_files and os.path.exists(raw_rassi_path):
        print("\n--- Computing Classical Reference Spectrum in Real-Time ---")
        orig_cwd = os.getcwd()
        try:
            with h5py.File(raw_rassi_path, "r") as f_raw:
                sfs_energies = np.array(f_raw["SFS_ENERGIES"])
            E_IP = sfs_energies[0]
            E_DIPs = sfs_energies[1:]
            HARTREE_TO_EV = 27.211386245988
            E_kinetics_ev = (E_IP - E_DIPs) * HARTREE_TO_EV
            
            sys.path.insert(0, os.path.abspath(os.path.join(get_current_file_dir(), "../../external/code/AugerOca")))
            # Monkey patch elmij in auger_oca to resolve gc == OCA_c mismatch bug
            import auger_oca.oca_integrals
            import auger_oca.rt2mzz
            orig_elmij = auger_oca.oca_integrals.elmij
            def patched_elmij(OCA_atom, OCA_c, c, i, j, l, m):
                return orig_elmij(OCA_atom, OCA_c, OCA_c, i, j, l, m)
            auger_oca.oca_integrals.elmij = patched_elmij
            auger_oca.rt2mzz.elmij = patched_elmij
            
            from auger_oca.initi import init2, init3
            from auger_oca.auger_driver import driver_auger
            
            os.chdir(r2tm_dir)
            computed_peaks = []
            for f in r2tm_files:
                parts = f.split("_")
                if len(parts) >= 3:
                    try:
                        state_num = int(parts[2])
                        state_idx = state_num - 2
                    except ValueError:
                        state_idx = 0
                else:
                    state_idx = 0
                    
                vals = list(init2(f))
                emitter_sym = symbols[emitter_atom_idx]
                if vals[2][0] == f"{emitter_sym} 1s":
                    vals[2][0] = f"{emitter_sym}1 1s"
                    
                hd5_file, basis_id_hd5, element = init3(vals[12], vals[5])
                
                driver_auger(
                    f, False, True, False, True,
                    vals[0], vals[2], vals[3], vals[4], vals[5],
                    vals[6], vals[9], vals[12], vals[15], vals[16],
                    vals[17], vals[18], vals[20], vals[21], vals[8],
                    vals[22], vals[23], hd5_file, element, basis_id_hd5
                )
                
                output_file = "Auger_OCA." + f + ".out"
                intensity = 0.0
                if os.path.exists(output_file):
                    with open(output_file, "r") as out_f:
                        for line in out_f:
                            if "Spectrum: BE(eV) and Intensity from OCA:" in line:
                                line_parts = line.strip().split()
                                if len(line_parts) >= 9:
                                    try:
                                        intensity = float(line_parts[8])
                                    except ValueError:
                                        intensity = 0.0
                    os.remove(output_file)
                    
                ke = E_kinetics_ev[state_idx] if state_idx < len(E_kinetics_ev) else 0.0
                computed_peaks.append((ke, intensity))
            
            if computed_peaks:
                classical_peaks = computed_peaks
                # Cache the computed peaks
                try:
                    cache_dir = os.path.dirname(spec_out_path)
                    if cache_dir:
                        os.makedirs(cache_dir, exist_ok=True)
                    with open(spec_out_path, "w") as f_spec:
                        f_spec.write("# Kinetic_Energy_eV Intensity\n")
                        for ke, intensity in classical_peaks:
                            f_spec.write(f"{ke:.6f} {intensity:.10f}\n")
                    print(f"Cached {len(classical_peaks)} computed peaks to: {spec_out_path}")
                except Exception as e:
                    print(f"Could not cache computed peaks: {e}")
                    
        except Exception as e:
            print(f"Error computing classical spectrum: {e}")
        finally:
            os.chdir(orig_cwd)


        
# Plot overlay results if classical peaks are available
df_spectrum_dict = {
    "Kinetic Energy (eV)": x_energies,
    "GQE (quantum)": spectrum_norm,
}

if len(classical_peaks) > 0:
    classical_spectrum = np.zeros_like(x_energies, dtype=float)
    for E_k, intensity in classical_peaks:
        classical_spectrum += float(intensity) * (gamma_width_ev / ((x_energies - E_k) ** 2 + gamma_width_ev ** 2))
    classical_spectrum_norm = classical_spectrum / np.max(classical_spectrum) if np.max(classical_spectrum) > 0 else classical_spectrum
    df_spectrum_dict["Classical Reference"] = classical_spectrum_norm

df_spectrum = pd.DataFrame(df_spectrum_dict)

import matplotlib.pyplot as plt
from IPython.display import display

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(df_spectrum["Kinetic Energy (eV)"], df_spectrum["GQE (quantum)"], label="GQE (quantum)", color="darkred")
if "Classical Reference" in df_spectrum.columns:
    ax.plot(df_spectrum["Kinetic Energy (eV)"], df_spectrum["Classical Reference"], "--", label="OpenMolcas RASSI", color="blue")
ax.set_title("Auger Spectrum: GQE vs Classical Reference")
ax.set_xlabel("Kinetic Energy (eV)")
ax.set_ylabel("Normalized Intensity")
ax.legend(loc="upper left")

ax.set_xlim(400, 575)
fig.tight_layout()

if SAVE_PLOTS:
    plot_path = os.path.join(current_file_dir, "emission_spectrum.png")
    fig.savefig(plot_path, dpi=300)
    print(f"Saved emission spectrum plot to {plot_path}")

display(fig)
plt.close(fig)

