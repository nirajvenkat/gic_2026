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
# # Quantum Chemistry for Auger Spectroscopy
# **Team Name:** Entangled Trio
#
# This notebook demonstrates the Generative Quantum Eigensolver (GQE) combined with the Quantum Self-Consistent Equation-of-Motion (q-sc-EOM) and One-Center Approximation (OCA) for simulating core-level Auger electron spectra for $\rm H_2O$.
#
# Inspired by the workflow in [Keithley et al.](https://arxiv.org/abs/2603.12859v1) shown below:

# %% [markdown]
# ![mi](../../img/workflow_fig.png)

# %% [markdown]
# ## Setup

# %%
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import os
import pickle

# Resolve absolute paths
def get_current_file_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        if os.path.exists("mitsubishi/phase_2/code"):
            return os.path.abspath("mitsubishi/phase_2/code")
        return os.getcwd()

current_file_dir = get_current_file_dir()

# %% [markdown]
# ## Parameter Settings
#
# Configure the execution parameters for the simulation:
#
# * **`USE_CUDAQ`**: If `True`, use NVIDIA's CUDA-Q backend for GPU-accelerated quantum simulation.
# * **`USE_DIT`**: If `True`, enable Diffusion Transformer (DIT) for GQE training as an alternative to the auto-regressive GPTnano.
# * **`USE_AVAS`**: If `True`, apply Active Space Selection (AVAS) to reduce the molecular active space size.
# * **`USE_H2O_OPTIMIZATIONS`**: If `True`, enable specialized performance optimizations for the $\rm H_2O$ molecule.

# %%
USE_CUDAQ = False
USE_DIT = False
USE_AVAS = False
USE_H2O_OPTIMIZATIONS = True

cache_dir = os.path.join(current_file_dir, "datasets", "qsceom")
cache_suffix = "_avas" if USE_AVAS else ""
cache_path = os.path.join(cache_dir, f"qsceom_cache{cache_suffix}.pkl")
has_cache = os.path.exists(cache_path)
openmolcas_filepath = os.path.abspath(os.path.join(current_file_dir, "datasets", "openmolcas", "h2o_aes.gqe_integrals.h5"))

seq_len = 4
trial_name = "trial_h2o"
if USE_AVAS:
    trial_name += "_avas"
save_dir = os.path.abspath(os.path.join(current_file_dir, "datasets", f"seq_len={seq_len}/{trial_name}"))


# %% [markdown]
# ## Molecular Data Generation
#
# We define a helper function to generate molecular data (Hamiltonian, operator pool, number of qubits/electrons, Hartree-Fock state) either using PennyLane's internal `qchem` dataset downloader or by fetching molecular geometry from PubChem and building the Hamiltonian.

# %%
import numpy as np
import pennylane as qml
import pubchempy as pcp

def generate_molecule_data(molecule_name="H2", source="qchem", local_dataset_path=None, use_avas=False):
    if local_dataset_path is None:
        local_dataset_path = os.path.join(get_current_file_dir(), "datasets")
    # Get the time set T
    op_times = np.sort(np.array([-2**k for k in range(1, 5)] + [2**k for k in range(1, 5)]) / 160)

    # Build operator set P for each molecule
    molecule_data = dict()
    
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
        
        if use_avas and USE_H2O_OPTIMIZATIONS and molecule_name == "H2O":
            active_electrons = 8
            active_orbitals = 6
            num_electrons = active_electrons
            num_qubits = 2 * active_orbitals
            
            # Rebuild Hamiltonian with active space
            hamiltonian, num_qubits = qml.qchem.molecular_hamiltonian(
                symbols, 
                coords, 
                active_electrons=active_electrons, 
                active_orbitals=active_orbitals
            )
            hf_state = qml.qchem.hf_state(active_electrons, num_qubits)
            
            # Compute active space expected ground state energy (FCI energy)
            import scipy.sparse.linalg
            H_sparse = hamiltonian.sparse_matrix()
            eigenvalues = scipy.sparse.linalg.eigsh(H_sparse, k=1, which='SA', return_eigenvectors=False)
            expected_ground_state_E = float(eigenvalues[0])
        else:
            num_electrons, num_qubits = molecule.n_electrons, 2 * molecule.n_orbitals
            hf_state = dataset.hf_state
            hamiltonian = dataset.hamiltonian
            expected_ground_state_E = dataset.fci_energy
        
    elif source == "pubchem":
        # Fetch from PubChem
        compounds = pcp.get_compounds(molecule_name, 'name', record_type='3d')
        if not compounds or not hasattr(compounds[0].atoms[0], 'x') or compounds[0].atoms[0].x is None:
            # Fallback to 2D if 3D is not available
            compounds = pcp.get_compounds(molecule_name, 'name')
            
        c = compounds[0]
        symbols = [atom.element for atom in c.atoms]
        # Extract coordinates, defaulting to 0.0 if not present
        coords = np.array([[getattr(atom, 'x', 0.0) or 0.0, 
                            getattr(atom, 'y', 0.0) or 0.0, 
                            getattr(atom, 'z', 0.0) or 0.0] for atom in c.atoms])
        
        if use_avas and USE_H2O_OPTIMIZATIONS and molecule_name == "H2O":
            active_electrons = 8
            active_orbitals = 6
            num_electrons = active_electrons
            num_qubits = 2 * active_orbitals
            
            # Build Hamiltonian with active space
            hamiltonian, num_qubits = qml.qchem.molecular_hamiltonian(
                symbols, 
                coords, 
                active_electrons=active_electrons, 
                active_orbitals=active_orbitals
            )
            hf_state = qml.qchem.hf_state(active_electrons, num_qubits)
            
            # Compute active space expected ground state energy (FCI energy)
            import scipy.sparse.linalg
            H_sparse = hamiltonian.sparse_matrix()
            eigenvalues = scipy.sparse.linalg.eigsh(H_sparse, k=1, which='SA', return_eigenvectors=False)
            expected_ground_state_E = float(eigenvalues[0])
        else:
            # Build Hamiltonian and molecule using PennyLane
            hamiltonian, num_qubits = qml.qchem.molecular_hamiltonian(symbols, coords)
            
            # Simple estimation of electrons (assuming neutral molecule)
            electron_map = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'I': 53}
            num_electrons = sum([electron_map.get(s, 0) for s in symbols]) 
            
            hf_state = qml.qchem.hf_state(num_electrons, num_qubits)
            expected_ground_state_E = None # FCI energy isn't directly pulled from PubChem

    singles, doubles = qml.qchem.excitations(num_electrons, num_qubits)
    double_excs = [qml.DoubleExcitation(time, wires=double) for double in doubles for time in op_times]
    single_excs = [qml.SingleExcitation(time, wires=single) for single in singles for time in op_times]
    identity_ops = [qml.exp(qml.I(range(num_qubits)), 1j*time) for time in op_times] # For Identity
    operator_pool = double_excs + single_excs + identity_ops
    
    molecule_data[molecule_name] = {
        "op_pool": np.array(operator_pool), 
        "num_qubits": num_qubits,
        "hf_state": hf_state,
        "hamiltonian": hamiltonian,
        "expected_ground_state_E": expected_ground_state_E,
        "symbols": symbols,
        "geometry": coords,
        "active_electrons": num_electrons,
        "active_orbitals": num_qubits // 2
    }
    
    return molecule_data


# %% [markdown]
# ## Loading and Configuring Molecular Parameters
#
# For comparison, we load the dataset for the target molecule (e.g., $H_2O$) via both the offline `qchem` database and `pubchem` to extract the coordinates, Hamiltonian terms, Hartree-Fock reference state, and full operator pool of single and double excitations.

# %%
target_molecule = "H2O"

# Helper to safely get the length of hamiltonian terms across PL versions
def get_hamiltonian_terms_len(h):
    return len(getattr(h, "ops", getattr(h, "operands", [])))

# 1. Load via qchem
qchem_molecule_data = generate_molecule_data(target_molecule, source="qchem", use_avas=USE_AVAS)
qchem_data = qchem_molecule_data[target_molecule]

print(f"--- {target_molecule} (qchem) ---")
print(f"Number of Qubits: {qchem_data['num_qubits']}")
print(f"Operator Pool Size: {len(qchem_data['op_pool'])}")
print(f"HF State: {qchem_data['hf_state']}")
print(f"FCI Energy: {qchem_data['expected_ground_state_E']}")
print(f"Hamiltonian terms: {get_hamiltonian_terms_len(qchem_data['hamiltonian'])}")

# 2. Load via pubchem
pubchem_molecule_data = generate_molecule_data(target_molecule, source="pubchem", use_avas=USE_AVAS)
pubchem_data = pubchem_molecule_data[target_molecule]

print(f"\n--- {target_molecule} (pubchem) ---")
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

# %% [markdown]
# ## Subsequence Energy Evaluation
#
# We define backend-specific functions to execute subsequence energy evaluations. For GPU acceleration, CUDA-Q observes the Hamiltonian directly via custom memories and operations. For standard runs, we use PennyLane's `lightning.qubit` simulator.
#
# ### CUDA-Q Execution Kernel (Optional)
#
# If `USE_CUDAQ` is enabled, we map the PennyLane Hamiltonians to CUDA-Q spin operators and construct specialized kernels to evaluate subsequence expectation values.

# %%
import pennylane as qml
import numpy as np

if USE_CUDAQ:
    import cudaq

    # 1. Map Hamiltonians
    def pl_hamiltonian_to_cudaq(pl_ham):
        ps = qml.pauli.pauli_sentence(pl_ham)
        cudaq_ham = None
        for pw, coeff in ps.items():
            term = None
            for wire, pauli in pw.items():
                p_op = getattr(cudaq.spin, pauli.lower())(wire)
                term = p_op if term is None else term * p_op
                    
            term = cudaq.spin.i(0) if term is None else term
            cudaq_ham = coeff * term if cudaq_ham is None else cudaq_ham + (coeff * term)
                
        return cudaq_ham

    ham_cudaq = pl_hamiltonian_to_cudaq(hamiltonian)

    # Helper to apply PL ops natively using generators in CUDA-Q
    def apply_pl_op_to_cudaq(kernel, qubits, op):
        if isinstance(op, qml.ops.Exp) and isinstance(op.base, qml.Identity): return
        
        param = op.parameters[0]
        ps = qml.pauli.pauli_sentence(op.generator())
        
        for pw, coeff in ps.items():
            term = None
            for wire, pauli in pw.items():
                p_op = getattr(cudaq.spin, pauli.lower())(wire)
                term = p_op if term is None else term * p_op
                    
            if term is not None:
                kernel.exp_pauli(float(param * coeff), qubits, term)

    # Helper to construct HF state kernel
    def build_base_kernel():
        kernel = cudaq.make_kernel()
        qubits = kernel.qalloc(num_qubits)
        for i, val in enumerate(init_state):
            if val == 1: kernel.x(qubits[i])
        return kernel, qubits

    # 2. Build the GQE CUDA-Q subsequence executor
    def get_subsequence_energies_cudaq(op_seq):
        energies = []
        for ops in op_seq:
            seq_es = []
            for step in range(1, len(ops) + 1):
                kernel, qubits = build_base_kernel()
                for op in ops[:step]:
                    apply_pl_op_to_cudaq(kernel, qubits, op)
                seq_es.append(cudaq.observe(kernel, ham_cudaq).expectation())
            energies.append(seq_es)
            
        return np.array(energies)

    # Verify with a tiny sequence
    print(get_subsequence_energies_cudaq([[op_pool[0], op_pool[1]]]))

# %% [markdown]
# ### PennyLane Simulator Execution
#
# We define the standard PennyLane QNode and subsequence collation function on `lightning.qubit` using snapshots to compute intermediate energies efficiently in a single simulator run.

# %%
dev = qml.device("lightning.qubit", wires=num_qubits)

@qml.qnode(dev)
def energy_circuit(gqe_ops):
    # Computes Eq. 1 from Nakaji et al. based on the selected unitary operators
    qml.BasisState(init_state, wires=range(num_qubits)) # Initial state <-- Hartree Fock state
    for op in gqe_ops:
        qml.Snapshot(measurement=qml.expval(hamiltonian))
        qml.apply(op) # Applies each of the unitary operators
    return qml.expval(hamiltonian)

energy_circuit = qml.snapshots(energy_circuit)

def get_subsequence_energies_pl(op_seq):
    # Collates the energies of each subsequence for a batch of sequences
    energies = []
    for ops in op_seq:
        es = energy_circuit(ops)
        energies.append(
            [es[k].item() for k in list(range(1, len(ops))) + ["execution_results"]]
        )
    return np.array(energies)

def get_subsequence_energies(op_seq):
    if USE_CUDAQ:
        return get_subsequence_energies_cudaq(op_seq)
    else:
        return get_subsequence_energies_pl(op_seq)

if not USE_CUDAQ:
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

cache_dir = os.path.join(current_file_dir, "datasets", "qsceom")
use_avas_val = globals().get("USE_AVAS", False)
cache_suffix = "_avas" if use_avas_val else ""
cache_path = os.path.join(cache_dir, f"qsceom_cache{cache_suffix}.pkl")
has_cache = os.path.exists(cache_path)

seq_len = 4
trial_name = "trial_h2o"
if use_avas_val:
    trial_name += "_avas"
save_dir = os.path.abspath(os.path.join(current_file_dir, "datasets", f"seq_len={seq_len}/{trial_name}"))

if not has_cache:
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
    """Multi-headed self-attention"""
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

if torch.backends.mps.is_available():
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
# # Step 2: Quantum Self-Consistent Equation-of-Motion (q-sc-EOM)
#
# We identify the best ansatz sequence generated by GQE and use it as a reference state. We then execute the q-sc-EOM algorithm to calculate:
# 1. Core-ionized energy levels (IP space, $N-1$ electrons).
# 2. Doubly ionized energy levels (DIP space, $N-2$ electrons).

# %%
if 'has_cache' not in globals():
    import os
    current_file_dir = get_current_file_dir()
    cache_dir = os.path.join(current_file_dir, "datasets", "qsceom")
    use_avas_val = globals().get("USE_AVAS", False)
    cache_suffix = "_avas" if use_avas_val else ""
    cache_path = os.path.join(cache_dir, f"qsceom_cache{cache_suffix}.pkl")
    has_cache = os.path.exists(cache_path)
    seq_len = 4
    trial_name = "trial_h2o"
    if use_avas_val:
        trial_name += "_avas"
    save_dir = os.path.abspath(os.path.join(current_file_dir, "datasets", f"seq_len={seq_len}/{trial_name}"))

if not has_cache:
    from qsceom import qscEOM

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
            params.append(float(op.parameters[0]))
            wires = op.wires.tolist()
            if USE_AVAS and USE_H2O_OPTIMIZATIONS and target_molecule == "H2O":
                # Shift wires by +2 to map 12-qubit active space to 14-qubit full space
                wires = [w + 2 for w in wires]
            ash_excitation.append(tuple(wires))

    # 3. Retrieve molecule details directly from the in-memory dictionary to prevent redundant downloads/DNS errors!
    symbols = qchem_data["symbols"]
    geometry = qchem_data["geometry"]
    if USE_AVAS and USE_H2O_OPTIMIZATIONS and target_molecule == "H2O":
        # qscEOM must run on the full space (10 e-, 7 orbitals) to compute core-ionized states
        active_electrons = 10
        active_orbitals = 7
    else:
        active_electrons = qchem_data["active_electrons"]
        active_orbitals = qchem_data["active_orbitals"]
    charge = 0 # Default calculation assumes a neutral molecule (e.g. H2)

    print(f"Preparing q-sc-EOM for {target_molecule}...")
    print(f"Number of excitations in reference ansatz from GQE: {len(params)}")

    # Resolve absolute path for pyscf datasets directory
    import os
    current_file_dir = get_current_file_dir()
    datasets_pyscf_dir = os.path.abspath(os.path.join(current_file_dir, "./datasets/pyscf"))
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
        outpath=datasets_pyscf_dir
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
        outpath=datasets_pyscf_dir
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
        "USE_AVAS": USE_AVAS if 'USE_AVAS' in globals() else False,
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
# # Step 2.5: Minimal Basis Projection of Atomic Integrals
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
    cache_dir = os.path.join(current_file_dir, "datasets", "qsceom")
    use_avas_val = globals().get("USE_AVAS", False)
    cache_suffix = "_avas" if use_avas_val else ""
    cache_path = os.path.join(cache_dir, f"qsceom_cache{cache_suffix}.pkl")
    if os.path.exists(cache_path):
        print(f"Loading variables from cache for Step 2.5: {cache_path}")
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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Computing integral data via PySCF with minimal basis projection...")
print("(Paper Eq. 19: D = T^{-1} U C, where T=MBS overlap, U=MBS-CGTO overlap, C=MO coefficients)")

# 1. Reconstruct molecule and compute RHF in full STO-3G basis
mol_str = "; ".join([f"{sym} {coord[0]} {coord[1]} {coord[2]}" for sym, coord in zip(symbols, geometry)])
mol = gto.M(atom=mol_str, basis="sto-3g", charge=charge, symmetry=False)
mf = scf.RHF(mol)
mf.kernel(verbose=0)

# 2. Identify emitter atom and its basis functions in STO-3G
if USE_H2O_OPTIMIZATIONS and target_molecule == "H2O":
    emitter_atom_idx = 0  # Oxygen in H2O
else:
    emitter_atom_idx = globals().get("EMITTER_ATOM_INDEX", 0)

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
active_indices = list(range(active_orbitals))

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
        sys.path.insert(0, "/Users/nvenkat/Desktop/Repos/OpenMolcas/Tools/AugerOca")
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
# # Step 3: One-Center Approximation (OCA) & Auger Transition Intensities
#
# We calculate the transition reduced density matrix (RDM) $\gamma_{pq}$ between the core-ionized state and double-ionization states, and contract it with the projected atomic integrals to compute Auger transition intensities under Fermi's Golden Rule.

# %%
import torch
import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas

HARTREE_TO_EV = 27.211386245988
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def compute_gamma_pq(C_IP, C_DIP, list_IP, list_DIP, core_idx, n_spin):
    """
    Classical mapping of the Transition RDM: < Phi_DIP | a^dagger_c a_p a_q | Phi_IP >
    Returns a tensor of shape (N_DIP_states, N_IP_states, n_spin, n_spin)
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
        target_ip_state_try = 0
        max_overlap_try = 0.0

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
# # Step 4: Auger Spectrum Broadening and Visualization
#
# We compute the Auger electron kinetic energies ($E_{\text{kin}} = E_{\text{IP}} - E_{\text{DIP}}$) and apply Lorentzian broadening with a typical experimental line-width parameter ($\Gamma = 1.5\text{ eV}$) to simulate the final Auger electron spectrum.

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
if E_ke_valid.size == 0:
    x_energies = np.linspace(0.0, 1.0, 1000)
    spectrum = np.zeros_like(x_energies, dtype=float)
else:
    x_min = max(0.0, float(E_ke_valid.min()) - 15.0)
    x_max = float(E_ke_valid.max()) + 15.0
    x_energies = np.linspace(x_min, x_max, 1500)
    spectrum = np.zeros_like(x_energies, dtype=float)

    gamma_width_ev = 1.5
    for E_k, intensity in zip(E_ke_valid, Gamma_ke_valid):
        spectrum += float(intensity) * (gamma_width_ev / ((x_energies - E_k) ** 2 + gamma_width_ev ** 2))

spectrum_norm = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum

# Construct classical reference spectrum for overlay
classical_peaks = []
if USE_H2O_OPTIMIZATIONS and target_molecule == "H2O":
    # Fallback to precomputed exact values for H2O if files are missing
    classical_peaks = [
        (500.49, 0.001367),
        (498.27, 0.001299),
        (492.82, 0.001971),
        (487.00, 0.000028),
        (478.74, 0.000010)
    ]

# Try to load from auger.spectrum.out if it is present
spec_out_path = os.path.join(current_file_dir, "datasets", "openmolcas", "auger.spectrum.out")
if not os.path.exists(spec_out_path):
    spec_out_path = "auger.spectrum.out"
    
if os.path.exists(spec_out_path):
    try:
        loaded_peaks = []
        with open(spec_out_path, "r") as f_spec:
            for line in f_spec:
                if line.strip() and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        # auger.spectrum.out contains: binding_energy intensity
                        loaded_peaks.append((float(parts[0]), float(parts[1])))
        if loaded_peaks:
            classical_peaks = loaded_peaks
    except Exception as e:
        print(f"Could not parse classical spectrum file: {e}")
        
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

gqe_plot = df_spectrum.hvplot.line(
    x="Kinetic Energy (eV)",
    y="GQE (quantum)",
    label="GQE (quantum)",
    color="darkred",
    width=900,
    height=450,
)

if "Classical Reference" in df_spectrum.columns:
    classical_plot = df_spectrum.hvplot.line(
        x="Kinetic Energy (eV)",
        y="Classical Reference",
        label="Classical Reference",
        color="blue",
        line_dash="dashed",
    )
    spectrum_fig = (gqe_plot * classical_plot).opts(
        title=f"Auger Spectrum: GQE vs Classical Reference",
        ylabel="Normalized Intensity",
        legend_position="top_right",
        fontscale=1.2,
    )
else:
    spectrum_fig = gqe_plot.opts(
        title=f"Auger Spectrum: GQE (quantum)",
        ylabel="Normalized Intensity",
        legend_position="top_right",
        fontscale=1.2,
    )

spectrum_fig

# %% [markdown]
# # Classical Reference Spectrum Printing
#
# If the raw OpenMolcas output is present in the repository, we can execute the classical post-processing calculations or print the precomputed benchmark energies and intensities for direct comparison.

# %%
import os
import subprocess
import shutil
import sys

def print_precomputed_benchmarks():
    print("\n--- Classical Reference Spectrum (Precomputed Benchmarks) ---")
    print("State 6  (KE = 500.49 eV): Intensity = 0.001367 (Lone Pair, gas phase experimental ~500 eV)")
    print("State 7  (KE = 498.27 eV): Intensity = 0.001299 (Lone Pair)")
    print("State 8  (KE = 492.82 eV): Intensity = 0.001971 (Inner Valence / CVV Peak)")
    print("State 9  (KE = 487.00 eV): Intensity = 0.000028 (Low-energy shoulder)")
    print("State 31 (KE = 478.74 eV): Intensity = 0.000010 (Low-energy tail)")

# Check if the OpenMolcas rassi.h5 file is present
raw_rassi_path = os.path.join(current_file_dir, "datasets", "openmolcas", "h2o_aes.rassi.h5")

# Look for any r2TM_ files in the current directory or datasets/openmolcas
r2tm_files = [f for f in os.listdir(".") if f.startswith("r2TM_")]
if not r2tm_files and os.path.exists(os.path.join(current_file_dir, "datasets", "openmolcas")):
    r2tm_files = sorted([f for f in os.listdir(os.path.join(current_file_dir, "datasets", "openmolcas")) if f.startswith("r2TM_")])
    r2tm_dir = os.path.join(current_file_dir, "datasets", "openmolcas")
else:
    r2tm_files = sorted(r2tm_files)
    r2tm_dir = "."

if os.path.exists(raw_rassi_path) and r2tm_files:
    print("\n--- Computing and Printing Classical Reference Spectrum ---")
    
    # Temporarily copy the raw rassi.h5 to the directory with r2TM_ files
    temp_rassi_path = os.path.join(r2tm_dir, "h2o_aes.rassi.h5")
    copied_temp = False
    if not os.path.exists(temp_rassi_path) or os.path.getsize(temp_rassi_path) != os.path.getsize(raw_rassi_path):
        shutil.copy2(raw_rassi_path, temp_rassi_path)
        copied_temp = True
        
    orig_cwd = os.getcwd()
    try:
        # Load energies from raw RASSI HDF5
        with h5py.File(raw_rassi_path, "r") as f_raw:
            sfs_energies = np.array(f_raw["SFS_ENERGIES"])
        E_IP = sfs_energies[0]
        E_DIPs = sfs_energies[1:]
        HARTREE_TO_EV = 27.211386245988
        E_kinetics_ev = (E_IP - E_DIPs) * HARTREE_TO_EV
        
        # Add local AugerOca directory to sys.path to get the fixed version instead of site-packages
        sys.path.insert(0, "/Users/nvenkat/Desktop/Repos/OpenMolcas/Tools/AugerOca")
        
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
        
        print("\nClassical Reference Spectrum (KE and Intensity from OCA):")
        print(f"{'State':<10}{'Kinetic Energy (eV)':<25}{'Intensity (a.u.)':<20}")
        print("-" * 55)
        
        for f in r2tm_files:
            # Parse state index from filename, e.g., r2TM_SDA_002_001 -> state 2 (1st double ionized state, index 0)
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
            # Modify scattering center to match basis labels
            if USE_H2O_OPTIMIZATIONS and target_molecule == "H2O":
                if vals[2][0] == "O 1s":
                    vals[2][0] = "O1 1s"
                
            hd5_file, basis_id_hd5, element = init3(vals[12], vals[5])
            
            # Execute the classical driver (writes output to Auger_OCA.<f>.out)
            driver_auger(
                f, False, True, False, True,
                vals[0], vals[2], vals[3], vals[4], vals[5],
                vals[6], vals[9], vals[12], vals[15], vals[16],
                vals[17], vals[18], vals[20], vals[21], vals[8],
                vals[22], vals[23], hd5_file, element, basis_id_hd5
            )
            
            # Read output, extract intensity
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
                                # Python 3 syntax error bug fix if it occurs, but it's just except ValueError
                                except ValueError:
                                    intensity = 0.0
                os.remove(output_file) # Clean up output file
                
            ke = E_kinetics_ev[state_idx] if state_idx < len(E_kinetics_ev) else 0.0
            print(f"State {state_idx+2:<5}{ke:<25.2f}{intensity:<20.6f}")
            
    except Exception as e:
        print(f"Error computing classical spectrum: {e}")
        # Fallback to precomputed benchmarks if calculation failed
        if USE_H2O_OPTIMIZATIONS and target_molecule == "H2O":
            print_precomputed_benchmarks()
        else:
            print("\nClassical reference calculations failed and no precomputed benchmarks are configured for this molecule.")
    finally:
        if copied_temp and os.path.exists(temp_rassi_path):
            os.remove(temp_rassi_path)
        os.chdir(orig_cwd)
else:
    # Print the precomputed classical reference peaks for comparison
    if USE_H2O_OPTIMIZATIONS and target_molecule == "H2O":
        print_precomputed_benchmarks()
    else:
        print("\nNo classical reference data found.")
