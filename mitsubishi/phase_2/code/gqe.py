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
# # Generative Quantum Eigensolver (GQE) & q-sc-EOM for Core-Level Spectroscopy
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

# Parameters
USE_CUDAQ = False
USE_DIT = False
USE_AVAS = True

cache_dir = os.path.join(current_file_dir, "datasets", "qsceom")
cache_suffix = "_avas" if USE_AVAS else ""
cache_path = os.path.join(cache_dir, f"qsceom_cache{cache_suffix}.pkl")
has_cache = os.path.exists(cache_path)

seq_len = 4
trial_name = "trial_h2o"
if USE_AVAS:
    trial_name += "_avas"
save_dir = os.path.abspath(os.path.join(current_file_dir, f"./seq_len={seq_len}/{trial_name}"))


# %% [markdown]
# ## Molecular Data Generation
#
# We define a helper function to generate molecular data (Hamiltonian, operator pool, number of qubits/electrons, Hartree-Fock state) either using PennyLane's internal `qchem` dataset downloader or by fetching molecular geometry from PubChem and building the Hamiltonian.

# %%
import numpy as np
import pennylane as qml
import pubchempy as pcp

def generate_molecule_data(molecule_name="H2", source="qchem", local_dataset_path="./datasets", use_avas=False):
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
        
        if use_avas and molecule_name == "H2O":
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
        
        if use_avas and molecule_name == "H2O":
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
# We load the dataset for the target molecule (e.g., $H_2O$) via both the offline `qchem` database and `pubchem` to extract the coordinates, Hamiltonian terms, Hartree-Fock reference state, and full operator pool of single and double excitations.

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

# Expose standard variables that can be used down the line (defaulting to qchem for now)
op_pool = qchem_data["op_pool"]
num_qubits = qchem_data["num_qubits"]
init_state = qchem_data["hf_state"]
hamiltonian = qchem_data["hamiltonian"]
grd_E = qchem_data["expected_ground_state_E"]
op_pool_size = len(op_pool)

# %% [markdown]
# ## Environment Configuration
#
# Define configuration flags to determine whether to use the GPU-accelerated **CUDA-Q** simulator back-end and whether to use the **Diffusion Transformer (DiT)** architecture instead of standard GPT.

# %%

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
save_dir = os.path.abspath(os.path.join(current_file_dir, f"./seq_len={seq_len}/{trial_name}"))

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

if not has_cache:
    try:
        losses = pd.read_csv(f"{save_dir}/losses.csv")["0"]
        loss_fig = losses.hvplot(
            title="Training loss progress", ylabel="loss", xlabel="Training epochs", logy=True
        ).opts(fig_size=600, fontscale=2, aspect=1.2)
        loss_fig
    except Exception as e:
        print(f"Skipping training loss plot: {e}")

# %% [markdown]
# ## GQE Energy Predictions vs. True Energies
#
# We plot the mean and range of the GQE-predicted energy sequences against their true quantum simulator energy values, demonstrating how the generator converges towards the ground-state energy.

# %%
if not has_cache:
    try:
        df_true = pd.read_csv(f"{save_dir}/true_Es_t.csv").iloc[:, 1:]
        df_pred = pd.read_csv(f"{save_dir}/pred_Es_t.csv").iloc[:, 1:]

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
        fig
    except Exception as e:
        print(f"Skipping evaluation stats plot: {e}")

# %% [markdown]
# ## Evaluation Summary
#
# We compare the statistics (average, minimum, and maximum energy) of random sequences against the outputs of the latest trained model and the best-performing model checkpoint.

# %%
if not has_cache:
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
        "Aves": [train_sub_seq_en[:, -1].mean(), true_Es_.mean(), loaded_true_Es_.mean()],
        "Mins": [train_sub_seq_en[:, -1].min(), true_Es_.min(), loaded_true_Es_.min()],
        "Maxs": [train_sub_seq_en[:, -1].max(), true_Es_.max(), loaded_true_Es_.max()],
        "Mins_error": [
            abs(train_sub_seq_en[:, -1].min() - grd_E),
            abs(true_Es_.min() - grd_E),
            abs(loaded_true_Es_.min() - grd_E),
        ],
    })
    df_compare_Es

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
    save_dir = os.path.abspath(os.path.join(current_file_dir, f"./seq_len={seq_len}/{trial_name}"))

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
            if USE_AVAS and target_molecule == "H2O":
                # Shift wires by +2 to map 12-qubit active space to 14-qubit full space
                wires = [w + 2 for w in wires]
            ash_excitation.append(tuple(wires))

    # 3. Retrieve molecule details directly from the in-memory dictionary to prevent redundant downloads/DNS errors!
    symbols = qchem_data["symbols"]
    geometry = qchem_data["geometry"]
    if USE_AVAS and target_molecule == "H2O":
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

# 2. Identify emitter atom (oxygen, index 0) and its basis functions in STO-3G
# STO-3G: O has 1s, 2s, 2p_x, 2p_y, 2p_z = 5 functions on oxygen (first 5 AO indices for H2O)
emitter_atom_idx = 0  # Oxygen in H2O
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
eri_ao = mol.intor("int2e")  # AO basis 2-electron integrals (nao, nao, nao, nao)
core_ao_idx = 0  # Oxygen 1s core AO in H2O is index 0
active_indices = list(range(active_orbitals))

# Use Oxygen valence atomic orbitals (2s, 2px, 2py, 2pz) as continuum proxies
f_orbitals = [1, 2, 3, 4]

# OCA approximation: contract the emitter AO-basis integrals with MBS projection C_mbs (D)
# For each continuum channel f, V_pq = sum_{μ,ν ∈ emitter} D_{μ,p} * D_{ν,q} * [ (f mu | c nu) - (f nu | c mu) ]
V_pq_channels = []
for f_ao in f_orbitals:
    V_pq_projected = np.zeros((active_orbitals, active_orbitals))
    for p_idx, p in enumerate(active_indices):
        for q_idx, q in enumerate(active_indices):
            val = 0.0
            for i, mu in enumerate(mbs_indices):
                for j, nu in enumerate(mbs_indices):
                    ao_int_direct = eri_ao[f_ao, mu, core_ao_idx, nu]
                    ao_int_exchange = eri_ao[f_ao, nu, core_ao_idx, mu]
                    val += C_mbs[i, p] * C_mbs[j, q] * (ao_int_direct - ao_int_exchange)
            V_pq_projected[p_idx, q_idx] = val
    V_pq_channels.append(V_pq_projected)

V_pq_numpy = np.array(V_pq_channels)  # Shape: (n_f, active_orbitals, active_orbitals)
V_pq = torch.tensor(V_pq_numpy, device=device, dtype=torch.float32)

print(f"Constructed projected V_pq tensor using OCA projection: shape {V_pq.shape}")

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
if not isinstance(V_pq, torch.Tensor):
    V_pq = torch.tensor(V_pq, dtype=torch.float64)
V_pq_cpu = V_pq.detach().cpu().to(torch.float64)

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

# 4. Spin-orbital expansion for classical integrals (V_pq)
# V_pq has shape (n_f, active_orbitals, active_orbitals)
V_pq_cpu = V_pq.detach().cpu().to(torch.float64)
V_lmpq_spin = torch.zeros((len(f_orbitals), 1, n_spin_full, n_spin_full), dtype=torch.float64)

# Populate all spin-channel pairs to avoid artificial zeroing from spin-map mismatch
# Use the local active space indices i and j to index V_lmpq_spin
for f_idx in range(len(f_orbitals)):
    for i, act_i in enumerate(active_indices):
        for j, act_j in enumerate(active_indices):
            val = V_pq_cpu[f_idx, i, j]
            for s1 in (0, 1):
                for s2 in (0, 1):
                    V_lmpq_spin[f_idx, 0, i * 2 + s1, j * 2 + s2] = val

# 5. Tensor contraction (OCA) in complex128 for numerical stability
amplitudes = torch.einsum(
    "kpq,lmpq->klm",
    gamma_pq.to(torch.complex128),
    V_lmpq_spin.to(torch.complex128),
)

# Apply Fermi's Golden Rule
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

# Plot primary unshifted result
df_spectrum = pd.DataFrame(
    {
        "Kinetic Energy (eV)": x_energies,
        "Intensity": spectrum,
        "Intensity (normalized)": spectrum_norm,
    }
)

spectrum_fig = df_spectrum.hvplot.line(
    x="Kinetic Energy (eV)",
    y="Intensity (normalized)",
    title=f"Simulated Auger Spectrum with {USE_DIT and 'DiT' or 'GPT'} Reference State",
    ylabel="Normalized Intensity",
    color="darkred",
    width=900,
    height=450,
).opts(fontscale=1.2)

spectrum_fig
