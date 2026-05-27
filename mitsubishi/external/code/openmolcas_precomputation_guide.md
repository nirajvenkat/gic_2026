# OpenMolcas & Auger-OCA Precomputation Guide

This guide details how to compile **OpenMolcas**, configure the environment, run the classical calculations for water ($H_2O$), and link the output directly to the hybrid quantum-classical GQE pipeline.

---

## 1. Compiling OpenMolcas & Environment Setup

Since OpenMolcas is a compiled C/Fortran suite, follow these steps to build and configure it on macOS:

### Prerequisites (using Homebrew)
Ensure you have compilers and build utilities installed:
```bash
brew install cmake gcc
```

### Cloning and Building
Choose a directory outside your git repository (e.g., your home folder or a `Repos/` directory) and clone/build OpenMolcas:
```bash
git clone https://gitlab.com/Molcas/OpenMolcas.git
cd OpenMolcas
git submodule update --init External/lapack
mkdir build
cd build
# Configure with internal Lapack/BLAS for simplicity
cmake .. -DLINALG=Internal
make -j$(sysctl -n hw.ncpu)
```

### Environment Configuration
Add the compiled binaries and environment variables to your shell configuration (e.g., `.zshrc`):
```bash
export PATH="/Users/nvenkat/Desktop/Repos/OpenMolcas/build/bin:$PATH"
export MOLCAS="/Users/nvenkat/Desktop/Repos/OpenMolcas"
```
*(Make sure to update the paths if you cloned OpenMolcas to a different directory).*

---

## 2. Creating Input Files

You need two files in your calculation directory: the geometry coordinate file (`H2O.xyz`) and the OpenMolcas instruction file (`h2o_aes.input`).

### Geometry File (`H2O.xyz`)
Create a file named `H2O.xyz` containing the water molecule coordinates:
```text
3
Water molecule geometry in STO-3G coords
O   0.000000   0.000000   0.117790
H   0.000000   0.755453  -0.471161
H   0.000000  -0.755453  -0.471161
```

### OpenMolcas Input File (`h2o_aes.input`)
Create a file named `h2o_aes.input` in the same directory:
```text
>>> EXPORT MOLCAS_PRINT = 2

&GATEWAY
  Title = Water molecule normal Auger calculation (Singlet final states)
  Coord = H2O.xyz
  Basis = STO-3G

&SEWARD

&SCF
  Title = Reference RHF ground state

* --- Step 1: Initial Core-Ionized State (Doublet, N-1 electrons) ---
* Active space contains core orbital (RAS1) and valence orbitals (RAS2)
&RASSCF
  Title = Core-ionized initial state
  Spin = 2
  Symmetry = 1
  Inactive = 0 0 0 0
  RAS1 = 1 0 0 0
  RAS2 = 3 1 0 2
  nActEl = 9 1 0
  HEXS 1 1

* --- Step 2: Final Doubly-Ionized States (Singlet, N-2 electrons) ---
* Valence-valence double-hole states (core 1s fully occupied and inactive)
&RASSCF
  Title = Double-hole final states
  Spin = 1
  Symmetry = 1
  Inactive = 1 0 0 0
  RAS2 = 3 1 0 2
  nActEl = 6 0 0
  CIROOT
  10 10 1

* --- Step 3: RASSI Transition Density Matrix Computation ---
&RASSI
  NrofJobIphs = 2
  AllStates
  Dyson
  TDYS = 1
  O 1s
```

---

## 3. Running the Simulation & Classical Reference

### Running the OpenMolcas calculation
Run the OpenMolcas calculation from your terminal:
```bash
pymolcas h2o_aes.input
```

This will run the calculation and output:
* A RASSI HDF5 property file: `h2o_aes.rassi.h5`
* Several Dyson transition density matrix files starting with `r2TM_` (representing core-valence transition densities).

### Running the Classical reference with `auger-oca`
The `auger-oca` package post-processes these files to compute classical Auger intensities using the One-Center Approximation.
Make sure you are in the directory containing `h2o_aes.rassi.h5` and the `r2TM_` files, and run:
```bash
# Activate the qi conda environment
conda activate qi

# Execute the auger-oca driver script
python3 $MOLCAS/Tools/AugerOca/auger_main.py -d ./ --aes --s --spec
```

This will produce:
* A directory named `auger_outputs/` containing classical intensities for all double-hole final states.
* An aggregated spectrum summary (often printed as `BE(eV) and Intensity from OCA`). This is your classical reference spectrum.

---

## 4. Extracting and Linking to the GQE Pipeline

In the hybrid quantum-classical GQE pipeline, the transition density matrix is computed quantumly. We only need the **atomic bound-continuum transition integrals** from the One-Center Approximation (OCA) database to project them into our molecular active space.

We have implemented an **automatic database fallback** directly in the loader of [gqe.py](file:///Users/nvenkat/Desktop/Repos/gic_2026/mitsubishi/phase_2/code/gqe.py):
* If the file `h2o_aes.rassi.h5` does not exist in `datasets/openmolcas/`, the loader will automatically import `auger_oca`, query its atomic integrals database for Oxygen, and generate the required HDF5 file on the fly.

Alternatively, you can manually run the helper script [prepare_openmolcas_integrals.py](file:///Users/nvenkat/Desktop/Repos/gic_2026/mitsubishi/phase_2/code/prepare_openmolcas_integrals.py) to compile the HDF5 file:
```bash
conda activate qi
cd /Users/nvenkat/Desktop/Repos/gic_2026/mitsubishi/phase_2/code
python3 prepare_openmolcas_integrals.py
```

This generates:
* **GQE input file:** `mitsubishi/phase_2/code/datasets/openmolcas/h2o_aes.rassi.h5`

> [!NOTE]
> The HDF5 file is populated with 9 continuum wave channels ($L=0, 1, 2$) and maps the Oxygen core and valence atomic orbitals to the PySCF STO-3G basis indexing.

---

## 5. Running and Verifying GQE

You can run the GQE simulation immediately without manually copying files or installing OpenMolcas:

```bash
# Activate conda environment
conda activate qi

# Run GQE (it will automatically generate the integrals file from auger_oca if missing)
cd /Users/nvenkat/Desktop/Repos/gic_2026/mitsubishi/phase_2/code
python3 gqe.py
```

With `USE_OPEN_MOLCAS = True` enabled in [gqe.py](file:///Users/nvenkat/Desktop/Repos/gic_2026/mitsubishi/phase_2/code/gqe.py), the pipeline will:
1. Load `h2o_aes.rassi.h5` (or auto-generate it from `auger_oca` if it doesn't exist).
2. Construct the projected $V_{pq}$ transition operator using the minimal basis set (MBS) projection matrix.
3. Compute the Auger transition intensities $\Gamma_k$ using the quantum state amplitudes from the Q-sc-EOM algorithm.
4. Output the quantum-computed intensities to compare directly against the classical reference spectrum.
