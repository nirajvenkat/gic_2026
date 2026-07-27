# Global Industry Challenge 2026

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_black.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/nirajvenkat/gic_2026.git&envSlug=etrio_fjqdnq)

**[Link to Repository](https://github.com/nirajvenkat/gic_2026.git)**

## Write-ups

[DOE OTC Challenge: Quantum-Enhanced Strategic Siting of Energy Storage and Microgrids](doe/phase_3/doc/doe_p3_doc.pdf)

[Mitsubishi/AIST Challenge: Harnessing the Generative Quantum Eigensolver for Next-Generation Materials Design](mitsubishi/phase_3/doc/mb_p3_doc.pdf)

## Setup instructions
To install the environment, run the following commands in a qBraid Lab Terminal or using qBraid CLI:
```bash
qbraid envs share redeem-code 1YDHDCR

qbraid envs install etrio_fjqdnq
```
After installation activate the environment:
```bash
qbraid envs activate etrio

qbraid kernels add etrio

cd gic_2026/
```
Convert scripts to Jupyter notebooks using:
```bash
# DOE challenge
jupytext --to notebook doe/phase_3/code/grid_opt.py

# Mitsubishi/AIST challenge
jupytext --to notebook mitsubishi/phase_3/code/euv_spectra.py
```

## DOE OTC Challenge specific instructions

- We use the Defining Point Algorithm (DPA) as the gold-standard classical benchmark. This binary must be built and placed near the `grid_opt` notebook. See the [DPA compilation guide](doe/external/code/dpa/dpa_compilation_guide.md). Alternatively, you may use this pre-compiled [Linux binary for DPA](https://drive.google.com/drive/folders/1jQLfPBWhmzgUGXMV4qMcVtOr92yZjjlm?usp=sharing).

- Samplomatic does not support the Qiskit Aer simulator. Therefore `USE_SAMPLOMATIC=True` with `USE_QPU=False` fails. We are exploring a workaround in a separate `smatic-noqpu` branch, but this is extremely slow (6 hours per run) and not recommended. We were not able to test the `USE_SAMPLOMATIC` setting on QPUs.

## Mitsubishi/AIST Challenge specific instructions

- If you have `UserWarning: CUDA initialization: The NVIDIA driver on your system is too old ...` after executing the second cell (Parameter Settings), try the following in a terminal:

    1. Uninstall existing PyTorch packages
        ```bash
        pip uninstall -y torch
        ```
    2. Install PyTorch with CUDA 12.1 compatibility
        ```bash
        pip install torch --index-url https://download.pytorch.org/whl/cu121
        ```
    
    *(If your environment still complains after this, replace cu121 with cu118 in the URL to use the ultra-compatible CUDA 11.8 build).*

- See [OpenMolcas Guide](mitsubishi/external/code/openmolcas_precomputation_guide.md) to plot a classical Auger spectrum for smaller molecules. The last cell in `euv_spectra` notebook can take OpenMolcas RASSI files to view the classical Auger spectrum alongside the quantum (GQE + q-sc-EOM + OCA) spectrum. Alternatively, for $\rm{LiH}$ and $\rm{H_2O}$, you may find the files here: [LiH Auger Data](https://drive.google.com/drive/folders/1X-CWoNeIbO0JL7Mm3dA7Lle5Dd58L12M?usp=sharing), [H2O Auger Data](https://drive.google.com/drive/folders/1wt1W31JQZ0AaxqLCQr5WzMl2fZ6KqoNg?usp=sharing). These files must be placed in `data/openmolcas` alongside the notebook.