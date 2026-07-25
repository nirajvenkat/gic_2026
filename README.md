# Global Industry Challenge 2026

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_black.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/nirajvenkat/gic_2026.git&envSlug=etrio_fjqdnq)


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

- Attempting to run `USE_SAMPLOMATIC=True` with `USE_QPU=False` is not supported by Qiskit Aer; we are exploring a workaround in a separate `smatic-noqpu` branch. However this is extremely slow (6 hours per run) and not recommended. Overall `USE_SAMPLOMATIC` remains untested due to QPU access issues.

- 

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

- [OpenMolcas Guide](mitsubishi/external/code/openmolcas_precomputation_guide.md)
In the last cell we need OpenMolcas RASSI files to view the classical Auger spectrum alongside the quantum (GQE + q-sc-EOM + OCA) one. 