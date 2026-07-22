# Global Industry Challenge 2026

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_black.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/nirajvenkat/gic_2026.git&envSlug=etrio_fjqdnq)


## Write-ups

[DOE OTC Challenge: Quantum-Enhanced Strategic Siting of Energy Storage and Microgrids](doe/phase_3/doc/doe_p3_doc.pdf)

[Mitsubishi/AIST Challenge: Harnessing the Generative Quantum Eigensolver for Next-Generation Materials Design](mitsubishi/phase_3/doc/mb_p3_doc.pdf)

## Setup instructions
To install the environment, run the following commands in a qBraid Terminal or using qBraid CLI:
```
qbraid envs share redeem-code 1YDHDCR
qbraid envs install etrio_fjqdnq
```
After installation you get an environment name `ENV_NAME`, activate it:
```
qbraid envs activate <ENV_NAME>
```
Convert scripts to Jupyter notebooks using:
```
jupytext --to notebook doe/phase_3/code/grid_opt.py # DOE challenge
jupytext --to notebook mitsubishi/phase_3/code/euv_spectra.py # Mitsubishi/AIST challenge
```

## DOE OTC Challenge specific instructions

## Mitsubishi/AIST Challenge specific instructions
