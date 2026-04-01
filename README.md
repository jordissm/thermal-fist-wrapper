# Van der Waals with conserved charges
Framework to estimate van der Waals equation of state parameters for nuclear matter with a dependency on conserved charges (baryon number $B$, strangeness $S$, and electric charge $Q$)

[![License: GPL v3](https://img.shields.io/badge/License-University_of_Illinois/NCSA_Open_Source-blue.svg)](LICENSE)

## Overview
This is a framework to calculate the Hadron Resonance Gas (HRG) susceptibilities using a van der Waals equation of state with varying parameters dependent on conserved charges (baryon number $B$, strangeness $S$, and electric charge $Q$) as implemented in to execute [`Thermal-FIST`](https://github.com/vlvovch/Thermal-FIST).

## Features
- Generation of vdW EoS interaction parameters with different charge dependence prescriptions
- Full integration with [`Thermal-FIST`](https://github.com/vlvovch/Thermal-FIST)
- Efficient output in HDF5 format

## Installation

The installation of the package is performed in three steps:

1) Create a virtual environment
2) Activate the virtual environment
3) Install the package

### 1) Create a virtual environment

To create a virtual environment,

```terminal
python3 -m venv .thermal-fit-wrapper-venv
```

This will create a (hidden) directory named `.thermal-fit-wrapper-venv` with the virtual environment.

### 2) Activate the virtual environment

On Linux/macOS, type

```terminal
source .thermal-fit-wrapper-venv/bin/activate
```

and on Windows (PowerShell),

```terminal
.\.thermal-fit-wrapper-venv\Scripts\Activate.ps1
```

### 3) Installing the package

```terminal
pip install -e '.[dev]'
```

## Usage
```terminal
thermal-fit-wrapper <SCRIPT-NAME>
```

## Citation
If you use this code, please cite:
```bibtex
@misc{identifier,
      title={}, 
      author={Jordi Salinas San Martín and Feyisola Nana and Jacquelyn Noronha-Hostler},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={nucl-th},
      url={https://arxiv.org/abs/}, 
}
```

## Contributing


## Authorship