# pypei

## This repo is a work in progress

`pypei` is a Python package that exposes some functionality on top of the CasADi stack that implements a variant of the generalised profiling method.

## Installation

```bash
python setup.py develop
```

### Prerequisites

```bash
pip install -r requirements.txt
```

or

```bash
conda install --channel conda-forge --file requirements.txt
```

For conda on Windows, you will need to use pip to install conda:

```powershell
conda install --file requirements_win_conda.txt
pip install -r requirements_win_pip.txt
```