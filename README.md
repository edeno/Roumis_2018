# Roumis_2018

### Installation
1. Install [miniconda](https://conda.io/miniconda.html) (or anaconda) if it isn't already installed.

2. Go to the local repository (`.../Roumis_2018`) and install the conda environment for the repository. Type into bash:
```bash
conda update -q conda  # Make sure conda is up to date
conda env create -f environment.yml  # create development environment
source activate Roumis_2018  # activate environment
python setup.py develop  # allow editing of source code
```
