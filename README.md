# Gaia Astrometric Exoplanet Orbits Detection

This repository contains the code and data accompanying the paper _[Machine learning-based identification of Gaia astrometric exoplanet orbits](https://ui.adsabs.harvard.edu/abs/2025MNRAS.tmp...25S/abstract)_.

<!-- TABLE OF CONTENTS -->
## Table of Contents

- [Gaia Astrometric Exoplanet Orbits Detection](#gaia-astrometric-exoplanet-orbits-detection)
  - [Table of Contents](#table-of-contents)
  - [About the Project](#about-the-project)
  - [Folder Structure](#folder-structure)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Setting up the Dataset](#setting-up-the-dataset)
    - [Running the Main Script](#running-the-main-script)
    - [Notebooks for Data Analysis](#notebooks-for-data-analysis)
  - [Contact](#contact)

## About the Project

This repository contains the code and resources used to reproduce the results from the paper _Machine learning-based identification of Gaia astrometric exoplanet orbits_. The paper proposes a novel approach to identify exoplanet candidates on Gaia DR3 astrometric data. It employs semi-supervised anomaly detection combined with machine learning classifiers to pinpoint potential low-mass companions in two-body systems. The key focus is the detection of substellar companions like exoplanets and brown dwarfs.

## Folder Structure

- The module `gaia_ad` implements the anomaly detection methods used.
- `compute_candidates.py`: Main script to compute and analyze candidate orbits using different classifiers and datasets.
- `compute_shap.py`: Contains functions to compute SHAP values for feature importance analysis.
- `create_paper_figures.ipynb`: Jupyter notebook to reproduce figures for the paper.
- `load_data.py`: Functions to load and preprocess the dataset.
- `plots.py`: Functions to generate various plots, including SHAP plots and score visualizations.
- `run_config.py`: Configures and runs the machine learning models on the dataset.
- `setup.py`: Setup script for the `gaia_ad` package.

## Getting Started

This section explains how to set up the repository and reproduce the results.

### Prerequisites

We recommend using [mamba](https://mamba.readthedocs.io/en/latest/) to manage dependencies. Clone this repository and install the necessary packages by running:

```bash
git clone <TODO>
cd <TODO>
mamba env create -f environment.yml
mamba activate gaia_exorbitml
```

### Setting up the Dataset

The dataset needed to reproduce the results is included in this repository under `data/0.0.6/`.

### Reproduce results

To compute and analyse candidate orbits, run the following command:

```bash
python compute_candidates.py
```

Results will be saved in `results/`. In particular there will be the following files:

- **`shap_candidates`** - Contains plots for the SHAP feature importance analysis of the candidates.

- **`shap_exo`** - Contains plots for the SHAP feature importance analysis of the known exoplanets / brown dwarfs.

- **`top_<model>_<dataset>_label_for_better_ssc=<value>.csv`**: Contains the top candidates identified by the model. The `<model>` can be `xgb` (XGBoost) or `rf` (Random Forest), and `<dataset>` can be `nss` (full dataset) or `ssc` (subset). The `label_for_better_ssc` indicates the label used for high-confidence substellar candidates.
  
- **`no_filter_top_<model>_<dataset>_label_for_better_ssc=<value>.csv`**: Similar to the `top` files but includes all candidates without filtering, providing a broader view of model predictions.

- **`worst_truepositive_top_<model>_<dataset>_label_for_better_ssc=<value>.csv`**: Used to investigate samples that were missed by the model, focusing on the worst true positives.

- **`scores_<model>_<dataset>_label_for_better_ssc=<value>.npy`**: Contains the scores for each model configuration, indicating how well the model performed in identifying known exoplanets and candidates.

- **`clfs_<model>_<dataset>_label_for_better_ssc=<value>.pk`**: Serialized trained classifiers for each model configuration, stored using Python's pickle module.

- **`split_index_<model>_<dataset>_label_for_better_ssc=<value>.csv`**: Stores the cross-validation folds used during model training, ensuring reproducibility of the results.

Paper figures can be reproduced using the notebook `create_paper_figures.ipynb` and are saved in the `paper/` folder.

Paper figures 3 can be reproduced by running the `generate_paper_figure3.py` script and are save in the `results/figures` folder.

## Contact

For further information, please contact the authors of the paper:

- Johannes Sahlmann (Johannes.Sahlmann at esa.int)
- Pablo Gómez (Pablo.Gomez at esa.int)
