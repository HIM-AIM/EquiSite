# EquiSite: Multi-Scale Equivariant Graph Learning for Robust Nucleic Acid Binding Site Prediction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[cite_start]This is the official PyTorch implementation of **EquiSite**[cite: 3].

**EquiSite** is an SE(3)-equivariant geometric graph neural network for accurate protein-DNA and protein-RNA binding site prediction. [cite_start]Unlike existing methods, EquiSite explicitly models **multi-scale protein geometry** (including atomic side-chain orientations) and **eliminates the need for computationally expensive evolutionary profiles (MSAs)**[cite: 10, 11].

[cite_start]EquiSite achieves state-of-the-art performance on both experimental structures and AlphaFold2-predicted models, and supports structure-guided molecular docking (e.g., HADDOCK3)[cite: 12, 13].

### 🌟 Graphical Abstract

![Graphical Abstract](Figures/graphical_abstract.png)
[cite_start]*(Note: Please ensure the image file is uploaded to the `Figures` folder)* [cite: 16]


---

## 🛠️ Installation

We recommend using Anaconda to manage the environment.

```bash
# 1. Create environment
conda create -n equisite python=3.9
conda activate equisite

# 2. Install PyTorch (Please adjust cuda version based on your driver)
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# 3. Install PyTorch Geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f [https://data.pyg.org/whl/torch-1.12.1+cu113.html](https://data.pyg.org/whl/torch-1.12.1+cu113.html)

# 4. Install other dependencies
pip install fair-esm h5py tqdm scipy sklearn biopython
