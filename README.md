# MLpipeline *-This repository is under development-*
This repository is a pipeline API to create machine learning (ML) models for Quantitative Structure-Activity Relationships (QSAR) and integtate them into a stacking model. It supports three ML algorithms (LGB: LightGBM, XGB: XGBoost, and NN: Neurak Network) and two types of input features (Mordred descriptors and Morgan fingerprint).  

This pipeline is consit of following four phases.  
1. Calculation of descriptors  
2. Optimization of hyper parameters  
3. Learning of individual ML models   
4. Creation of a tacking model  

Sample data for classification model is mutagenicity data set build by [Hansen *et al* (2009)](https://pubs.acs.org/doi/10.1021/ci900161g)., and data for regression model is [solubility data set from RDkit](https://github.com/rdkit/rdkit/tree/master/Docs/Book/data).  

# Direcroty
```
.
├── original_data
│     └── sample
│           ├──mutagenicity
│           └──solubility
├── processed_data
│     └── sample
│           ├──mutagenicity
│           └──solubility


├── figure
├── images
├── model
│     ├── checkpoint_model.pth
│     └── config.ini
├── src
│     ├── mol2graph.py
│     ├── callbacks.py
│     ├── network.py
│     ├── integrated_gradients.py
│     └── train.py
├── install_packages.sh
├── train.sh
├── visualize.sh
└── README.md
```

# Installation of packages
You can easily install required libraries by running this command in the ```root``` directory.  
```bash
bash install_libraries.sh
```

<br>

# Usage
### Phase 1
In the phase 1, you can calculate descriptors or binaryized finger prints according to SMILES string which is the linear representation of chemical structure.  
In oder to work this program, 