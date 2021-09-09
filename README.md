# MLpipeline *-This repository is under development-*
This repository is a framework to create machine learning (ML) models for Quantitative Structure-Activity Relationships (QSAR) and integtate them into a stacking model. It supports three ML algorithms (LGB: LightGBM, XGB: XGBoost, and NN: Neurak Network) and two types of input features (Mordred descriptors and Morgan fingerprint).  

This pipeline is consit of following five components.  
1. Calculation of descriptors  
2. Optimization of hyper parameters  
3. Learning of individual ML models   
4. Creation of a stacking model  
5. Feature analysis  

In this tutorial, we adopt sample data set of mutagenicity  build by [Hansen *et al.* (2009)](https://pubs.acs.org/doi/10.1021/ci900161g) to create classification models, and data set of [solubility data set from RDkit](https://github.com/rdkit/rdkit/tree/master/Docs/Book/data) to create regression model.  
# Direcroty
```bash
.
├── original_data
│     └── sample # sample data set from Hansen et al. and RDkit
│           ├──mutagenicity
│           └──solubility
├── processed_data # descriptor-calculated data sets
│     └── sample
│           ├──mutagenicity
│           └──solubility
├── results
│     ├── pjxxxx # sample project
│     │     ├──best_params
│     │     ├──trained_models
│     │     └──valid_metrics
│     ├── project_build_by_yourself # edit yourself
│     .     ├──best_params
│     .     ├──trained_models
│     .     └──valid_metric 
.     .
.     .
.     
├── src # main python codes
│     ├── descriptirs
│     └── pipeline
├── install_libraries.sh
└── README.md
```
# Installation of packages
Creating a new environment.  
```
conda create -n stkQSAR python=3.8.5 -y
conda activate stkQSAR
```
You can easily install required packages by running this command.  
```bash
bash install_libraries.sh
```
# Usage
### 1. Start new project
In oder to start new project, run following command. In this tutotial, we adopt a new project : ```pj_sample```.
```bash
python start_project.py pj_sample
```
Running this command will generate unique project directory in the root directory, which contains programs to create stacking QSAR model. After creating sample project, please move to the project directory.  
<br>

### 2. Download sample data
Before starting this tutorial, please download sample data. In the directory of ```pj_sample```, run following command.

```bash
wget -P ./original_data https://github.com/TeddyGlass/sample_data/raw/main/mutagenicity.zip
unzip ./original_data/mutagenicity.zip

wget -P ./original_data https://github.com/TeddyGlass/sample_data/raw/main/solubility.zip
unzip ./original_data/solubility.zip
```
<br>

### 3. Calculate descriptors
In the phase 1, you can calculate descriptors or binaryized finger prints according to SMILES string which is the linear representation of chemical structure. In oder to work this program, it requies **SMILES linear strings and corresponding experimental results listed as continuous or binary values, which must be stored in the CSV format with each column name**. 