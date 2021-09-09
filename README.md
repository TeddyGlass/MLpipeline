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
In oder to start new project, run following command. 
```bash
python start_project.py pj_sample
```
Running this command will generate unique project directory in the root directory, which contains programs to create stacking QSAR model. After creating sample project, please move to the project directory.  
<br>

### 2. Download sample data
Before starting this tutorial, please download sample data. In the directory of ```pj_sample```, run following command.

```bash
wget -P ./original_data https://github.com/TeddyGlass/sample_data/raw/main/mutagenicity.zip
unzip ./original_data/mutagenicity.zip　-d　./original_data/

wget -P ./original_data https://github.com/TeddyGlass/sample_data/raw/main/solubility.zip
unzip ./original_data/solubility.zip　-d　./original_data/
```
<br>

### 3. Calculate descriptors
You can calculate descriptors or binaryized fingerprints according to SMILES string which is the linear representation of chemical structure. In oder to work this program, it requies **SMILES strings and corresponding experimental results listed as continuous or binary values, which must be saved in the CSV format with each column name**. Here, we demonstrate example using sample data set of solubility.  
<br>

**I.**  Change the directory from ```pj_sample``` to ```pj_sample/src```, edit ```ettings.ini``` in the project directory. You must write profiles of parameters used for descriptor calculation.  

```ini
[mordred_descriptirs]
csv_path = ../original_data/solubility/solubility_train.csv
col_smiles = SMILES
col_property = LogSOL
ignore_3D = True

[morgan_fp]
csv_path = ../original_data/solubility/solubility_train.csv
col_smiles = SMILES
col_property = LogSOL
radius = 2
nBits = 1024
```
**Parameters of Mordred descriptirs**
> * **csv_path** : Path from ```/pj_sample/src``` to CSV file which save SMILES strings and experimental results.  
> * **col_smiles** : Column name corresponding to the SMILES string.  
> * **col_property** : Column name corresponding to the experimental results.  
> * **ignore_3D** : If you do not calculate 3D descriptors, set it to ```True```.

**Parameters of Morgan fingerprint**
> * **csv_path** : Path from ```/pj_sample/src``` to CSV file which saves SMILES strings and experimental results.  
> * **col_smiles** : Column name corresponding to the SMILES string.  
> * **col_property** : Column name corresponding to the experimental results.  
> * **radius** : Radius of the path to get the fingerprint.
> * **nBots** : Number of dimensions in which fingerprints of substructures are encoded.

<br>

**II.** Run the python code (```calc_mordred_descriptors.py```). 
```bash
conf=settings.ini
python ./descriptors/calc_mordred_descriptors.py $conf
```
<br>

### 4. Hyperparameter optimization
You can conduct searching for optimal hyper parameters with K-fold crass validation.  

**I.** Edit ```/src/settings.ini``` in the project directory as following. Before runing the hyper parameter optimization, you must write parameters used.  
```ini
[general]
train_data = ../processed_data/solubility_train_mordred_ignore_3D.csv

[optimizer]
random_state = 4321
n_splits = 3
n_trials = 10
n_jobs = -1
early_stopping_rounds = 10
```
**Parameters of general settings**  
> * **train_data** : Path from ```/pj_sample/src``` to CSV file which saves calculated descriptors and experimental results.  

**Parameters of hyper parameter optimization**
 > * **random_state** : Random seed used in the cross validation of hyper parameter search.  
> * **n_splits** : Number of folds of cross validation.  
> * **n_jobs** : Number of threds for hyperparameter search with optuna.
> * **early_stopping_rounds** : The model will train until the validation score stops improving. Validation score needs to improve at least every ```early_stopping_rounds``` round(s) to continue training.  

<br>

**II.** Run the python code (```hyperparameters_optimization.py```). When, you must set the action related to the model type selection from among ```--LGB``` or ```--XGB``` or ```--NN```.
```bash
conf=settings.ini
python ./pipeline/hyperparameters_optimization.py $conf --LGB
```
<br>
