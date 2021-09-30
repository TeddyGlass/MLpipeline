# MLpipeline
This repository is a framework to create machine learning (ML) models for Quantitative Structure-Activity Relationships (QSAR) and integtate them into a stacking model. It supports three ML algorithms (LGB: LightGBM, XGB: XGBoost, and NN: Neurak Network) and two types of input features (Mordred descriptors and Morgan fingerprint).  

This pipeline is consit of following four components.  
1. Calculation of descriptors  
2. Optimization of hyper parameters  
3. Learning of individual ML models   
4. Creation of a stacking model  

In this tutorial, we adopt sample data of mutagenicity  build by [Hansen *et al.* (2009)](https://pubs.acs.org/doi/10.1021/ci900161g) to create classification models, and data of [solubility data set from RDkit](https://github.com/rdkit/rdkit/tree/master/Docs/Book/data) to create regression model.  
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
bash install_packages.sh
```
<br>

# Tutorials
### 1. Start new project
In order to start new project, run the following command. 
```bash
python start_project.py pj_sample
```
Running this command will generate unique project directory in the root directory, which contains programs to create stacking QSAR model. After creating directory of sample project, you will need to move to the project directory. Note that almost all parameters related to building QSAR models are centrally managed by ```settings.ini``` in the project directory.  
<br>

### 2. Download sample data
Before starting this tutorial, please download sample data. In the directory of ```pj_sample```, please run the following commands.

```bash
wget -P ./original_data https://github.com/TeddyGlass/sample_data/raw/main/mutagenicity.zip
unzip ./original_data/mutagenicity.zip　-d　./original_data/

wget -P ./original_data https://github.com/TeddyGlass/sample_data/raw/main/solubility.zip
unzip ./original_data/solubility.zip　-d　./original_data/
```
<br>

### 3. Calculate descriptors
You can calculate descriptors and binaryized fingerprints according to SMILES string which is the linear representation of chemical structure. Before running this program, it requies **SMILES strings and corresponding experimental results listed as continuous or binary values which are saved in the CSV format with each column name**. Here, we demonstrate example using sample data of solubility CSV file which have already downloaded.  
<br>

**I.**  Change the directory from ```/pj_sample``` to ```/pj_sample/src```, edit ```settings.ini```. You must write parameters used for calculation of descriptors.  

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
You can conduct searching for optimal hyper parameters with K-fold crass validation using bayesian optimization.  

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
python ./pipeline/hyperparameters_optimization.py $conf --XGB
python ./pipeline/hyperparameters_optimization.py $conf --NN
```
After the hyperparameter optimization is complete, ```.pkl``` extension file that records optimal hyper parameters will be saved in ```/pj_sample/results/best_params/```.

<br>

### 5. Training of individual ML models
To train ML models, we need to set up baseline hyperparameters.  
It is possible to train ML models with or without optimal hyper parameters. If you train without optimal hyperparameters, baseline hyperparameters are adopted. If you train with optimal hyperparameters, baseline hyperparameters will be overwritten by optimal hyperparameters.  

**I.** Edit ```/src/settings.ini``` as following.
```ini
[general]
train_data = ../processed_data/solubility_train_mordred_ignore_3D.csv

[trainer]
random_state = 1234
n_splits = 3
early_stopping_rounds = 10

[lgb_params]
learning_rate = 1e-3
n_estimators = 1000000
max_depth = -1
num_leaves = 31
subsample = 0.65
colsample_bytree = 0.65
bagging_freq = 10
min_child_weight = 10
min_child_samples = 10
min_split_gain = 0.01
n_jobs = -1
best_prams_path = ../results/best_params/LGBMRegressor_bestparams.pkl

[xgb_params]
learning_rate = 1e-3
n_estimators = 1000000
max_depth = 9
subsample = 0.65
colsample_bytree = 0.65
gamma = 1
min_child_weight = 10
n_jobs = -1
best_prams_path = ../results/best_params/XGBRegressor_bestparams.pkl

[nn_params]
standardization = True
learning_rate = 1e-3
epochs = 100000
hidden_units = 256
batch_size = 32
input_dropout = 0.1
hidden_dropout = 0.1
hidden_layers = 2
batch_norm = before_act
best_prams_path = ../results/best_params/NNRegressor_bestparams.pkl
``` 

<br>

**Parameters of general settings**  
> * **train_data** : Path from ```/pj_sample/src``` to CSV file which saves calculated descriptors and experimental results. 

<br>

**Parameters of trainer**: These parameters are related to training process.
> * **random_state** : Random seed used in the cross validation.  
> * **n_splits** : Number of folds of cross validation.  
> * **early_stopping_rounds** : The model will train until the validation score stops improving. Validation score needs to improve at least every ```early_stopping_rounds``` round(s) to continue training.  

<br>

**Parameters of lgb_params**: Almost all parameter names corresond to the documentation of lightgbm. Details of othe parameter is available in the [documentation of lightgbm](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html).
> * **best_prams_path** : Path from ```/pj_sample/src``` to the ```.pkl``` extension file that records the best parameters.  

<br>

**Parameters of xgb_params**: Almost all parameter names corresond to the documentation of xgboost. Details of othe parameter is available in the [documentation of xgboost](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn).
> * **best_prams_path** : Path from ```/pj_sample/src``` to the ```.pkl``` extension file that records the best parameters.

<br>

**Parameters of nn_params**
> * **standardization**: If ```True```, features used for the inputs will be forcibly converted to normal distributions.  
> * **learning_rate**:  Details are available in the description of ```learning_rate``` of [keras documentation](https://keras.io/ja/).  
> * **epochs**: Details are available in the description of ```epochs``` of [keras documentation](https://keras.io/ja/).  
> * **hidden_units**: Number of units (dimensions) of hidden layers. Details are available in the description of ```units``` of [keras documentation](https://keras.io/ja/).  
> * **batch_size**: Details are available in the description of ```batch_size``` of [keras documentation](https://keras.io/ja/).  
> * **input_dropout**: Drop out rate [0.0-1.0] of the input layer.  
> * **hidden_dropout**: Drop out rate [0.0-1.0] of the hidden layer.
> * **hidden_layers**: Number of hidden layers.  
> * **before_act**: If ```before_act```, batch normalization is conducted before activation by activation function (ReLU).  If you not adopt  batch normalization, please set it to ```None```. 
> * **best_prams_path** : Path from ```/pj_sample/src``` to the ```.pkl``` extension file that records the best parameters.

<br>

**II.** Run the python code (```train_models.py```). When, you must set the action related to the model type selection from among ```--LGB``` or ```--XGB``` or ```--NN```. Moreover, if you train models with optimal hyperparameters, please set an another action: ```--use_best_params```.  
```bash
conf=settings.ini
python ./pipeline/train_models.py $conf --LGB --use_best_params
python ./pipeline/train_models.py $conf --XGB --use_best_params
python ./pipeline/train_models.py $conf --NN --use_best_params
```
After the training is complete, each K kind of ```.pkl``` extension file that records trained model which was build in the K-fold CV will be saved in ```/pj_sample/results/trained_models```. In the event of keras model, ```.json``` file (one kind) which records the information of netowork architecture and ```.h5``` file (K kinds) which records the best weight will be saved in ```/pj_sample/results/trained_models```. Moreover, after training, monioted training processes (learning curves) and predictive performances of the validation sets will be saved in ```/pj_sample/results/trained_models```.
