conf=settings.ini
# # calculagte descriptors
# python ./descriptors/calc_mordred_descriptors.py $conf
# python ./descriptors/calc_morgan_fingerprint.py $conf

# # optimization
# python ./pipeline/hyperparameters_optimization.py $conf --LGB
# python ./pipeline/hyperparameters_optimization.py $conf --XGB
# python ./pipeline/hyperparameters_optimization.py $conf --NN

# # training
# python ./pipeline/train_models.py $conf --LGB --use_best_params
# python ./pipeline/train_models.py $conf --XGB --use_best_params
# python ./pipeline/train_models.py $conf --NN --use_best_params

# stacking
python ./pipeline/train_stacking.py $conf