# 20210907
config=pj20210907_settings.ini
python ./pipeline/train_models.py $config --LGB --use_best_params
python ./pipeline/train_models.py $config --XGB --use_best_params
python ./pipeline/train_models.py $config --NN --use_best_params
