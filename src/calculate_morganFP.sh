path_to_csv=../original_data/sample/solubility/solubility_test.csv
out_dir=../processed_data/sample/solubility
smiles_col=SMILES
property_col=LogSOL
radius=2
nBits=1024
python ./descriptors/calc_morgan_fingerprint.py $path_to_csv $out_dir $smiles_col $property_col $radius $nBits

path_to_csv=../original_data/sample/solubility/solubility_train.csv
out_dir=../processed_data/sample/solubility
smiles_col=SMILES
property_col=LogSOL
radius=2
nBits=1024
python ./descriptors/calc_morgan_fingerprint.py $path_to_csv $out_dir $smiles_col $property_col $radius $nBits

path_to_csv=../original_data/sample/mutagenicity/mutagenicity_train.csv
out_dir=../processed_data/sample/mutagenicity
smiles_col=SMILES
property_col=Mutagenicity
radius=2
nBits=1024
python ./descriptors/calc_morgan_fingerprint.py $path_to_csv $out_dir $smiles_col $property_col $radius $nBits

path_to_csv=../original_data/sample/mutagenicity/mutagenicity_test.csv
out_dir=../processed_data/sample/mutagenicity
smiles_col=SMILES
property_col=Mutagenicity
radius=2
nBits=1024
python ./descriptors/calc_morgan_fingerprint.py $path_to_csv $out_dir $smiles_col $property_col $radius $nBits