path_to_csv=../original_data/sample/solubility/solubility_test.csv
out_dir=../processed_data/sample/solubility
smiles_col=SMILES
property_col=LogSOL
python ./descriptors/calc_mordred_descriptors.py $path_to_csv $out_dir $smiles_col $property_col --ignore_3D
