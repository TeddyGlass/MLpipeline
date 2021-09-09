from utils import mol2fp

import os
import pandas as pd
import numpy as np
from rdkit import Chem
import configparser
import argparse



if __name__ == '__main__':

    # argments
    parser = argparse.ArgumentParser()
    parser.add_argument('conf')
    args = parser.parse_args()

    # load config file
    conf_file = args.conf
    section = 'morgan_fp'
    config = configparser.ConfigParser()
    config.read(conf_file)
    csv_path = config.get(section, 'csv_path')
    radius = int(config.get(section, 'radius'))
    nBits = int(config.get(section, 'nBits'))
    col_smiles = config.get(section, 'col_smiles')
    col_property = config.get(section, 'col_property')

    df = pd.read_csv(csv_path)
    smiles = df[col_smiles].tolist()
    mols = [Chem.AddHs(Chem.MolFromSmiles(smi)) for smi in smiles]
    if None in mols:
        print('Error! SMILES have not been convertrd to Mol onjects properly.')
    else:
        print(f'Calculating Morgan FP. (radius={radius}, nBits={nBits})')
        morganFP_arrs = np.array(
            [mol2fp(mol, radius, nBits)[0] for mol in mols],
            dtype=int
            )
        print('Completed.')
    df_morganFP = pd.DataFrame(morganFP_arrs)
    df_morganFP.insert(0, 'property', df[col_property])
    df_morganFP.insert(0, 'SMILES', smiles)

    out_root = '../processed_data'
    file_name = csv_path.split('/')[-1].split('.csv')[0] \
        + f'_morganfp_radius{radius}_nBits{nBits}.csv'
    out_path = os.path.join(out_root, file_name)
    df_morganFP.to_csv(out_path, index=False)
    print(f'Result file was stored in "{out_path}".')
