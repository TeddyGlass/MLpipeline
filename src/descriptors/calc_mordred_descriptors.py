from utils import smiles2descriptor

import os
import pandas as pd
import configparser
import argparse


if __name__ == '__main__':

    # argments
    parser = argparse.ArgumentParser()
    parser.add_argument('conf')
    args = parser.parse_args()

    # load config file
    conf_file = args.conf
    section = 'mordred_descriptirs'
    config = configparser.ConfigParser()
    config.read(conf_file)
    csv_path = config.get(section, 'csv_path')
    ignore_3D = config.getboolean(section, 'ignore_3D')
    col_smiles = config.get(section, 'col_smiles')
    col_property = config.get(section, 'col_property')

    df = pd.read_csv(csv_path)
    smiles = df[col_smiles].tolist()
    df_descriptors = smiles2descriptor(smiles, ignore_3D)
    df_descriptors.insert(0, 'property', df[col_property])
    df_descriptors.insert(0, 'SMILES', smiles)

    out_root = '../processed_data'
    if ignore_3D:
        file_name = csv_path.split('/')[-1].split('.csv')[0] \
            + '_mordred_ignore_3D.csv'
    else:
        file_name = csv_path.split('/')[-1].split('.csv')[0] \
            + '_mordred.csv'
    out_path = os.path.join(out_root, file_name)
    df_descriptors.to_csv(out_path, index=False)
    print(f'Result file was stored in "{out_path}".')
