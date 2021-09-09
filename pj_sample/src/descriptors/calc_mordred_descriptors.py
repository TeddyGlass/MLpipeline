from utils import smiles2descriptor

import pandas as pd
import configparser


if __name__ == '__main__':

    # load config file
    conf_file= 'settings.ini'
    section = 'mordred_descriptirs'
    config = configparser.ConfigParser()
    config.read(conf_file)
    csv_path = config.get(section, 'csv_path')
    ignore_3D = config.getboolean(section, 'ignore_3D')

    df = pd.read_csv(path_to_csv)
    smiles = df[args.smiles_col].tolist()
    df_descriptors = smiles2descriptor(smiles, args.ignore_3D)
    df_descriptors.insert(0, 'property', df[args.property_col])
    df_descriptors.insert(0, 'SMILES', smiles)

    out_path = f'{args.out_dir}/' + args.path_to_csv.split('/')[-1].split('.csv')[0] \
        + '_mordred' + f'_ignore3D({args.ignore_3D}).csv'
    df_descriptors.to_csv(out_path, index=False)