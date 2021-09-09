import pandas as pd
import argparse
from utils import smiles2descriptor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_csv')
    parser.add_argument('out_dir')
    parser.add_argument('smiles_col')
    parser.add_argument('property_col')
    parser.add_argument('--ignore_3D', action='store_true',
    help='If you calculate 3D descriptors, please set to --ignore_3D.')
    args = parser.parse_args()

    df = pd.read_csv(args.path_to_csv)
    smiles = df[args.smiles_col].tolist()
    df_descriptors = smiles2descriptor(smiles, args.ignore_3D)
    df_descriptors.insert(0, 'property', df[args.property_col])
    df_descriptors.insert(0, 'SMILES', smiles)

    out_path = f'{args.out_dir}/' + args.path_to_csv.split('/')[-1].split('.csv')[0] \
        + '_mordred' + f'_ignore3D({args.ignore_3D}).csv'
    df_descriptors.to_csv(out_path, index=False)