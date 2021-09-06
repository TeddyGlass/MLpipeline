from rdkit import Chem
from utils import mol2fp
import pandas as pd
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_csv')
    parser.add_argument('out_dir')
    parser.add_argument('smiles_col')
    parser.add_argument('property_col')
    parser.add_argument('radius', type=int)
    parser.add_argument('nBits', type=int)
    args = parser.parse_args()

    df = pd.read_csv(args.path_to_csv)
    smiles = df[args.smiles_col].tolist()
    mols = [Chem.AddHs(Chem.MolFromSmiles(smi)) for smi in smiles]
    if None in mols:
        print('Error! SMILES have not been convertrd to Mol onjects properly.')
    else:
        print(f'Calculating Morgan FP. (radius={args.radius}, nBits={args.nBits})')
        morganFP_arrs = np.array(
            [mol2fp(mol, args.radius, args.nBits)[0] for mol in mols],
             dtype=int
             )
        print('Completed.')
    df_morganFP = pd.DataFrame(morganFP_arrs)
    df_morganFP.insert(0, 'property', df[args.property_col])
    df_morganFP.insert(0, 'SMILES', smiles)

    out_path = f'{args.out_dir}/' + args.path_to_csv.split('/')[-1].split('.csv')[0] \
        + '_morganFP' + f'_radius{args.radius}_nBits{args.nBits}.csv'
    df_morganFP.to_csv(out_path, index=False)
    print(f'Result file was stored in "{out_path}".')
