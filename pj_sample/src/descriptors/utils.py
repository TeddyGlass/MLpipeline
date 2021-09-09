from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from mordred import descriptors, Calculator
import numpy as np


def smiles2descriptor(smiles, ignore_3D):
    mols = [Chem.AddHs(Chem.MolFromSmiles(smi)) for smi in smiles]
    if None in mols:
        return 'Error! SMILES have not been convertrd to Mol onjects properly.'
    else:
        if ignore_3D:
            print('Claculating 2D descriptors')
            calc = Calculator(descriptors, ignore_3D=True)
        else:
            print('Claculating 3D descriptors')
            calc = Calculator(descriptors)
        df = calc.pandas(mols)
        df = df.astype(str)
        masks = df.apply(
            lambda d: d.str.contains('[a-zA-Z]', na=False))
        df = df[~masks]
        df = df.astype(float)
    return df


def mol2fp(mol, radius=2, nBits=1024):
    bitInfo={}
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
         nBits=nBits,
          bitInfo=bitInfo
          )
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr, bitInfo