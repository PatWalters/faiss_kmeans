#!/usr/bin/env python

import sys
import numpy as np
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit import Chem, DataStructs
import numpy as np
import time
from tqdm import tqdm
import h5py
from functools import wraps
from time import time

# https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
# https://codereview.stackexchange.com/questions/169870/decorator-to-measure-execution-time-of-a-function
def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print(f.__name__,f"Elapsed time: {end-start:.2f} sec")
        return result
    return wrapper

@timing
def make_np_array(lst):
    return np.array(lst,dtype=np.float32)

@timing
def save_data(fp_array,smiles_list,name_list,outfile_name):
    h5f = h5py.File(outfile_name, 'w')
    dt = h5py.special_dtype(vlen=bytes)
    h5f.create_dataset('fp_list', data=fp_array)
    h5f.create_dataset('smiles_list', (len(smiles_list),1),dt, smiles_list)
    h5f.create_dataset('name_list', (len(name_list),1),dt, name_list)
    h5f.close()

@timing
def generate_fingerprints(infile_name):
    ifs = open(infile_name)
    fp_list = []
    smiles_list = []
    name_list = []
    for line in tqdm(ifs):
        toks = line.strip().split(" ",1)
        if len(toks) >= 2:
            smiles, name = toks
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = rdmd.GetMorganFingerprintAsBitVect(mol, 2,256)
                arr = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                fp_list.append(arr)
                smiles_list.append(smiles.encode("ascii", "ignore"))
                name_list.append(name.encode("ascii", "ignore"))
    return fp_list,smiles_list,name_list

@timing
def main():
    fp_list, smiles_list, name_list = generate_fingerprints(sys.argv[1])
    outfile_name = sys.argv[2]
    fp_array = make_np_array(fp_list)
    save_data(fp_array,smiles_list,name_list,outfile_name)

main()




