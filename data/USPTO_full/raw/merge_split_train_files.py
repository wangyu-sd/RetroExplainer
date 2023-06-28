# train.csv is too large to upload to Github. We split it into serval smaller
# files and then merge these files into the original train.csv file.
# 
# To merge train files, in the data directory:
# python merge_split_train_files.pt --merge
#

import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--merge", action='store_true', help="Merge small files into a large file")
args = parser.parse_args()



num_files = 4
train_data = 'train.csv'
train_data_split = ['train-{}.csv'.format(i+1) for i in range(num_files)]

if args.merge:
    rxn_smiles = []
    for tdata in train_data_split:
        rxn = pd.read_csv(tdata)['rxn_smiles'].tolist()
        rxn_smiles.extend(rxn)
        os.remove(tdata)
    df = pd.DataFrame({'rxn_smiles': rxn_smiles})
    df.to_csv(train_data, index=False, sep='\t', encoding='utf-8')

else:
    rxn = pd.read_csv(train_data)['rxn_smiles'].tolist()
    total_size = len(rxn)
    chunk_size = total_size // num_files + 1
    for i in range(1, num_files + 1):
        df = pd.DataFrame({'rxn_smiles': rxn[chunk_size*(i-1):chunk_size*i]})
        df.to_csv(train_data_split[i-1], index=False, sep='\t', encoding='utf-8')

