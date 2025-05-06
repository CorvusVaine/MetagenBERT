import os
import numpy as np
import torch
import random
import argparse

def subsample_one(sample_path,n_to_load,save_path):
    data=[]
    loaded=0
    tr=0
    L_files = os.listdir(sample_path)
    random.shuffle(L_files)
    while loaded < n_to_load and tr < len(L_files):
        dat= np.load(os.path.join(sample_path,L_files[tr]))
        random.shuffle(dat)
        if loaded+len(dat) > n_to_load:
            data.append(dat[:n_to_load-loaded])
            loaded+=len(dat[:n_to_load-loaded])
        else:
            data.append(dat)
            loaded+=len(dat)
        tr+=1
    data = np.concatenate(data)
    np.save(os.path.join(save_path,"sub_"+str(n_to_load)+".npy"),data)

def subsample_everywhere(data_path, n_to_load, save_path):
    os.makedirs(save_path,exist_ok=True)
    for s_path in os.listdir(data_path):
        s_path = os.path.join(data_path,s_path)
        os.makedirs(os.path.join(save_path,s_path.split('/')[-1]),exist_ok=True)
        subsample_one(str(s_path), n_to_load, os.path.join(save_path,s_path.split("/")[-1]))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample embeddings")
    parser.add_argument("data_path", type=str, help="Path to the data directory")
    parser.add_argument("n_to_load", type=int, help="Number of samples to load")
    parser.add_argument("save_path", type=str, help="Path to save the subsampled data")
    args = parser.parse_args()
    
    subsample_everywhere(args.data_path, args.n_to_load, args.save_path)