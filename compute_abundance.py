import os
import torch
import numpy as np
import json
import multiprocessing

def get_abundance(sample,number):
    abundance = {}
    for i in range(number):
        abundance[i] = 0
    for assign in os.listdir(sample):
        print(assign)
        if "assign" in assign:
            assign = os.path.join(sample, assign)
            assigned = np.load(assign)
            for read in assigned:
                ele = read[0]
                if ele in abundance:
                    abundance[ele] += 1
    total_assigned = sum(abundance.values())
    abundance = {ele: count / total_assigned for ele, count in abundance.items()}
    print(abundance)
    print(len(abundance))
    abundance_list = [abundance[i] for i in range(number)]
    print(sum(abundance_list))
    print(len(abundance_list))
    return abundance_list

def get_abundance(sample,number):
    print(sample)
    abundance = np.zeros((number))
    for assign in os.listdir(sample):
        if "assign" in assign:
            assign = os.path.join(sample, assign)
            assigned = np.load(assign)
            assigned = assigned[:len(assigned)//10]
            for read in assigned:
                ele = read[0]
                if ele < number:
                    abundance[ele] += 1
    total_assigned = np.sum(abundance)
    abundance = abundance / total_assigned
    np.save(sample + "/abundance_10.npy", abundance)
    print(sample,"saved")
    return abundance

"""def all_samples(samples_dir,number):
    samples = os.listdir(samples_dir)
    samples.sort()
    for sample in samples:
        print(sample)
        sample = os.path.join(samples_dir, sample)
        abundance = get_abundance(sample,number)
        np.save(sample + "/abundance.npy", abundance)
        print(sample,"saved")
"""
def all_samples_parallel(samples_dir,number):
    samples = os.listdir(samples_dir)
    samples.sort()
    with multiprocessing.Pool(processes=32) as pool:
        for sample in samples:
            sample = os.path.join(samples_dir, sample)
            pool.apply_async(get_abundance, args=(sample,number))
        pool.close()
        pool.join()

def all_numbers(numbers_dir):
    numbers = os.listdir(numbers_dir)
    numbers.sort()
    for number in numbers:
        print(number)
        #if "4096" in number:
        #    continue
        samples_dir = os.path.join(os.path.join(numbers_dir, number),"Fold_0/all")
        #all_samples_parallel(samples_dir,int(number))
        #samples_dir = os.path.join(os.path.join(numbers_dir, number),"Fold_0/test")
        all_samples_parallel(samples_dir,int(number))

#print("cirrhosis DNABERT_2")
#all_numbers("/data/db/deepintegromics/passoli_datasets/reads/cirrhosis/DNABERT_2/clusters_global_ordered/mean/")
#print("cirrhosis DNABERT_S")
#all_numbers("/data/db/deepintegromics/passoli_datasets/reads/cirrhosis/DNABERT_S/clusters_global_ordered/mean/")
#print("t2d DNABERT_2")
all_numbers("/data/db/deepintegromics/passoli_datasets/reads/t2d/clusters_global_ordered/mean/")
#print("t2d DNABERT_S")
#all_numbers("/data/db/deepintegromics/passoli_datasets/reads/t2d/DNABERT_S/clusters_global_ordered/mean/")
#all_samples("/data/db/deepintegromics/passoli_datasets/reads/cirrhosis/clusters_global_ordered/mean/4096/Fold_0/all")