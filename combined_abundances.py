from sklearn.linear_model import LogisticRegressionCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
import torch
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
from pprint import pprint
import math

# Pasolli abundance from Giu or Predomics
"""
#Giu
abundance_dir = "/data/db/deepintegromics/passoli_datasets/reads/cirrhosis/Pasolli_abundances/giu/"
d1 = pd.read_csv(abundance_dir+"cir_train-2.csv", sep=",", index_col=0)
d2 = pd.read_csv(abundance_dir+"cir_test-2.csv", sep=",", index_col=0)

#Predomics
abundance_dir = "/data/db/deepintegromics/passoli_datasets/reads/cirrhosis/Pasolli_abundances/predomics/"
d1 = pd.read_csv(abundance_dir+"cir_train_predomics.csv", sep=",", index_col=0)
d2 = pd.read_csv(abundance_dir+"cir_test_predomics.csv", sep=",", index_col=0)

#Shared
# Rename columns by their last character
d1.columns = ["".join(col.split(".")[-2:]) for col in d1.columns]
d2.columns = ["".join(col.split(".")[-2:]) for col in d2.columns]

if "labels" in d1.index:
    d1 = d1.drop(index="labels")
if "labels" in d2.index:
    d2 = d2.drop(index="labels")


# Sort the lines by alphabetical order
d1 = d1.sort_index(axis=1)
d2 = d2.sort_index(axis=1)

# Merge train and test column-wise
merged_data = pd.concat([d1, d2], axis=1)

# Pasolli abundance from github

abundance_dir = "/data/db/deepintegromics/passoli_datasets/reads/cirrhosis/Pasolli_abundances/git/"

merged_data=pd.read_csv(abundance_dir+"abundance_cirrhosis.csv", sep="\t", index_col=0,header=1)
# Filter rows that start with 'k' or have 'Sample ID' in them
merged_data = merged_data[merged_data.index.str.startswith('k__') | merged_data.index.str.contains('sampleID')]

# Remove the "-" characters in column names
merged_data.columns = [col.replace("-", "") for col in merged_data.columns]
"""

# T2D abundance from github

abundance_dir = "/data/db/deepintegromics/passoli_datasets/reads/t2d/Pasolli_abundance/"
merged_data=pd.read_csv(abundance_dir+"abundance_t2d_long-t2d_short.txt", sep="\t", index_col=0,header=1)
merged_data = merged_data[merged_data.index.str.startswith('k__') | merged_data.index.str.contains('sampleID')]
merged_data.columns = [col.replace("-", "") for col in merged_data.columns]


# Delete columns LD12, LD33, LD45, LV16, LV9
#merged_data = merged_data.drop(columns=['LD12', 'LD33', 'LD45', 'LV16', 'LV9','LD16','LV12'])
abundance = merged_data.to_numpy()

abundance = abundance.T


# Our Abundance

n_clust=16384
random_state = 42

#samples_dir="/data/db/deepintegromics/passoli_datasets/reads/t2d/clusters_global_ordered/mean/"+str(n_clust)+"/Fold_0/all"
#samples_dir="/data/db/deepintegromics/passoli_datasets/reads/t2d/clusters_global_ordered/mean/"+str(n_clust)+"/Fold_0/all"
#samples_dir="/data/db/deepintegromics/passoli_datasets/reads/cirrhosis/DNABERT_S/clusters_global_ordered/mean/"+str(n_clust)+"/Fold_0/all"
samples_dir="/data/db/deepintegromics/passoli_datasets/reads/t2d/clusters_global_ordered/mean/"+str(n_clust)+"/Fold_0/all/"


# If aggregations are used
#samples_dir="/data/db/deepintegromics/passoli_datasets/aggregations/t2d/dnaS/aggregations/dataset/"

dirr = samples_dir
data = []
labels = []

samples = os.listdir(dirr)
samples.sort()
"""
#Cirrhosis
#for s in samples :
for s in list(merged_data.columns):
    #print(s)
    s = os.path.join(dirr, s)
    #data.append(np.load(os.path.join(s,"centroids.npy")))
    #data.append(np.load(os.path.join(s,"sub_"+str(sub)+".npy")))
    #data.append(np.load(os.path.join(s,"mean_embedding.npy")))
    data.append(np.load(os.path.join(s,"abundance_10.npy")))
    #data.append(np.load(s+".npy"))
    if "L" in s.split("/")[-1]:
        labels.append(1)
    if "H" in s.split("/")[-1]:
        labels.append(0)
"""

#T2D

sam_to_lab = json.load(open("/data/db/deepintegromics/passoli_datasets/reads/t2d/sample_to_label.json", "r"))
for s in samples :
    s=s.replace(".npy","")
    if s == "T2D-108":
        continue
#    print(s)
    sam = os.path.join(samples_dir, s)
    data.append(np.load(os.path.join(sam,"abundance.npy")))
    #data.append(np.load(sam+".npy"))
    labels.append(sam_to_lab[s])
labels = [1 if label == "Y" else 0 for label in labels]


for d in data : 
    if sum(d)<0.99:
        print(sum(d))
## Correcting abundance, don't keep this part when you compute it better
data = np.array(data)
labels = np.array(labels)

# Normalize the data
scaler_abu = StandardScaler()
scaler_clust = StandardScaler()

#Scaling across all samples

## When using simple aggregations or abundance
abu=scaler_abu.fit_transform(abundance)
data = scaler_clust.fit_transform(data)
#data = data.reshape(data.shape[0],1,data.shape[1])

#data = np.concatenate((data,abu),axis=1)

## Using only abundance
#data = abu

print(data.shape)


# Shuffle the data and labels
shuffle_indices = np.random.permutation(len(data))
abundance = abundance[shuffle_indices]
data = data[shuffle_indices]
labels = labels[shuffle_indices]

# Store results
accuracy_list = []
auc_list = []

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

# Logistic Regression with LASSO and built-in cross-validation
model = LogisticRegressionCV(
    penalty='l1', 
    solver='liblinear', 
    cv=cv,               # 10-fold cross-validation
    scoring='accuracy', # Use accuracy for model selection
    random_state=random_state,
)

# Fit the model
model.fit(data, labels)
err_list=[]
best_list=[]
#print("Fold-specific accuracies:")

scores = list(model.scores_.values())[0]
mean_scores = np.mean(scores, axis=0)
std_scores = np.std(scores, axis=0)
maxi = max(np.mean(scores,axis =0))
idx = np.argmax(np.mean(scores,axis =0))
for fold_idx, score in enumerate(model.scores_[1], start=1):  # '1' refers to the positive class
    best_accuracy = score.max()  # Max accuracy for the fold
    best_list.append(best_accuracy)
#    print(f"Fold {fold_idx}: Best Accuracy = {best_accuracy:.4f}")
    fold_err = score[idx]
    err_list.append((fold_err*(1-fold_err))/10)
std_err = math.sqrt(sum(err_list)/len(data))

print("Mean Model Accuracy:", maxi)
print("Standard Deviation of Mean Model Accuracy:", std_scores[idx])
print("Standard Error of Mean Model Accuracy:", std_err)
print("Mean of Best :",np.mean(best_list))
print("Standard Deviation of Best :",np.std(best_list))

# Logistic Regression with LASSO and built-in cross-validation
model = LogisticRegressionCV(
    penalty='l1', 
    solver='liblinear', 
    cv=cv,               # 10-fold cross-validation
    scoring='roc_auc', # Use accuracy for model selection
    random_state=random_state,
)

# Fit the model
model.fit(data, labels)

best_list=[]
err_list=[]
#print("Fold-specific auc:")

print(model)
scores = list(model.scores_.values())[0]
mean_scores = np.mean(scores, axis=0)
std_scores = np.std(scores, axis=0)
maxi = max(np.mean(scores,axis =0))
idx = np.argmax(np.mean(scores,axis =0))
std_err = math.sqrt(sum(err_list)/len(data))

for fold_idx, score in enumerate(model.scores_[1], start=1):  # '1' refers to the positive class
    best_auc = score.max()  # Max accuracy for the fold
    best_list.append(best_auc)
    #print(f"Fold {fold_idx}: Best AUC = {best_auc:.4f}")
    fold_err = score[idx]
    err_list.append((fold_err*(1-fold_err))/10)
std_err = math.sqrt(sum(err_list)/len(data))

print("Mean Model AUC:", maxi)
print("Standard Deviation of Mean Model AUC:", std_scores[idx])
print("Standard Error of Mean Model AUC:", std_err)

print("Mean of Best :",np.mean(best_list))
print("Standard Deviation of Best :",np.std(best_list))


