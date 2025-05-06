import numpy as np
import pandas as pd

import time
import math
import json
import torch
from torch import nn
import torch.nn.functional as F
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class DeepSets(nn.Module):
    def __init__(self, phi, rho, mil_layer, device):
        super(DeepSets, self).__init__()
        self.phi = phi
        self.rho = rho
        self.mil_layer = mil_layer
        self.device = device
        if mil_layer == "attention":
            self.attention = nn.Sequential(
                nn.Linear(self.phi.last_hidden_size, self.phi.last_hidden_size // 3),
                nn.Tanh(),
                nn.Linear(self.phi.last_hidden_size // 3, 1),
            ).to(self.device)
        self.criterion = (
            nn.BCEWithLogitsLoss()
            if self.rho.output_size <= 2
            else nn.CrossEntropyLoss()
        )

    def forward(self, x):
        # compute the representation for each data point
        x = self.phi.forward(x)
        A = None
        # sum up the representations
        if self.mil_layer == "sum":
            x = torch.sum(x, dim=1, keepdim=True)
        if self.mil_layer == "max":
            x = torch.max(x, dim=1, keepdim=True)[0]
        if self.mil_layer == "mean":
            x = torch.mean(x, dim=1, keepdim=True)
        if self.mil_layer == "attention":
            A = self.attention(x)
            A = F.softmax(A, dim=1)
            x = torch.bmm(torch.transpose(A, 2, 1), x)
        # compute the output
        out = self.rho.forward(x)
        return out, A

class Phi(nn.Module):
    def __init__(self, embed_size, hidden_init=200, n_layer=1, dropout=0.2):
        super(Phi, self).__init__()
        layer_size = [embed_size, hidden_init]
        n_layer -= 1
        for i in range(n_layer):
            hidden_init = hidden_init // 2
            layer_size.append(hidden_init)
        self.layers = []
        for i in range(len(layer_size) - 1):
            self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Dropout(dropout))
        self.nets = nn.Sequential(*self.layers[:-1])  # Remove the last drop out
        self.last_hidden_size = layer_size[-1]

    def forward(self, x):
        return self.nets(x)


class Rho(nn.Module):
    def __init__(
        self, phi_hidden_size, hidden_init=100, n_layer=1, dropout=0.2, output_size=1
    ):
        super(Rho, self).__init__()
        self.output_size = output_size
        layer_size = [phi_hidden_size, hidden_init]
        n_layer -= 1
        for i in range(n_layer):
            hidden_init = hidden_init // 2
            layer_size.append(hidden_init)
        self.layers = []
        for i in range(len(layer_size) - 1):
            self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            self.layers.append(nn.LeakyReLU()),
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(layer_size[-1], output_size))
        self.nets = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.nets(x)


# Load the data

# If centroids are used

#n_clust=16384
#samples_dir="/data/db/deepintegromics/passoli_datasets/reads/cirrhosis/clusters_p2_ordered/mean/"+str(n_clust)
#samples_dir="/data/db/deepintegromics/passoli_datasets/reads/cirrhosis/DNABERT_2/mean/"+str(n_clust)
#samples_dir="/data/db/deepintegromics/passoli_datasets/reads/t2d/clusters_p2_ordered/mean/"+str(n_clust)
#samples_dir="/data/db/deepintegromics/passoli_datasets/reads/t2d/DNABERT_S/clusters_p2_ordered/"+str(n_clust)+"/dataset/"

#samples_dir="/data/db/deepintegromics/passoli_datasets/reads/t2d/clusters_global_ordered/mean/"+str(n_clust)+"/Fold_0/train"

# If subsamples are used    
sub = 512
samples_dir="/data/projects/deepintegromics/scratch/subsamples/sub_dna2_t2d/"
#samples_dir="/data/db/deepintegromics/passoli_datasets/reads/cirrhosis/subsamples/mean/"+str(sub)

#If abundance is used on 90% of the data
#n_clust=16384
#samples_dir="/data/db/deepintegromics/passoli_datasets/reads/cirrhosis/clusters_global/mean/"+str(n_clust)+"/Fold_0/train"

# If aggregations are used
#samples_dir="/data/db/deepintegromics/passoli_datasets/reads/cirrhosis/aggregations/mean/all/train/"

dirr = samples_dir
data = []
labels = []

samples = os.listdir(dirr)
samples.sort()

## LABELS for CIRRHOSIS
"""
for s in samples :
    print(s)
    s = os.path.join(dirr, s)
    if s.split("/")[-1] in ["LV12", "LD33", "LD45", "LD16", "LV9","Fold_0"]:
        print("Skipping", s.split("/")[-1])
        continue
    data.append(np.load(os.path.join(s,"centroids.npy")))
    #data.append(np.load(os.path.join(s,"sub_"+str(sub)+".npy")))
    #data.append(np.load(os.path.join(s,"mean_embedding.npy")))
    #data.append(np.load(os.path.join(s,"abundance.npy")))
    if "L" in s.split("/")[-1]:
        labels.append(1)
    if "H" in s.split("/")[-1]:
        labels.append(0)

        """
## LABELS for T2D

sam_to_lab = json.load(open("/data/db/deepintegromics/passoli_datasets/reads/t2d/sample_to_label.json", "r"))

for s in samples :
    print(s)
    sam = os.path.join(dirr, s)
    #data.append(np.load(os.path.join(sam,"centroids.npy")))
    data.append(np.load(os.path.join(sam,"sub_"+str(sub)+".npy")))
    #data.append(np.load(os.path.join(s,"mean_embedding.npy")))
    #data.append(np.load(os.path.join(sam,"abundance.npy")))
    labels.append(sam_to_lab[s])
   

labels = [1 if label == "Y" else 0 for label in labels]


data = np.array(data)
labels = np.array(labels)
# Normalize the data
scaler = StandardScaler()

#Scaling across all samples

## When using clustering
data_2d = data.reshape(data.shape[0]*data.shape[1],data.shape[2])
data_2d = scaler.fit_transform(data_2d)
data = data_2d.reshape(data.shape[0],data.shape[1],data.shape[2])

## When using simple aggregations or abundance
#data = scaler.fit_transform(data)
#data = data.reshape(data.shape[0],1,data.shape[1])

# Shuffle the data and labels
print(data.shape)
shuffle_indices = np.random.permutation(len(data))
data = data[shuffle_indices]
labels = labels[shuffle_indices]


# Train the model
n_epochs = 250
batch_size = 32
splits = 10
best_auc_list = []
best_acc_list = []
best_conf_matrix_list = []

# Split data and labels into train and test sets
kf = KFold(n_splits=splits)  # Change the number of splits as needed
for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
    phi = Phi(data.shape[-1], hidden_init=512, n_layer=1, dropout=0.4)
    rho = Rho(phi.last_hidden_size, hidden_init=256, n_layer=1, dropout=0.2, output_size=1)
    model = DeepSets(phi, rho, "mean", "cuda")
    model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    best_auc = 0
    best_acc = 0
    print(f"Starting Fold {fold + 1}/{kf.n_splits}")
    
    # Split data
    train_data, test_data = data[train_idx], data[val_idx]
    train_labels, test_labels = labels[train_idx], labels[val_idx]
   
    #train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.1, random_state=42)
    print("Train data shape:", train_data.shape)
    print("Train labels shape:", train_labels.shape)
    print("Test data shape:", test_data.shape)
    print("Test labels shape:", test_labels.shape)

    # Convert data and labels to PyTorch tensors
    train_data = torch.tensor(train_data, dtype=torch.float32).to("cuda")
    train_labels = torch.tensor(train_labels, dtype=torch.float32).to("cuda")
    test_data = torch.tensor(test_data, dtype=torch.float32).to("cuda")
    test_labels = torch.tensor(test_labels, dtype=torch.float32).to("cuda")
    n_batches = math.ceil(len(train_data) / batch_size)

    for epoch in range(n_epochs):
        model.train()
        start_time = time.time()
        for i in range(n_batches):
            optimizer.zero_grad()
            batch_data = train_data[i * batch_size : (i + 1) * batch_size]
            batch_labels = train_labels[i * batch_size : (i + 1) * batch_size]
            output, _ = model.forward(batch_data)
            loss = model.criterion(output.squeeze(), batch_labels)
            loss.backward()
            optimizer.step()
        if (epoch+1)%10==0:
            model.eval()
            with torch.no_grad():
                output, _ = model.forward(test_data)
                test_labels = torch.tensor(test_labels, dtype=torch.float32).to("cuda")
                test_loss = model.criterion(output.squeeze(), test_labels)
                test_pred = torch.sigmoid(output.squeeze()).cpu().numpy()
                test_labels = test_labels.cpu().numpy()
                print(test_labels)
                print(test_pred)
                test_auc = roc_auc_score(test_labels, test_pred)
                test_acc = accuracy_score(test_labels, test_pred > 0.5)
                test_f1 = f1_score(test_labels, test_pred > 0.5)
                test_precision = precision_score(test_labels, test_pred > 0.5)
                test_recall = recall_score(test_labels, test_pred > 0.5)
                conf_matrix = confusion_matrix(test_labels, test_pred > 0.5)
            print(
                f"Epoch {epoch + 1}/{n_epochs} ({time.time() - start_time:.1f}s): "
                f"Train loss: {loss:.4f}, "
                f"Test loss: {test_loss:.4f}, "
                f"Test AUC: {test_auc:.4f}, "
                f"Test accuracy: {test_acc:.4f}, "
                f"Test F1: {test_f1:.4f}, "
                f"Test precision: {test_precision:.4f}, "
                f"Test recall: {test_recall:.4f}",
                f"Confusion matrix: {conf_matrix}",
            )
            if test_acc > best_acc:
                best_acc = test_acc
                best_conf_matrix = conf_matrix
            if test_auc > best_auc:
                best_auc = test_auc
    best_acc_list.append(best_acc)
    best_auc_list.append(best_auc)
    best_conf_matrix_list.append(best_conf_matrix)
err_list = []
for accuracy in best_acc_list:
    fold_err = accuracy
    err_list.append((fold_err*(1-fold_err))/10)
std_err_acc = math.sqrt(sum(err_list)/len(data))
err_list = []
for auc in best_auc_list:
    fold_err = auc
    err_list.append((fold_err*(1-fold_err))/10)
std_err_auc = math.sqrt(sum(err_list)/len(data))





print(f"Best accuracy for each fold: {best_acc_list}")
print(f"Best AUC for each fold: {best_auc_list}")
print(f"Best confusion matrix for each fold: {best_conf_matrix_list}")
print(f"Mean accuracy: {np.mean(best_acc_list)}")
print(f"Mean AUC: {np.mean(best_auc_list)}")
print(f"Standard deviation of accuracy: {np.std(best_acc_list)}")
print(f"Standard deviation of AUC: {np.std(best_auc_list)}")
print(f"Standard error of accuracy: {std_err_acc}")
print(f"Standard error of AUC: {std_err_auc}")
                