import os
import torch
import numpy as np
import faiss
import argparse
import time
import json
import random
import glob

def load_data_from_one(s_path, use_perc=False, file_of_lens=None, raw=10000, perc = 0.01):
    L_files = os.listdir(s_path)
    random.shuffle(L_files)
    data = []
    use_perc = False
    if use_perc:
        dict_lens = json.load(open(file_of_lens))
        n_to_load = dict_lens[s_path.split("/")[-1]]*perc
    else:
        n_to_load = raw
    loaded=0
    tr=0
    while loaded < n_to_load and tr < len(L_files):
        if ".pt" in L_files[tr]:
            dat = torch.load(os.path.join(s_path,L_files[tr])).numpy()
        else:
            dat= np.load(os.path.join(s_path,L_files[tr]))
        if loaded+len(dat) > n_to_load:
            data.append(dat[:n_to_load-loaded])
            loaded+=len(dat[:n_to_load-loaded])
        else:
            data.append(dat)
            loaded+=len(dat)
        tr+=1
    data = np.concatenate(data)
    return data

def load_data_from_everywhere(list_of_paths, use_perc=False, file_of_lens=None, raw=10000,perc = 0.01):
    data = []
    for s_path in list_of_paths:
        data.append(load_data_from_one(str(s_path),use_perc, file_of_lens, raw,perc))
    data = np.concatenate(data)
    print(data.shape)
    return data
            
def train_kmeans_faiss_multi_gpu(data, save_path, n_clusters, n_iter=20, verbose=True,min_points=32, max_points=1024):
    """
    Runs K-means using FAISS on multiple GPUs.
    
    Args:
        data (np.ndarray): The data array to cluster, shape (num_samples, num_features).
        n_clusters (int): The number of clusters to form.
        n_iter (int): Number of iterations for the K-means algorithm.
        verbose (bool): Whether to print the output during clustering.
        
    Returns:
        tuple: Cluster centroids and the assignments of each data point.
    """
    # Ensure data is in float32 format for FAISS
    data = data.astype(np.float32)
    
    # Get the number of available GPUs
    num_gpus = faiss.get_num_gpus()
    print(f"Using {num_gpus} GPUs for K-means clustering")

    # Create a resource object for each GPU
    res = [faiss.StandardGpuResources() for _ in range(num_gpus)]
    
    # Build a GPU index for clustering
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0  # Use GPU 0 as the primary device

    # Set up the K-means with FAISS using multi-GPU resources
    kmeans = faiss.Clustering(data.shape[1], n_clusters)
    kmeans.niter = n_iter
    kmeans.verbose = verbose
    kmeans.max_points_per_centroid = max_points
    kmeans.min_points_per_centroid = min_points

    # Create the multi-GPU index
    gpu_index = faiss.index_cpu_to_all_gpus(
        faiss.IndexFlatL2(data.shape[1]),  # Flat (L2) index
        co=None                            # Use all GPUs
    )

    # Train the K-means model using the multi-GPU index
    kmeans.train(data, gpu_index)

    # Get the cluster centroids and assignments for the input data
    centroids = faiss.vector_to_array(kmeans.centroids).reshape(n_clusters, -1)
    np.save(os.path.join(save_path,'centroids.npy'), centroids)
    #np.save(gpu_index, os.path.join(save_path,'gpu_index'))
    return gpu_index

def assign_kmeans_faiss_multi_gpu(data, gpu_index, save_path,num_batch):
    """
    Assigns data points to clusters using a pre-trained K-means model.
    
    Args:
        data (np.ndarray): The data array to cluster, shape (num_samples, num_features).
        gpu_index (faiss.GpuMultipleCloner): Pre-trained K-means model.
        
    Returns:
        np.ndarray: Assignments of each data point to a cluster.
    """
    # Assign data points to clusters
    _, assignments = gpu_index.search(data, 1)
    np.save(os.path.join(save_path,"assignments_"+str(num_batch)+".npy"), assignments)
    return assignments

def assign_by_batch(data_path, gpu_index, save_path, nb_batch=200):
    print(data_path)
    os.makedirs(save_path, exist_ok=True)
    files=[]
    for f in os.listdir(data_path):
        if isinstance(f, bytes):
            f = os.fsdecode(f)  # Convert bytes to string
        files.append(os.path.join(data_path,f))
    files.sort()
    tr=0
    while tr < len(files):
        batch=[]
        n=0
        while n < nb_batch and tr < len(files):
            if ".pt" in files[tr]:
                batch.append(torch.load(files[tr]).numpy())
            else :
                batch.append(np.load(files[tr]))
            n+=1
            tr+=1
        batch = np.concatenate(batch)
        assign_kmeans_faiss_multi_gpu(batch, gpu_index, save_path, tr)


def cross_val(path, save_path, n_clusters, n_iter, verbose, min_points, max_points, use_perc=False, file_of_lens=None, raw=10000, perc=0.1, nb_batch=200):
    os.makedirs(save_path, exist_ok=True)
    samples=[os.path.join(path,f) for f in os.listdir(path)]
    np.random.shuffle(samples)
    Folds = np.array_split(samples, 10)
    i=0
    tdeb=time.time()
    while i < 1:
        tepoch=time.time()
        test = Folds[i]
        if i==0:
            train = np.concatenate(Folds[1:])
        elif i==9:
            train = np.concatenate(Folds[:9])
        else:
            train = np.concatenate(Folds[:i]+Folds[i+1:])
        save_path_i=os.path.join(save_path,"Fold_"+str(i))
        train_save_path_i = os.path.join(save_path_i,"train")
        test_save_path_i = os.path.join(save_path_i,"test")
        os.makedirs(save_path_i, exist_ok=True)
        os.makedirs(train_save_path_i, exist_ok=True)
        os.makedirs(test_save_path_i, exist_ok=True)
        data = load_data_from_everywhere(train, use_perc, file_of_lens, raw, perc)
        print("Loading over :",time.time()-tdeb,"seconds")
        tload=time.time()
        gpu_index = train_kmeans_faiss_multi_gpu(data, save_path_i, n_clusters, n_iter, verbose, min_points, max_points)
        print("Training over :",time.time()-tload,"seconds")
        ttrain=time.time()
        for f in train:
            assign_by_batch(f, gpu_index, os.path.join(train_save_path_i, f.split("/")[-1]), nb_batch)
        for f in test:
            assign_by_batch(f, gpu_index, os.path.join(test_save_path_i, f.split("/")[-1]), nb_batch)
        print("Assigning over :",time.time()-ttrain,"seconds")
        print("Epoch over :",time.time()-tepoch,"seconds")
        i+=1
    print("Total time :",time.time()-tdeb,"seconds")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering using K-means with FAISS on multiple GPUs")
    parser.add_argument("data_path", type=str, help="Path to the data to cluster")
    parser.add_argument("save_path", type=str, help="Path to save the clustering results")
    parser.add_argument("n_clusters", type=int, help="Number of clusters to form")
    parser.add_argument("n_iter", type=int, default=20, help="Number of iterations for the K-means algorithm")
    parser.add_argument("verbose", type=bool, default=True, help="Whether to print the output during clustering")
    parser.add_argument("min_points", type=int, default=32, help="Minimum number of points per cluster")
    parser.add_argument("max_points", type=int, default=1024, help="Maximum number of points per cluster")
    parser.add_argument("use_perc", type=bool, default=False, help="Whether to use a percentage of the data")
    parser.add_argument("file_of_lens", type=str, help="File containing the number of data points to use for each class")
    parser.add_argument("raw", type=int, default=10000, help="Number of data points to use if not using a percentage")
    parser.add_argument("perc", type=float, default=0.01, help="Percentage of the data to use")
    parser.add_argument("nb_batch", type=int, default=10, help="Number of batches to split the data into")
    args = parser.parse_args()
    cross_val(args.data_path, args.save_path, args.n_clusters, args.n_iter, args.verbose, args.min_points, args.max_points, args.use_perc, args.file_of_lens, args.raw, args.perc, args.nb_batch)
