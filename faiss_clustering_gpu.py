import os
import torch
import numpy as np
import faiss
import argparse
import time

def load_data(file_paths,number_to_load=0):
    """
    Loads data from a list of file paths, which can be either .pt, .pth, or .npy files.
    
    Args:
        file_paths (list of str): Paths to the data files.
        
    Returns:
        list of np.ndarray: List of loaded datasets as numpy arrays.
    """
    datasets = []
    i=0
    l = len(file_paths)
    tdeb = time.time()
    for path in file_paths:
        if i%100==0:
            print(f"Loading dataset {i}/{l} in {time.time()-tdeb} seconds")
        if path.endswith(('.pt', '.pth')):
            # Load PyTorch tensor and convert to numpy array
            data = torch.load(path,map_location=torch.device("cpu")).cpu().numpy()[:number_to_load]
        elif path.endswith('.npy'):
            # Load numpy array directly
            data = np.load(path)[:number_to_load]
        else:
            raise ValueError(f"Unsupported file format: {path}")
        datasets.append(data)
        i+=1
    print(f"Loading time: {time.time()-tdeb}")
    return datasets

def concatenate_datasets(datasets):
    """
    Concatenates a list of numpy arrays along the first axis.
    
    Args:
        datasets (list of np.ndarray): List of datasets to concatenate.
        
    Returns:
        np.ndarray: Concatenated dataset.
    """
    return np.concatenate(datasets, axis=0)

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
    print(f"Centroids shape: {centroids.shape}")
    return gpu_index

def assign_kmeans_faiss_multi_gpu(data, idx_data, gpu_index, save_path,nb_batch):
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
    np.save(os.path.join(save_path,"assignments_"+str(nb_batch)+".npy"), assignments)
    np.save(os.path.join(save_path,"idx_"+str(nb_batch)+".npy"), idx_data)
    print(f"Assignments shape: {assignments.shape}")
    return assignments
    


def treat_one_sample(data_dir, save_path, n_clusters, n_iter=20, min_points=32, max_points=1024,nb_file_batch=10):
    """
    Main function to load data, concatenate, and perform K-means clustering.
    
    Args:
        data_dir (str): Directory containing the .pt, .pth, or .npy files.
        n_clusters (int): Number of clusters for K-means.
        n_iter (int): Number of iterations for K-means.
    """
    tdeb_sample = time.time()
    os.makedirs(save_path, exist_ok=True)
    # Get all file paths in the directory
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.endswith(('.pt', '.pth', '.npy'))]
### ADAPT THE SORTING FUNCTION TO YOUR FILE NAMES
    file_paths = sorted(file_paths, key=lambda x: (
    int(x.split("/")[-1].split(".")[0].split('_')[1][3:]),   # Number after ERR
    int(x.split("/")[-1].split(".")[0].split('_')[2]),       # Number just after
    int(x.split("/")[-1].split(".")[0].split('_')[4]),       # Number AFTER EMBEDDINGS
    int(x.split("/")[-1].split(".")[0].split('_')[5]),       # Number before .pt
))
#    file_paths = sorted(file_paths, key=lambda x: (
#    int(x.split("/")[-1].split(".")[0].split('_')[1]),       # Number after run
#    int(x.split("/")[-1].split(".")[0].split('_')[3]),       # Number AFTER EMBEDDINGS
#    int(x.split("/")[-1].split(".")[0].split('_')[4]),       # Number before .pt
#))
    idx_paths = [f.replace("mean","idx").replace("embeddings","idx") for f in file_paths]
    print(len(file_paths))
    print(len(idx_paths))
    print(file_paths[:100])
    print(idx_paths[:100])
    # Number of points to load per file
    number_to_load = (max_points*n_clusters)
    print(f"Number of points to load: {number_to_load}")
    tr=0
    data = []
    n_points = 0
    while n_points<number_to_load and tr<len(file_paths):
        if file_paths[tr].endswith(('.pt', '.pth')):
            # Load PyTorch tensor and convert to numpy array
            dat = torch.load(file_paths[tr],map_location=torch.device("cpu")).cpu().numpy()
            
            if number_to_load-n_points>len(dat):
                data.append(dat)
            else:
                data.append(dat[:number_to_load-n_points])
        elif file_paths[tr].endswith('.npy'):
            # Load numpy array directly
            dat = np.load(file_paths[tr])[:number_to_load]
            if number_to_load-n_points>len(dat):
                data.append(dat)
            else:
                data.append(dat[:number_to_load-n_points])
        else:
            raise ValueError(f"Unsupported file format: {file_paths[tr]}")
        n_points+=len(dat)
        print(n_points)
        print(tr)
        tr+=1
    data = concatenate_datasets(data)
    print(f"Data shape after concatenation: {data.shape}")
    print(f"Loading time: {time.time()-tdeb_sample}")
    index = train_kmeans_faiss_multi_gpu(data, save_path, n_clusters, n_iter, True, min_points, max_points)
    i=0
    #tdeb_assign = time.time()
    #while i < len(file_paths):
    #    j=0
    #    assign_data =[]
    #    idx_data = []
    #    while j<nb_file_batch and i<len(file_paths):
    #        if file_paths[i].endswith(('.pt', '.pth')):
    #            # Load PyTorch tensor and convert to numpy array
    #            dat = torch.load(file_paths[i],map_location=torch.device("cpu")).cpu().numpy()
    #            idx = torch.load(idx_paths[i],map_location=torch.device("cpu")).cpu().numpy()
    #        elif file_paths[i].endswith('.npy'):
    #            # Load numpy array directly
    #            dat = np.load(file_paths[i])
    #            idx = np.load(idx_paths[i])
    #        else:
    #            raise ValueError(f"Unsupported file format: {file_paths[i]}")
    #        assign_data.append(dat)
    #        idx_data.append(idx)
    #        print(file_paths[i])
    #        j+=1
    #        i+=1
    #    assign_data = concatenate_datasets(assign_data)
    #    idx_data = concatenate_datasets(idx_data)
    #    print(f"Data shape after concatenation: {assign_data.shape}")
    #    assign_kmeans_faiss_multi_gpu(assign_data, idx_data,index, save_path,i)
    #print(f"Assigning time: {time.time()-tdeb_assign}")
        

def main(data_dir, save_path,n_clusters, n_iter=20, min_points=32, max_points=1024,nb_file_batch=10):
    save_path = os.path.join(save_path, str(n_clusters))
    os.makedirs(save_path, exist_ok=True)
    tdeb = time.time()
    c=0
    for sample in os.listdir(data_dir):
        ttemp = time.time()
        if sample in os.listdir(save_path):
            print(sample,"already treated")
            c+=1
            continue
        print(c)
        treat_one_sample(os.path.join(data_dir,sample), os.path.join(save_path,sample), n_clusters, n_iter, min_points, max_points,nb_file_batch)
        print(f"Sample time: {time.time()-ttemp}")
    print(f"Total time: {time.time()-tdeb}")
    print(f"Time by sample: {(time.time()-tdeb)/len(os.listdir(data_dir))}")    

if __name__ == "__main__":
    parsearg = argparse.ArgumentParser()
    parsearg.add_argument('data_dir', type=str, help='Directory containing the data files.')
    parsearg.add_argument('save_path', type=str, help='Directory to save the clustering results.')
    parsearg.add_argument('n_clusters', type=int, help='Number of clusters for K-means.')
    parsearg.add_argument('n_iter', type=int, default=20, help='Number of iterations for K-means.')
    parsearg.add_argument('min_points', type=int, default=32, help='Minimum number of points per centroid.')
    parsearg.add_argument('max_points', type=int, default=1024, help='Maximum number of points per centroid.')
    parsearg.add_argument('nb_file_batch', type=int, default=10, help='Number of files per batch')
    args = parsearg.parse_args()
    main(args.data_dir, args.save_path, args.n_clusters, args.n_iter, args.min_points, args.max_points,args.nb_file_batch)
