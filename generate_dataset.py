# Generate Dataset
# This script generates a dataset of Calabi-Yau (CY) hypersurfaces in toric varieties using CYTools (https://cy.tools/).
# We use the GKZ vector corresponding to each triangulation as features and the log of the CY volume
# at the tip of the stretched Kahler cone as labels.

# Warning: This script was written to work with CYTools version 1.0.1. (https://github.com/LiamMcAllisterGroup/cytools/releases/tag/v1.0.1)
# More recent versions of CYTools may prevent ray from working.

from cytools import fetch_polytopes
import numpy as np
import os
import csv
import ray # We use ray to parallelize the computation. Install with `pip install ray`.
from collections import defaultdict
import random
from tqdm import tqdm

# Parameters
h11 = 30 # Higher h11 will require more compute resources.
h21 = 42 # Polytopes in the Kreuzer-Skarke dataset are ordered. If h21 is not specified, the polytope with the smallest h21 is returned first.
num_CYs = 1e6 # Number of unique Calabi-Yau manifolds to be generated. Remember that different triangulations can give rise to the same CY.
max_dataset_size = int(1e10) # Maximum number of triangulations to include in the dataset.
train_test_split = 0.8 
num_threads = 16 # Number of threads to use for parallelization.
compute_chunk_size = int(1e2) # Number of CYs to be generated by each thread at a time.
dataset_dir = "/home/cytools/mounted_volume/LocalStorage/MLProjects/CY/Datasets/GKZ/" # Change this to your storage directory

# Create dataset directory
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Fetch polytopes
poly_gen = fetch_polytopes(h11=h11, h21=h21, lattice='N', limit=1, favorable=True)
POLY = next(poly_gen)

# ray returns an error when it runs inside the CYTools docker image, and the function inputs/outputs a custom class (e.g. a CY class).
# To avoid this issue, we check for duplicate triangulations using the GKZ vector, and duplicate CY phases using the intersection numbers.
@ray.remote
def generate_CYs(N):
    seed = np.random.randint(0, 2**32-1)
    try:
        triangulations_list = POLY.random_triangulations_fast(N, c=2.5, as_list=True, seed=seed, progress_bar=False, backend='cgal')
    except:
        try:
            triangulations_list = POLY.random_triangulations_fast(N, c=2.5, as_list=True, seed=seed, progress_bar=False, backend='qhull')
        except:
            return []
    cy_data = []
    for t in triangulations_list:
        gkz = tuple(t.gkz_phi())[1:] # The first element is the GKZ value of the origin, which is constant.
        cy = t.get_cy()
        intersection_numbers = cy.intersection_numbers(in_basis=True, format='coo')
        intersection_numbers = np.sort(intersection_numbers, axis=0)
        intersection_numbers = tuple(map(tuple, intersection_numbers))
        kahler_cone_tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
        cy_volume = cy.compute_cy_volume(kahler_cone_tip)
        cy_data += [[intersection_numbers, gkz, np.log10(cy_volume)]]
    return cy_data

CY_dict = defaultdict(lambda: [])
triangulation_ctr = 0
pbar = tqdm(total=num_CYs)
while len(CY_dict) < num_CYs and triangulation_ctr < max_dataset_size:
    cy_data_ray = ray.get([generate_CYs.remote(compute_chunk_size) for _ in range(num_threads)])
    for cy_data in cy_data_ray:
        for d in cy_data:
            CY_dict[d[0]] += [[d[1], d[2]]]
    pbar.update(n=len(CY_dict)-pbar.n)
    triangulation_ctr += num_threads * compute_chunk_size

dataset = list(CY_dict.values())
CY_dict = None # Free up memory
random.shuffle(dataset)

# Split dataset into training and testing sets
train_dataset_size = int(len(dataset) * train_test_split)
train_dataset = dataset[:train_dataset_size]
test_dataset = dataset[train_dataset_size:]

# Flatten
train_dataset = [d for c in train_dataset for d in c]
test_dataset = [d for c in test_dataset for d in c]

# Normalize Features
train_features = np.array([d[0] for d in train_dataset])
test_features = np.array([d[0] for d in test_dataset])
train_labels = np.array([d[1] for d in train_dataset])
test_labels = np.array([d[1] for d in test_dataset])

tol = 1e-10
features_mean = train_features.mean(axis=0)
features_std = train_features.std(axis=0)
train_features = (train_features - features_mean) / (features_std + tol)
test_features = (test_features - features_mean) / (features_std + tol)

print("Number of unique CYs: {}".format(len(dataset)))
print("Training set size: {}".format(len(train_dataset)))
print("Testing set size: {}".format(len(test_dataset)))

# Save dataset
print("Saving dataset...")
with open(dataset_dir + "train_features.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(train_features)
with open(dataset_dir + "test_features.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(test_features)
with open(dataset_dir + "train_labels.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(train_labels[:,None])
with open(dataset_dir + "test_labels.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(test_labels[:,None])
print("Done!")