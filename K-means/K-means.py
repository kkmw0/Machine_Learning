#!/usr/bin/env python3

import numpy as np
import random
import matplotlib.pyplot as plt

def init():
    """
    Initialization Function
    """
    random.seed(20211208)
    # To reproduce the same result,
    # we intentionally set the seed value of random.seed()
    # If you want to get radom numbers,
    # call random.seed() without any argument

def euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))

def find_closet_centroid(distance, data_idx):
    closet_centroid = min(distance, key=distance.get)
    return [closet_centroid, data_idx]

def compute_new_centroid_value(cluster, data, k):
    tmp = []
    
    for i in range(k):
        sum_distance = [0, 0]
        cnt = 0
        for j in range(len(cluster)):
            if cluster[j][0] == i:
                cnt += 1
                sum_distance[0] += data[cluster[j][1]][0]
                sum_distance[1] += data[cluster[j][1]][1]
        tmp.append([sum_distance[0] / cnt, sum_distance[1] / cnt])
        
    return tmp

def labeling(centroid, data, data_idx):
    tmp = [[] for _ in range(len(centroid))]
    for i in range(len(centroid)):
        for j in range(len(data)):
            if data_idx[j][0] == i:
                tmp[i].append(data[data_idx[j][1]])
        tmp[i].insert(0, centroid[i])
    return tmp

def kmeans(data, k, seed=None, niter=30):
    if seed is None:
        seed = sorted(random.sample(range(data.shape[0]), k))
    assert k == len(seed), "Need k seed numbers"

    # Implement k-means clustering algorihtm
    # Cluster Format
    # [
    #    [ center_point, the list of the cluster entries ]
    #    ...
    # ]

    centroid = []

    for i in range(len(seed)):
        centroid.append(data[seed[i]])

    for i in range(niter):
        # Need to be implemented
        tmp = []
        for j in range(len(data)):
            distance = {}
            for k in range(len(centroid)):
                distance[k] = euclidean_distance(data[j], centroid[k])            
            tmp.append(find_closet_centroid(distance, j))

        centroid = compute_new_centroid_value(tmp, data, len(centroid))
        clusters = labeling(centroid, data, tmp)

    return sorted(clusters, key=lambda x: x[0])

if __name__ == "__main__":
    init()
    # Load data
    data = np.load("k_means.data.npy")
    clusters = kmeans(data, 3)
    
    # Plot Clustered dataset with matplotlib.
    # Plot each cluster in a different color with a 2-dimentional graph.
    colors = ['red', 'blue', 'green']
    labels = ['cluster1', 'cluster2', 'cluster3']

    for i in range(3):
        for j in range(1, len(clusters[i])):
            plt.scatter(clusters[i][j][0], clusters[i][j][1], c = colors[i])
        plt.scatter(clusters[i][0][0], clusters[i][0][1], c = 'black')
    
    plt.show()
