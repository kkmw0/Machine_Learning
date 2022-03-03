#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def pca(data, k):
    '''
    Step 1.
    Implement PCA algorithm
    return k PCs from the given data
    '''
    m = data.shape
    norm_data = data - data.mean()
    norm_data = norm_data / data.std(axis = 0)
    res = np.cov(norm_data.T)
    
    return res

def transform_w_pca(data, pcs):
    '''
    Step 2.
    Express the original data with PCs.
    '''
    m = data.shape
    norm_data = data - data.mean()
    norm_data = norm_data / data.std(axis = 0)
    
    eigen_val, eigen_vec = np.linalg.eig(pcs)
    z1 = eigen_vec[:, 0][0] * norm_data[:, 0] + eigen_vec[:, 0][1] * norm_data[:, 1] + eigen_vec[:, 0][2] * norm_data[:, 2]
    z2 = eigen_vec[:, 1][0] * norm_data[:, 0] + eigen_vec[:, 1][1] * norm_data[:, 1] + eigen_vec[:, 1][2] * norm_data[:, 2]
    pca_res = np.vstack([z1, z2]).T
    converted_data = pca_res[:]

    return converted_data
    
if __name__ == '__main__':
    data = np.loadtxt('HR_comma_sep.csv', delimiter=',', skiprows=1, usecols=range(8))
    pcs = pca(data, 2)
    converted_data = transform_w_pca(data, pcs)
    
    # Step 3.
    # Plot transformed data with matplotlib.
    # You can plot the data with a 2-dimentional graph.
    plt.scatter(converted_data[:, 0], converted_data[:, 1])
    plt.show()
