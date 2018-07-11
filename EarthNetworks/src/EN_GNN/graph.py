import numpy as np


def get_matrix(region):
    file="../../Data/parsed/" + region + "/distance.csv"
    return np.genfromtxt(file, delimiter=",")


def gaussian_kernel(A, thres=0.01, binary=False, sigma=0.1):
    A = np.exp(-sigma * np.power(A, 2))  # type: np.ndarray  # apply kernel
    A[np.diag_indices(A.shape[0])] = 0
    A[A < thres] = 0
    if binary:
        A[A > 0] = 1
    return A


def get_distance_graph(region, thres=0.01, binary=False, sigma=0.1):
    return gaussian_kernel(get_matrix(region), thres, binary, sigma)
