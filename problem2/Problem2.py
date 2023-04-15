import warnings
warnings.filterwarnings('ignore')

# Standard libraries
import os
import numpy as np
import pandas as pd

# Loading .mat files
import scipy.io

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

def PFBS(
    X : np.ndarray,
    A : np.ndarray,
    m : int,
    n1 : int,
    n2 : int,
    tau : int = 250,
    max_iteration : int = 500
) -> np.ndarray:
    '''
    '''
    X0 = X.copy()

    Y = np.zeros((n1, n2))
    delta = n1 * n2 / m

    vec = np.zeros((max_iteration,))
    err = vec.copy()
    for i in range(max_iteration):
        U, S, V = np.linalg.svd(A)

        S_t = (S - tau)
        S_t[S_t < 0] = 0

        Z = U @ S_t @ V.T
        P = X - Z

        Y0 = Y.copy()
        Y = Y0 + delta * P

        vec[i] = np.sum(np.sum((Y - Y0)**2))
        err[i] = np.sum(np.sum((X0 - Z)**2)) / np.sum(np.sum((X0)**2))
    return err

if __name__ == '__main__':
    np.random.seed(1234)

    current_path = os.path.abspath(__file__)
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Image.mat'))
    X = scipy.io.loadmat(file_directory)['X']

    A = np.random.rand(*X.shape) >= 0.80
    X[A] = 0
    m = np.sum(np.sum(A == 0))

    error = PFBS(X, A, m, *X.shape)
    print(error)