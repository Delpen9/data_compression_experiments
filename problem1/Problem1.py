# Standard libraries
import os
import numpy as np

# Loading .mat files
import scipy.io

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Scipy libraries
from scipy.optimize import linprog

from minimization import l1eq_pd

def noiselet(
    n : int
) -> np.ndarray:
    '''
    '''
    if n < 2 or (np.log2(n) - np.floor(np.log2(n))) != 0:
        raise ValueError('The input argument should be of form 2^k')

    N = 0.5 * np.array([[1 - 1j, 1 + 1j],
                        [1 + 1j, 1 - 1j]])

    for i in range(2, int(np.log2(n)) + 1):
        N1 = N.copy()
        N = np.zeros(
            (2**i, 2**i),
            dtype = complex
        )
        for k in range(1, 2**i, 2):
            N[k - 1, :] = 0.5 * np.kron([1 - 1j, 1 + 1j], N1[(k + 1) // 2 - 1, :])

        for k in range(2, 2**i + 1, 2):
            N[k - 1, :] = 0.5 * np.kron([1 + 1j, 1 - 1j], N1[k // 2 - 1, :])

    return N

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'full.mat'))
    X = scipy.io.loadmat(file_directory)['X']

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'haar.mat'))
    psi = scipy.io.loadmat(file_directory)['W']

    n = [600, 700, 800, 900]
    sBasis = noiselet(1024)
    q = np.random.permutation(1024)

    phi = [sBasis[q[:i], :].copy() for i in n]

    y = [phi[i] @ X for i in np.arange(4).astype(int)]
    x0 = [psi.T @ (phi[i].T @ y[i]) for i in np.arange(4).astype(int)]

    A = [phi[i] @ psi for i in np.arange(4).astype(int)]

    ## TODO: We just need to recover the signal now and compare
    ## =======================
    ## 
    ## =======================
