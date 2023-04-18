import warnings
warnings.filterwarnings('ignore')

# Standard libraries
import os
import numpy as np
import pandas as pd

# Loading .mat files
import scipy.io

# Sklearn libraries
from sklearn.metrics import mean_squared_error

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

def rpca(
    M : np.ndarray,
    _lambda : float = 1e-2,
    tolerance : float = 1e-7,
    max_iteratons : int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    '''
    '''
    m, n = M.shape

    Y = M.copy()
    norm_two = np.linalg.norm(Y)
    norm_inf = np.linalg.norm(Y, ord=np.inf) / _lambda

    Y /= norm_inf

    A_hat = np.zeros((m, n))
    E_hat = np.zeros((m, n))

    mu = 1.25 / norm_two
    mu_bar = mu * 1e+7

    rho = 1.5

    d_norm = np.linalg.norm(M, 'fro')

    total_svd = 0
    converged = False
    stopCriterion = 1
    while not converged:
        temp_T = M - A_hat + (1 / mu) * Y
        E_hat = np.max(temp_T - _lambda / mu, 0)
        E_hat += np.min(temp_T + _lambda / mu, 0)

        U, S, V = np.linalg.svd(M - E_hat + (1 / mu) * Y)

        S = np.diag(S)
        diagS = np.diag(S)
        svp = len(np.where(diagS > 1 / mu)[0])

        A_hat = U[:, :svp] @ np.diag(diagS[:svp] - (1 / mu)) @ V[:svp, :]
        total_svd += 1

        Z = M - A_hat - E_hat
        Y += mu * Z

        mu = min(mu * rho, mu_bar)

        stopCriterion = np.linalg.norm(Z, 'fro') / d_norm

        print(stopCriterion)

        if stopCriterion < tolerance:
            converged = True

    return A_hat, E_hat

if __name__ == '__main__':
    np.random.seed(1234)

    current_path = os.path.abspath(__file__)
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Image_anomaly.mat'))
    X = scipy.io.loadmat(file_directory)['X']

    L, S = rpca(X)

    mse = mean_squared_error(X, L)