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

# Sklearn libraries
from sklearn.metrics import mean_squared_error

# Other libraries
import cvxpy as cp

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

def minimize(
    A : np.ndarray,
    y : np.ndarray
) -> np.ndarray:
    '''
    '''
    x = cp.Variable((1024, 1))
    objective = cp.Minimize(cp.norm1(x))

    constraints = [y == A @ x]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver = cp.OSQP, verbose = False)

    x_optimal = x.value

    return x_optimal

if __name__ == '__main__':
    np.random.seed(1234)

    current_path = os.path.abspath(__file__)
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'full.mat'))
    X = scipy.io.loadmat(file_directory)['X']

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'haar.mat'))
    psi = scipy.io.loadmat(file_directory)['W']

    n = [600, 700, 800, 900]
    sBasis = noiselet(1024)
    q = np.random.permutation(1024)

    phi = [sBasis[q[:i], :].copy() for i in n]

    y = [phi[i] @ X for i in np.arange(len(n)).astype(int)]
    A = [phi[i] @ psi.T for i in np.arange(len(n)).astype(int)]

    xprec_results = []
    mse_results_normal = []
    for i in range(len(n)):
        xp = minimize(A[i], y[i])
        xprec = (np.linalg.pinv(psi) @ xp).real
        mse = mean_squared_error(X, xprec)
        mse_results_normal.append(mse)
        print(fr'''(Original Data) The mean-square error for n = {n[i]} is: {mse}''')
        xprec_results.append(xprec)

    indices = np.arange(0, 1024).reshape(-1, 1)
    x_df = pd.DataFrame(np.hstack((indices, X)), columns = ['Index', 'Value'])

    for i in range(len(n)):
        xprec_df = pd.DataFrame(np.hstack((indices, xprec_results[i])), columns = ['Index', 'Value'])

        sns.lineplot(x = 'Index', y = 'Value', data = xprec_df)
        sns.lineplot(x = 'Index', y = 'Value', data = x_df)
        plt.title(fr'n = {600 + i * 100}; Xprec VS. X')
        plt.xlabel('Index')
        plt.ylabel('Value')

        file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', fr'n_{600 + i * 100}_vs_x.png'))
        plt.savefig(file_directory, dpi = 100)

        plt.clf()
        plt.cla()

    ## =================
    ## Adding random noise
    ## =================
    mu = 0
    sigma = 10
    X += np.random.normal(mu, sigma, 1024).reshape(-1, 1)

    y = [phi[i] @ X for i in np.arange(len(n)).astype(int)]
    A = [phi[i] @ psi.T for i in np.arange(len(n)).astype(int)]

    xprec_results = []
    mse_results_noisy = []
    for i in range(len(n)):
        xp = minimize(A[i], y[i])
        xprec = (np.linalg.pinv(psi) @ xp).real
        mse = mean_squared_error(X, xprec)
        mse_results_noisy.append(mse)
        print(fr'''(Normally Distributed Noise) The mean-square error for n = {n[i]} is: {mse}''')
        xprec_results.append(xprec)

    indices = np.arange(0, 1024).reshape(-1, 1)
    x_df = pd.DataFrame(np.hstack((indices, X)), columns = ['Index', 'Value'])

    for i in range(len(n)):
        xprec_df = pd.DataFrame(np.hstack((indices, xprec_results[i])), columns = ['Index', 'Value'])

        sns.lineplot(x = 'Index', y = 'Value', data = xprec_df)
        sns.lineplot(x = 'Index', y = 'Value', data = x_df)
        plt.title(fr'n = {600 + i * 100}; Xprec VS. X Noisy')
        plt.xlabel('Index')
        plt.ylabel('Value')

        file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', fr'n_{600 + i * 100}_vs_x_with_noise.png'))
        plt.savefig(file_directory, dpi = 100)

        plt.clf()
        plt.cla()

    ## =================
    ## Mean Square Error Plotting
    ## =================
    indices = np.arange(600, 1000, 100).reshape(-1, 1)
    mse_normal_df = pd.DataFrame(np.hstack((indices, np.array(mse_results_normal).reshape(-1, 1))), columns = ['n-value', 'Error'])

    sns.lineplot(x = 'n-value', y = 'Error', data = mse_normal_df)
    plt.title(fr'n-value VS. Errors')
    plt.xlabel('n-value')
    plt.ylabel('Error')

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'normal_n_value_vs_errors.png'))
    plt.savefig(file_directory, dpi = 100)

    plt.clf()
    plt.cla()

    indices = np.arange(600, 1000, 100).reshape(-1, 1)
    mse_noisy_df = pd.DataFrame(np.hstack((indices, np.array(mse_results_noisy).reshape(-1, 1))), columns = ['n-value', 'Error'])

    sns.lineplot(x = 'n-value', y = 'Error', data = mse_noisy_df)
    plt.title(fr'n-value VS. Errors')
    plt.xlabel('n-value')
    plt.ylabel('Error')

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'noisy_n_value_vs_errors.png'))
    plt.savefig(file_directory, dpi = 100)

    plt.clf()
    plt.cla()