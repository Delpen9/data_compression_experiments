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

# Saving Images
import cv2

def block_coordinate_descent(
    M : np.ndarray,
    _lambda : float = 1e-9,
    _gamma : float = 1e-6,
    max_iterations : float = 100,
    verbose : bool = False
) -> tuple[np.ndarray, np.ndarray]:
    '''
    '''
    L = np.random.rand(*M.shape)
    S = np.random.rand(*M.shape)

    m_norm = np.linalg.norm(M, 'fro')

    mu = 2
    iteration = 0

    stop_algorithm = False

    objective_function_values = []

    while not stop_algorithm:
        iteration += 1

        U, SV, V = np.linalg.svd(M - S)
        SV = np.diag(SV)
        diagS = np.diag(SV)
        alpha_L = U[:, :mu] @ np.diag(diagS[:mu]) @ V[:mu, :]

        gamma_L = 0.5 * _gamma * np.linalg.norm(L, 'nuc')
        
        L[alpha_L >= gamma_L] = alpha_L[alpha_L >= gamma_L] - gamma_L
        L[(alpha_L >= -gamma_L) & (alpha_L < gamma_L)] = 0
        L[alpha_L < -gamma_L] = alpha_L[alpha_L < -gamma_L] + gamma_L

        alpha_S = M - L
        gamma_S = 0.5 * _lambda * np.linalg.norm(S, ord = 1)

        S[alpha_S >= gamma_S] = alpha_S[alpha_S >= gamma_S] - gamma_S
        S[(alpha_S >= -gamma_S) & (alpha_S < gamma_S)] = 0
        S[alpha_S < -gamma_S] = alpha_S[alpha_S < -gamma_S] + gamma_S

        objective = np.linalg.norm(M - L - S, ord = 'fro')**2
        objective += _gamma * np.linalg.norm(L, 'nuc')
        objective += _lambda * np.linalg.norm(S, ord = 1)

        objective_function_values.append(objective)

        if verbose:
            print(objective)

        if iteration == max_iterations:
            stop_algorithm = True

    objective_function_values = np.array(objective_function_values)
    return (L, S, objective_function_values)

def rpca(
    M : np.ndarray,
    _lambda : float = 1e-2,
    tolerance : float = 1e-7
) -> tuple[np.ndarray, np.ndarray]:
    '''
    '''
    m, n = M.shape

    Y = M.copy()
    norm_two = np.linalg.norm(Y)
    norm_inf = np.linalg.norm(Y, ord = np.inf) / _lambda

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

        if stopCriterion < tolerance:
            converged = True

    return (A_hat, E_hat)

if __name__ == '__main__':
    np.random.seed(1234)

    current_path = os.path.abspath(__file__)
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Image_anomaly.mat'))
    X = scipy.io.loadmat(file_directory)['X']

    X = ((X - X.min()) / (X.max() - X.min())) * 255

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'X_problem3.png'))
    cv2.imwrite(file_directory, X.astype(np.uint8))

    # L, S = rpca(X)

    # mse = mean_squared_error(X, L + S)
    # print(mse)

    # L = ((L - L.min()) / (L.max() - L.min())) * 255

    # file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'L_rpca.png'))
    # cv2.imwrite(file_directory, L.astype(np.uint8))

    # S = ((S - S.min()) / (S.max() - S.min())) * 255

    # file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'S_rpca.png'))
    # cv2.imwrite(file_directory, S.astype(np.uint8))

    L, S, objective_function_values = block_coordinate_descent(X, verbose = True)

    mse = mean_squared_error(X, L + S)
    print(fr'The mean square error for block coordinate descent is {mse}.')

    L = ((L - L.min()) / (L.max() - L.min())) * 255

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'L_block_coordinate_descent.png'))
    cv2.imwrite(file_directory, L.astype(np.uint8))

    S = ((S - S.min()) / (S.max() - S.min())) * 255

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'S_block_coordinate_descent.png'))
    cv2.imwrite(file_directory, S.astype(np.uint8))

    # Plotting Objective Function
    indices = np.arange(1, len(objective_function_values) + 1).reshape(-1, 1)
    objective_function_df = pd.DataFrame(np.hstack((indices, np.array(objective_function_values).reshape(-1, 1))), columns = ['Iteration', 'Objective Function'])

    sns.lineplot(x = 'Iteration', y = 'Objective Function', data = objective_function_df)
    plt.title(fr'Iteration VS. Objective Function')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function')

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'iteration_vs_objective_function.png'))
    plt.savefig(file_directory, dpi = 100)

    plt.clf()
    plt.cla()