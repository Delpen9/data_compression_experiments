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
    y : np.ndarray,
    n : int
) -> np.ndarray:
    '''
    '''
    c = np.hstack([np.zeros(1024), np.ones(1024)])

    # Constraint 1: x - z <= 0
    A1 = np.hstack([np.identity(1024), -np.identity(1024)])
    b1 = np.zeros(1024)

    # Constraint 2: -x - z <= 0
    A2 = np.hstack([-np.identity(1024), -np.identity(1024)])
    b2 = np.zeros(1024)

    A_ineq = np.vstack([A1, A2])
    b_ineq = np.hstack([b1, b2])

    # Constraint 3: Ax = y
    A_eq = np.hstack([A, np.zeros((n, 1024))])
    b_eq = y.flatten()

    # Solve the linear programming problem
    result = linprog(
        c,
        # A_ub = A_ineq,
        # b_ub = b_ineq,
        A_eq = A_eq,
        b_eq = b_eq,
        method = 'highs'
    )

    x_opt = result.x[:1024].reshape(1024, 1)

    return x_opt

def relative_reconstruction_error(
    original: np.ndarray,
    reconstructed : np.ndarray
) -> float:
    '''
    '''
    diff = original - reconstructed
    frobenius_norm_diff = np.linalg.norm(diff, ord='fro')
    frobenius_norm_original = np.linalg.norm(original, ord='fro')
    
    return frobenius_norm_diff / frobenius_norm_original


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

    y = [phi[i] @ X for i in np.arange(4).astype(int)]
    x0 = [psi.T @ (phi[i].T @ y[i]) for i in np.arange(4).astype(int)]

    A = [phi[i] @ psi for i in np.arange(4).astype(int)]

    ## TODO: We just need to recover the signal now and compare
    ## =======================
    xp = minimize(A[0], y[0], 600)
    xprec = (-np.linalg.pinv(psi) @ xp).real

    print(relative_reconstruction_error(X, xprec))

    xp = minimize(A[1], y[1], 700)
    xprec = (-np.linalg.pinv(psi) @ xp).real

    print(relative_reconstruction_error(X, xprec))

    xp = minimize(A[2], y[2], 800)
    xprec = (-np.linalg.pinv(psi) @ xp).real

    print(relative_reconstruction_error(X, xprec))

    xp = minimize(A[3], y[3], 900)
    xprec = (-np.linalg.pinv(psi) @ xp).real

    print(relative_reconstruction_error(X, xprec))
    ## =======================
