# Standard Libraries
import numpy as np
import pandas as pd

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
    n = [600, 700, 800, 900]
    sBasis = noiselet(1024)
    sMatrix = [
        sBasis[np.random.permutation(1024)[:i], :].copy()
        for i in n
    ]

    print(sMatrix[0].shape)