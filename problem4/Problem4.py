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

def gaussian_kernel_1d(
    x : np.ndarray,
    bandwidth : int = 1
) -> np.ndarray:
    '''
    '''
    return np.exp(-0.5 * (x / bandwidth)**2)

def gram_matrix(
)-> np.ndarray:
    '''
    '''
    x = np.linspace(0, 10, 100)

    def squared_euclidean_distance(x, y):
        return np.sum((x - y)**2)

    distances = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            distances[i, j] = squared_euclidean_distance(x[i], x[j])

    kernel_1d = gaussian_kernel_1d(distances, bandwidth = 1)

    gram_matrix = np.kron(kernel_1d, kernel_1d).reshape(10000, 10000)
    return gram_matrix

def RKHS(
    y : np.ndarray
) -> np.ndarray:
    '''
    '''
    _lambda = 0.9
    K = gram_matrix()
    _alpha = np.linalg.inv(K + _lambda) @ y
    y_hat = K @ _alpha
    return y_hat

if __name__ == '__main__':
    np.random.seed(1234)

    current_path = os.path.abspath(__file__)
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Peaks.mat'))
    y = scipy.io.loadmat(file_directory)['Y']

    y_flattened = y.copy().flatten()
    y_hat_flattened = RKHS(y_flattened)

    y_hat = y_hat_flattened.reshape(y.shape)

    y_hat = ((y_hat - y_hat.min()) / (y_hat.max() - y_hat.min())) * 255
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'problem_4_denoised_image.png'))
    cv2.imwrite(file_directory, y_hat)

    y = ((y - y.min()) / (y.max() - y.min())) * 255
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'problem_4_image.png'))
    cv2.imwrite(file_directory, y)