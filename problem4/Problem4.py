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

# Scipy Libraries
from scipy.spatial.distance import cdist

def gaussian_kernel_1d(
    x : np.ndarray,
    bandwidth : int = 1.0
) -> np.ndarray:
    '''
    '''
    return np.exp(-0.5 * (x / bandwidth)**2)

def gram_matrix(
)-> np.ndarray:
    '''
    '''
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    distances = cdist(X, X, metric = 'euclidean')

    kernel_1d = gaussian_kernel_1d(distances)

    gram_matrix = np.kron(kernel_1d, kernel_1d).reshape(10000, 10000)
    return gram_matrix

def RKHS(
    y : np.ndarray
) -> np.ndarray:
    '''
    '''
    _lambda = 0.1
    K = gram_matrix()
    _alpha = np.linalg.inv(K + _lambda * np.eye(K.shape[0])) @ y
    y_hat = K @ _alpha
    return y_hat

if __name__ == '__main__':
    np.random.seed(1234)

    current_path = os.path.abspath(__file__)
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Peaks.mat'))
    y = scipy.io.loadmat(file_directory)['Y']

    y_hat_flattened = RKHS(y.flatten())
    y_hat = y_hat_flattened.reshape(y.shape)

    y_hat = ((y_hat - y_hat.min()) / (y_hat.max() - y_hat.min())) * 255
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'problem_4_denoised_image.png'))
    cv2.imwrite(file_directory, y_hat)

    y = ((y - y.min()) / (y.max() - y.min())) * 255

    smoothed_standard_deviation = np.std(y_hat)
    original_standard_deviation = np.std(y)

    noise_standard_deviation = np.abs(original_standard_deviation - smoothed_standard_deviation)

    print(fr'The standard deviation of the smoothed image is: {smoothed_standard_deviation}')
    print(fr'The standard deviation of the original image is: {original_standard_deviation}')
    print(fr'The standard deviation of the noise is: {noise_standard_deviation}')


