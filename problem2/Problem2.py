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

if __name__ == '__main__':
    np.random.seed(1234)

    current_path = os.path.abspath(__file__)
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Image.mat'))
    X = scipy.io.loadmat(file_directory)['X']

    # Make 15% of the pixels = 0
    zero_count = int(X.flatten().shape[0] * 0.15)
    indices = np.random.choice(X.flatten().shape[0], size = zero_count, replace = False)

    shape = X.shape
    X = X.flatten()
    X[indices] = 0
    X = X.reshape(shape)

    print(X)