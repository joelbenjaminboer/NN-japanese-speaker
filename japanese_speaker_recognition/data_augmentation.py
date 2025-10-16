import numpy as np


def add_gaussian_noise(X, noise_factor=0.001):
    """
    add_gaussian_noise adds Gaussian noise to the input data X.
    """
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise


def random_scaling(X, scale_range=(0.95, 1.05)):
    """
    random_scaling randomly scales the input data X by a factor
    within the specified scale_range.
    """
    scale = np.random.uniform(*scale_range)
    return X * scale


def time_masking(X, max_mask_percentage=0.01):
    """
    time_masking applies time masking to the input data X.
    max_mask_percentage defines the maximum percentage of the time dimension to mask.
    """
    n_samples, maxlen, n_features = X.shape
    mask_len = int(maxlen * max_mask_percentage)

    for i in range(n_samples):
        start = np.random.randint(0, maxlen - mask_len)
        X[i, start : start + mask_len, :] = 0
    return X


def frequency_masking(X, max_mask_percentage=0.01):
    """
    frequency_masking applies frequency masking to the input data X.
    max_mask_percentage defines the maximum percentage of the feature dimension to mask.
    """
    n_samples, maxlen, n_features = X.shape
    mask_len = int(n_features * max_mask_percentage)

    for i in range(n_samples):
        start = np.random.randint(0, n_features - mask_len)
        X[i, :, start : start + mask_len] = 0
    return X
