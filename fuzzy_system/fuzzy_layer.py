import numpy as np


def gaussian_membership(x, mean, sigma):

    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


def fuzzy_transform(features, params):

    mean1, sigma1, mean2, sigma2 = params

    features = features.numpy()

    low_membership = gaussian_membership(features, mean1, sigma1)

    high_membership = gaussian_membership(features, mean2, sigma2)

    fused_features = (low_membership + high_membership) / 2

    return fused_features