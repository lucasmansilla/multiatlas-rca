import numpy as np

is_sim_metric = {
    'sad': False,
    'ssd': False,
    'ncc': True,
    'mi': True
}


def sad(x, y):
    """Sum of Absolute Differences (SAD) between two images."""
    return np.sum(np.abs(x - y))


def ssd(x, y):
    """Sum of Squared Differences (SSD) between two images."""
    return np.sum((x - y) ** 2)


def ncc(x, y):
    """Normalized Cross Correlation (NCC) between two images."""
    return np.mean((x - x.mean()) * (y - y.mean())) / (x.std() * y.std())


def mi(x, y):
    """Mutual Information (MI) between two images."""
    from sklearn.metrics import mutual_info_score
    return mutual_info_score(x.ravel(), y.ravel())
