import numpy as np
from sklearn.metrics import mutual_info_score

is_similarity = {
    'mean_absolute_error': False,
    'mean_squared_error': False,
    'norm_corr': True,
    'mutual_info': True
}


def mean_absolute_error(true_image, pred_image):
    """ Mean absolute error (MAE). """
    return np.mean(np.abs(true_image - pred_image))


def mean_squared_error(true_image, pred_image):
    """ Mean squared error (MSE). """
    return np.mean((true_image - pred_image) ** 2)


def norm_corr(true_image, pred_image):
    """ Normalized cross correlation (NCC). """
    t0 = true_image - true_image.mean()
    p0 = pred_image - pred_image.mean()
    return np.mean(t0 * p0) / (true_image.std() * pred_image.std())


def mutual_info(true_image, pred_image):
    """ Mutual information (MI). """
    return mutual_info_score(true_image.ravel(), pred_image.ravel())
