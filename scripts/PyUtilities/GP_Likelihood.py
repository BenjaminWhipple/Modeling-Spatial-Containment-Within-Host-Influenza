import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import multivariate_normal

def fit_gp_and_evaluate(
    X, y, t_eval,
    y_std=None,
    kernel=None
):
    """
    Fit GP to training data, evaluate on t_eval, optionally compute log-likelihood and save.

    Parameters
    ----------
    X : array-like, shape (n_samples, 1)
        Training input (e.g., time points)
    y : array-like, shape (n_samples,)
        Training outputs (measurements)
    t_eval : array-like, shape (m, 1)
        Evaluation points for GP prediction
    y_std : array-like, shape (n_samples,), optional
        Measurement standard deviations for each y_i (heteroscedastic noise)
    kernel : sklearn.gaussian_process.kernels.Kernel, optional
        Custom kernel (defaults to Matern + WhiteKernel)
    Returns
    -------
    mu : ndarray, shape (m,)
        GP posterior mean at t_eval
    cov : ndarray, shape (m, m)
        GP posterior covariance at t_eval
    """
    if kernel is None:
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=1.5) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e0))

    alpha=1e-6

    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=100)
    gp.fit(X, y)

    mu, cov = gp.predict(t_eval, return_cov=True)
    std = np.sqrt(np.diag(cov))

    return mu, cov