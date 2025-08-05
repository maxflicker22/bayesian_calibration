#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~ helper_functions gaussian_procss_functions.py~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MF~~~~~#

# Filename: helper_functions/gaussian_procss_functions.py
# Author: Markus Flicker
# Date: 2023-08-05
# Description: 
#                This module provides helper functions for Gaussian Process (GP) regression using JAX.
#                Functions:
#                ----------
#                custom_rbf_kernel(x1, x2, lengthscale, variance)
#                    Computes the Radial Basis Function (RBF) kernel (also known as the squared exponential kernel)
#                    between two sets of input points.
#                    Parameters:
#                        x1 (jnp.ndarray): First set of input points, shape (N, D).
#                        x2 (jnp.ndarray): Second set of input points, shape (M, D).
#                        lengthscale (float or jnp.ndarray): Lengthscale parameter(s) of the kernel.
#                        variance (float): Variance (amplitude squared) parameter of the kernel.
#                    Returns:
#                        jnp.ndarray: Kernel matrix of shape (N, M).
#                gp_posterior(X_pred, X_train, y_train, lengthscale, variance, noise)
#                    Computes the posterior mean and covariance of a Gaussian Process at test points,
#                    given training data and kernel hyperparameters.
#                    Parameters:
#                       X_pred (jnp.ndarray): Test input points, shape (N*, D).
#                       X_train (jnp.ndarray): Training input points, shape (N, D).
#                       y_train (jnp.ndarray): Training targets, shape (N,).
#                       lengthscale (float or jnp.ndarray): Lengthscale parameter(s) of the kernel.
#                       variance (float): Variance (amplitude squared) parameter of the kernel.
#                       noise (float): Standard deviation of observation noise.
#                   Returns:
#                       mu (jnp.ndarray): Posterior mean at test points, shape (N*,).
#                       cov (jnp.ndarray): Posterior covariance matrix at test points, shape (N*, N*).

import jax
import jax.numpy as jnp


@jax.jit
def custom_rbf_kernel(x1, x2, lengthscale, variance):
    """
    Computes the Radial Basis Function (RBF) kernel (also known as the squared exponential kernel) between two sets of input vectors.
    Args:
        x1 (jnp.ndarray): An array of shape (N, D) representing N input vectors of dimension D.
        x2 (jnp.ndarray): An array of shape (M, D) representing M input vectors of dimension D.
        lengthscale (float or jnp.ndarray): The lengthscale parameter(s) of the kernel. Can be a scalar or a vector of shape (D,).
        variance (float): The variance (amplitude squared) parameter of the kernel.
    Returns:
        jnp.ndarray: A kernel matrix of shape (N, M) where each entry (i, j) is the RBF kernel value between x1[i] and x2[j].
    """
    
    x1 = x1 / lengthscale
    x2 = x2 / lengthscale
    sqdist = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    return variance * jnp.exp(-0.5 * sqdist)


@jax.jit
def gp_posterior(X_pred, X_train, y_train, lengthscale, variance, noise):
    """
    Computes the posterior predictive mean and covariance of a Gaussian Process (GP) regression model
    using the Radial Basis Function (RBF) kernel.

    This function implements GP regression to estimate the predictive distribution p(f* | X_pred, X_train, y_train),
    where f* are the function values at the prediction points.

    Args:
        X_pred (jnp.ndarray): Array of shape (N*, D) containing the prediction input points 
                              where the GP posterior is evaluated.
        X_train (jnp.ndarray): Array of shape (N, D) containing the training input points.
        y_train (jnp.ndarray): Array of shape (N,) containing the observed target values at X_train.
        lengthscale (float or jnp.ndarray): Lengthscale parameter(s) of the RBF kernel.
                                            Controls the smoothness of the function. Can be scalar or shape (D,).
        variance (float): Variance (signal amplitude squared) parameter of the RBF kernel.
                          Determines the overall scale of function variations.
        noise (float): Observation noise standard deviation (Gaussian i.i.d. noise).

    Returns:
        tuple:
            - mu (jnp.ndarray): Predictive mean vector of shape (N*,), representing E[f* | X_pred].
            - cov (jnp.ndarray): Predictive covariance matrix of shape (N*, N*), representing 
                                 Cov[f* | X_pred].

    """
    
    jitter = 1e-4  # Small value to ensure numerical stability

    K_xx = custom_rbf_kernel(X_train, X_train, lengthscale, variance) + (noise**2 + jitter) * jnp.eye(X_train.shape[0])
    K_xs = custom_rbf_kernel(X_train, X_pred, lengthscale, variance)
    K_ss = custom_rbf_kernel(X_pred, X_pred, lengthscale, variance)

    # JAX-native Cholesky
    L = jnp.linalg.cholesky(K_xx)
    # Solve for alpha: L @ L.T @ alpha = y_train
    alpha = jax.scipy.linalg.solve_triangular(L, y_train, lower=True)
    alpha = jax.scipy.linalg.solve_triangular(L.T, alpha, lower=False)

    # Predictive mean
    mu = jnp.dot(K_xs.T, alpha)

    # v = L \ K_xs
    v = jax.scipy.linalg.solve_triangular(L, K_xs, lower=True)
    cov = K_ss - jnp.dot(v.T, v)

    return mu, cov