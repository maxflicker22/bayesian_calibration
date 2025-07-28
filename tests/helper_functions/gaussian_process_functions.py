import jax
import jax.numpy as jnp
######--GP- Functions--######

# RBF kernel for Gaussian-Process
@jax.jit
def custom_rbf_kernel(x1, x2, lengthscale, variance):
    x1 = x1 / lengthscale
    x2 = x2 / lengthscale
    sqdist = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    return variance * jnp.exp(-0.5 * sqdist)

# Posterior computation for Gaussian-Process
@jax.jit
def gp_posterior(X_pred, X_train, y_train, lengthscale, variance, noise):
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