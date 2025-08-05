
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
analyze_parameter_cov.py

Author: Markus Flicker
Date: 2023-08-05
#~~~~~~~~~~~~~~~~ helper_functions analyze_parameter_cov.py~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MF~~~~~#

# Filename: helper_functions/analyze_parameter_cov.py
# Author: Markus Flicker
# Date: 2023-08-05
# Description: 
#           This module provides helper functions for analyzing the covariance matrix of parameter samples,
#            typically from a Gaussian process model or MCMC sampling. It includes functions for computing
#            the covariance matrix and performing eigenvalue decomposition using JAX for efficient computation.
#            Functions:
#            ----------
#            covariance_matrix(samples: jnp.ndarray) -> jnp.ndarray
#                Computes the covariance matrix for a set of samples with shape (N, D), where N is the number
#                of samples and D is the number of parameters. Returns a (D, D) covariance matrix.
#            eigen_decomposition(cov: jnp.ndarray)
#                Performs eigenvalue decomposition on a symmetric covariance matrix, returning the eigenvalues
#               and eigenvectors.
#           Usage:
#            ------
#            - Loads parameter samples from a .npz file.
#            - Stacks the samples into a parameter matrix.
#            - Computes the covariance matrix of the parameters.
#            - Performs eigenvalue decomposition to analyze principal directions of variance.
#            - Optionally reconstructs the covariance matrix from its eigen-decomposition for validation.
#            Dependencies:
#            -------------
#            - os
#            - glob
#            - pandas
#            - matplotlib.pyplot
#            - jax
#            - jax.numpy
#            - numpy


import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import numpy as np


@jax.jit
def covariance_matrix(samples: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the covariance matrix of the given samples.
    Args:
        samples (jnp.ndarray): A 2D array of shape (N, D), where N is the number of samples and D is the number of parameters.
    Returns:
        jnp.ndarray: The covariance matrix of shape (D, D).
    Notes:
        - The function centers the data by subtracting the mean of each parameter.
        - The covariance is computed as (centered.T @ centered) / (N - 1), where N is the number of samples.
    """
    
    # Mittelwert über die Samples (Achse 0 -> pro Parameter)
    mean = jnp.mean(samples, axis=0, keepdims=True)  # Shape (1, D)
    
    # Zentrierte Daten
    centered = samples - mean  # Shape (N, D)
    
    # Kovarianzmatrix: (D x D)
    cov = (centered.T @ centered) / (samples.shape[0] - 1)
    return cov

@jax.jit
def eigen_decomposition(cov: jnp.ndarray):
    """
    Performs eigenvalue decomposition of a symmetric covariance matrix.
    Given a symmetric covariance matrix `cov`, this function computes its eigenvalues and eigenvectors using `jnp.linalg.eigh`, which is optimized for Hermitian (symmetric) matrices. The decomposition satisfies: cov = W Λ W^T, where W contains the eigenvectors and Λ is a diagonal matrix of eigenvalues.
    Args:
        cov (jnp.ndarray): Symmetric covariance matrix of shape (N, N).
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: 
            - eigvals: Array of eigenvalues in ascending order.
            - eigvecs: Matrix whose columns are the normalized eigenvectors.
    Eigenwertzerlegung der Kovarianzmatrix: C = W Λ W^T
    """
    
    # Symmetrische Matrix -> eigh (statt eig) nutzen
    eigvals, eigvecs = jnp.linalg.eigh(cov)  # liefert aufsteigende Eigenwerte
    
    # Eigenwerte und Eigenvektoren in absteigender Reihenfolge sortieren
    #idx = jnp.argsort(eigvals)[::-1]
    #eigvals = eigvals[idx]
    #eigvecs = eigvecs[:, idx]
    
    return eigvals, eigvecs

# Directory with the input files
input_dir = "tests/output_pcb_trace_impedance"

# Specific file to read
fname = os.path.join(input_dir, "mcmc_samples__1.npz")
print(f"Reading file: {fname}")

# Load the samples from the .npz file
samples = jnp.load(fname)

# Stack samples together
samples_param_matrix = jnp.column_stack([samples[p].flatten() for p in samples.files])

# Calculate covariance matrix - returns a (D, D) matrix 
cov_matrix = covariance_matrix(samples_param_matrix)

# Caluclate desending sorted eigenvalues and eigenvectors
eigvals, eigvecs = eigen_decomposition(cov_matrix)

print("Kovarianzmatrix C:\n", cov_matrix)
print("\nEigenwerte Λ:\n", eigvals)
print("\n Parameter-Strings:\n", samples.files)

# Kontrolle: Rekonstruiere C ≈ W Λ W^T
C_reconstructed = eigvecs @ jnp.diag(eigvals) @ eigvecs.T
print("\nRekonstruktionsfehler:", jnp.linalg.norm(cov_matrix - C_reconstructed))
