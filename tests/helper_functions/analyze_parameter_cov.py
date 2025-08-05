import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import numpy as np

# This script is for analyzing the covariance matrix of parameter samples from a Gaussian process model.



@jax.jit
def covariance_matrix(samples: jnp.ndarray) -> jnp.ndarray:
    """
    Berechnet die Kovarianzmatrix für Samples mit Shape (N, D).
    N = Anzahl Samples, D = Anzahl Parameter.
    Gibt eine (D, D)-Kovarianzmatrix zurück.
    """
    print(f"Shape der Samples: {samples.shape}")  # (N, D)
    
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
    Eigenwertzerlegung der Kovarianzmatrix: C = W Λ W^T
    """
    # Symmetrische Matrix -> eigh (statt eig) nutzen
    eigvals, eigvecs = jnp.linalg.eigh(cov)  # liefert aufsteigende Eigenwerte
    
    # Eigenwerte und Eigenvektoren in absteigender Reihenfolge sortieren
    idx = jnp.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    return eigvals, eigvecs

# Directory with the input files
input_dir = "mcmc_output_capacitor"

# Specific file to read
fname = os.path.join(input_dir, "mcmc_samples__45.npz")
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
print("\nEigenvektoren W (Spalten):\n", eigvecs)

# Kontrolle: Rekonstruiere C ≈ W Λ W^T
C_reconstructed = eigvecs @ jnp.diag(eigvals) @ eigvecs.T
print("\nRekonstruktionsfehler:", jnp.linalg.norm(cov_matrix - C_reconstructed))
