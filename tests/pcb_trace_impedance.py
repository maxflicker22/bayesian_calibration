#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~ Tests - PCB-Trace-Impedance ~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MF~~~~~#

# Filename: tests/pcb_trace_impedance.py
# Author: Markus Flicker
# Date: 2023-08-05
# Description:
#           This script tests the functionality of the BayesCalibrator class
#           for calibrating parameters of the impedance of an pcb trace using Gaussian Process regression.
#           It generates synthetic observed data, trains a Gaussian Process emulator,
#           performs Bayesian inference using MCMC sampling, and compares the true and found parameters.
#
# !Note!:
#           In this example, no unique solution is found. Cause we pointing from 4 parameters to 1 output the
#           system is underdetermined. The posterior distribution of the parameters is not as sharp as in the complete case.


#-----------------------------------------------------------------------------------------------------
######################--Imports--######################
import os
import arviz as az
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yaml
from bacali.sampler import BayesCalibrator


def set_jax_platform():
    try:
        import importlib
        # Nur das jaxlib.xla_extension Modul laden, nicht JAX selbst
        xla = importlib.import_module("jaxlib.xla_extension")
        # Prüfen, ob eine GPU sichtbar ist
        gpu_devices = [d for d in xla.get_local_devices() if d.device_kind == "Gpu"]
        if gpu_devices:
            os.environ["JAX_PLATFORM_NAME"] = "gpu"
            print("GPU gefunden und als Standard-Device gesetzt.")
        else:
            print("Keine GPU gefunden. Nutze CPU.")
    except Exception as e:
        print("Konnte jaxlib.xla_extension nicht importieren:", e)
        print("Nutze CPU.")

set_jax_platform()

import jax
# set double precision for JAX
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, cho_factor

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary
from sklearn.preprocessing import MinMaxScaler
from pyDOE import lhs
import json
from helper_functions.functions import load_pcb_config, min_max_scale_jax
from helper_functions.functions import inverse_min_max_scale_jax, impedance_pcb_trace
from helper_functions.gaussian_process_functions import gp_posterior
from helper_functions.analysis import store_model_values_results, store_posterior_covariance, mean_comparison

import bacali
print(bacali.__file__)
from bacali import BayesCalibrator
print(BayesCalibrator.__init__.__code__.co_varnames)

# print available devices for parallel
print(jax.devices())


    
#-----------------------------------------------------------------------------------------------------
######################--Load True Material Data--######################
config = load_pcb_config('tests/config_pcb_trace_impedance.yaml')

# Store true values from config for later use
true_eps_r = config['pcb_trace']['epsilon_r']
true_h     = config['pcb_trace']['height_mm']
true_t     = config['pcb_trace']['thickness_mm']
true_w     = config['pcb_trace']['width_mm']
true_material = config['material']['name']

# Load ranges from config
ranges = config['pcb_trace_ranges']
eps_r_min, eps_r_max = ranges['epsilon_r']
h_min, h_max = ranges['height_mm']
t_min, t_max = ranges['thickness_mm']
w_min, w_max = ranges['width_mm']
impedance_min, impedance_max = ranges['impedance']

# Print of true values
print("Trace width (mm):", true_w)
print("Trace thickness (mm):", true_t)
print("Substrate height (mm):", true_h)
print("Epsilon r:", true_eps_r)
print("Material:", true_material)

#-----------------------------------------------------------------------------------------------------
######################--Generate Observed-Data--######################

# Number of Observed samples
n_samples = 1

# Scale true values using your scaling function
scaled_true_eps_r = min_max_scale_jax(true_eps_r, eps_r_min, eps_r_max)
scaled_true_h     = min_max_scale_jax(true_h, h_min, h_max)
scaled_true_t     = min_max_scale_jax(true_t, t_min, t_max)
scaled_true_w     = min_max_scale_jax(true_w, w_min, w_max)

# Parameter strings
para_strings = ["eps_scaled", "h_scaled", "t_scaled", "w_scaled"]

# In ein gemeinsames Array stapeln
real_true_paras_obs = jnp.stack([true_eps_r, true_h, true_t, true_w], axis=0)  # shape: (n_samples, n_paras)
scaled_paras_obs = jnp.stack([scaled_true_eps_r, scaled_true_h, scaled_true_t, scaled_true_w], axis=0)# shape: (n_samples, n_paras)

# Generate Observational Data (True)
y_obs_noise_free = jnp.array([impedance_pcb_trace(*real_true_paras_obs)])

# Add White Noise
key_noise = jax.random.PRNGKey(43)
noise = 0.1000 * jax.random.normal(key_noise, shape=n_samples)
true_y_obs = y_obs_noise_free + noise

# scale y_obs values between 0 and 1
scaled_y_obs = min_max_scale_jax(true_y_obs, impedance_min, impedance_max)

#-----------------------------------------------------------------------------------------------------
######################--Generate Training-Data--######################

# Number of training Points and Parameters
n_train = 110
n_params = real_true_paras_obs.shape

# LHS in Range [0,1] for every Dimension
x_lhs = lhs(n=n_params[0], samples=n_train)  # shape (n_train, n_params)
x_lhs = jnp.array(x_lhs)

# Store scaled training variables for GP (before inverse scaling)
scaled_x_train = jnp.zeros_like(x_lhs)
scaled_x_train = scaled_x_train.at[:, 0].set(x_lhs[:, 0])  # already in [0,1]
scaled_x_train = scaled_x_train.at[:, 1].set(x_lhs[:, 1])
scaled_x_train = scaled_x_train.at[:, 2].set(x_lhs[:, 2])
scaled_x_train = scaled_x_train.at[:, 3].set(x_lhs[:, 3])

# Transform LHS samples to real parameter space using min/max from config
x_train = jnp.zeros_like(x_lhs)
x_train = x_train.at[:, 0].set(inverse_min_max_scale_jax(x_lhs[:, 0], eps_r_min, eps_r_max))  # eps_r
x_train = x_train.at[:, 1].set(inverse_min_max_scale_jax(x_lhs[:, 1], h_min, h_max))          # h
x_train = x_train.at[:, 2].set(inverse_min_max_scale_jax(x_lhs[:, 2], t_min, t_max))          # t
x_train = x_train.at[:, 3].set(inverse_min_max_scale_jax(x_lhs[:, 3], w_min, w_max))          # w

# Calculate y_train values out of real space x_train
y_train = jnp.array([
    impedance_pcb_trace(*paras)
    for paras in x_train
])

# Skaliere y_train auf [0, 1] mit min_max_scale_jax
scaled_y_train = min_max_scale_jax(y_train, impedance_min, impedance_max)

#-----------------------------------------------------------------------------------------------------
######################--Set Up GP-Model--######################
length_scales = [1e-4, 1.0, 1e-4, 1.0]
length_bounds = [(1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2)]  # jetzt korrekt: 2 Tupel für 2 Parameter

# Define a 2D RBF kernel with separate length scales per dimension
rbf_kernel = RBF(length_scale=length_scales, length_scale_bounds=length_bounds)

# Add signal variance (amplitude) and noise
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * rbf_kernel + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1))

gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

#-----------------------------------------------------------------------------------------------------
######################--Train GP Emulator--######################

# Fit the Gaussian Process model to the training data
gp.fit(scaled_x_train, scaled_y_train)

# Extrahiere t Hyperparameter
lengthscale = gp.kernel_.k1.k2.length_scale
variance = gp.kernel_.k1.k1.constant_value
gp_noise = gp.kernel_.k2.noise_level

# Convert to jax
lengthscale = jnp.array(lengthscale, dtype=float)
variance = jnp.array(variance, dtype=float)
gp_noise = jnp.array(gp_noise, dtype=float)

#-----------------------------------------------------------------------------------------------------
######################--Baysian Inference & MCMC-Sampling--######################

Bacali = BayesCalibrator(
    model_function=gp_posterior,
    model_parameters_string=para_strings,
    model_function_parameters={
        "X_train": scaled_x_train,
        "y_train": scaled_y_train,
        "lengthscale": lengthscale,
        "variance": variance,
        "noise": gp_noise
    },
    observed_data=scaled_y_obs,
    output_dir="tests/output_pcb_trace_impedance")

# Optional: adjust Prior 
Bacali.adjust_prior(use_uniform_prior=True) # Uniform prior for all parameters

# Sample from chain
Bacali.sample_from_chain(num_samples=5000, num_chains=2)


#-----------------------------------------------------------------------------------------------------
######################--Plotting-Results--######################

# --- GP Prediction ---
scaled_y_pred_mean, scaled_y_pred_std = gp.predict(scaled_paras_obs.reshape(1, -1), return_std=True)

scaled_eps = scaled_paras_obs[0]
scaled_h = scaled_paras_obs[1]
scaled_t = scaled_paras_obs[2]
scaled_w = scaled_paras_obs[3]

# Re-transform GP predictions from [0, 1] back to real impedance space
y_pred_mean_real = inverse_min_max_scale_jax(scaled_y_pred_mean, impedance_min, impedance_max)
y_pred_std_real = scaled_y_pred_std * (impedance_max - impedance_min)  # Varianz/Std skaliert mit der Range!

real_true_y_obs = jnp.mean(true_y_obs)

mae = abs(float(real_true_y_obs) - float(y_pred_mean_real.item()))
rmse = mae  # Für einen einzelnen Wert sind MAE und RMSE identisch

mae_scaled = abs(float(scaled_y_obs.item()) - float(scaled_y_pred_mean.item()))

print("\n--- GP Emulator Prediction Error ---")
print(f"Absolute Error:       {mae:.4f}")
print(f"Absolute Error scaled range:       {mae_scaled:.4f}")
print()

# --- Compare Impedance ---
# Load Mcmc summarize Results
mcmc_summary = Bacali.last_mcmc_summary
print(mcmc_summary["mean"][:-1])

# Helper functions (assumed to be defined elsewhere)
# inverse_min_max_scale_jax, impedance_pcb_trace, true_y_obs,
# eps_r_min, eps_r_max, h_min, h_max, t_min, t_max, w_min, w_max

# Get posterior mean for each scaled parameter
posterior_means = mcmc_summary["mean"]

# Rescale to real parameter space
found_eps_r = inverse_min_max_scale_jax(posterior_means["eps_scaled"], eps_r_min, eps_r_max)
found_h     = inverse_min_max_scale_jax(posterior_means["h_scaled"], h_min, h_max)
found_t     = inverse_min_max_scale_jax(posterior_means["t_scaled"], t_min, t_max)
found_w     = inverse_min_max_scale_jax(posterior_means["w_scaled"], w_min, w_max)

found_parameters = jnp.stack([found_eps_r, found_h, found_t, found_w])

# Calculate impedance for found parameters
found_impedance = impedance_pcb_trace(found_eps_r, found_h, found_t, found_w)
true_y_obs = impedance_pcb_trace(true_eps_r, true_h, true_t, true_w)

# Values for plotting
true_val = float(true_y_obs)
found_val = float(found_impedance)
abs_error = abs(true_val - found_val)
rel_error = abs_error / true_val

# Create bar plot
labels = ["True Impedance", "Found Impedance"]
values = [true_val, found_val]

plt.figure(figsize=(6, 5))
plt.bar(labels, values, color=["green", "blue"], alpha=0.7)
plt.ylabel("Impedance")
plt.title("Impedance Comparison")

# Annotate bars with values
for i, v in enumerate(values):
    plt.text(i, v + 0.01 * max(values), f"{v:.4f}", ha='center', fontsize=10)

# Add text for error values and found parameters
param_text = "\n".join([f"{name}: {val:.4f}" for name, val in zip(para_strings, found_parameters)])
plt.figtext(0.5, -0.2,
            f"Absolute Error: {abs_error:.4f} | Relative Error: {rel_error:.4f}\nFound Parameters:\n{param_text}",
            wrap=True, horizontalalignment='center', fontsize=10)

# Save plot
plt.tight_layout()
plt.savefig("tests/output_pcb_trace_impedance/impedance_comparison.png", dpi=150, bbox_inches="tight")
plt.close()





