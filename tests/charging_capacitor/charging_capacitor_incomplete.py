
#-----------------------------------------------------------------------------------------------------
######################--Imports--######################
import os
import arviz as az
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import xarray as xr
import pymc as pm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import product
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
from helper_functions.functions import inverse_min_max_scale_jax, charging_capacitor_model, charging_capacitor_taylor_7, charging_capacitor_taylor_3
from helper_functions.functions import charging_capacitor_taylor_2, charging_capacitor_taylor_1
from helper_functions.gaussian_process_functions import gp_posterior

# print available devices for parallel
print(jax.devices())

    
#-----------------------------------------------------------------------------------------------------
######################--Load True Material Data--######################
config = load_pcb_config('tests/config_capacitor.yaml')

# Store true values from config for later use
true_u_0 = config['charging_capacitor']['u_0']
true_tau     = config['charging_capacitor']['tau']

t_a = jnp.array(config['charging_capacitor']['t_a'])
# Load ranges from config
ranges = config['charging_capacitor_ranges']
u_0_min, u_0_max = ranges['u_0']
tau_min, tau_max = ranges['tau']
u_t_min, u_t_max = ranges['u_t']


# Print of true values
print("true_u_0:", true_u_0)
print("true_tau:", true_tau)

#-----------------------------------------------------------------------------------------------------
######################--Generate Observed-Data--######################

# Number of Observed samples
n_samples = 100

# Scale true values using your scaling function
scaled_true_u_0 = min_max_scale_jax(true_u_0, u_0_min, u_0_max)
scaled_true_tau     = min_max_scale_jax(true_tau, tau_min, tau_max)


# Parameter strings
para_strings = ["u0_scaled", "tau_scaled"]

# In ein gemeinsames Array stapeln
real_true_paras_obs = jnp.array([true_u_0, true_tau])  # shape (2,)
scaled_paras_obs = jnp.stack([scaled_true_u_0, scaled_true_tau], axis=0)# shape: (n_samples, n_paras)

# Berechne die "wahre" Modellkurve einmal
y_obs_noise_free = charging_capacitor_model(t_a, *real_true_paras_obs)  # shape: (len(t_a),)

# Erweitere für Broadcasting auf (1, len(t_a))
y_obs_noise_free = y_obs_noise_free[None, :]  # shape: (1, len(t_a))

# Generiere White Noise für alle Samples
key_noise = jax.random.PRNGKey(43)
noise = 0.0001 * jax.random.normal(key_noise, shape=(n_samples, len(t_a)))  # shape: (n_samples, len(t_a))

# Addiere Noise zu jeder Kurve
true_y_obs = y_obs_noise_free + noise  # shape: (n_samples, len(t_a))

# scale y_obs values between 0 and 1
scaled_y_obs = min_max_scale_jax(true_y_obs, u_t_min, u_t_max)

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


# Transform LHS samples to real parameter space using min/max from config
x_train = jnp.zeros_like(x_lhs)
x_train = x_train.at[:, 0].set(inverse_min_max_scale_jax(x_lhs[:, 0], u_0_min, u_0_max))  # eps_r
x_train = x_train.at[:, 1].set(inverse_min_max_scale_jax(x_lhs[:, 1], tau_min, tau_max))          # h

# Calculate y_train values out of real space x_train
y_train = jnp.array([
    charging_capacitor_taylor_1(t_a, *paras)
    for paras in x_train
])

# Skaliere y_train auf [0, 1] mit min_max_scale_jax
scaled_y_train = min_max_scale_jax(y_train, u_t_min, u_t_max)

#-----------------------------------------------------------------------------------------------------
######################--Set Up GP-Model--######################
length_scales = [1e-4, 1.0]
length_bounds = [(1e-2, 1e2), (1e-2, 1e2)]  # jetzt korrekt: 2 Tupel für 2 Parameter

# Define a 2D RBF kernel with separate length scales per dimension
rbf_kernel = RBF(length_scale=length_scales, length_scale_bounds=length_bounds)

# Add signal variance (amplitude) and noise
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * rbf_kernel + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1))

gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

#-----------------------------------------------------------------------------------------------------
######################--Train GP Emulator--######################
print("scaled_y_train:", scaled_y_train)
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
    output_dir="mcmc_output_capacitor")

# Optional: adjust Prior 
Bacali.adjust_prior() # Uniform prior for all parameters

# Sample from chain
Bacali.sample_from_chain(num_samples=10000, num_chains=4)


#-----------------------------------------------------------------------------------------------------
######################--Plotting-Results--######################

# --- GP Prediction ---
scaled_y_pred_mean, scaled_y_pred_std = gp.predict(scaled_paras_obs.reshape(1, -1), return_std=True)

scaled_u_0 = scaled_paras_obs[0]
scaled_tau = scaled_paras_obs[1]

# Re-transform GP predictions from [0, 1] back to real impedance space
y_pred_mean_real = inverse_min_max_scale_jax(scaled_y_pred_mean, u_t_min, u_t_max)
y_pred_std_real = scaled_y_pred_std * (u_t_max - u_t_min)  # Varianz/Std skaliert mit der Range!

real_true_y_obs = jnp.mean(true_y_obs)
y_pred_mean_samples_real = jnp.mean(y_pred_mean_real)


mae = abs(float(real_true_y_obs) - float(y_pred_mean_samples_real))
#mae_scaled = abs(float(scaled_y_obs.item()) - float(scaled_y_pred_mean.item()))

print("\n--- GP Emulator Prediction Error ---")
print(f"Absolute Error:       {mae:.4f}")
#print(f"Absolute Error scaled range:       {mae_scaled:.4f}")
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

# Mean comparison
Bacali.mean_comparison(
    mean_true_parameters=scaled_paras_obs,
    posterior_sampled_means=posterior_means,
    param_names=Bacali.model_parameters_string)

# Rescale to real parameter space
found_u_0 = inverse_min_max_scale_jax(posterior_means["u0_scaled"], u_0_min, u_0_max)
found_tau     = inverse_min_max_scale_jax(posterior_means["tau_scaled"], tau_min, tau_max)


# Calculate impedance for found parameters
found_u_t = charging_capacitor_taylor_1(t_a, found_u_0, found_tau).squeeze()
true_y_obs = charging_capacitor_taylor_1(t_a, true_u_0, true_tau).squeeze()

# Compare to true impedance
print("\n--- Impedance Comparison ---")

print(f"True U(t)):      {true_y_obs}")
print(f"Found U(t):     {found_u_t}")
print(f"Absolute Error:      {abs(true_y_obs - found_u_t)}")
print(f"Relativ Error:      {abs(true_y_obs - found_u_t)/abs(true_y_obs)}")


# Save Data for experiments
# Store True, Found, Covariance
Bacali.store_model_values_results(
    true_value=true_y_obs,
    found_value=found_u_t
)


# Plot Model: Compare true vs. found parameters, plot vertical lines for all t_a values

# Zeitbereich für Plot
t_plot = np.linspace(0, float(t_a.max()) + 0.1, 200)

# Berechne Spannung für true und found Parameter über t_plot
v_true = charging_capacitor_taylor_1(t_plot, float(true_u_0), float(true_tau))
v_found = charging_capacitor_taylor_1(t_plot, float(found_u_0), float(found_tau))

plt.figure(figsize=(8, 5))
plt.plot(t_plot, v_true, label=f"True: $u_0$={float(true_u_0):.3f}, $\\tau$={float(true_tau):.3f}", color="green", linewidth=2)
plt.plot(t_plot, v_found, label=f"Found: $u_0$={float(found_u_0):.3f}, $\\tau$={float(found_tau):.3f}", color="red", linestyle="--", linewidth=2)

# Vertikale Linien für alle t_a Werte
for t_val in np.array(t_a):
    plt.axvline(float(t_val), color='blue', linestyle=':', alpha=0.7, label='t_a' if t_val == t_a[0] else None)
    plt.scatter(t_a, true_y_obs, color='green', marker='o', label='True Observations' if t_val == t_a[0] else None, s=50)
    plt.scatter(t_a, y_pred_mean_real, color='black', marker='x', label='GP Prediction' if t_val == t_a[0] else None, s=50)


plt.xlabel("Time $t$")
plt.ylabel("Voltage $U(t)$")
plt.title(f"Comparison of True vs. Found Parameters Over Time With {len(t_a)} Values")
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save plot as PNG in output directory
output_dir = "mcmc_output_capacitor"
plt.savefig(os.path.join(output_dir, f"true_vs_found_model_comparison_{len(t_a)}_time_values.png"))
plt.show()

"""
# Store configurations within 1% of found impedance
Bacali.store_configurations_within_delta(
    true_model_function=charging_capacitor_model,
    ref_value=found_u_t,
    true_model_func_params=(t_a),
    delta=0.01,
    inverse_fn=inverse_min_max_scale_jax,
    params=[(u_0_min, u_0_max),
      (tau_min, tau_max)],
    fname_suffix="found_u_t"
)

# Store configurations within 1% of found impedance
Bacali.store_configurations_within_delta(
    true_model_function=charging_capacitor_model,
    ref_value=true_y_obs,
    delta=0.01,
    true_model_func_params=(t_a),
    inverse_fn=inverse_min_max_scale_jax,
    params=[(u_0_min, u_0_max),
      (tau_min, tau_max)],
    fname_suffix="true_u_t"
)

"""