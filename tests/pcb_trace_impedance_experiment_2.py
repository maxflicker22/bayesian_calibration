
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
from helper_functions.pcb_trace_impedance_functions import load_pcb_config, min_max_scale_jax
from helper_functions.pcb_trace_impedance_functions import inverse_min_max_scale_jax, impedance_pcb_trace
from helper_functions.gaussian_process_functions import gp_posterior

# Load config and parameter ranges
config = load_pcb_config('tests/config_pcb_trace_impedance.yaml')
ranges = config['pcb_trace_ranges']
eps_r_min, eps_r_max = ranges['epsilon_r']
h_min, h_max = ranges['height_mm']
t_min, t_max = ranges['thickness_mm']
w_min, w_max = ranges['width_mm']
impedance_min, impedance_max = ranges['impedance']

para_strings = ["eps_scaled", "h_scaled", "t_scaled", "w_scaled"]

# Generate Training-Data for GP
n_train = 110
n_params = 4

x_lhs = lhs(n=n_params, samples=n_train)
x_lhs = jnp.array(x_lhs)

scaled_x_train = x_lhs

x_train = jnp.zeros_like(x_lhs)
x_train = x_train.at[:, 0].set(inverse_min_max_scale_jax(x_lhs[:, 0], eps_r_min, eps_r_max))
x_train = x_train.at[:, 1].set(inverse_min_max_scale_jax(x_lhs[:, 1], h_min, h_max))
x_train = x_train.at[:, 2].set(inverse_min_max_scale_jax(x_lhs[:, 2], t_min, t_max))
x_train = x_train.at[:, 3].set(inverse_min_max_scale_jax(x_lhs[:, 3], w_min, w_max))

y_train = jnp.array([impedance_pcb_trace(*paras) for paras in x_train])
scaled_y_train = min_max_scale_jax(y_train, impedance_min, impedance_max)

# Dummy GP Hyperparameter (replace with your GP fit if needed)
length_scales = [1e-4, 1.0, 1e-4, 1.0]
variance = 1.0
gp_noise = 1e-2

def run_experiment_2():
    # 10 Werte von 0.9 bis 0.1 (inklusive) für scaled_y_obs
    scaled_y_obs_values = jnp.linspace(0.8, 0.2, 10)

    for idx, scaled_y_obs in enumerate(scaled_y_obs_values):
        print(f"Run {idx+1}: scaled_y_obs = {scaled_y_obs:.2f}")

        Bacali = BayesCalibrator(
            model_function=gp_posterior,
            model_parameters_string=para_strings,
            model_function_parameters={
                "X_train": scaled_x_train,
                "y_train": scaled_y_train,
                "lengthscale": jnp.array(length_scales),
                "variance": jnp.array(variance),
                "noise": jnp.array(gp_noise)
            },
            observed_data=jnp.array([scaled_y_obs])
        )

        Bacali.adjust_prior(use_uniform_prior=True)
        Bacali.sample_from_chain(num_samples=100000, num_chains=4)

        mcmc_summary = Bacali.last_mcmc_summary
        posterior_means = mcmc_summary["mean"]

        found_eps_r = inverse_min_max_scale_jax(posterior_means["eps_scaled"], eps_r_min, eps_r_max)
        found_h     = inverse_min_max_scale_jax(posterior_means["h_scaled"], h_min, h_max)
        found_t     = inverse_min_max_scale_jax(posterior_means["t_scaled"], t_min, t_max)
        found_w     = inverse_min_max_scale_jax(posterior_means["w_scaled"], w_min, w_max)
        found_impedance = impedance_pcb_trace(found_eps_r, found_h, found_t, found_w)
        
         # Berechne den echten (realen) Impedanzwert zu diesem scaled_y_obs
        true_impedance = inverse_min_max_scale_jax(scaled_y_obs, impedance_min, impedance_max)

        Bacali.store_model_values_results(
            true_value=float(true_impedance),
            found_value=float(found_impedance)
        )

        Bacali.store_configurations_within_delta(
            true_model_function=impedance_pcb_trace,
            ref_value=float(found_impedance),
            delta=0.01,
            inverse_fn=inverse_min_max_scale_jax,
            params=[(eps_r_min, eps_r_max),
                    (h_min, h_max),
                    (t_min, t_max),
                    (w_min, w_max)],
            fname_suffix=f"found_impedance_{idx+1}"
        )

if __name__ == "__main__":
    run_experiment_2()