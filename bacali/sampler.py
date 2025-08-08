#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~ BaCali - BayesCalibrator~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MF~~~~~#

# Filename: bacali/sampler.py
# Author: Markus Flicker
# Date: 2023-08-05
# Description:
#   This script contains the BayesCalibrator class, which performs Bayesian model calibration
#   using Hamiltonian Markov Chain Monte Carlo (HMC) with NumPyro.
#   It allows for flexible model definitions (model_function, prior distributions, etc.)


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

import os
import arviz
import numpy as np
from typing import Callable, List, Dict, Optional
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary


print(jax.__version__)   # sollte zu jaxlib passen
import jaxlib
print(jaxlib.__version__)
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

class BayesCalibrator:
    """
    Performs Bayesian model calibration using Hamiltonian Markov Chain Monte Carlo.

    This class samples from the unknown posterior distribution of parameter of an 
    underlying model based on the provided observed Data.
    

    Attributes:
    
        model_function : Callable
            The function representing the system to calibrate.
            It must return two JAX arrays: the predictive mean and covariance matrix.
        model_parameters_string : list of str
            List of parameter names as strings, e.g. ["alpha", "beta", "gamma"].

        observed_data : jax.numpy.ndarray
            Observed values to calibrate against.
            Shape (N,) for scalar outputs, or (N, D) for D-dimensional outputs.
            
        model_function_parameters : dict, optional
        
        output_dir : str
            Directory to save MCMC results and figures.
        
        dense_mass : bool
            If True, uses dense mass matrix for NUTS sampling.

    Methods:
    
        _build_model():
            Sets up the model with the necessary parameters for sampling.
    """
    
    def __init__(
        self,
        model_function: Callable[..., tuple],
        model_parameters_string: List[str],
        observed_data: jnp.ndarray,
        model_function_parameters: Optional[Dict] = None,
        output_dir: str = "mcmc_output",
        dense_mass: bool = False):

        # Initialize attributes
        self.model_function = model_function
        self.model_parameters_string = model_parameters_string
        self.observed_data = jnp.asarray(observed_data)
        self.model_function_parameters = model_function_parameters or {}
        self.output_dir = output_dir
        self.dense_mass = dense_mass
        
        # Set default prior: Normal(0.5, 0.2) with truncation [0, 1]
        self.normal_prior = True
        self.prior_range = (0.0, 1.0)
        
        # Default Prior settings for parameters
        param_count = len(self.model_parameters_string)
        self.prior_mean = jnp.full((param_count,), 0.5)
        self.prior_std = jnp.full((param_count,), 0.2)
        self.white_noise_std = 0.001
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory set to: {self.output_dir}")
        
    def set_output_dir(self, output_dir: str):
        """
        Set a custom output directory for MCMC results.
        
        Args:
            output_dir (str): Path to the output directory.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def adjust_prior(self, prior_mean=None, prior_std=None, prior_range=None, white_noise_std=None, use_uniform_prior=False):
        """
        Adjust the prior settings for the model parameters.
        
        Args:
            prior_mean (float or list): Mean of the prior distribution.
            prior_std (float or list): Standard deviation of the prior distribution.
            prior_range (tuple): Range for the prior distribution (low, high).
            white_noise_std (float): Standard deviation of the white noise.
            use_uniform_prior (bool): If True, uses a uniform prior instead of normal.
        """
        
        # Adjust prior mean
        if prior_mean is not None:
            if jnp.isscalar(prior_mean):
                self.prior_mean = jnp.full((len(self.model_parameters_string),), prior_mean)
            elif len(prior_mean) == len(self.model_parameters_string):
                self.prior_mean = jnp.asarray(prior_mean)
            else:
                raise ValueError("prior_mean must be a scalar or an array of the same length as model_parameters_string")
    
        # Adjust prior std
        if prior_std is not None:
            if jnp.isscalar(prior_std):
                self.prior_std = jnp.full((len(self.model_parameters_string),), prior_std)
            elif len(prior_std) == len(self.model_parameters_string):
                self.prior_std = jnp.asarray(prior_std)
            else:
                raise ValueError("prior_std must be a scalar or an array of the same length as model_parameters_string")

        # Adjust prior range
        if prior_range is not None:
            if isinstance(prior_range, (tuple, list)) and len(prior_range) == 2:
                self.prior_range = tuple(prior_range)
            else:
                raise ValueError("prior_range must be a tuple or list of length 2")
            
        # Adjust white noise std
        if white_noise_std is not None:
            if jnp.isscalar(white_noise_std):
                self.white_noise_std = white_noise_std
            else:
                raise ValueError("white_noise_std must be a scalar value")
            
        # Adjust prior type
        if use_uniform_prior:
            self.normal_prior = False
            
    def baysian_model(self):
        """
        Internal method which declares the NumPyro model for MCMC sampling.
        """
        params = {}

        for para_string in self.model_parameters_string:
            params[para_string] = numpyro.sample(
                para_string,
                dist.TruncatedNormal(
                    self.prior_mean[self.model_parameters_string.index(para_string)],
                    self.prior_std[self.model_parameters_string.index(para_string)],
                    low=self.prior_range[0],
                    high=self.prior_range[1]
                ) if self.normal_prior else dist.Uniform(self.prior_range[0], self.prior_range[1])
            )

        # Declare White Noise
        white_noise = numpyro.sample("white_noise", dist.Normal(0, self.white_noise_std))

        # Combine into jnp array
        jnp_params = jnp.array([[params[name] for name in self.model_parameters_string]])

        # Get predictive mean and variance from model
        mu_func, var_func = self.model_function(jnp_params, **self.model_function_parameters)

        # Add White Noise to variance
        sigma = jnp.sqrt(var_func + white_noise ** 2)

        # Likelihood
        numpyro.sample("likelihood", dist.Normal(mu_func, sigma), obs=self.observed_data)
        
    def sample_from_chain(self, num_samples, num_warmup=None, num_chains=None, rng_seed: int = 0):
        """
        Sample from the posterior distribution using MCMC with NUTS.
        
        Args:
            num_samples (int): Number of samples to draw from the posterior.
            num_warmup (int, optional): Number of warmup iterations. Defaults to num_samples // 5.
            num_chains (int, optional): Number of chains to run. Defaults to 1.
            rng_seed (int): Random seed for reproducibility.
        """
        
        # Build the model
        kernel = NUTS(self.baysian_model, dense_mass=self.dense_mass)
        mcmc = MCMC(
            kernel,
            num_samples=num_samples or 50000,
            num_warmup=num_warmup or (num_samples // 5),
            num_chains=num_chains or 1,
        )

        # Run MCMC
        rng_key = jax.random.PRNGKey(rng_seed)
        mcmc.run(rng_key)
        self.last_mcmc = mcmc  
        mcmc.print_summary()
        
        # Get samples | Grouped and flat 
        samples_grouped = mcmc.get_samples(group_by_chain=True)
        self.last_samples_grouped = samples_grouped
        samples_flat = mcmc.get_samples(group_by_chain=False)
        self.last_samples_flat = samples_flat
        
        # Save samples as .npz
        npz_fname = self.get_unique_filename(self.output_dir, "mcmc_samples", para_string="", ext=".npz")
        np.savez(npz_fname, **{k: np.array(v) for k, v in samples_grouped.items()})

        # Convert to InferenceData and save as .nc
        idata = az.from_numpyro(mcmc)
        self.last_mcmc_idata = idata
        # Every File according to an sample share the same end index
        idata_fname = self.get_unique_filename(self.output_dir, "mcmc_results", para_string="", ext=".nc", fixed_index=self.last_sample_index)
        idata.to_netcdf(idata_fname)

        # Compute and save summary
        summary = az.summary(idata)
        self.last_mcmc_summary = summary
        summary_fname = self.get_unique_filename(self.output_dir, "mcmc_summary", para_string="", ext=".csv", fixed_index=self.last_sample_index)
        summary.to_csv(summary_fname)
        
        # Plot and save chain stats
        for para_string in self.model_parameters_string:
            self.plot_chain_stats(idata, para_string)
        
    def plot_chain_stats(self, idata, para_string):
        """
        Plot and save chain statistics for a given parameter.
        
        Args:
            idata: InferenceData object containing MCMC results.
            para_string (str): Name of the parameter to plot.
        """
        try:
            # Define output directory
            figures_dir = os.path.join(self.output_dir, "figures")
            os.makedirs(figures_dir, exist_ok=True)

            # --- Autocorrelation plot ---
            az.plot_autocorr(idata, var_names=[para_string])
            plt.title(rf"Autokorrelation für ${para_string}$")
            plt.tight_layout()
            fname = self.get_unique_filename(figures_dir, "autocorr", para_string, ext=".png", fixed_index=self.last_sample_index)
            plt.savefig(fname, dpi=150)
            plt.close()

            # --- Traceplot ---
            az.plot_trace(idata, var_names=[para_string])
            plt.suptitle(rf"Traceplot für ${para_string}$", fontsize=16)
            plt.tight_layout()
            fname = self.get_unique_filename(figures_dir, "trace", para_string, ext=".png", fixed_index=self.last_sample_index)
            plt.savefig(fname, dpi=150)
            plt.close()

            # --- Rankplot ---
            az.plot_rank(idata, var_names=[para_string])
            plt.title(rf"Rankplot für ${para_string}$")
            plt.tight_layout()
            fname = self.get_unique_filename(figures_dir, "rank", para_string, ext=".png", fixed_index=self.last_sample_index)
            plt.savefig(fname, dpi=150)
            plt.close()
            
        except Exception as e:
            print(f"Caught an error on {para_string}: {e}")
            
    def get_unique_filename(self, base_path, prefix, para_string, ext=".png", fixed_index=None):
        """
        Generate a unique filename in the specified directory.
        If a fixed index is provided, it will be used directly.
        Otherwise, it will increment the index until a unique filename is found.
        
        Args:
            base_path (str): Base directory to save the file.
            prefix (str): Prefix for the filename.
            para_string (str): Parameter string to include in the filename.
            ext (str): File extension.
            fixed_index (int, optional): If provided, use this index directly.
        Returns:
            str: Full path to the unique filename.
        """
        # Check if fixed_index is provided
        if fixed_index is not None:
            # If a fixed index is provided, use it directly
            filename = f"{prefix}_{para_string}_{fixed_index}{ext}"
            return os.path.join(base_path, filename)
        i = 1
        # Increment index until a unique filename is found
        while True:
            filename = f"{prefix}_{para_string}_{i}{ext}"
            full_path = os.path.join(base_path, filename)
            if not os.path.exists(full_path):
                self.last_sample_index = i  # Store the last used index
                return full_path
            i += 1
            
   