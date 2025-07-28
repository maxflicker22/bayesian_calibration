import jax.numpy as jnp
import os
import arviz
import numpy as np
from typing import Callable, List, Dict, Optional
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary
import jax
import arviz as az
import matplotlib.pyplot as plt

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

    Methods:
    
        _build_model():
            Sets up the model with the necessary parameters for sampling.
    """
    
    def __init__(
        self,
        model_function: Callable[..., tuple],
        model_parameters_string: List[str],
        observed_data: jnp.ndarray,
        model_function_parameters: Optional[Dict] = None
        ):
        """
        Args:
            model_function (Callable): Function that returns (mean, cov)
            model_parameters_string (list of str): Names of sampled parameters
            observed_data (jax.numpy.ndarray): Target data
            model_function_parameters (dict, optional): Additional fixed parameters passed to the model_function
        """
        self.model_function = model_function
        self.model_parameters_string = model_parameters_string
        self.observed_data = jnp.asarray(observed_data)
        self.model_function_parameters = model_function_parameters or {}
        self.output_dir = "mcmc_output"
        
        # Set default prior: Normal(0.5, 0.2) with truncation [0, 1]
        self.normal_prior = True
        self.prior_range = (0.0, 1.0)
        
        # Default Prior settings for parameters
        param_count = len(self.model_parameters_string)
        self.prior_mean = jnp.full((param_count,), 0.5)
        self.prior_std = jnp.full((param_count,), 0.2)
        self.white_noise_std = 0.2
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def adjust_prior(self, prior_mean=None, prior_std=None, prior_range=None, white_noise_std=None, use_uniform_prior=True):
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
        kernel = NUTS(self.baysian_model)
        mcmc = MCMC(
            kernel,
            num_samples=num_samples or 50000,
            num_warmup=num_warmup or (num_samples // 5),
            num_chains=num_chains or 1,
        )

        rng_key = jax.random.PRNGKey(rng_seed)
        mcmc.run(rng_key)  
        mcmc.print_summary()
        
        # Get samples
        samples = mcmc.get_samples(group_by_chain=True)

        # Save samples as .npz
        npz_fname = self.get_unique_filename(self.output_dir, "mcmc_samples", para_string="", ext=".npz")
        np.savez(npz_fname, **{k: np.array(v) for k, v in samples.items()})

        # Convert to InferenceData and save as .nc
        idata = az.from_numpyro(mcmc)
        idata_fname = self.get_unique_filename(self.output_dir, "mcmc_results", para_string="", ext=".nc")
        idata.to_netcdf(idata_fname)

        # Compute and save summary
        summary = az.summary(idata)
        summary_fname = self.get_unique_filename(self.output_dir, "mcmc_summary", para_string="", ext=".csv")
        summary.to_csv(summary_fname)
        
        # Plot and save chain stats
        for para_string in self.model_parameters_string:
            self.plot_chain_stats(idata, para_string)
        
    def plot_chain_stats(self, idata, para_string):
        try:
            # Define output directory
            figures_dir = os.path.join(self.output_dir, "figures")
            os.makedirs(figures_dir, exist_ok=True)

            # --- Autocorrelation plot ---
            az.plot_autocorr(idata, var_names=[para_string])
            plt.title(rf"Autokorrelation für ${para_string}$")
            plt.tight_layout()
            fname = self.get_unique_filename(figures_dir, "autocorr", para_string)
            plt.savefig(fname, dpi=150)
            plt.close()

            # --- Traceplot ---
            az.plot_trace(idata, var_names=[para_string])
            plt.suptitle(rf"Traceplot für ${para_string}$", fontsize=16)
            plt.tight_layout()
            fname = self.get_unique_filename(figures_dir, "trace", para_string)
            plt.savefig(fname, dpi=150)
            plt.close()

            # --- Rankplot ---
            az.plot_rank(idata, var_names=[para_string])
            plt.title(rf"Rankplot für ${para_string}$")
            plt.tight_layout()
            fname = self.get_unique_filename(figures_dir, "rank", para_string)
            plt.savefig(fname, dpi=150)
            plt.close()
            
        except Exception as e:
            print(f"Caught an error on {para_string}: {e}")
            
    def get_unique_filename(self, base_path, prefix, para_string, ext=".png"):
        i = 1
        while True:
            filename = f"{prefix}_{para_string}_{i}{ext}"
            full_path = os.path.join(base_path, filename)
            if not os.path.exists(full_path):
                return full_path
            i += 1