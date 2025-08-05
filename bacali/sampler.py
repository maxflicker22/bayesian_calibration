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
        
    def adjust_prior(self, prior_mean=None, prior_std=None, prior_range=None, white_noise_std=None, use_uniform_prior=False):
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
        self.last_mcmc = mcmc  
        mcmc.print_summary()
        
        # Get samples
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
        if fixed_index is not None:
            # If a fixed index is provided, use it directly
            filename = f"{prefix}_{para_string}_{fixed_index}{ext}"
            return os.path.join(base_path, filename)
        i = 1
        while True:
            filename = f"{prefix}_{para_string}_{i}{ext}"
            full_path = os.path.join(base_path, filename)
            if not os.path.exists(full_path):
                self.last_sample_index = i  # Store the last used index
                return full_path
            i += 1
            
    # Analysis of the means (true and posterior)
    def mean_comparison(self, mean_true_parameters, posterior_sampled_means, param_names):
        """
        Compare true means vs posterior means for multiple parameters and save results to a txt file.
        
        Args:
            mean_true_parameters (array-like): Array of true mean values.
            posterior_sampled_means (array-like): Array of posterior mean values.
            param_names (list): Names of the parameters (same length as means).
            outdir (str): Output directory.
        """
        # Prepare result string
        result_lines = ["\n--- Mean Comparison - Scaled Range ---\n"]
        for name, true_mean, post_mean in zip(param_names, mean_true_parameters, posterior_sampled_means):
            abs_diff = abs(true_mean - post_mean)
            result_lines.append(
                f"True mean of {name} (observations):      {true_mean:.4f}\n"
                f"Posterior mean of {name} (samples):     {post_mean:.4f}\n"
                f"Absolute difference ({name}):           {abs_diff:.4f}\n"
                "------------------------------------------------------------\n"
            )

        result = "".join(result_lines)

        # Print to console
        #print(result)

        # Save to file
        fname = self.get_unique_filename(self.output_dir, "mean_comparison", para_string="", ext=".txt", fixed_index=self.last_sample_index)
        with open(fname, "w") as f:
            f.write(result)
            
    # Experiment 1 - Store true and found model values
    def store_model_values_results(self, true_value, found_value, fname_suffix=""):
        """
        Store true value, found value, and covariance matrix of posterior samples in CSV.
        """
        # 1. Save True and Found impedance + covariance matrix
        data = {
            "True_Value": [true_value],
            "Found_Value": [found_value]
        }
        df_main = pd.DataFrame(data)
        

        # Create output folder
        analysis_model_values_dir = os.path.join(self.output_dir, "model_analysis/experiment_1/model_values")
        os.makedirs(analysis_model_values_dir, exist_ok=True)
        
        # Store impedance values
        fname = self.get_unique_filename(analysis_model_values_dir, "model_values", para_string=fname_suffix, ext=".csv", fixed_index=self.last_sample_index)
        df_main.to_csv(fname, index=False)
        print(f"Stored model values to: {fname}")
    
        
    def store_posterior_covariance(self, fname_suffix=""):
        """
        Store covariance matrix of posterior samples in CSV.
        """
        # 1. Covariance matrix from posterior samples
        param_names = self.model_parameters_string  # e.g. ["eps_scaled", "h_scaled", ...]
        samples = self.last_samples_flat  # Use the last flat samples
        param_matrix = jnp.column_stack([samples[p] for p in param_names])
        cov_matrix = jnp.cov(param_matrix, rowvar=False)
        
        # 2. Save covariance matrix as DataFrame
        cov_df = pd.DataFrame(cov_matrix, index=param_names, columns=param_names)
        
        # Create output folder
        analysis_posterior_covariance_dir = os.path.join(self.output_dir, "model_analysis/experiment_1/posterior_covariance")
        os.makedirs(analysis_posterior_covariance_dir, exist_ok=True)
        
        # Store covariance matrix separately
        fname = self.get_unique_filename(analysis_posterior_covariance_dir, "posterior_covariance", para_string="", ext=".csv", fixed_index=self.last_sample_index)
        cov_df.to_csv(fname, index=False)
        print(f"Stored model values and covariance matrix in '{analysis_posterior_covariance_dir}'")

        
        
        
    def store_configurations_within_delta(self, true_model_function, ref_value, delta=0.01, inverse_fn=None, params=None, fname_suffix=""):
        """
        Check all posterior samples and store those configurations whose impedance is within ±delta of ref_value.
        Allows passing a dynamic inverse scaling function with *params for flexibility.
        """
        # Extract posterior samples
        samples = self.last_samples_flat  # Use the last flat samples
        param_names = self.model_parameters_string

        # Dynamically inverse transform parameters if inverse_fn is provided
        if inverse_fn is not None and params is not None:
            real_params = [inverse_fn(samples[p], *pp) for p, pp in zip(param_names, params)]
        else:
            # If no inverse_fn is provided, keep parameters as they are
            real_params = [samples[p] for p in param_names]

        # Compute impedance for all posterior samples (unpack real parameters)
        model_values = true_model_function(*real_params)

        # Filter configurations within delta interval
        lower = ref_value * (1 - delta)
        upper = ref_value * (1 + delta)
        mask = (model_values >= lower) & (model_values <= upper)

        # Collect matching configurations dynamically
        config_dict = {p: jnp.array(r)[mask] for p, r in zip(param_names, real_params)}
        config_dict["model_values"] = jnp.array(model_values)[mask]
        config_dict["rel_error"] = jnp.abs((config_dict["model_values"] - ref_value) / ref_value)

        matching_configs = pd.DataFrame(config_dict)

        # Create output directory
        analysis_dir = os.path.join(self.output_dir, "model_analysis/experiment_1/posterior_configs")
        os.makedirs(analysis_dir, exist_ok=True)

        # Store results in CSV
        fname = self.get_unique_filename(
            analysis_dir, "posterior_configs_within_delta", para_string=fname_suffix, ext=".csv", fixed_index=self.last_sample_index
        )
        matching_configs.to_csv(fname, index=False)

        print(f"Found {len(matching_configs)} configurations within ±{delta*100:.1f}% of reference impedance.")
        print(f"Saved results to: {analysis_dir}")


