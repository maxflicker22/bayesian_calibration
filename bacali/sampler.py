import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary
import jax
import jax.numpy as jnp
import os
import aviz
import numpy as np

class BayesCalibrator:
    """
    Performs Bayesian model calibration using Hamiltonian Markov Chain Monte Carlo.

    This class samples from the unknown posterior distribution of parameter of an 
    underlying model based on the provided observed Data.
    

    Attributes:
    
        model_function : Callable
            The function representing the system to calibrate.
            It must return two JAX arrays: the predictive mean and covariance matrix.
            W
        model_parameters_string : list of str
            List of parameter names as strings, e.g. ["alpha", "beta", "gamma"].

        observed_data : jax.numpy.ndarray
            Observed values to calibrate against.
            Shape (N,) for scalar outputs, or (N, D) for D-dimensional outputs.



    Methods:
    
        _build_model():
            Sets up the model with the necessary parameters for sampling.
    """
    
    def __init__(self, model_function, model_parameters_string, observed_data, model_function_parameters=None):
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
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        
       
    def _model(observed_data, model_parameters_string, model_function, model_function_parameters):
        """
        Internal method which declares the NumPyro model for MCMC sampling.
        """
        params = {}

        for para_string in model_parameters_string:
            params[para_string] = numpyro.sample(
                para_string,  # <- fixed 'name' bug
                dist.TruncatedNormal(loc=0.5, scale=0.2, low=0.0, high=1.0)
            )

        # Declare White Noise
        white_noise = numpyro.sample("white_noise", dist.Normal(0, 0.2))

        # Combine into jnp array
        jnp_params = jnp.array([[params[name] for name in model_parameters_string]])  # shape: (1, N)

        # Get predictive mean and variance from model
        mu_func, var_func = model_function(jnp_params, **model_function_parameters)

        # Add White Noise to variance
        sigma = jnp.sqrt(var_func + white_noise ** 2)

        # Likelihood
        numpyro.sample("likelihood", dist.Normal(mu_func, sigma), obs=observed_data)
        
    
    def sample_from_chain(num_samples, num_warmup=None, num_chains=None):
        # Define kwargs dictionary
        kwargs = {
            "num_samples": num_samples or 50000,
            "num_warmup": num_warmup or num_samples // 5, # Default 20 % of num_samples
            "num_chains": num_chains 
        }
        
        # Remove entries that are None
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        kernel = NUTS(_model)
        mcmc = MCMC(kernel, **kwargs)
        mcmc.run(
            observed_data=self.observed_data,
            model_parameters_string=self.model_parameters_string,
            model_function=self.model_function,
            model_function_parameters=self.model_function_parameters
        )
        
        # Print Summary
        mcmc.print_summary()
        
        # Get samples
        samples = mcmc.get_samples(group_by_chain=True)

        # Save samples as .npz
        np.savez(f"{self.output_dir}/mcmc_samples.npz", **{k: np.array(v) for k, v in samples.items()})

        # Convert to InferenceData and save as .nc
        idata = az.from_numpyro(mcmc)
        idata.to_netcdf(f"{self.output_dir}/mcmc_results.nc")

        # Compute and save summary
        summary = az.summary(idata)
        summary.to_csv(f"{self.output_dir}/mcmc_summary.csv")
        
        # Plot and save chain stats
        for para_string in self.model_parameters_string:
            self.plot_chain_stats(idata, para_string)
        
        
    # Plot MCMC Sampling Plots (Autocorrelation, Traceplot, Histogram, Rank)
    def plot_chain_stats(idata, para_string):
        try:
            # Define Output directories
            figures_dir = os.path.join(self.output_dir, "figures")
            # Ensure output directory exists
            os.makedirs(figures_dir, exist_ok=True)

            # Autocorrelation plot
            az.plot_autocorr(idata, var_names=[para_string])
            plt.title(rf"Autokorrelation für ${para_string}$")
            plt.tight_layout()
            fname = os.path.join(figures_dir, f"autocorr_{para_string}.png")
            plt.savefig(fname, dpi=150)

            # Traceplot
            az.plot_trace(idata, var_names=[para_string])
            plt.suptitle(rf"Traceplot für ${para_string}$", fontsize=16)
            plt.tight_layout()
            fname = os.path.join(figures_dir, f"trace_{para_string}.png")
            plt.savefig(fname, dpi=150)

            # Rankplot
            az.plot_rank(idata, var_names=[para_string])
            plt.title(rf"Rankplot für ${para_string}$")
            plt.tight_layout()
            fname = os.path.join(figures_dir, f"rank_{para_string}.png")
            plt.savefig(fname, dpi=150)

        except Exception as e:
            print(f"Caught an error on {para_string}: {e}")

        
        
        
        
        