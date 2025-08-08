
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~ helper_functions analysis.py~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MF~~~~~#

# Filename: helper_functions/analysis.py
# Author: Markus Flicker
# Date: 2023-08-05
# Description: 
#        This module provides helper functions for analyzing posterior samples and model results, particularly in the context of Bayesian inference and parameter estimation. The functions support loading and processing samples from .npz files, comparing true and posterior means, storing model values and covariance matrices, filtering posterior configurations within a specified delta of a reference value, and plotting posterior distributions from both raw samples and summary statistics.
#        Functions:
#        ----------
#        - mean_comparison(sample_fname, mean_true_parameters):
#        Compares true means with posterior means for multiple parameters and saves the results to a text file.
#        - store_model_values_results(true_value, found_value, output_dir, fname):
#        Stores the true value, found value, and (optionally) the covariance matrix of posterior samples in a CSV file.
#        - store_posterior_covariance(self, sample_fname, output_dir, fname):
#        Stores the covariance matrix of posterior samples in a CSV file.
#        - store_configurations_within_delta(true_model_function, ref_value, sample_fname, true_model_func_params=None, delta=0.01, inverse_fn=None, params=None, fname_suffix=""):
#        Identifies and stores posterior sample configurations whose model values are within a specified delta of a reference value.
#        - load_samples_from_npz(file_path):
#        Loads samples from a .npz file and returns them as a JAX array along with parameter names.
#        - construct_output_path(input_file, prefix):
#        Constructs an output file path with a consistent index based on the input file name.
#        - plot_posterior_from_samples(file_path):
#        Plots normalized histograms of posterior samples for each parameter and saves the plots as PNG files.
#        - plot_posterior_from_summary_csv(summary_file):
#        Plots normalized probability density functions (PDFs) for each parameter using summary statistics from a CSV file and saves the plots as PNG files.
#


import os
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import norm
import re


def jnp_trapz(y, x):
    dx = x[1:] - x[:-1]
    return jnp.sum(0.5 * (y[1:] + y[:-1]) * dx)

# Plot posterior samples dynamically with normalized histogram
def plot_posterior_from_samples(file_path):
    """
    Plots posterior distributions for parameters from MCMC or sampling results stored in a .npz file.
    Args:
        file_path (str): Path to the .npz file containing sample arrays and parameter names.
    The function:
        - Loads samples and parameter names from the specified .npz file.
        - Plots a histogram for each parameter's posterior distribution.
        - Sets axis labels, titles, and limits for clarity.
        - Saves the resulting figure to an output path constructed from the input file path.
    """
    
    
    
    sample_array, param_names = load_samples_from_npz(file_path)
    num_params = len(param_names)
    fig, axes = plt.subplots(1, num_params, figsize=(4 * num_params, 4))

    if num_params == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, param_names)):
        counts, bins, patches = ax.hist(sample_array[:, i], bins=30, density=True, alpha=0.7, color="skyblue", edgecolor="black")
        ax.set_title(f"Posterior: {name}")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density (Normalized)")
        ax.set_ylim(0, max(counts) * 1.1)

    plt.tight_layout()
    out_path = construct_output_path(file_path, "posterior_distributions_samples")
    plt.savefig(out_path, dpi=150)
    plt.close()

# Plot posterior from summary CSV file with normalized PDF
def plot_posterior_from_summary_csv(summary_file):
    """
    Plots posterior distributions as normal approximations for parameters summarized in a CSV file.
    The function reads a summary CSV file containing parameter names, means, and standard deviations,
    then plots the normal distribution for each parameter using the provided statistics. Each plot is
    normalized and displayed side-by-side in a single figure. The resulting figure is saved to disk.
    Args:
        summary_file (str or Path): Path to the CSV file containing parameter summaries. The file must
            have the parameter names in the first column, and columns named "mean" and "sd" for the
            mean and standard deviation of each parameter.
    Saves:
        A PNG image of the posterior distributions, with the output path constructed based on the input
        file and the suffix "posterior_distributions".
    """
    
    
    
    df = pd.read_csv(summary_file)
    param_names = df.iloc[:, 0].tolist()
    means = df["mean"].values
    stds = df["sd"].values

    num_params = len(param_names)
    fig, axes = plt.subplots(1, num_params, figsize=(4 * num_params, 4))

    if num_params == 1:
        axes = [axes]

    x = jnp.linspace(0, 1, 500)
    for ax, name, mu, sigma in zip(axes, param_names, means, stds):
        y = norm.pdf(x, mu, sigma)
        y /= jnp_trapz(y, x)  # Ensure normalization
        ax.plot(x, y, color="blue")
        ax.fill_between(x, y, color="skyblue", alpha=0.4)
        ax.set_title(f"Normal Approx: {name}")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density (Normalized)")
        ax.set_ylim(0, max(y) * 1.1)

    plt.tight_layout()
    out_path = construct_output_path(summary_file, "posterior_distributions")
    plt.savefig(out_path, dpi=150)
    plt.close()
    

def mean_comparison(sample_fname, mean_true_parameters):
    """
    Compare true means vs posterior means for multiple parameters and save results to a txt file.
    
    Args:
        mean_true_parameters (array-like): Array of true mean values.
        posterior_sampled_means (array-like): Array of posterior mean values.
        param_names (list): Names of the parameters (same length as means).
        outdir (str): Output directory.
    """
    
    # Load samples from the provided .npz file
    samples, param_names = load_samples_from_npz(sample_fname)
    # Calculate posterior means
    posterior_sampled_means = jnp.mean(samples, axis=0)
    
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
        

def store_model_values_results(true_value, found_value, output_dir, fname):
    """
    Store true value, found value, and covariance matrix of posterior samples in CSV.
    
    Args:
        true_value (float): True value of the model parameter.
        found_value (float): Found value from the posterior samples.
        fname_suffix (str): Suffix to append to the filename.
    
    """
    # 1. Save True and Found impedance + covariance matrix
    data = {
        "True_Value": [true_value],
        "Found_Value": [found_value]
    }
    df_main = pd.DataFrame(data)
    
    # Create output folder
    analysis_model_values_dir = os.path.join(output_dir, "model_analysis/model_values")
    os.makedirs(analysis_model_values_dir, exist_ok=True)
    
    # Store impedance values
    complete_fname = os.path.join(analysis_model_values_dir, fname)
    df_main.to_csv(complete_fname, index=False)
    print(f"Stored model values to: {complete_fname}")

    
def store_posterior_covariance(self, sample_fname, output_dir, fname):
    """
    Computes and stores the posterior covariance matrix from parameter samples.
    This method loads posterior samples from a specified .npz file, computes the covariance matrix of the parameters,
    and saves the resulting matrix as a CSV file in a designated output directory.
    Args:
        sample_fname (str): Path to the .npz file containing posterior samples and parameter names.
        output_dir (str): Directory where the covariance matrix CSV will be saved.
        fname (str): Filename for the saved covariance matrix CSV.
    Notes:
        - The .npz file is expected to contain both the parameter samples and their corresponding names.
        - The output CSV will be saved under 'model_analysis/posterior_covariance' within the specified output directory.
    """
    
    # 1. Covariance matrix from posterior samples
            # Extract posterior samples
            
    # Load samples from the provided .npz file
    samples, param_names = load_samples_from_npz(sample_fname)
        
    cov_matrix = jnp.cov(param_matrix, rowvar=False)
    
    # 2. Save covariance matrix as DataFrame
    cov_df = pd.DataFrame(cov_matrix, index=param_names, columns=param_names)
    
    # Create output folder
    analysis_posterior_covariance_dir = os.path.join(output_dir, "model_analysis/posterior_covariance")
    os.makedirs(analysis_posterior_covariance_dir, exist_ok=True)

    complete_fname = os.path.join(analysis_posterior_covariance_dir, fname)
    
    cov_df.to_csv(complete_fname, index=False)
    print(f"Stored model values and covariance matrix in '{complete_fname}'")

    
    
    
def store_configurations_within_delta(true_model_function, ref_value, sample_fname, true_model_func_params=None, delta=0.01, inverse_fn=None, params=None, fname_suffix=""):
    """
    Filters and stores model parameter configurations whose computed model values fall within a specified delta range of a reference value.
    Parameters:
        true_model_function (callable): The function representing the true model, used to compute model values from parameters.
        ref_value (float): The reference value to compare computed model values against.
        sample_fname (str): Path to the .npz file containing posterior samples and parameter names.
        true_model_func_params (iterable, optional): Additional parameters to be passed to the true model function for each evaluation. Defaults to None.
        delta (float, optional): The relative tolerance (as a fraction) for selecting configurations within ±delta of the reference value. Defaults to 0.01 (i.e., ±1%).
        inverse_fn (callable, optional): Function to inverse-transform sampled parameters, if required. Defaults to None.
        params (iterable, optional): Parameters to be passed to the inverse function for each parameter. Defaults to None.
        fname_suffix (str, optional): Suffix to append to the output filename. Defaults to "".
    Returns:
        None. The function saves a CSV file containing the filtered configurations and prints summary information.
    """
  

    # Load samples from the provided .npz file
    samples, param_names = load_samples_from_npz(sample_fname)

    # Dynamically inverse transform parameters if inverse_fn is provided
    if inverse_fn is not None and params is not None:
        real_params = [inverse_fn(samples[p], *pp) for p, pp in zip(param_names, params)]
    else:
        # If no inverse_fn is provided, keep parameters as they are
        real_params = [samples[p] for p in param_names]

    # Compute impedance for all posterior samples (unpack real parameters)
    if true_model_func_params is None:
        model_values = true_model_function(*real_params)
    else:
        model_values = [
            true_model_function(*model_params, *real_params)
            for model_params in true_model_func_params
        ]

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
    
def load_samples_from_npz(file_path):
    """
    Load samples from a .npz file and return as a JAX array.
    
    Args:
        file_path (str): Path to the .npz file.
        
    Returns:
        jax.numpy.ndarray: Samples loaded from the file.
    """
    samples = jnp.load(file_path)
    return jnp.column_stack([samples[p].flatten() for p in samples.files]), list(samples.files)

# Helper to construct output file path with same index
def construct_output_path(input_file, prefix):
    """
    Constructs an output file path based on the input file path and a given prefix.
    The function extracts the directory and base name from the input file path. It then searches
    for an underscore followed by digits before the file extension in the base name to determine
    an index. If such a pattern is not found, it defaults the index to "1". The output path is
    constructed by joining the directory with a new file name composed of the prefix, the index,
    and a ".png" extension.
    Args:
        input_file (str): The path to the input file.
        prefix (str): The prefix to use for the output file name.
    Returns:
        str: The constructed output file path with the ".png" extension.
    """
    
    
    
    directory = os.path.dirname(input_file)
    base_name = os.path.basename(input_file)
    match = re.search(r'_(\d+)\.[A-Za-z]+$', base_name)
    index = match.group(1) if match else "1"
    return os.path.join(directory, f"{prefix}_{index}.png")





# Plot Distributions of Posterior Samples
#plot_posterior_from_samples("tests/output_capacitor_incomplete/mcmc_samples__9.npz")
#plot_posterior_from_summary_csv("tests/output_capacitor_incomplete/mcmc_summary__9.csv")
#plot_posterior_from_samples("tests/output_capacitor_incomplete/mcmc_samples__7.npz")
#plot_posterior_from_summary_csv("tests/output_capacitor_incomplete/mcmc_summary__7.csv")
plot_posterior_from_samples("tests/output_capacitor_complete/mcmc_samples__12.npz")
plot_posterior_from_summary_csv("tests/output_capacitor_complete/mcmc_summary__12.csv")
#plot_posterior_from_samples("tests/output_capacitor_complete/mcmc_samples__8.npz")
#plot_posterior_from_summary_csv("tests/output_capacitor_complete/mcmc_summary__8.csv")
#plot_posterior_from_samples("tests/output_capacitor_complete/mcmc_samples__9.npz")
#plot_posterior_from_summary_csv("tests/output_capacitor_complete/mcmc_summary__9.csv")
#plot_posterior_from_samples("tests/output_pcb_trace_impedance/mcmc_samples__1.npz")
#plot_posterior_from_summary_csv("tests/output_pcb_trace_impedance/mcmc_summary__1.csv")



