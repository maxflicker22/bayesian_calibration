import os
import pandas as pd
import jax
import jax.numpy as jnp
 
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
    Store covariance matrix of posterior samples in CSV.
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
    Check all posterior samples and store those configurations whose impedance is within ±delta of ref_value.
    Allows passing a dynamic inverse scaling function with *params for flexibility.
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


