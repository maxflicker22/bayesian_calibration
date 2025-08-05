
    def store_model_values_results(self, true_value, found_value, fname_suffix=""):
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

        
        
        
    def store_configurations_within_delta(self, true_model_function, ref_value, true_model_func_params=None, delta=0.01, inverse_fn=None, params=None, fname_suffix=""):
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