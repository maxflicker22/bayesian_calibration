import yaml
import jax
import jax.numpy as jnp

# min_max_scale_jax / lineare bijectiv transformation
def min_max_scale_jax(x, x_min=None, x_max=None, feature_min=0.0, feature_max=1.0):
    """
    Min-max scale array x to [feature_min, feature_max] using JAX.
    If x_min or x_max are not provided, they're computed from x.
    """
    x = jnp.asarray(x)
    if x_min is None:
        x_min = jnp.min(x)
    if x_max is None:
        x_max = jnp.max(x)
    scaled = feature_min + (x - x_min) * (feature_max - feature_min) / (x_max - x_min)
    return scaled

# revert min_max_scale_jax / lineare bijectiv transformation
def inverse_min_max_scale_jax(scaled_x, x_min, x_max, feature_min=0.0, feature_max=1.0):
    """
    Inverse min-max scaling: scales data from [feature_min, feature_max] back to [x_min, x_max].
    """
    scaled_x = jnp.asarray(scaled_x)
    return x_min + (scaled_x - feature_min) * (x_max - x_min) / (feature_max - feature_min)

# load pcb configs from config file
def load_pcb_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# True underlying pcb_trace_impedance model
def impedance_pcb_trace(eps_r, h, t, w):
    """
    Impedance Formular for pcb trace
    eps_r...    relativ permitivity
    h...        height of dieelectric
    t...        thickness of trace
    w...        widness of trace
    Z...        Impedance from PCB Trace at one length uni
    """
    prefactor = 87 / (jnp.sqrt(eps_r + 1.41))
    Z = prefactor * jnp.log((5.98 * h) / (0.8 * w + t)) # Impedance
    return Z

