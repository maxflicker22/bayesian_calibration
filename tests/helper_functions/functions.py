#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~ helper_functions functions.py~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MF~~~~~#

# Filename: helper_functions/functions.py
# Author: Markus Flicker
# Date: 2023-08-05
# Description: 
#          helper_functions/functions.py
#            This module provides helper functions for modeling the voltage across a charging capacitor,
#            performing Taylor expansions of the capacitor charging equation, min-max scaling using JAX,
#            loading PCB configuration files, and calculating the impedance of PCB traces.
#            Functions:
#            ----------
#            - charging_capacitor_model(t, u_0, tau):
#                Computes the voltage across a charging capacitor at time t.
#            - charging_capacitor_taylor_1(t, u_0, tau):
#                First-order Taylor expansion approximation of the charging capacitor voltage.
#           - charging_capacitor_taylor_2(t, u_0, tau):
#                Second-order Taylor expansion approximation of the charging capacitor voltage.
#           - charging_capacitor_taylor_3(t, u_0, tau):
#                Third-order Taylor expansion approximation of the charging capacitor voltage.
#            - charging_capacitor_taylor_7(t, u_0, tau):
#                Seventh-order Taylor expansion approximation of the charging capacitor voltage.
#            - min_max_scale_jax(x, x_min=None, x_max=None, feature_min=0.0, feature_max=1.0):
#                Scales input array x to a specified range [feature_min, feature_max] using min-max scaling with JAX.
#            - inverse_min_max_scale_jax(scaled_x, x_min, x_max, feature_min=0.0, feature_max=1.0):
#                Reverts min-max scaling, transforming data from [feature_min, feature_max] back to [x_min, x_max].
#            - load_pcb_config(config_path):
#                Loads PCB configuration from a YAML file.
#            - impedance_pcb_trace(eps_r, h, t, w):
#                Calculates the impedance of a PCB trace based on relative permittivity, dielectric height, trace thickness, and width.


import yaml
import jax
import jax.numpy as jnp


def charging_capacitor_model(t, u_0, tau):
    """
    This function models the voltage across a capacitor being charged in an RC circuit,
    using the equation: V(t) = u_0 * (1 - exp(-t / tau)), where u_0 is the supply voltage,
    t is the time, and tau is the time constant of the circuit.

    Parameters
    ----------
    t : float or array-like
        Time(s) at which to evaluate the capacitor voltage.
    u_0 : float
        The supply voltage (maximum voltage across the capacitor).
    tau : float
        The time constant of the RC circuit (tau = R * C).

    Returns
    -------
    float or array-like
        The voltage across the capacitor at time t.
    
    """
    return u_0 * (1 - jnp.exp(-t / tau))


def charging_capacitor_taylor_1(t, u_0, tau):
    # Taylor expansion with 1 term
    # u ≈ u_0 * (t/tau)
    return u_0 * (t / tau)


def charging_capacitor_taylor_2(t, u_0, tau):
    # Taylor expansion with 2 terms
    # u ≈ u_0 * [ (t/tau) - (t/tau)^2/2! ]
    return u_0 * ((t / tau) - (t / tau)**2 / 2)


def charging_capacitor_taylor_3(t, u_0, tau):
    # Taylor expansion with 3 terms
    # u ≈ u_0 * [ (t/tau) - (t/tau)^2/2! + (t/tau)^3/3! ]
    return u_0 * ((t / tau) - (t / tau)**2 / 2 + (t / tau)**3 / 6)


def charging_capacitor_taylor_7(t, u_0, tau):
    # Taylor expansion with 7 terms
    # u ≈ u_0 * [ (t/tau) - (t/tau)^2/2! + (t/tau)^3/3! - ... + (t/tau)^7/7! ]
    x = t / tau
    return u_0 * (
        x
        - x**2 / 2
        + x**3 / 6
        - x**4 / 24
        + x**5 / 120
        - x**6 / 720
        + x**7 / 5040
    )


def min_max_scale_jax(x, x_min=None, x_max=None, feature_min=0.0, feature_max=1.0):
    """
    Scales the input array `x` to a specified range [feature_min, feature_max] using min-max normalization with JAX.
    Parameters:
        x (array-like): Input data to be scaled.
        x_min (float or array-like, optional): Minimum value(s) for scaling. If None, computed from `x`.
        x_max (float or array-like, optional): Maximum value(s) for scaling. If None, computed from `x`.
        feature_min (float, optional): Lower bound of the desired feature range. Default is 0.0.
        feature_max (float, optional): Upper bound of the desired feature range. Default is 1.0.
    Returns:
        jax.numpy.ndarray: Scaled array with values in the range [feature_min, feature_max].
    """
    
    x = jnp.asarray(x)
    if x_min is None:
        x_min = jnp.min(x)
    if x_max is None:
        x_max = jnp.max(x)
    scaled = feature_min + (x - x_min) * (feature_max - feature_min) / (x_max - x_min)
    return scaled


def inverse_min_max_scale_jax(scaled_x, x_min, x_max, feature_min=0.0, feature_max=1.0):
    
    """
    Inverse min-max scaling for JAX arrays.

    This function reverses the min-max scaling transformation, converting data from a normalized range 
    [feature_min, feature_max] back to its original range [x_min, x_max].

    Args:
        scaled_x (array-like): The scaled data to be inverse transformed. Can be a JAX array or array-like object.
        x_min (float or array-like): The minimum value(s) of the original data range.
        x_max (float or array-like): The maximum value(s) of the original data range.
        feature_min (float, optional): The minimum value of the scaled feature range. Default is 0.0.
        feature_max (float, optional): The maximum value of the scaled feature range. Default is 1.0.

    Returns:
        jax.numpy.ndarray: The data rescaled back to the original range [x_min, x_max].

    """
    scaled_x = jnp.asarray(scaled_x)
    return x_min + (scaled_x - feature_min) * (x_max - x_min) / (feature_max - feature_min)

def load_pcb_config(config_path):

    """
    Loads a PCB configuration from a YAML file.
    Args:
        config_path (str): The file path to the YAML configuration file.
    Returns:
        dict: The configuration data loaded from the YAML file.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def impedance_pcb_trace(eps_r, h, t, w):
    """
    Calculates the characteristic impedance of a PCB microstrip trace.
    Parameters:
        eps_r (float): Relative permittivity (dielectric constant) of the PCB material.
        h (float): Height (thickness) of the dielectric layer between the trace and the reference plane (in the same units as w and t).
        t (float): Thickness of the PCB trace (in the same units as h and w).
        w (float): Width of the PCB trace (in the same units as h and t).
    Returns:
        float: The characteristic impedance (in ohms) of the PCB trace.
    Notes:
        This function uses an empirical formula for microstrip impedance calculation.
        All dimensions should be provided in consistent units.
    """
    prefactor = 87 / (jnp.sqrt(eps_r + 1.41))
    Z = prefactor * jnp.log((5.98 * h) / (0.8 * w + t)) # Impedance
    return Z



