# Library of activation functions for general learning tasks and
# custom activations for learning implicit neural representations.
#
# create_activation_module() can be used to create activation modules
# based on string names, callable functions, or pre-initialized nn.Module
# instances. It includes custom activations like SIREN, SPDER, and others.
# The custom activations are designed to work with a frequency parameter
# omega, which can be specified when creating the activation module.
#
# February 2025
#
# Prasanth B. Nair
# University of Toronto Institute for Aerospace Studies (UTIAS)
# WWW: http://arrow.utias.utoronto.ca/~pbn
# Email: prasanth.nair@utoronto.ca

import torch
import torch.nn as nn

from typing import Union, Callable, Optional


def create_activation_module(
    activation_spec: Union[str, Callable, nn.Module],
    omega_for_custom_activ: Optional[float] = 30.0,
) -> nn.Module:
    """
    Create an activation module instance.

    Prioritizes direct use of torch.nn activations if specified by string.
    Falls back to custom modules for 'siren', 'spder_abs', 'spder_arctan'.
    Wraps callables in CustomTorchActivation. Returns pre-instantiated nn.Modules directly.

    Args:
        activation_spec (Union[str, Callable, nn.Module]): This argument take the following forms:
            - str: 'relu', 'tanh', 'sigmoid', 'leakyrelu', 'elu', 'gelu', siren', 'spder_abs',
              'spder_arctan', 'identity', 'none'. Standard PyTorch names are case-insensitive.
            - Callable:  A function like lambda x: torch.tanh(x).
            - nn.Module: A pre-initialized activation module.
        omega_for_custom_activ (Optional[float]): Omega value to be used if the activation is one of the
            custom types that require it (e.g., 'siren', 'spder_abs', 'spder_arctan').
            Ignored for standard PyTorch activations or if the custom activation doesn't use omega.

    Returns:
        nn.Module: An instance of an activation module.
    """

    if isinstance(activation_spec, nn.Module):
        return activation_spec
    if callable(activation_spec):
        return CustomTorchActivation(activation_spec)
    if not isinstance(activation_spec, str):
        raise ValueError(
            f"activation_spec must be a string, callable, or nn.Module, got {type(activation_spec)}"
        )

    act_type_str = activation_spec.lower()

    if act_type_str == "siren":
        assert (
            omega_for_custom_activ is not None
        ), "omega_for_custom_activ must be specified for 'siren' activation."
        return SineActivation(omega=omega_for_custom_activ)
    elif act_type_str == "spder_abs":
        assert (
            omega_for_custom_activ is not None
        ), "omega_for_custom_activ must be specified for 'spder_abs' activation."
        return SpderAbsActivation(omega=omega_for_custom_activ)
    elif act_type_str == "spder_arctan":
        assert (
            omega_for_custom_activ is not None
        ), "omega_for_custom_activ must be specified for 'spder_arctan' activation."
        return SpderArctanActivation(omega=omega_for_custom_activ)
    elif act_type_str == "none":
        return nn.Identity()

    pytorch_activations_map = {  # Only listing common activations
        "identity": nn.Identity,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "sigmoid": nn.Sigmoid,
        "leakyrelu": nn.LeakyReLU,  # Uses default negative_slope=0.01
        "elu": nn.ELU,  # Uses default alpha=1.0
        "gelu": nn.GELU,  # Uses default approximate='none'
        "softplus": nn.Softplus,
    }

    if act_type_str in pytorch_activations_map:
        activation_class = pytorch_activations_map[act_type_str]
        return activation_class()

    raise ValueError(
        f"Unknown or unsupported activation_spec: '{activation_spec}'. "
        f"Supported strings include {list(pytorch_activations_map.keys())}, "
        f"'siren', 'spder_abs', 'spder_arctan', 'none'. "
        f"You can also provide a callable or an nn.Module instance."
    )


class SineActivation(nn.Module):
    """
    SIREN sinusoidal activation function with frequency parameter omega
    """

    def __init__(self, omega=30.0):
        super().__init__()
        self.omega = omega

    def forward(self, x):
        return torch.sin(self.omega * x)


class SpderAbsActivation(nn.Module):
    """
    SPDER activation function with |x| damping and frequency parameter omega
    """

    def __init__(self, omega=30.0):
        super().__init__()
        self.omega = omega
        self.register_buffer("eps", torch.tensor(1e-6))

    def forward(self, x):
        scaled_x = self.omega * x
        return torch.sin(scaled_x) * torch.sqrt(
            torch.maximum(torch.abs(scaled_x), self.eps)
        )


class SpderArctanActivation(nn.Module):
    """
    SPDER activation function with arctan(x) damping and frequency parameter omega
    """

    def __init__(self, omega=30.0):
        super().__init__()
        self.omega = omega

    def forward(self, x):
        scaled_x = self.omega * x
        return torch.sin(scaled_x) * torch.arctan(scaled_x)


class CustomTorchActivation(nn.Module):
    """
    A wrapper for custom activation functions that are callable
    (e.g., lambda functions or user-defined functions).
    """

    def __init__(self, activation_fn):
        super().__init__()
        if not callable(activation_fn):
            raise ValueError(
                "activation_fn must be a callable (e.g., a lambda or function)"
            )
        self.activation_fn = activation_fn

    def forward(self, x):
        return self.activation_fn(x)
