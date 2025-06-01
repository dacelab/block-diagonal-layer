# BlockDiagonalLayer is a PyTorch class that defines the linear layer of an ensemble of
# decoupled MLPs using a block diagonal weight matrix (stored as a rank-3 tensor) and
# applies the user-defined activation function. This approach allows for parallel forward
# pass and gradient computation, and updating all networks simultaneously during training
# (without a for loop).
#
# Each MLP is assumed to have the same topology but different weights and biases that take
# different/same views/projections of the data as input.
#
# I have found this layer useful for a variety of applications, including ensemble implicit
# neural representations, multihead MLPs, and learning decoupled latent representations of
# high-dimensional functions.
#
# October 11, 2024: initial prototype
# March 21, 2025: Refactored to support general initialization methods
#
# Prasanth B. Nair
# University of Toronto Institute for Aerospace Studies (UTIAS)
# WWW: http://arrow.utias.utoronto.ca/~pbn
# Email: prasanth.nair@utoronto.ca

import torch
import torch.nn as nn

import math

from typing import Optional, Union, Dict, List


class BlockDiagonalLayer(nn.Module):
    r"""
    A custom PyTorch layer for decoupled MLPs using a block diagonal weight matrix.

    This layer is designed for architectures involving multiple independent MLPs, where each MLP has
    the same topology but different weights and biases. Each MLP takes different or the same views/projections
    of the data as input. The linear transformation is performed using a block diagonal weight
    matrix and a user-defined activation function is applied to the output.

    This implementation follows the PyTorch convention of defining the weight matrix, W, with shape
    (out_features, in_features) so that given x with shape (batch_size, in_features), the output: x W^T + b
    is of shape (batch_size, out_features), where b is the bias vector with shape (out_features,).

    Args:
        num_networks (int): Number of independent neural networks.
        input_features_per_network (int): Number of input features per network.
        output_features_per_network (int): Number of output features per network.
        activation (nn.Module or str): Activation function to be applied after the linear transformation.
            Options:
                - nn.Module instance: Pre-instantiated activation module (e.g., nn.ReLU(), nn.Identity()).
                - 'heterogeneous_siren': Special case for SIREN activations with different omega values per network.
                - 'heterogeneous_spder_abs': Special case for SPDER abs (sin(omega*x)\sqrt(|omega*x|)) activations
                   with different omega values per network.
                - 'heterogeneous_spder_arctan': Special case for SPDER arctan (sin(omega*x)arctan(omega*x)) activations
                   with different omega values per network.
                All three heterogenous activations requires weight_init_method='siren_first' or 'siren_hidden' and
                weight_init_params to be a list of omega values of length num_networks.
            Notes:
                - Activation function is applied to the concatenated output tensor of shape
                  (batch_size, num_networks * output_features_per_network).
                - For a linear layer without an explicit activation function, pass nn.Identity().
        use_bias (bool, optional): Whether to include a bias term.
        weight_init_method (str, optional): Initialization method for weight matrices.
            Options:
                - `'pytorch_default'`: Use PyTorch default approach for `nn.Linear` (Kaiming uniform with `a=sqrt(5)`).
                - `'siren_first'`: SIREN initialization for the first layer.
                - `'siren_hidden'`: SIREN initialization for hidden layers (requires `omega` in `weight_init_params`).
                  When activation='heterogeneous_siren' or 'heterogeneous_spder_abs' or 'heterogeneous_spder_arctan,
                  weight_init_params must be a list of omega values.
                - Any valid method name from `torch.nn.init` (e.g., `'xavier_uniform'`, `'normal'`).
                  The trailing underscore for inplace operations will be added automatically if not present.
                  For these methods, `input_features_per_network` is `fan_in` and `output_features_per_network`
                  is `fan_out` for each network's weight matrix.
            Defaults to `'pytorch_default'`.
        weight_init_params (float, dict, or List[float], optional): Parameters for the weight initialization method.
            Notes:
                - For `'siren_hidden'` with regular activation: provide a float value for `omega` (frequency factor),
                  or a dict `{'omega': float_value}`.
                - For `'siren_hidden'` with activation='heterogeneous_siren': provide a list of omega values
                  (one per network). All values must be positive and list length must equal num_networks.
                - For `torch.nn.init` methods: provide a dict of parameters (e.g., `{'gain': 1.0, 'a': 0.01}`).
                - If `None`, PyTorch's default settings for the specified `'weight_init_method'` are used.
                - Ignored if `'weight_init_method'` is `'pytorch_default'` or `'siren_first'`.
            Defaults to `None`.
        bias_init_method (str, optional): Initialization method for bias parameters, if `use_bias` is True.
            Options:
                - `'pytorch_default'`: Use PyTorch's default approach for `nn.Linear`, i.e.,
                  U \sim (-1/sqrt(input_features_per_network), 1/sqrt(input_features_per_network))
                - `'zeros'`: Initializes biases to zero.
            Defaults to `'pytorch_default'`.

    Attributes:
        num_networks (int): Number of independent networks
        input_features_per_network (int): Number of input features per network
        output_features_per_network (int): Number of output features per network
        activation (nn.Module): Activation function to be applied after the linear transformation
        use_bias (bool, optional): Whether to include a bias term
        weight_init_method (str): Initialization method used for weights, `siren_first`, `siren_hidden`,
            `pytorch_default` or any valid method from torch.nn.init.
        weight_init_params (dict): Initialization parameters corresponding to weight_init_method.
        bias_init_method (str): Initialization method for bias parameters, `pytorch_default` or `zeros`.
        weight (nn.Parameter): Block diagonal weight matrix stored as
            a rank-3 tensor (num_networks, output_features_per_network, input_features_per_network).
        bias (Optional[nn.Parameter]): Matrix of bias vectors stored as (num_networks, output_features_per_network)
            matrix, if `use_bias` is True, otherwise None.
        _batched_matvec (function): Function to optimize batched matrix-vector operations in the forward pass.

    Methods:
        _validate_init_method(): Validates the weight initialization method and required parameters.
        reset_parameters(): Initializes the weights and biases using the specified method.
        forward(x: torch.Tensor): Forward pass through the BlockDiagonalLayer.
            Takes input tensor of shape (batch_size, num_networks * input_features_per_network) and returns
            output tensor of shape (batch_size, num_networks * output_features_per_network).

    Raises:
        ValueError: If `num_networks`, `input_features_per_network`, or `output_features_per_network` are not positive.
        ValueError: If `activation` is not provided, not an `nn.Module`, or invalid string value.
        ValueError: If heterogeneous INR activation is used but weight_init_method is not 'siren_first' or 'siren_hidden'.
        ValueError: If `weight_init_method` or `bias_init_method` is invalid.
        ValueError: If required parameters for `weight_init_method` (e.g., `omega` for `'siren_hidden'`)
            are missing or invalid in `weight_init_params`.
        TypeError: If `weight_init_params` is of an unsupported type.
    """

    def __init__(
        self,
        num_networks: int,
        input_features_per_network: int,
        output_features_per_network: int,
        activation: Union[nn.Module, str],
        use_bias: bool = True,
        weight_init_method: str = "pytorch_default",
        weight_init_params: Optional[Union[float, Dict, List[float]]] = None,
        bias_init_method: str = "pytorch_default",
    ):
        super().__init__()

        if num_networks <= 0:
            raise ValueError("num_networks must be positive")
        if input_features_per_network <= 0 or output_features_per_network <= 0:
            raise ValueError("Feature dimensions must be positive")
        if activation is None:
            raise ValueError("Activation function must be provided.")

        if activation in {
            "heterogeneous_siren",
            "heterogeneous_spder_abs",
            "heterogeneous_spder_arctan",
        }:
            if weight_init_method not in {"siren_hidden", "siren_first"}:
                raise ValueError(
                    "Heterogeneous INR activations (heterogeneous_siren, heterogeneous_spder_abs, "
                    "heterogeneous_spder_arctan) require weight_init_method='siren_first' or 'siren_hidden'."
                )
            if weight_init_params is None:
                raise ValueError(
                    "weight_init_params must be provided as a list of omega values when "
                    "using heterogeneous INR activations (heterogeneous_siren, heterogeneous_spder_abs, "
                    "heterogeneous_spder_arctan)."
                )
            if not isinstance(weight_init_params, (list, tuple)):
                raise ValueError(
                    "weight_init_params must be a list or tuple of omega values when "
                    "using heterogeneous INR activations (heterogeneous_siren, heterogeneous_spder_abs, "
                    "heterogeneous_spder_arctan)."
                )
            if len(weight_init_params) != num_networks:
                raise ValueError(
                    f"weight_init_params must have length {num_networks}, got {len(weight_init_params)}."
                )
            if not all(
                isinstance(w, (int, float)) and w > 0 for w in weight_init_params
            ):
                raise ValueError(
                    "All omega values in weight_init_params must be positive numbers."
                )

            self.activation = _HeterogeneousINRActivation(
                activation,
                weight_init_params,
                num_networks,
                output_features_per_network,
            )
        else:
            if not isinstance(activation, nn.Module):
                raise ValueError(
                    "Activation must be an instance of nn.Module (e.g., nn.ReLU(), nn.Identity()) "
                    "or one of the heterogeneous INR activation strings: 'heterogeneous_siren', "
                    "'heterogeneous_spder_abs', 'heterogeneous_spder_arctan'."
                )
            self.activation = activation

        self.num_networks = num_networks
        self.input_features_per_network = input_features_per_network
        self.output_features_per_network = output_features_per_network
        self.use_bias = use_bias
        self.weight_init_method = weight_init_method
        self.bias_init_method = bias_init_method

        if activation in {
            "heterogeneous_siren",
            "heterogeneous_spder_abs",
            "heterogeneous_spder_arctan",
        }:
            self.weight_init_params = {
                f"omega_{i}": float(omega) for i, omega in enumerate(weight_init_params)
            }
            self._omega_values = list(weight_init_params)
        else:
            if weight_init_params is None:
                self.weight_init_params = {}
            elif isinstance(weight_init_params, dict):
                self.weight_init_params = weight_init_params.copy()
            elif isinstance(weight_init_params, (int, float)):
                if self.weight_init_method == "siren_hidden":
                    self.weight_init_params = {"omega": float(weight_init_params)}
                else:
                    raise TypeError(
                        f"weight_init_params as a float/int is only supported when weight_init_method is 'siren_hidden'. "
                        f"For '{self.weight_init_method}', use a dictionary for parameters or None."
                    )
            else:
                raise TypeError(
                    "weight_init_params must be a number (for siren_hidden), dict, list of floats "
                    "(for heterogeneous_siren, heterogeneous_spder_abs, heterogenous_spder_arctan), or None."
                )

        self._validate_init_method()

        self.weight = nn.Parameter(
            torch.empty(
                num_networks, output_features_per_network, input_features_per_network
            )
        )
        if self.use_bias:
            self.bias = nn.Parameter(
                torch.empty(num_networks, output_features_per_network)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        if self.input_features_per_network == 1:
            self._batched_matvec = _batched_matvec_input_one
        elif self.output_features_per_network == 1:
            self._batched_matvec = _batched_matvec_output_one
        else:
            self._batched_matvec = _batched_matvec_general

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass of BlockDiagonalLayer.

        The input tensor `x` can be either a 2D or 3D tensor:
            (a) 2D tensor: `(batch_size, num_networks * input_features_per_network)`
                This shape assumes that input features for all networks are concatenated along the feature dimension.
            (b) 3D tensor: `(batch_size, num_networks, input_features_per_network)`
                This shape directly provides per-network inputs.

        Args:
            x (torch.Tensor): Input tensor.
                Shape: `(batch_size, num_networks * input_features_per_network)` or
                       `(batch_size, num_networks, input_features_per_network)`.

        Returns:
            torch.Tensor: Output tensor after linear transformation and activation.
                Shape: `(batch_size, num_networks * output_features_per_network)`.
                Use `reshape` to convert to `(batch_size, num_networks, output_features_per_network)` if needed.
        """
        batch_size = x.shape[0]
        if x.ndim == 2:
            expected_features = self.num_networks * self.input_features_per_network
            if x.shape[1] != expected_features:
                raise ValueError(
                    f"Input dimension mismatch for 2D input. Expected {expected_features} features "
                    f"with shape({batch_size}, {expected_features})), got {x.shape[1]} (shape {x.shape})."
                )
            x_reshaped = x.view(
                batch_size, self.num_networks, self.input_features_per_network
            )
        elif x.ndim == 3:
            expected_shape = (
                batch_size,
                self.num_networks,
                self.input_features_per_network,
            )
            if (
                x.shape[1] != self.num_networks
                or x.shape[2] != self.input_features_per_network
            ):
                raise ValueError(
                    f"Input dimension mismatch for 3D input. Expected shape "
                    f"{expected_shape}, got {x.shape}."
                )
            x_reshaped = x
        else:
            raise ValueError(
                f"Input tensor must be 2D or 3D. Got {x.ndim}D tensor with shape {x.shape}."
            )

        output = self._batched_matvec(self.weight, x_reshaped)

        if self.bias is not None:
            output = output + self.bias.unsqueeze(0)

        output = self.activation(output.reshape(batch_size, -1))

        return output

    def _validate_init_method(self):
        r"""
        Validate initialization method and parameters for weights and biases.
        """
        if self.weight_init_method in {"siren_hidden"}:
            if hasattr(self, "_omega_values"):
                pass
            else:
                if "omega" not in self.weight_init_params:
                    raise ValueError(
                        f"omega parameter required for siren_hidden initialization. "
                        f"Provide it in init_params as a float or dict with 'omega' key. "
                    )
                if self.weight_init_params["omega"] <= 0:
                    raise ValueError(
                        "omega must be a positive number for siren_hidden initialization."
                    )
        elif self.weight_init_method in {
            "siren_first",
            "pytorch_default",
        }:  # No additional parameters needed
            pass
        else:
            method_name = self.weight_init_method
            if not method_name.endswith("_"):
                method_name += "_"
            if not hasattr(torch.nn.init, method_name):
                raise ValueError(
                    f"Unknown weight_init_method: {self.weight_init_method}. "
                    f"Must be 'siren_first', 'siren_hidden', 'pytorch_default', "
                    f"or a valid method name from torch.nn.init (e.g., 'kaiming_uniform'). "
                )

        if self.bias_init_method not in ["pytorch_default", "zeros"]:
            raise ValueError(
                f"Unknown bias_init_method: {self.bias_init_method}. "
                f"bias_init_method must be 'pytorch_default' or 'zeros'. "
            )

    def reset_parameters(self):
        r"""
        Initialize weights and biases using the specified method(s).
        """
        with torch.no_grad():
            if (
                self.weight_init_method == "siren_first"
            ):  # SIREN first layer initialization
                bound = 1.0 / self.input_features_per_network
                self.weight.uniform_(-bound, bound)
            elif (
                self.weight_init_method == "siren_hidden"
            ):  # SIREN hidden layer initialization
                if hasattr(self, "_omega_values"):
                    # Heterogeneous INR activation: use different omega for each network
                    for i, omega in enumerate(self._omega_values):
                        bound = math.sqrt(6.0 / self.input_features_per_network) / omega
                        self.weight[i].uniform_(-bound, bound)
                else:
                    # Standard SIREN: same omega for all networks
                    omega = self.weight_init_params["omega"]
                    bound = math.sqrt(6.0 / self.input_features_per_network) / omega
                    self.weight.uniform_(-bound, bound)
            elif self.weight_init_method == "pytorch_default":
                bound = 1.0 / math.sqrt(self.input_features_per_network)
                self.weight.uniform_(-bound, bound)
            else:
                init_fn_name = self.weight_init_method
                if not init_fn_name.endswith("_"):
                    init_fn_name += "_"
                if not hasattr(torch.nn.init, init_fn_name):
                    raise ValueError(
                        f"Inplace initializer {init_fn_name} not found in torch.nn.init."
                    )

                init_fn = getattr(torch.nn.init, init_fn_name)
                for i in range(self.num_networks):
                    try:
                        if self.weight_init_params:
                            init_fn(self.weight[i], **self.weight_init_params)
                        else:
                            init_fn(self.weight[i])
                    except TypeError as e:
                        raise ValueError(
                            f"Invalid parameters for {init_fn_name}: {e}. "
                            f"Parameters provided: {self.weight_init_params}. "
                            f"Ensure they are valid for torch.nn.init.{init_fn_name}. "
                        )

            if self.use_bias:
                if self.bias_init_method == "pytorch_default":
                    bound = 1.0 / math.sqrt(self.input_features_per_network)
                    self.bias.uniform_(-bound, bound)
                elif self.bias_init_method == "zeros":
                    self.bias.zero_()

    def extra_repr(self):
        r"""
        Return a string with the extra representation of the module.
        """
        repr = [
            f"num_networks={self.num_networks}",
            f"input_features_per_network={self.input_features_per_network}",
            f"output_features_per_network={self.output_features_per_network}",
            f"activation={self.activation}",
            f"use_bias={self.use_bias}",
        ]
        if self.use_bias:
            repr.append(f"bias_init_method='{self.bias_init_method}'")

        repr.extend(
            [
                f"weight_init_method='{self.weight_init_method}'",
                f"weight_init_params={self.weight_init_params}",
            ]
        )
        return ", ".join(repr)


@torch.compile
def _batched_matvec_input_one(weights, x):
    r"""
    Batched matrix-vector multiplication for the special case when input_features_per_network = 1.

    This function implements a block diagonal operation where each of N independent networks
    transforms a scalar input to a d_out-dimensional output using network-specific weights.
    For each network i and batch element b, we have
            y[b, i, :] = x[b, i, 0] * w_i  for i = 1, ..., N,
    where w_i is the weight vector for network i with length d_out. In flattened form, this
    corresponds to the operation y = x W^T, where W is a block diagonal (N*d_out, N) matrix
    with blocks w_1, w_2, ..., w_N, x is (batch_size, N), and y is (batch_size, N*d_out).

    Args:
        weights (torch.Tensor): Block diagonal weight matrix with shape (N, d_out, 1),
            where weights[i, :, 0] contains the weight vector for network i
        x (torch.Tensor): Input tensor of shape (batch_size, N, 1),
            where x[b, i, 0] is the scalar input to network i for batch element b

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, N, d_out),
            where output[b, i, :] = x[b, i, 0] * weights[i, :, 0]

    """
    return x * weights.squeeze(2).unsqueeze(0)


@torch.compile
def _batched_matvec_output_one(weights, x):
    r"""
    Batched matrix-vector multiplication for the special case when output_features_per_network = 1.

    This function implements a block diagonal operation where each of N independent networks
    transforms a d_in-dimensional input to a scalar output using network-specific weights.
    For each network i and batch element b, we have
            y[b, i, 0] = sum_{j=0}^{d_in-1} x[b, i, j] * w_i[j] for i = 1, ..., N,
    where w_i is the weight vector for network i with length d_in. In flattened form, this
    corresponds to the operation y = x W^T, where W is a block diagonal (N, N*d_in) matrix
    with blocks w_1, w_2, ..., w_N, x is (batch_size, N*d_in), and y is (batch_size, N).

    Args:
        weights (torch.Tensor): Block diagonal weight matrix with shape (N, 1, d_in),
            where weights[i, 0, :] contains the weight vector for network i
        x (torch.Tensor): Input tensor of shape (batch_size, N, d_in),
            where x[b, i, :] is the d_in-dimensional input to network i for batch element b

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, N, 1),
            where output[b, i, 0] = sum_{j} x[b, i, j] * weights[i, 0, j]
    """
    return torch.sum(x * weights.squeeze(1).unsqueeze(0), dim=2, keepdim=True)


@torch.compile
def _batched_matvec_general(weights, x):
    r"""
    Batched matrix-vector multiplication for the general case with arbitrary input and output dimensions.

    This function implements a block diagonal operation where each of N independent networks transforms
    a d_in-dimensional input to a d_out-dimensional output using network-specific weight matrices.
    For each network i and batch element b, we have
            y[b, i, :] = x[b, i, :] @ W_i^T for i = 1, ..., N,
    where W_i is the weight matrix for network i with shape (d_out, d_in). In flattened form,
    this corresponds to the operation y = x W^T, where W is a block diagonal (N*d_out, N*d_in)
    matrix with blocks W_1, W_2, ..., W_N, x is (batch_size, N*d_in), and y is (batch_size, N*d_out).

    Args:
        weights (torch.Tensor): Block diagonal weight matrix with shape (N, d_out, d_in),
            where weights[i, :, :] contains the weight matrix for network i
        x (torch.Tensor): Input tensor of shape (batch_size, N, d_in),
            where x[b, i, :] is the d_in-dimensional input to network i for batch element b

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, N, d_out),
            where output[b, i, :] = x[b, i, :] @ weights[i, :, :]^T
    """
    return torch.einsum("bdi, dji -> bdj", x, weights)


class _HeterogeneousINRActivation(nn.Module):
    """
    Internal class for heterogeneous SIREN and SPDER (abs and arctan damping) activations with
    different omega values per network.
    """

    def __init__(
        self,
        activation: str,
        omega_values: List[float],
        num_networks: int,
        output_features_per_network: int,
    ):
        super().__init__()
        self.register_buffer(
            "omega_values", torch.tensor(omega_values, dtype=torch.float32)
        )
        self.num_networks = num_networks
        self.output_features_per_network = output_features_per_network

        activation_to_pure_fn = {
            "heterogeneous_siren": self._siren_activation_pure,
            "heterogeneous_spder_abs": self._spder_abs_activation_pure,
            "heterogeneous_spder_arctan": self._spder_arctan_activation_pure,
        }

        # Create vmapped activation function (vmap over omega dimension)
        if activation in activation_to_pure_fn:
            self.vmapped_activation = torch.vmap(
                activation_to_pure_fn[activation], in_dims=(1, 0), out_dims=1
            )
        else:
            raise ValueError(
                f"Unknown activation type: {activation}. "
                f"Supported types are {', '.join(activation_to_pure_fn.keys())}."
            )

    @staticmethod
    def _siren_activation_pure(x: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """
        Pure function for SIREN activation: sin(omega * x)
        """
        return torch.sin(omega * x)

    @staticmethod
    def _spder_abs_activation_pure(
        x: torch.Tensor, omega: torch.Tensor
    ) -> torch.Tensor:
        """
        Pure function for SPDER activation: sin(omega * x) * sqrt(|omega * x|)
        """
        eps = torch.tensor(1e-6)
        scaled_x = omega * x
        return torch.sin(scaled_x) * torch.sqrt(torch.maximum(torch.abs(scaled_x), eps))

    @staticmethod
    def _spder_arctan_activation_pure(
        x: torch.Tensor, omega: torch.Tensor
    ) -> torch.Tensor:
        """
        Pure function for SPDER activation: sin(omega * x) * arctan(omega * x)
        """
        scaled_x = omega * x
        return torch.sin(scaled_x) * torch.arctan(scaled_x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply heterogeneous INR activations.

        Args:
            x: Input tensor of shape (batch_size, num_networks * output_features_per_network)

        Returns:
            Output tensor of same shape with heterogeneous INR activations applied per network
        """
        batch_size = x.shape[0]

        x_reshaped = x.view(
            batch_size, self.num_networks, self.output_features_per_network
        )

        output = self.vmapped_activation(x_reshaped, self.omega_values)

        return output.reshape(batch_size, -1)
