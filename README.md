# BlockDiagonalLayer

A PyTorch class that defines the linear layer of an ensemble of decoupled MLPs using a block diagonal weight matrix (stored as a rank-3 tensor) and applies the user-defined activation function. This approach allows for parallel forward pass and gradient computation, and updating all networks simultaneously during training (without a for-loop).

Each MLP is assumed to have the same topology but different weights and biases that take different/same views/projections of the data as input.

This layer is useful for a variety of applications, including ensemble implicit neural representations, multi-head MLPs, and learning decoupled latent representations of high-dimensional functions.

## Key Features

- **Massive Parallelization**: Process thousands of decoupled MLPs simultaneously with block diagonal operations
- **Memory Efficient for GPUs**: Stores weights as rank-3 tensors instead of individual layer objects, enabling around 100x speedups over naive for-loop implementations
- **Flexible Initialization**: Supports PyTorch default, SIREN, and all `torch.nn.init` methods

## Requirements

- PyTorch >= 2.0
- Python >= 3.8

## Installation

```bash
git clone https://github.com/dacelab/block-diagonal-layer.git
cd block-diagonal-layer

# Install dependencies
pip install torch pytest
```

## Usage Example

Here's how to create an ensemble of 1000 MLPs with architecture [10 → 50 → 50 → 2] and ReLU activation:

```python
import torch
import torch.nn as nn
from block_diagonal_layer import BlockDiagonalLayer

# Define number of MLPs and architecture of each MLP
num_networks = 1000
architecture = [10, 50, 50, 2]  # 10 inputs -> 50 -> 50 -> 2 outputs

# Create ensemble of MLPs using BlockDiagonalLayer
model = nn.Sequential(
    # Layer 1: 10 -> 50 (num_networks times)
    BlockDiagonalLayer(
        num_networks=num_networks,
        input_features_per_network=architecture[0],
        output_features_per_network=architecture[1],
        activation=nn.ReLU(),
    ),
    
    # Layer 2: 50 -> 50 (num_networks times)
    BlockDiagonalLayer(
        num_networks=num_networks,
        input_features_per_network=architecture[1],
        output_features_per_network=architecture[2],
        activation=nn.ReLU(),
    ),
    
    # Layer 3: 50 -> 2 (num_networks times)
    BlockDiagonalLayer(
        num_networks=num_networks,
        input_features_per_network=architecture[2],
        output_features_per_network=architecture[3],
        activation=nn.Identity(),  # No activation for output layer
    )
)

batch_size = 64
total_input_dim = num_networks * architecture[0]  # 1000 * 10 = 10,000

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
else:
    device = torch.device("cpu")

model = model.to(device)
x = torch.randn(batch_size, total_input_dim, device=device)  # Shape: (64, 10000)

output = model(x)  # Shape: (64, 2000) - 1000 networks × 2 outputs each

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Activation Functions

The `activation` parameter must be a `nn.Module` instance, e.g., `nn.ReLU()`, `nn.Tanh()`, `nn.Identity()`. See `activation_utils.py` which provides a utility
function to create other activation functions. Examples:

```python
from activation_utils import create_activation_module
activation_fn = create_activation_module('siren', 30.0)
activation_fn = create_activation_module('spder_abs', 10.0)
activation_fn = create_activation_module('spder_arctan', 20.0)
```

Also supports different frequency parameter for each decoupled MLP when using SIREN or SPDER (abs and arctan damping) activation; see docstring for details.

### Weight Initialization Methods

```python
# PyTorch default 
weight_init_method="pytorch_default"

# SIREN initialization
weight_init_method="siren_first"     # For first layer (weight_init_params not needed)
weight_init_method="siren_hidden"    # For hidden layers
weight_init_params=30.0              # Omega parameter for SIREN

# Standard PyTorch initializers, for example:
weight_init_method="xavier_uniform"
weight_init_params={"gain": 1.0}

weight_init_method="kaiming_normal"
weight_init_params={"mode": "fan_out", "nonlinearity": "relu"}

weight_init_method="normal"
weight_init_params={"mean": 0.0, "std": 0.01}
```

### Input Tensor Formats

BlockDiagonalLayer accepts both 2D and 3D input tensors:

```python
# 2D tensor (concatenated features)
x_2d = torch.randn(batch_size, num_networks * input_features_per_network)

# 3D tensor (structured per-network)
x_3d = torch.randn(batch_size, num_networks, input_features_per_network)

# Both produce identical outputs with shape (batch_size, num_networks * output_features_per_network) # which can be reshaped to (batch_size, num_networks, output_features_per_network) if needed.
output_2d = layer(x_2d)
output_3d = layer(x_3d)
```

## Performance Benchmarks

Performance comparison on **NVIDIA GeForce RTX 4090** showing speedups over a for-loop implementation:

| Configuration | BlockDiagonal Time | For-loop Time | Speedup | Throughput |
|---------------|-------------------|---------------|---------|------------|
| 2000 layers, 32->32, batch=128 | 0.454 ms | 331.446 ms | **730x** | 1154.8 GFLOPS |
| 1000 layers, 32->32, batch=256 | 0.427 ms | 165.089 ms | **386x** | 1227.0 GFLOPS |
| 100 layers, 128->256, batch=128 | 0.434 ms | 16.463 ms | **38x** | 1931.1 GFLOPS |

*BlockDiagonalLayer is designed for architectures involving thousands of decoupled MLPs where massive parallelization provides substantial computational advantages.*

## Testing


```bash
pytest -v --capture=no
```

The test suite includes: checks for equivalence with a stack of standard `nn.Linear` layers, gradient correctness verification, performance benchmarks, and GPU utilization tests

## Training

**Standard use-case:** A standard PyTorch training pipeline with a single instance of torch.optim can be used when the outputs from decoupled MLPs are combined to compute a single (aggregated) loss function.

**Decoupled learning:** Efficiently performing decoupled learning where each MLP is assigned a **different** loss function is straightforward when using SGD; see steps outlined below for one SGD step.  
```python
# Consider the earlier example with num_networks = 1000 and each MLP architecture [10, 50, 50, 2]
d_in = architecture[0]    # 10 inputs per MLP
d_out = architecture[-1]  # 2 outputs per MLP

# Steps involved in one iteration of mini-batch gradient descent  
batch_size = 64
learning_rate = 1e-3
x = torch.randn(batch_size, num_networks, d_in)         # (64, 1000, 10)  
targets = torch.randn(batch_size, num_networks, d_out)  # (64, 1000, 2)

# Forward pass through all MLPs simultaneously 
predictions = model(x)  # (64, 2000)
predictions = predictions.view(batch_size, num_networks, d_out)  # (64, 1000, 2)

# Compute total loss across all MLPs
total_loss = 0
for i in range(num_networks):
    loss_i = torch.nn.functional.mse_loss(predictions[:, i, :], targets[:, i, :])
    total_loss += loss_i

# A single backward pass can be used to compute the gradients. Due to the block-diagonal structure, gradients for MLP i's parameters depend only on loss_i (since predictions[:, i, :] depends only on MLP i's parameters across all layers).
model.zero_grad()
total_loss.backward()

for param in model.parameters():
    param.data -= learning_rate * param.grad
```
Efficient implementation of truly decoupled learning using `num_networks` instances of adaptive optimizers such as Adam is not straightforward. 



## Applications

BlockDiagonalLayer is particularly useful for:

- **Ensemble neural networks**: Training thousands of diverse models simultaneously
- **Implicit neural representations**: Learning multiple coordinate-based functions
- **Multi-head MLPs**: Parallel processing of different data views
- **Dynamical systems**: Learning decoupled latent representations
- **Meta-learning**: Training on multiple related tasks simultaneously

##

**Prasanth B. Nair**  
University of Toronto Institute for Aerospace Studies (UTIAS)  
Website: http://arrow.utias.utoronto.ca/~pbn  
Email: prasanth.nair@utoronto.ca

## License

MIT

