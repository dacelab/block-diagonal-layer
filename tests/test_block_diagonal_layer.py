import pytest
import torch
import torch.nn as nn

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from block_diagonal_layer import (
    BlockDiagonalLayer,
    _batched_matvec_input_one,
    _batched_matvec_output_one,
    _batched_matvec_general,
)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture
def default_params():
    return {
        "num_networks": 500,
        "input_features_per_network": 10,
        "output_features_per_network": 20,
        "activation": nn.ReLU(),
        "use_bias": True,
        "weight_init_method": "pytorch_default",
        "bias_init_method": "pytorch_default",
    }


@pytest.fixture
def layer_instance(default_params):
    return BlockDiagonalLayer(**default_params)


def test_successful_initialization(default_params):
    try:
        layer = BlockDiagonalLayer(**default_params)
        assert isinstance(layer, BlockDiagonalLayer)
        assert layer.num_networks == default_params["num_networks"]
        assert (
            layer.input_features_per_network
            == default_params["input_features_per_network"]
        )
        assert (
            layer.output_features_per_network
            == default_params["output_features_per_network"]
        )
        assert isinstance(layer.activation, nn.Module)
        assert layer.use_bias == default_params["use_bias"]
        assert layer.weight.shape == (
            default_params["num_networks"],
            default_params["output_features_per_network"],
            default_params["input_features_per_network"],
        )
        if default_params["use_bias"]:
            assert layer.bias is not None
            assert layer.bias.shape == (
                default_params["num_networks"],
                default_params["output_features_per_network"],
            )
        else:
            assert layer.bias is None
    except Exception as e:
        pytest.fail(f"Initialization failed with default params: {e}")


def test_initialization_no_bias(default_params):
    params_no_bias = default_params.copy()
    params_no_bias["use_bias"] = False
    layer = BlockDiagonalLayer(**params_no_bias)
    assert layer.use_bias is False
    assert layer.bias is None
    assert layer.bias_init_method == params_no_bias["bias_init_method"]


@pytest.mark.parametrize(
    "param_name, invalid_value, expected_error_type, expected_error_msg_part",
    [
        ("num_networks", 0, ValueError, "num_networks must be positive"),
        ("num_networks", -1, ValueError, "num_networks must be positive"),
        (
            "input_features_per_network",
            0,
            ValueError,
            "Feature dimensions must be positive",
        ),
        (
            "output_features_per_network",
            0,
            ValueError,
            "Feature dimensions must be positive",
        ),
        ("activation", None, ValueError, "Activation function must be provided"),
        (
            "activation",
            "not_a_module",
            ValueError,
            "Activation must be an instance of nn.Module",
        ),
        (
            "weight_init_method",
            "invalid_method",
            ValueError,
            "Unknown weight_init_method",
        ),
        (
            "bias_init_method",
            "invalid_bias_method",
            ValueError,
            "Unknown bias_init_method",
        ),
        (
            "weight_init_params",
            30.0,
            TypeError,
            "weight_init_params as a float/int is only supported when weight_init_method is 'siren_hidden'",
        ),
    ],
)
def test_invalid_initialization_args(
    default_params,
    param_name,
    invalid_value,
    expected_error_type,
    expected_error_msg_part,
):
    params = default_params.copy()
    params[param_name] = invalid_value

    if param_name == "weight_init_params" and isinstance(invalid_value, (int, float)):
        if params["weight_init_method"] != "siren_hidden":
            pass
        else:
            pytest.skip(
                "Skipping specific weight_init_params float test for siren_hidden context here"
            )

    with pytest.raises(expected_error_type, match=expected_error_msg_part):
        BlockDiagonalLayer(**params)


def test_siren_hidden_missing_omega(default_params):
    params = default_params.copy()
    params["weight_init_method"] = "siren_hidden"
    params["weight_init_params"] = {}
    with pytest.raises(ValueError, match="omega parameter required for siren_hidden"):
        BlockDiagonalLayer(**params)


def test_siren_hidden_invalid_omega(default_params):
    params = default_params.copy()
    params["weight_init_method"] = "siren_hidden"
    params["weight_init_params"] = {"omega": 0}
    with pytest.raises(ValueError, match="omega must be a positive number"):
        BlockDiagonalLayer(**params)


def test_siren_hidden_correct_omega_float(default_params):
    params = default_params.copy()
    params["weight_init_method"] = "siren_hidden"
    params["weight_init_params"] = 30.0
    try:
        layer = BlockDiagonalLayer(**params)
        assert layer.weight_init_params["omega"] == 30.0
    except Exception as e:
        pytest.fail(f"Initialization failed for siren_hidden with float omega: {e}")


def test_siren_hidden_correct_omega_dict(default_params):
    params = default_params.copy()
    params["weight_init_method"] = "siren_hidden"
    params["weight_init_params"] = {"omega": 30.0}
    try:
        layer = BlockDiagonalLayer(**params)
        assert layer.weight_init_params["omega"] == 30.0
    except Exception as e:
        pytest.fail(f"Initialization failed for siren_hidden with dict omega: {e}")


def test_xavier_initialization(default_params):
    params = default_params.copy()
    params["weight_init_method"] = "xavier_uniform"
    params["weight_init_params"] = {"gain": 1.0}

    layer = BlockDiagonalLayer(**params)
    # Just verify it initializes without error
    assert layer.weight is not None


@pytest.mark.parametrize(
    "input_shape, expected_error_msg",
    [
        ((32,), "Input tensor must be 2D or 3D"),  # 1D input
        ((32, 10, 20, 5), "Input tensor must be 2D or 3D"),  # 4D input
        ((32, 999), "Input dimension mismatch for 2D input"),  # Wrong 2D size
        ((32, 999, 10), "Input dimension mismatch for 3D input"),  # Wrong 3D networks
        ((32, 500, 999), "Input dimension mismatch for 3D input"),  # Wrong 3D features
    ],
)
def test_invalid_input_shapes(default_params, input_shape, expected_error_msg):
    layer = BlockDiagonalLayer(**default_params)
    x = torch.randn(*input_shape)

    with pytest.raises(ValueError, match=expected_error_msg):
        layer(x)


def test_3d_input_support(default_params):
    """Test that 3D input works correctly and produces same results as 2D input."""
    layer = BlockDiagonalLayer(**default_params)
    batch_size = 32

    # 2D input: (batch_size, num_networks * input_features_per_network)
    x_2d = torch.randn(
        batch_size,
        default_params["num_networks"] * default_params["input_features_per_network"],
    )

    # 3D input: (batch_size, num_networks, input_features_per_network)
    x_3d = x_2d.view(
        batch_size,
        default_params["num_networks"],
        default_params["input_features_per_network"],
    )

    output_2d = layer(x_2d)
    output_3d = layer(x_3d)

    assert torch.allclose(
        output_2d, output_3d, atol=1e-6
    ), "2D and 3D inputs should produce identical outputs"


def test_functional_equivalence_and_gradients(default_params):
    """
    Test to check that the outputs and gradients of BlockDiagonalLayer are equivalent to those
    computed by a stack of nn.Linear layers.
    """
    set_seed(42)

    num_networks = default_params["num_networks"]
    input_feats = default_params["input_features_per_network"]
    output_feats = default_params["output_features_per_network"]
    activation_fn = nn.Tanh()
    use_bias = default_params["use_bias"]

    block_diag_layer = BlockDiagonalLayer(
        num_networks,
        input_feats,
        output_feats,
        activation_fn,
        use_bias=use_bias,
        weight_init_method="pytorch_default",
        bias_init_method="pytorch_default",
    )
    block_diag_layer.train()

    reference_layers = nn.ModuleList()
    for i in range(num_networks):
        linear_layer = nn.Linear(input_feats, output_feats, bias=use_bias)
        reference_layers.append(linear_layer)
    reference_layers.train()

    # Set identical weights and biases
    new_weights = torch.randn_like(block_diag_layer.weight.data)
    block_diag_layer.weight.data.copy_(new_weights)

    if use_bias:
        new_biases = torch.randn_like(block_diag_layer.bias.data)
        block_diag_layer.bias.data.copy_(new_biases)

    # Copy weights to reference layers
    for i in range(num_networks):
        reference_layers[i].weight.data.copy_(block_diag_layer.weight.data[i])
        if use_bias:
            reference_layers[i].bias.data.copy_(block_diag_layer.bias.data[i])

    batch_size = 50
    x_input = torch.randn(batch_size, num_networks * input_feats, requires_grad=True)

    output_block_diag = block_diag_layer(x_input)

    x_reshaped = x_input.view(batch_size, num_networks, input_feats)
    outputs_ref_list = []
    for i in range(num_networks):
        output_linear = reference_layers[i](x_reshaped[:, i, :])
        outputs_ref_list.append(output_linear)

    output_ref_stacked = torch.stack(outputs_ref_list, dim=1)
    output_ref_activated = activation_fn(output_ref_stacked.reshape(batch_size, -1))

    assert torch.allclose(
        output_block_diag, output_ref_activated, atol=1e-6
    ), "Outputs of BlockDiagonalLayer and reference nn.Linear stack do not match."

    grad_output = torch.randn_like(output_block_diag)

    block_diag_layer.zero_grad()
    for layer in reference_layers:
        layer.zero_grad()
    x_input.grad = None

    output_block_diag.backward(gradient=grad_output, retain_graph=True)
    block_diag_grad_input = x_input.grad.clone()

    x_input.grad = None

    output_ref_activated.backward(gradient=grad_output)
    ref_grad_input = x_input.grad

    for i in range(num_networks):
        assert (
            block_diag_layer.weight.grad is not None
        ), "BlockDiagonalLayer weight grad is None"
        assert (
            reference_layers[i].weight.grad is not None
        ), f"Reference layer {i} weight grad is None"
        assert torch.allclose(
            block_diag_layer.weight.grad[i], reference_layers[i].weight.grad, atol=1e-5
        ), f"Weight gradients for network {i} do not match."

    if use_bias:
        assert (
            block_diag_layer.bias.grad is not None
        ), "BlockDiagonalLayer bias grad is None"
        for i in range(num_networks):
            assert (
                reference_layers[i].bias.grad is not None
            ), f"Reference layer {i} bias grad is None"
            assert torch.allclose(
                block_diag_layer.bias.grad[i], reference_layers[i].bias.grad, atol=1e-5
            ), f"Bias gradients for network {i} do not match."

    assert block_diag_grad_input is not None, "BlockDiagonalLayer input grad is None"
    assert ref_grad_input is not None, "Reference input grad is None"
    assert torch.allclose(
        block_diag_grad_input, ref_grad_input, atol=1e-5
    ), "Input gradients do not match."


class TestEdgeCases:
    """
    Not a comprehensive test suite.
    """

    def test_single_network(self):
        layer = BlockDiagonalLayer(
            num_networks=1,
            input_features_per_network=5,
            output_features_per_network=3,
            activation=nn.Identity(),
        )

        x = torch.randn(4, 5)
        output = layer(x)
        assert output.shape == (4, 3)

    def test_large_number_of_networks(self):
        layer = BlockDiagonalLayer(
            num_networks=1000,
            input_features_per_network=2,
            output_features_per_network=3,
            activation=nn.ReLU(),
        )

        x = torch.randn(8, 2000)
        output = layer(x)
        assert output.shape == (8, 3000)

    def test_single_input_feature(self):
        layer = BlockDiagonalLayer(
            num_networks=50,
            input_features_per_network=1,
            output_features_per_network=10,
            activation=nn.Tanh(),
        )

        x = torch.randn(16, 50)
        output = layer(x)
        assert output.shape == (16, 500)

        assert layer._batched_matvec == _batched_matvec_input_one

    def test_single_output_feature(self):
        layer = BlockDiagonalLayer(
            num_networks=50,
            input_features_per_network=10,
            output_features_per_network=1,
            activation=nn.Sigmoid(),
        )

        x = torch.randn(16, 500)
        output = layer(x)
        assert output.shape == (16, 50)

        assert layer._batched_matvec == _batched_matvec_output_one

    def test_general_case_matvec(self):
        layer = BlockDiagonalLayer(
            num_networks=50,
            input_features_per_network=10,
            output_features_per_network=20,
            activation=nn.Sigmoid(),
        )

        x = torch.randn(16, 500)
        output = layer(x)
        assert output.shape == (16, 1000)

        assert layer._batched_matvec == _batched_matvec_general


class TestPerformance:

    def _get_device_and_sync(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            sync_fn = torch.cuda.synchronize
            device_name = f"CUDA ({torch.cuda.get_device_name()})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            sync_fn = lambda: None
            device_name = "Apple Silicon MPS"
        else:
            device = torch.device("cpu")
            sync_fn = lambda: None
            device_name = "CPU"

        return device, sync_fn, device_name

    @pytest.mark.parametrize(
        "num_networks,input_feats,output_feats,batch_size",
        [
            (2000, 32, 32, 128),  # Large problem
            (1000, 32, 32, 256),  # Many small networks
            (100, 128, 256, 128),  # Fewer large networks
        ],
    )
    def test_gpu_utilization_scaling(
        self, num_networks, input_feats, output_feats, batch_size, capsys
    ):
        """
        Test GPU utilization and speedups with different problem sizes.
        """
        device, sync_fn, device_name = self._get_device_and_sync()

        if device.type not in ["cuda", "mps"]:
            pytest.skip("GPU utilization test requires GPU")

        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch._dynamo.config.cache_size_limit = 64

        layer = BlockDiagonalLayer(
            num_networks=num_networks,
            input_features_per_network=input_feats,
            output_features_per_network=output_feats,
            activation=nn.ReLU(),
        ).to(device)

        for_loop_layers = nn.ModuleList(
            [nn.Linear(input_feats, output_feats) for _ in range(num_networks)]
        ).to(device)

        for i, for_loop_layer in enumerate(for_loop_layers):
            for_loop_layer.weight.data.copy_(layer.weight[i])
            if layer.bias is not None:
                for_loop_layer.bias.data.copy_(layer.bias[i])

        x = torch.randn(batch_size, num_networks * input_feats, device=device)
        x_reshaped = x.view(batch_size, num_networks, input_feats)

        total_ops = batch_size * num_networks * input_feats * output_feats * 2
        total_params = num_networks * input_feats * output_feats

        num_iterations = 10

        print(f"\n.......Running for varying problem size on: {device_name}")
        print(
            f"Shape: {num_networks} networks, {input_feats} -> {output_feats}, batch={batch_size}"
        )
        print(f"Total parameters: {total_params:,}")
        print(f"FLOPS per iteration: {total_ops/1e9:.2f} GFLOP")

        for _ in range(5):  # Warm-up iterations
            _ = layer(x)

        sync_fn()
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            output_block = layer(x)
        sync_fn()
        block_time = time.perf_counter() - start_time

        for _ in range(5):  # Warm-up iterations
            outputs = torch.empty(batch_size, num_networks, output_feats, device=device)
            for i in range(num_networks):
                outputs[:, i, :] = for_loop_layers[i](x_reshaped[:, i, :])
            _ = nn.ReLU()(outputs.reshape(batch_size, -1))

        sync_fn()
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            outputs = torch.empty(batch_size, num_networks, output_feats, device=device)
            for i in range(num_networks):
                outputs[:, i, :] = for_loop_layers[i](x_reshaped[:, i, :])
            output_for_loop = nn.ReLU()(outputs.reshape(batch_size, -1))
        sync_fn()
        for_loop_time = time.perf_counter() - start_time

        block_avg_ms = (block_time / num_iterations) * 1000
        for_loop_avg_ms = (for_loop_time / num_iterations) * 1000
        speedup = for_loop_time / block_time

        block_gflops = (total_ops * num_iterations / 1e9) / block_time
        for_loop_gflops = (total_ops * num_iterations / 1e9) / for_loop_time

        print(f"\n{'Method':<15} {'Time (ms)':<12} {'GFLOPS':<10}")
        print(f"{'-'*36}")
        print(f"{'BlockDiagonal':<15} {block_avg_ms:<12.3f} {block_gflops:<10.1f} ")
        print(f"{'For-loop':<15} {for_loop_avg_ms:<12.3f} {for_loop_gflops:<10.1f} ")
        print(
            f"\nBlockDiagonal is {speedup:.2f}x faster than a for-loop ({block_gflops/for_loop_gflops:.2f}x throughput)"
        )

        if device.type == "cuda":
            memory_used = torch.cuda.memory_allocated(device) / 1024**2
            print(f"GPU Memory: {memory_used:.1f} MB")

        assert torch.allclose(
            output_block, output_for_loop, atol=1e-4
        ), "Outputs don't match between implementations"

    def test_sustained_gpu_load(self, capsys):
        device, sync_fn, device_name = self._get_device_and_sync()

        if device.type != "cuda":
            pytest.skip("Sustained load test designed for CUDA")

        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch._dynamo.config.cache_size_limit = 64

        num_networks = 5000
        input_feats = 128
        output_feats = 128
        batch_size = 64

        layer = BlockDiagonalLayer(
            num_networks=num_networks,
            input_features_per_network=input_feats,
            output_features_per_network=output_feats,
            activation=nn.ReLU(),
        ).to(device)

        x = torch.randn(batch_size, num_networks * input_feats, device=device)

        total_params = num_networks * input_feats * output_feats
        total_ops = batch_size * num_networks * input_feats * output_feats * 2

        print(f"\n.......Sustained GPU Load Test")
        print(f"Problem size: {total_params/1e6:.1f}M parameters")
        print(f"Operations per forward pass: {total_ops/1e9:.2f} GFLOP")
        print("Running for 5 seconds to measure sustained performance...")

        iterations = 0
        start_time = time.perf_counter()
        target_duration = 5.0

        while (time.perf_counter() - start_time) < target_duration:
            _ = layer(x)
            iterations += 1
            if iterations % 500 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"  {elapsed:.1f}s: {iterations} iterations")

        sync_fn()
        total_time = time.perf_counter() - start_time

        avg_time = total_time / iterations * 1000
        sustained_gflops = (total_ops * iterations / 1e9) / total_time

        print(f"  Total iterations: {iterations}")
        print(f"  Average time per iteration: {avg_time:.3f}ms")
        print(f"  Sustained throughput: {sustained_gflops:.1f} GFLOPS")

        memory_used = torch.cuda.memory_allocated(device) / 1024**2
        print(f"  GPU Memory Used: {memory_used:.1f} MB")

        assert (
            sustained_gflops > 100
        ), f"Expected >100 sustained GFLOPS, got {sustained_gflops:.1f}"
