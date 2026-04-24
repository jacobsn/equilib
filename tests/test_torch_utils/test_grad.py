#!/usr/bin/env python3

"""Gradient tests for torch_utils rotation and intrinsic matrix functions.

Uses torch.autograd.gradcheck (double precision, central finite differences)
to verify that the analytical Jacobians are numerically correct.
"""

import pytest
import torch

from equilib.torch_utils.rotation import (
    create_rotation_matrix,
    create_rotation_matrix_at_once,
)
from equilib.torch_utils.intrinsic import create_intrinsic_matrix


@pytest.mark.parametrize("z_down", [True, False])
def test_rotation_matrix_gradcheck(z_down: bool) -> None:
    """create_rotation_matrix: analytical Jacobians match finite differences."""
    dtype = torch.float64
    device = torch.device("cpu")

    roll = torch.tensor(0.1, dtype=dtype, requires_grad=True)
    pitch = torch.tensor(0.2, dtype=dtype, requires_grad=True)
    yaw = torch.tensor(0.3, dtype=dtype, requires_grad=True)

    def fn(r, p, y):
        return create_rotation_matrix(
            r, p, y, z_down=z_down, dtype=dtype, device=device
        )

    torch.autograd.gradcheck(
        fn, (roll, pitch, yaw), eps=1e-6, rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("z_down", [True, False])
def test_rotation_matrix_at_once_gradcheck(z_down: bool) -> None:
    """create_rotation_matrix_at_once: analytical Jacobians match finite differences."""
    dtype = torch.float64
    device = torch.device("cpu")

    roll = torch.tensor(0.1, dtype=dtype, requires_grad=True)
    pitch = torch.tensor(0.2, dtype=dtype, requires_grad=True)
    yaw = torch.tensor(0.3, dtype=dtype, requires_grad=True)

    def fn(r, p, y):
        return create_rotation_matrix_at_once(
            r, p, y, z_down=z_down, dtype=dtype, device=device
        )

    torch.autograd.gradcheck(
        fn, (roll, pitch, yaw), eps=1e-6, rtol=1e-4, atol=1e-5
    )


def test_intrinsic_matrix_gradcheck() -> None:
    """create_intrinsic_matrix: analytical Jacobians match finite differences."""
    dtype = torch.float64
    device = torch.device("cpu")

    fov_x = torch.tensor(90.0, dtype=dtype, requires_grad=True)
    skew = torch.tensor(0.0, dtype=dtype, requires_grad=True)

    def fn(f, s):
        return create_intrinsic_matrix(
            height=64, width=64, fov_x=f, skew=s, dtype=dtype, device=device
        )

    torch.autograd.gradcheck(
        fn, (fov_x, skew), eps=1e-6, rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("param_name", ["roll", "pitch", "yaw"])
def test_rotation_matrix_grad_nonzero(param_name: str) -> None:
    """Each rotation angle produces a non-zero gradient in the output matrix."""
    dtype = torch.float64
    device = torch.device("cpu")

    vals = {"roll": 0.1, "pitch": 0.2, "yaw": 0.3}
    tensors = {
        k: torch.tensor(v, dtype=dtype, requires_grad=(k == param_name))
        for k, v in vals.items()
    }

    R = create_rotation_matrix(
        tensors["roll"],
        tensors["pitch"],
        tensors["yaw"],
        z_down=True,
        dtype=dtype,
        device=device,
    )
    R.sum().backward()

    grad = tensors[param_name].grad
    assert grad is not None, f"{param_name}.grad is None"
    assert grad.abs() > 0, f"{param_name}.grad is zero"


def test_intrinsic_matrix_grad_nonzero() -> None:
    """fov_x produces a non-zero gradient in the intrinsic matrix."""
    dtype = torch.float64
    device = torch.device("cpu")

    fov_x = torch.tensor(90.0, dtype=dtype, requires_grad=True)
    K = create_intrinsic_matrix(
        height=64, width=64, fov_x=fov_x, skew=0.0, dtype=dtype, device=device
    )
    K.sum().backward()

    assert fov_x.grad is not None
    assert fov_x.grad.abs() > 0
