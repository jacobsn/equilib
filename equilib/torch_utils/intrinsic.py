#!/usr/bin/env python3

from typing import Union

import torch

pi = torch.Tensor([3.14159265358979323846])


def deg2rad(tensor: torch.Tensor) -> torch.Tensor:
    """Function that converts angles from degrees to radians"""
    return tensor * pi.to(tensor.device).type(tensor.dtype) / 180.0


def create_intrinsic_matrix(
    height: int,
    width: int,
    fov_x: Union[float, torch.Tensor],
    skew: Union[float, torch.Tensor],
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create intrinsic matrix

    params:
    - height, width (int)
    - fov_x (float or torch.Tensor): make sure it's in degrees
    - skew (float or torch.Tensor): 0.0
    - dtype (torch.dtype): torch.float32
    - device (torch.device): torch.device("cpu")

    returns:
    - K (torch.tensor): 3x3 intrinsic matrix
    """
    if not isinstance(fov_x, torch.Tensor):
        fov_x = torch.tensor(fov_x, dtype=dtype, device=device)
    if not isinstance(skew, torch.Tensor):
        skew = torch.tensor(skew, dtype=dtype, device=device)

    f = width / (2 * torch.tan(deg2rad(fov_x) / 2))
    f = f.squeeze()  # ensure scalar (0-d) shape for scalar fov_x inputs

    zeros = f.new_zeros(())
    ones = f.new_ones(())
    width_half = f.new_tensor(width / 2)
    height_half = f.new_tensor(height / 2)

    K = torch.stack(
        [
            torch.stack([f, skew, width_half]),
            torch.stack([zeros, f, height_half]),
            torch.stack([zeros, zeros, ones]),
        ]
    )
    return K.to(device)
