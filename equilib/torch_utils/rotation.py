#!/usr/bin/env python3

from typing import Dict, List, Union

import torch


def create_global2camera_rotation_matrix(
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Rotation from global (world) to camera coordinates"""
    R_XY = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],  # X <-> Y
        dtype=dtype,
    )
    R_YZ = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],  # Y <-> Z
        dtype=dtype,
    )
    R = R_XY @ R_YZ
    return R.to(device)


def create_rotation_matrix(
    roll: Union[float, torch.Tensor],
    pitch: Union[float, torch.Tensor],
    yaw: Union[float, torch.Tensor],
    z_down: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create Rotation Matrix

    params:
    - roll, pitch, yaw (float or torch.Tensor): in radians
    - z_down (bool): flips pitch and yaw directions
    - dtype (torch.dtype): data types

    returns:
    - R (torch.Tensor): 3x3 rotation matrix
    """

    if not isinstance(roll, torch.Tensor):
        roll = torch.tensor(roll, dtype=dtype, device=device)
    if not isinstance(pitch, torch.Tensor):
        pitch = torch.tensor(pitch, dtype=dtype, device=device)
    if not isinstance(yaw, torch.Tensor):
        yaw = torch.tensor(yaw, dtype=dtype, device=device)

    zeros = roll.new_zeros(())
    ones = roll.new_ones(())

    # calculate rotation about the x-axis
    R_x = torch.stack(
        [
            torch.stack([ones, zeros, zeros]),
            torch.stack([zeros, torch.cos(roll), -torch.sin(roll)]),
            torch.stack([zeros, torch.sin(roll), torch.cos(roll)]),
        ]
    )
    # calculate rotation about the y-axis
    if not z_down:
        pitch = -pitch
    R_y = torch.stack(
        [
            torch.stack([torch.cos(pitch), zeros, torch.sin(pitch)]),
            torch.stack([zeros, ones, zeros]),
            torch.stack([-torch.sin(pitch), zeros, torch.cos(pitch)]),
        ]
    )
    # calculate rotation about the z-axis
    if not z_down:
        yaw = -yaw
    R_z = torch.stack(
        [
            torch.stack([torch.cos(yaw), -torch.sin(yaw), zeros]),
            torch.stack([torch.sin(yaw), torch.cos(yaw), zeros]),
            torch.stack([zeros, zeros, ones]),
        ]
    )
    R = R_z @ R_y @ R_x
    return R.to(device)


def create_rotation_matrix_at_once(
    roll: Union[float, torch.Tensor],
    pitch: Union[float, torch.Tensor],
    yaw: Union[float, torch.Tensor],
    z_down: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create rotation matrix at once"

    params:
    - roll, pitch, yaw (float or torch.Tensor): in radians
    - z_down (bool): flips pitch and yaw directions
    - dtype (torch.dtype): data types
    - device (torch.device): torch.device("cpu")

    returns:
    - R (torch.Tensor): 3x3 rotation matrix

    NOTE: same results as `create_rotation_matrix` but a little bit faster
    """

    if not isinstance(roll, torch.Tensor):
        roll = torch.tensor(roll, dtype=dtype, device=device)
    if not isinstance(pitch, torch.Tensor):
        pitch = torch.tensor(pitch, dtype=dtype, device=device)
    if not isinstance(yaw, torch.Tensor):
        yaw = torch.tensor(yaw, dtype=dtype, device=device)

    if not z_down:
        pitch = -pitch
        yaw = -yaw

    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    cos_p = torch.cos(pitch)
    sin_p = torch.sin(pitch)
    cos_r = torch.cos(roll)
    sin_r = torch.sin(roll)

    return torch.stack(
        [
            torch.stack(
                [
                    cos_y * cos_p,
                    cos_y * sin_p * sin_r - sin_y * cos_r,
                    cos_y * sin_p * cos_r + sin_y * sin_r,
                ]
            ),
            torch.stack(
                [
                    sin_y * cos_p,
                    sin_y * sin_p * sin_r + cos_y * cos_r,
                    sin_y * sin_p * cos_r - cos_y * sin_r,
                ]
            ),
            torch.stack(
                [
                    -sin_p,
                    cos_p * sin_r,
                    cos_p * cos_r,
                ]
            ),
        ]
    ).to(device)


def create_rotation_matrices(
    rots: List[Dict[str, Union[float, torch.Tensor]]],
    z_down: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create rotation matrices from batch of rotations

    This methods creates a bx3x3 torch.Tensor where `b` refers to the number
    of rotations (rots) given in the input
    """

    return torch.stack(
        [
            create_rotation_matrix(
                **rot, z_down=z_down, dtype=dtype, device=device
            )
            for rot in rots
        ]
    )


def create_rotation_matrix_dep(
    x: Union[float, torch.Tensor],
    y: Union[float, torch.Tensor],
    z: Union[float, torch.Tensor],
    z_down: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create Rotation Matrix

    NOTE: DEPRECATED
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=dtype, device=device)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=dtype, device=device)
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z, dtype=dtype, device=device)

    zeros = x.new_zeros(())
    ones = x.new_ones(())

    # calculate rotation about the x-axis
    R_x = torch.stack(
        [
            torch.stack([ones, zeros, zeros]),
            torch.stack([zeros, torch.cos(x), -torch.sin(x)]),
            torch.stack([zeros, torch.sin(x), torch.cos(x)]),
        ]
    )
    # calculate rotation about the y-axis
    if not z_down:
        y = -y
    R_y = torch.stack(
        [
            torch.stack([torch.cos(y), zeros, -torch.sin(y)]),
            torch.stack([zeros, ones, zeros]),
            torch.stack([torch.sin(y), zeros, torch.cos(y)]),
        ]
    )
    # calculate rotation about the z-axis
    if not z_down:
        z = -z
    R_z = torch.stack(
        [
            torch.stack([torch.cos(z), torch.sin(z), zeros]),
            torch.stack([-torch.sin(z), torch.cos(z), zeros]),
            torch.stack([zeros, zeros, ones]),
        ]
    )
    R = R_z @ R_y @ R_x
    return R.to(device)
