#!/usr/bin/env python3

from typing import Optional

import numpy as np


def create_intrinsic_matrix(
    height: int,
    width: int,
    fov_x: float,
    skew: float,
    dtype: np.dtype = np.dtype(np.float32),
    fov_y: Optional[float] = None,
) -> np.ndarray:
    """Create intrinsic matrix

    params:
    - height, width (int)
    - fov_x (float): horizontal field of view in degrees
    - skew (float): 0.0
    - dtype (np.dtype): np.float32
    - fov_y (float, optional): vertical field of view in degrees.
      When provided, ``fy`` is computed directly from ``fov_y`` which
      allows for non-square pixels or explicit vertical-FOV control.
      When ``None`` (default), ``fy`` is derived from ``fov_x`` and the
      aspect ratio, which is equivalent to assuming square pixels.

    returns:
    - K (np.ndarray): 3x3 intrinsic matrix
    """

    # horizontal focal length
    fx = width / (2.0 * np.tan(np.radians(fov_x).astype(dtype) / 2.0))
    # vertical focal length: use fov_y if given, otherwise derive from fov_x
    if fov_y is not None:
        fy = height / (2.0 * np.tan(np.radians(fov_y).astype(dtype) / 2.0))
    else:
        fy = height / (
            2.0
            * np.tan(
                np.arctan(
                    np.tan(np.radians(fov_x).astype(dtype) / 2.0) * height / width
                )
            )
        )
    # transform between camera frame and pixel coordinates
    K = np.array(
        [[fx, skew, width / 2], [0.0, fy, height / 2], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )

    return K
