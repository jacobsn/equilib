#!/usr/bin/env python3

from copy import deepcopy
import os

import numpy as np

import pytest

import torch

from equilib.equi2pers.base import equi2pers

from tests.helpers.image_io import (
    load2numpy,
    load2torch,
    # save,
)
from tests.helpers.timer import func_timer

# from tests.helpers.rot_path import (
#     create_rots,
#     create_rots_pitch,
#     create_rots_yaw,
# )

IMG_ROOT = "tests/data"
SAVE_ROOT = "tests/test_equi2pers/results"
IMG_NAME = "test.jpg"


def get_numpy_img(dtype: np.dtype = np.float32):
    path = os.path.join(IMG_ROOT, IMG_NAME)
    img = load2numpy(path, dtype=dtype, is_cv2=False)
    return img


def get_torch_img(dtype: torch.dtype = torch.float32):
    path = os.path.join(IMG_ROOT, IMG_NAME)
    img = load2torch(path, dtype=dtype, is_cv2=False)
    return img


def make_batch(img, bs: int = 1):
    if isinstance(img, np.ndarray):
        imgs = np.empty((bs, *img.shape), dtype=img.dtype)
        for b in range(bs):
            imgs[b, ...] = deepcopy(img)
    elif torch.is_tensor(img):
        imgs = torch.empty((bs, *img.shape), dtype=img.dtype)
        for b in range(bs):
            imgs[b, ...] = img.clone()
    else:
        raise ValueError()
    return imgs


def numpy_single(
    height: int,
    width: int,
    fov_x: float,
    z_down: bool,
    mode: str,
    dtype: np.dtype,
) -> None:
    # print parameters for debugging
    print()
    print("grid(height, width, fov_x):", (height, width, fov_x))
    print("dtype:", dtype)
    print("z_down:", z_down)
    print("mode:", mode)

    # just a single image and rotation dictionary
    img = get_numpy_img(dtype=dtype)
    rot = {
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    }

    out = func_timer(equi2pers)(
        equi=img,
        rots=rot,
        height=height,
        width=width,
        fov_x=fov_x,
        skew=0.0,
        mode=mode,
        z_down=z_down,
    )

    assert out.shape == (3, height, width)
    assert out.dtype == dtype


def torch_single(
    height: int,
    width: int,
    fov_x: float,
    z_down: bool,
    mode: str,
    dtype: torch.dtype,
) -> None:
    # print parameters for debugging
    print()
    print("grid(height, width, fov_x):", (height, width, fov_x))
    print("dtype:", dtype)
    print("z_down:", z_down)
    print("mode:", mode)

    # just a single image and rotation dictionary
    img = get_torch_img(dtype=dtype)
    rot = {
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    }

    out = func_timer(equi2pers)(
        equi=img.clone(),
        rots=rot,
        height=height,
        width=width,
        fov_x=fov_x,
        skew=0.0,
        mode=mode,
        z_down=z_down,
    )

    assert out.shape == (3, height, width)
    assert out.dtype == dtype


@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [64])
@pytest.mark.parametrize("fov_x", [90.0])
@pytest.mark.parametrize("z_down", [False])
@pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_single(
    height: int,
    width: int,
    fov_x: float,
    z_down: bool,
    mode: str,
    dtype: np.dtype,
) -> None:
    numpy_single(
        height=height,
        width=width,
        fov_x=fov_x,
        z_down=z_down,
        mode=mode,
        dtype=dtype,
    )


@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [64])
@pytest.mark.parametrize("fov_x", [90.0])
@pytest.mark.parametrize("z_down", [False])
@pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_torch_single(
    height: int,
    width: int,
    fov_x: float,
    z_down: bool,
    mode: str,
    dtype: torch.dtype,
) -> None:
    torch_single(
        height=height,
        width=width,
        fov_x=fov_x,
        z_down=z_down,
        mode=mode,
        dtype=dtype,
    )


def test_numpy_fov_y() -> None:
    """equi2pers with explicit fov_y produces the correct output shape and dtype."""
    img = get_numpy_img(dtype=np.float32)
    rot = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
    height, width = 32, 64
    fov_x = 90.0
    # Derive expected fov_y from aspect ratio (should give identical result to
    # omitting fov_y since square pixels are assumed by default).
    import math
    fov_y = math.degrees(2 * math.atan(math.tan(math.radians(fov_x) / 2) * height / width))

    out_default = equi2pers(equi=img, rots=rot, height=height, width=width, fov_x=fov_x)
    out_fov_y = equi2pers(equi=img, rots=rot, height=height, width=width, fov_x=fov_x, fov_y=fov_y)

    assert out_fov_y.shape == (3, height, width)
    assert out_fov_y.dtype == np.float32
    np.testing.assert_allclose(out_default, out_fov_y, rtol=1e-5, atol=1e-5)

    # Verify that a different fov_y produces a different result
    out_wide = equi2pers(equi=img, rots=rot, height=height, width=width, fov_x=fov_x, fov_y=fov_y * 1.5)
    assert not np.allclose(out_default, out_wide, rtol=1e-5, atol=1e-5)


def test_torch_fov_y() -> None:
    """equi2pers with explicit fov_y produces the correct output shape and dtype."""
    img = get_torch_img(dtype=torch.float32)
    rot = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
    height, width = 32, 64
    fov_x = 90.0
    import math
    fov_y = math.degrees(2 * math.atan(math.tan(math.radians(fov_x) / 2) * height / width))

    out_default = equi2pers(equi=img.clone(), rots=rot, height=height, width=width, fov_x=fov_x)
    out_fov_y = equi2pers(equi=img.clone(), rots=rot, height=height, width=width, fov_x=fov_x, fov_y=fov_y)

    assert out_fov_y.shape == (3, height, width)
    assert out_fov_y.dtype == torch.float32
    torch.testing.assert_close(out_default, out_fov_y, rtol=1e-4, atol=1e-4)

    # Verify that a different fov_y produces a different result
    out_wide = equi2pers(equi=img.clone(), rots=rot, height=height, width=width, fov_x=fov_x, fov_y=fov_y * 1.5)
    assert not torch.allclose(out_default, out_wide, rtol=1e-5, atol=1e-5)
