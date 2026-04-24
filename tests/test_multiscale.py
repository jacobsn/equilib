#!/usr/bin/env python3

"""Tests for equilib.multiscale.make_equi_pyramid.

Covers:
- Output list length equals num_levels.
- Spatial dimensions decrease monotonically at each level.
- Batch dimension is preserved (or stripped for single-image inputs).
- dtype is preserved across levels.
- Correct behaviour for numpy arrays and torch tensors.
- Gradient flow: torch pyramid is fully differentiable — gradients propagate
  from a coarse-level loss back to the original-resolution tensor.
- Integration with equi2pers: coarse-level warping gradients reach rotation
  parameters.
- Error handling: invalid num_levels / scale_factor raise ValueError.
"""

import pytest
import numpy as np
import torch

from equilib import make_equi_pyramid
from equilib.equi2pers.torch import run as equi2pers_run

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_B, _C, _H, _W = 2, 3, 64, 128
_DTYPE_TORCH = torch.float32
_DTYPE_NP = np.float32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_torch(batch=True, dtype=_DTYPE_TORCH, requires_grad=False):
    t = torch.rand(_B, _C, _H, _W, dtype=dtype)
    if not batch:
        t = t[0]
    if requires_grad:
        t.requires_grad_(True)
    return t


def _make_numpy(batch=True, dtype=_DTYPE_NP):
    arr = np.random.rand(_B, _C, _H, _W).astype(dtype)
    if not batch:
        arr = arr[0]
    return arr


# ---------------------------------------------------------------------------
# Shape / length tests (torch)
# ---------------------------------------------------------------------------

class TestTorchPyramidShape:
    def test_length_equals_num_levels(self):
        pyr = make_equi_pyramid(_make_torch(), num_levels=4)
        assert len(pyr) == 4

    def test_single_level_returns_original(self):
        img = _make_torch()
        pyr = make_equi_pyramid(img, num_levels=1)
        assert len(pyr) == 1
        assert pyr[0].shape == img.shape

    def test_level_zero_is_original_shape(self):
        img = _make_torch()
        pyr = make_equi_pyramid(img, num_levels=3)
        assert pyr[0].shape == img.shape

    def test_spatial_dims_decrease(self):
        pyr = make_equi_pyramid(_make_torch(), num_levels=4)
        for i in range(1, len(pyr)):
            assert pyr[i].shape[-2] <= pyr[i - 1].shape[-2]
            assert pyr[i].shape[-1] <= pyr[i - 1].shape[-1]

    def test_batch_dim_preserved(self):
        pyr = make_equi_pyramid(_make_torch(batch=True), num_levels=3)
        for level in pyr:
            assert level.shape[0] == _B

    def test_single_image_no_batch_dim(self):
        img = _make_torch(batch=False)  # (C, H, W)
        pyr = make_equi_pyramid(img, num_levels=3)
        for level in pyr:
            assert level.ndim == 3, f"Expected 3 dims, got {level.ndim}"

    def test_dtype_preserved_float32(self):
        pyr = make_equi_pyramid(_make_torch(dtype=torch.float32), num_levels=3)
        for level in pyr:
            assert level.dtype == torch.float32

    def test_dtype_preserved_float64(self):
        pyr = make_equi_pyramid(_make_torch(dtype=torch.float64), num_levels=3)
        for level in pyr:
            assert level.dtype == torch.float64

    def test_dtype_preserved_uint8(self):
        img = (torch.rand(_B, _C, _H, _W) * 255).to(torch.uint8)
        pyr = make_equi_pyramid(img, num_levels=3)
        for level in pyr:
            assert level.dtype == torch.uint8

    def test_scale_factor_half(self):
        pyr = make_equi_pyramid(_make_torch(), num_levels=3, scale_factor=0.5)
        # Level 1 should be approximately half of level 0
        assert pyr[1].shape[-2] == _H // 2
        assert pyr[1].shape[-1] == _W // 2

    def test_scale_factor_quarter(self):
        pyr = make_equi_pyramid(_make_torch(), num_levels=2, scale_factor=0.25)
        assert pyr[1].shape[-2] == _H // 4
        assert pyr[1].shape[-1] == _W // 4

    @pytest.mark.parametrize("mode", ["bilinear", "bicubic", "area"])
    def test_interpolation_modes(self, mode):
        pyr = make_equi_pyramid(_make_torch(), num_levels=3, mode=mode)
        assert len(pyr) == 3


# ---------------------------------------------------------------------------
# Shape / length tests (numpy)
# ---------------------------------------------------------------------------

class TestNumpyPyramidShape:
    def test_length_equals_num_levels(self):
        pyr = make_equi_pyramid(_make_numpy(), num_levels=4)
        assert len(pyr) == 4

    def test_single_level_returns_original(self):
        arr = _make_numpy()
        pyr = make_equi_pyramid(arr, num_levels=1)
        assert len(pyr) == 1
        assert pyr[0].shape == arr.shape

    def test_level_zero_is_original_shape(self):
        arr = _make_numpy()
        pyr = make_equi_pyramid(arr, num_levels=3)
        assert pyr[0].shape == arr.shape

    def test_spatial_dims_decrease(self):
        pyr = make_equi_pyramid(_make_numpy(), num_levels=4)
        for i in range(1, len(pyr)):
            assert pyr[i].shape[-2] <= pyr[i - 1].shape[-2]
            assert pyr[i].shape[-1] <= pyr[i - 1].shape[-1]

    def test_batch_dim_preserved(self):
        pyr = make_equi_pyramid(_make_numpy(batch=True), num_levels=3)
        for level in pyr:
            assert level.shape[0] == _B

    def test_single_image_no_batch_dim(self):
        arr = _make_numpy(batch=False)  # (C, H, W)
        pyr = make_equi_pyramid(arr, num_levels=3)
        for level in pyr:
            assert level.ndim == 3, f"Expected 3 dims, got {level.ndim}"

    def test_dtype_preserved_float32(self):
        pyr = make_equi_pyramid(_make_numpy(dtype=np.float32), num_levels=3)
        for level in pyr:
            assert level.dtype == np.float32

    def test_dtype_preserved_float64(self):
        pyr = make_equi_pyramid(_make_numpy(dtype=np.float64), num_levels=3)
        for level in pyr:
            assert level.dtype == np.float64

    def test_dtype_preserved_uint8(self):
        arr = (np.random.rand(_B, _C, _H, _W) * 255).astype(np.uint8)
        pyr = make_equi_pyramid(arr, num_levels=3)
        for level in pyr:
            assert level.dtype == np.uint8

    def test_scale_factor_half(self):
        pyr = make_equi_pyramid(_make_numpy(), num_levels=3, scale_factor=0.5)
        assert pyr[1].shape[-2] == _H // 2
        assert pyr[1].shape[-1] == _W // 2


# ---------------------------------------------------------------------------
# Differentiability tests (torch)
# ---------------------------------------------------------------------------

class TestTorchPyramidGrad:
    def test_grad_flows_from_coarse_level_to_input(self):
        """Gradient of a coarse-level loss reaches the original tensor."""
        img = _make_torch(requires_grad=True)
        pyr = make_equi_pyramid(img, num_levels=4)
        loss = pyr[-1].sum()  # loss at the coarsest level
        loss.backward()
        assert img.grad is not None, "img.grad is None"
        assert img.grad.abs().sum() > 0, "img.grad is all-zero"

    def test_grad_flows_from_intermediate_level(self):
        """Gradient flows from an intermediate level (not just the coarsest)."""
        img = _make_torch(requires_grad=True)
        pyr = make_equi_pyramid(img, num_levels=4)
        loss = pyr[2].sum()
        loss.backward()
        assert img.grad is not None
        assert img.grad.abs().sum() > 0

    def test_equi_grad_through_pyramid_and_warp(self):
        """Gradient of equi2pers on a pyramid level reaches the original equi."""
        equi = torch.rand(1, 3, 32, 64, dtype=torch.float64, requires_grad=True)
        pyr = make_equi_pyramid(equi, num_levels=3)
        # apply equi2pers on the coarsest level
        coarse = pyr[-1]
        out = equi2pers_run(
            equi=coarse,
            rots=[{"roll": 0.0, "pitch": 0.0, "yaw": 0.0}],
            height=8,
            width=8,
            fov_x=90.0,
            skew=0.0,
            z_down=True,
            mode="bilinear",
        )
        out.sum().backward()
        assert equi.grad is not None
        assert equi.grad.abs().sum() > 0

    def test_rotation_grad_through_pyramid_level(self):
        """Rotation parameter gradient flows when optimising on a pyramid level."""
        equi = torch.rand(1, 3, 32, 64, dtype=torch.float64)
        pyr = make_equi_pyramid(equi, num_levels=3)

        yaw = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        out = equi2pers_run(
            equi=pyr[-1],
            rots=[{"roll": 0.0, "pitch": 0.0, "yaw": yaw}],
            height=8,
            width=8,
            fov_x=90.0,
            skew=0.0,
            z_down=True,
            mode="bilinear",
        )
        out.sum().backward()
        assert yaw.grad is not None
        assert yaw.grad.abs() > 0

    def test_pyramid_level_is_tensor(self):
        """Every level in the pytorch pyramid is a torch.Tensor."""
        pyr = make_equi_pyramid(_make_torch(), num_levels=4)
        for level in pyr:
            assert torch.is_tensor(level)

    def test_pyramid_level_is_ndarray(self):
        """Every level in the numpy pyramid is a numpy.ndarray."""
        pyr = make_equi_pyramid(_make_numpy(), num_levels=4)
        for level in pyr:
            assert isinstance(level, np.ndarray)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestMakeEquiPyramidErrors:
    def test_invalid_num_levels_zero(self):
        with pytest.raises(ValueError, match="num_levels"):
            make_equi_pyramid(_make_torch(), num_levels=0)

    def test_invalid_num_levels_negative(self):
        with pytest.raises(ValueError, match="num_levels"):
            make_equi_pyramid(_make_torch(), num_levels=-1)

    def test_invalid_scale_factor_ge_one(self):
        with pytest.raises(ValueError, match="scale_factor"):
            make_equi_pyramid(_make_torch(), num_levels=2, scale_factor=1.0)

    def test_invalid_scale_factor_zero(self):
        with pytest.raises(ValueError, match="scale_factor"):
            make_equi_pyramid(_make_torch(), num_levels=2, scale_factor=0.0)

    def test_invalid_scale_factor_negative(self):
        with pytest.raises(ValueError, match="scale_factor"):
            make_equi_pyramid(_make_torch(), num_levels=2, scale_factor=-0.5)

    def test_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError):
            make_equi_pyramid("not an image", num_levels=2)  # type: ignore


# ---------------------------------------------------------------------------
# Import test
# ---------------------------------------------------------------------------

def test_importable_from_equilib():
    """make_equi_pyramid is importable from the top-level equilib package."""
    from equilib import make_equi_pyramid as fn  # noqa: F401
    assert callable(fn)
