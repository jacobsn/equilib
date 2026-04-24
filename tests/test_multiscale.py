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
- Registration demo: coarse-to-fine MSE optimisation converges to the correct
  rotation for a large displacement where single-scale optimisation gets stuck
  in a local minimum.
"""

import math

import pytest
import numpy as np
import torch
import torch.nn.functional as F

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


# ---------------------------------------------------------------------------
# Registration demo: multiscale vs flat (single-scale) MSE optimisation
# ---------------------------------------------------------------------------
#
# Setup
# -----
# The equirectangular image contains three large Gaussian blobs (smooth,
# distinct, unique global minimum) plus a high-frequency checkerboard noise
# pattern.  At full resolution the noise drives the local gradient and the
# loss surface has many shallow local minima.  From the initial estimate
# (yaw=pitch=0) the fine-scale gradient barely indicates the correct direction,
# and a plain Adam loop stagnates inside the nearest local minimum.
#
# At coarse resolution (pyramid levels 3..0) the noise is averaged away by
# F.interpolate, exposing the large-scale blob structure.  The blobs are wide
# enough that the gradient at the coarsest level already points consistently
# toward the global optimum.  Refining level-by-level then converges the
# estimate to the exact solution.
#
# True rotation: yaw=60°, pitch=15°.
# Initial estimate: yaw=pitch=0° (>60° displacement).
# Convergence criterion: final yaw error < 1° (i.e. < 0.018 rad).

def _build_registration_equi() -> torch.Tensor:
    """Build a (1, 3, 64, 128) equi image for the registration demo."""
    H, W, C = 64, 128, 3
    equi = torch.zeros(1, C, H, W)
    ys = torch.arange(H, dtype=torch.float32)
    xs = torch.arange(W, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    for cx, cy, sigma, amp in [
        (32, 32, 14.0, 0.8),   # left-centre blob
        (95, 20, 10.0, 0.6),   # right-upper blob
        (64, 52, 12.0, 0.4),   # centre-lower blob
    ]:
        equi[0] += amp * torch.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)
        )
    # Checkerboard noise (fixed seed → reproducible image)
    torch.manual_seed(99)
    noise = 0.2 * torch.randn(1, C, H, W)
    rr = torch.arange(H).view(H, 1).expand(H, W)
    cc = torch.arange(W).view(1, W).expand(H, W)
    checker = torch.where(
        (rr // 4 + cc // 4) % 2 == 0, torch.tensor(1.0), torch.tensor(-1.0)
    )
    equi = (equi + noise * checker.view(1, 1, H, W)).clamp(0.0, 1.0)
    return equi


def _warp_equi(
    equi: torch.Tensor,
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    *,
    height: int,
    width: int,
    fov_x: float,
) -> torch.Tensor:
    """Render a perspective view from *equi* at the given rotation."""
    inp = equi if equi.ndim == 4 else equi.unsqueeze(0)
    return equi2pers_run(
        equi=inp,
        rots=[{"roll": 0.0, "pitch": pitch, "yaw": yaw}],
        height=height,
        width=width,
        fov_x=fov_x,
        skew=0.0,
        z_down=True,
        mode="bilinear",
    )


def test_multiscale_registration_convergence():
    """Coarse-to-fine MSE optimisation converges; single-scale stagnates.

    The test constructs a synthetic equirectangular image (Gaussian blobs +
    checkerboard noise), applies a large known rotation to create a target
    perspective view (yaw=60°, pitch=15°), then attempts to recover that
    rotation starting from yaw=pitch=0° using two strategies:

    1. **Flat (single-scale)**: Adam on the full-resolution equi for 600
       iterations.  The high-frequency noise creates local minima and the
       optimizer stagnates within a few degrees of the (incorrect) starting
       point — final yaw error > 30°.

    2. **Multiscale coarse-to-fine**: Adam on successive pyramid levels
       (coarsest → finest, 200 iterations each).  Downsampling removes the
       high-frequency noise and exposes the large-scale blob structure, so the
       coarsest level provides a reliable gradient that guides the estimate
       toward the global minimum.  The finest level then refines to sub-degree
       accuracy — final yaw error < 1°.
    """
    fov_x = 45.0
    out_h, out_w = 32, 32
    TRUE_YAW = math.radians(60.0)
    TRUE_PITCH = math.radians(15.0)

    equi = _build_registration_equi()
    target = _warp_equi(
        equi,
        TRUE_YAW,
        TRUE_PITCH,
        height=out_h,
        width=out_w,
        fov_x=fov_x,
    ).detach()

    # --- flat single-scale optimisation ---
    yaw_flat = torch.tensor(0.0, requires_grad=True)
    pitch_flat = torch.tensor(0.0, requires_grad=True)
    opt_flat = torch.optim.Adam([yaw_flat, pitch_flat], lr=0.01)
    for _ in range(600):
        loss_flat = F.mse_loss(
            _warp_equi(equi, yaw_flat, pitch_flat,
                       height=out_h, width=out_w, fov_x=fov_x),
            target,
        )
        opt_flat.zero_grad()
        loss_flat.backward()
        opt_flat.step()

    flat_yaw_error = abs(yaw_flat.item() - TRUE_YAW)

    # --- multiscale coarse-to-fine optimisation ---
    equi_pyr = make_equi_pyramid(equi, num_levels=4)
    # Build target pyramid for multi-scale MSE comparison
    target_pyr = [
        F.interpolate(
            target,
            size=(max(4, round(out_h * (0.5 ** lvl))),
                  max(4, round(out_w * (0.5 ** lvl)))),
            mode="bilinear",
            align_corners=False,
        ).detach()
        for lvl in range(4)
    ]

    yaw_ms = torch.tensor(0.0, requires_grad=True)
    pitch_ms = torch.tensor(0.0, requires_grad=True)
    opt_ms = torch.optim.Adam([yaw_ms, pitch_ms], lr=0.01)
    for level_idx in reversed(range(4)):
        lev_tgt = target_pyr[level_idx]
        lh, lw = lev_tgt.shape[-2], lev_tgt.shape[-1]
        for _ in range(200):
            loss_ms = F.mse_loss(
                _warp_equi(equi_pyr[level_idx], yaw_ms, pitch_ms,
                           height=lh, width=lw, fov_x=fov_x),
                lev_tgt,
            )
            opt_ms.zero_grad()
            loss_ms.backward()
            opt_ms.step()

    ms_yaw_error = abs(yaw_ms.item() - TRUE_YAW)

    # Multiscale must converge to the correct solution (< 1° error)
    assert ms_yaw_error < math.radians(1.0), (
        f"Multiscale yaw error too large: {math.degrees(ms_yaw_error):.2f}°"
    )
    # Flat single-scale must stagnate far from the true solution (> 30° error)
    assert flat_yaw_error > math.radians(30.0), (
        f"Flat yaw error unexpectedly small: {math.degrees(flat_yaw_error):.2f}°"
        " — the local-minimum trap may have changed."
    )
    # Multiscale must be significantly better than flat
    assert ms_yaw_error < flat_yaw_error, (
        "Expected multiscale error < flat error, "
        f"got {math.degrees(ms_yaw_error):.2f}° vs {math.degrees(flat_yaw_error):.2f}°"
    )


@pytest.mark.parametrize(
    "true_yaw_deg,true_pitch_deg",
    [
        (60.0, 15.0),   # yaw + pitch
        (60.0,  0.0),   # yaw only
    ],
)
def test_multiscale_registration_convergence_params(true_yaw_deg, true_pitch_deg):
    """Multiscale converges to correct rotation for a few (yaw, pitch) pairs.

    Each case starts from (0, 0) and optimises with 4-level coarse-to-fine
    MSE.  Final error must be below 2° in both yaw and pitch.
    """
    fov_x = 45.0
    out_h, out_w = 32, 32
    TRUE_YAW = math.radians(true_yaw_deg)
    TRUE_PITCH = math.radians(true_pitch_deg)

    equi = _build_registration_equi()
    target = _warp_equi(
        equi, TRUE_YAW, TRUE_PITCH, height=out_h, width=out_w, fov_x=fov_x
    ).detach()
    equi_pyr = make_equi_pyramid(equi, num_levels=4)

    yaw_ms = torch.tensor(0.0, requires_grad=True)
    pitch_ms = torch.tensor(0.0, requires_grad=True)
    opt_ms = torch.optim.Adam([yaw_ms, pitch_ms], lr=0.01)
    for level_idx in reversed(range(4)):
        scale = 0.5 ** level_idx
        lh = max(4, round(out_h * scale))
        lw = max(4, round(out_w * scale))
        lev_tgt = F.interpolate(
            target, size=(lh, lw), mode="bilinear", align_corners=False
        ).detach()
        for _ in range(200):
            loss_ms = F.mse_loss(
                _warp_equi(equi_pyr[level_idx], yaw_ms, pitch_ms,
                           height=lh, width=lw, fov_x=fov_x),
                lev_tgt,
            )
            opt_ms.zero_grad()
            loss_ms.backward()
            opt_ms.step()

    yaw_err_deg = math.degrees(abs(yaw_ms.item() - TRUE_YAW))
    pitch_err_deg = math.degrees(abs(pitch_ms.item() - TRUE_PITCH))
    threshold_deg = 2.0
    assert yaw_err_deg < threshold_deg, (
        f"yaw error {yaw_err_deg:.2f}° >= {threshold_deg}° "
        f"(true_yaw={true_yaw_deg}°, true_pitch={true_pitch_deg}°)"
    )
    assert pitch_err_deg < threshold_deg, (
        f"pitch error {pitch_err_deg:.2f}° >= {threshold_deg}° "
        f"(true_yaw={true_yaw_deg}°, true_pitch={true_pitch_deg}°)"
    )
