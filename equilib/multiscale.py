#!/usr/bin/env python3

"""Multiscale (image-pyramid) utilities for equirectangular images.

Building a Gaussian-style pyramid is the standard strategy for coarse-to-fine
optimisation.  When two panoramas have a large relative rotation the
optimisation landscape at full resolution has many local minima; starting at a
coarsely downsampled scale and progressively refining is far more robust.

Usage example (PyTorch, coarse-to-fine registration)::

    import torch
    from equilib import make_equi_pyramid, equi2pers

    equi = ...  # (B, C, H, W) float32 tensor
    pyramid = make_equi_pyramid(equi, num_levels=4)
    # pyramid[0] is the original (finest); pyramid[-1] is the coarsest.

    roll  = torch.tensor(0.0, requires_grad=True)
    pitch = torch.tensor(0.0, requires_grad=True)
    yaw   = torch.tensor(0.0, requires_grad=True)
    opt   = torch.optim.Adam([roll, pitch, yaw], lr=1e-2)

    # coarse-to-fine optimisation loop
    for level_img in reversed(pyramid):
        for _ in range(100):
            out = equi2pers(level_img, {"roll": roll, "pitch": pitch, "yaw": yaw},
                            height=64, width=64, fov_x=90.0, mode="bilinear")
            loss = some_loss(out)
            opt.zero_grad()
            loss.backward()
            opt.step()

Usage example (NumPy)::

    import numpy as np
    from equilib import make_equi_pyramid

    equi = ...  # (C, H, W) uint8 array
    pyramid = make_equi_pyramid(equi, num_levels=4)
    # Each level is a NumPy array at a progressively lower resolution.
"""

from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["make_equi_pyramid"]

ArrayLike = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _torch_pyramid(
    equi: torch.Tensor,
    num_levels: int,
    scale_factor: float,
    mode: str,
) -> List[torch.Tensor]:
    """Build a pyramid for a ``(B, C, H, W)`` torch tensor.

    All resize operations use :func:`torch.nn.functional.interpolate`, which
    is fully differentiable.  Gradients therefore propagate back through the
    pyramid into any upstream computation graph.

    params:
    - equi: (B, C, H, W) tensor
    - num_levels: total number of levels (>= 1)
    - scale_factor: ratio between consecutive levels (0 < scale_factor < 1)
    - mode: interpolation mode accepted by F.interpolate

    returns:
    - list of tensors, index 0 = original (finest)
    """
    pyramid: List[torch.Tensor] = [equi]
    current = equi

    # F.interpolate requires a floating-point dtype; we cast uint8 temporarily
    # and restore the original dtype afterwards to preserve consistent behaviour
    # with the rest of the library.
    needs_cast = current.dtype == torch.uint8
    if needs_cast:
        current = current.float()

    for _ in range(num_levels - 1):
        new_h = max(1, round(current.shape[-2] * scale_factor))
        new_w = max(1, round(current.shape[-1] * scale_factor))

        align_corners = (
            False if mode in ("bilinear", "bicubic") else None
        )
        kwargs = {} if align_corners is None else {"align_corners": align_corners}
        current = F.interpolate(
            current,
            size=(new_h, new_w),
            mode=mode,
            **kwargs,
        )

        level = current.to(equi.dtype) if needs_cast else current
        pyramid.append(level)

    return pyramid


def _numpy_downsample(
    equi: np.ndarray,
    new_h: int,
    new_w: int,
) -> np.ndarray:
    """Bilinear downsampling for ``(B, C, H, W)`` numpy arrays.

    Uses only numpy (no scipy / skimage dependency) via vectorised bilinear
    interpolation.
    """
    _B, _C, H, W = equi.shape

    src = equi.astype(np.float64)

    # target pixel centres mapped back to source coordinates
    row_s = np.linspace(0.0, H - 1.0, new_h)  # (new_h,)
    col_s = np.linspace(0.0, W - 1.0, new_w)  # (new_w,)

    r0 = np.floor(row_s).astype(np.intp).clip(0, H - 1)
    r1 = (r0 + 1).clip(0, H - 1)
    c0 = np.floor(col_s).astype(np.intp).clip(0, W - 1)
    c1 = (c0 + 1).clip(0, W - 1)

    wr1 = (row_s - np.floor(row_s)).astype(np.float64)  # (new_h,)
    wr0 = 1.0 - wr1
    wc1 = (col_s - np.floor(col_s)).astype(np.float64)  # (new_w,)
    wc0 = 1.0 - wc1

    # Advanced indexing: src[:, :, r0[:, None], c0[None, :]] → (B, C, new_h, new_w)
    I00 = src[:, :, r0[:, np.newaxis], c0[np.newaxis, :]]
    I01 = src[:, :, r0[:, np.newaxis], c1[np.newaxis, :]]
    I10 = src[:, :, r1[:, np.newaxis], c0[np.newaxis, :]]
    I11 = src[:, :, r1[:, np.newaxis], c1[np.newaxis, :]]

    # Combine with bilinear weights (broadcast over B and C dims)
    w00 = wr0[:, np.newaxis] * wc0[np.newaxis, :]  # (new_h, new_w)
    w01 = wr0[:, np.newaxis] * wc1[np.newaxis, :]
    w10 = wr1[:, np.newaxis] * wc0[np.newaxis, :]
    w11 = wr1[:, np.newaxis] * wc1[np.newaxis, :]

    out = w00 * I00 + w01 * I01 + w10 * I10 + w11 * I11  # (B, C, new_h, new_w)

    if np.issubdtype(equi.dtype, np.integer):
        out = np.round(out).astype(equi.dtype)
    else:
        out = out.astype(equi.dtype)

    return out


def _numpy_pyramid(
    equi: np.ndarray,
    num_levels: int,
    scale_factor: float,
) -> List[np.ndarray]:
    """Build a pyramid for a ``(B, C, H, W)`` numpy array."""
    pyramid: List[np.ndarray] = [equi]
    current = equi

    for _ in range(num_levels - 1):
        new_h = max(1, round(current.shape[-2] * scale_factor))
        new_w = max(1, round(current.shape[-1] * scale_factor))
        current = _numpy_downsample(current, new_h, new_w)
        pyramid.append(current)

    return pyramid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_equi_pyramid(
    equi: ArrayLike,
    num_levels: int,
    scale_factor: float = 0.5,
    mode: str = "bilinear",
) -> List[ArrayLike]:
    """Build a multiscale pyramid from an equirectangular image.

    Each successive level is ``scale_factor`` times the spatial size of the
    previous one.  Level 0 is the original (finest) resolution; level
    ``num_levels - 1`` is the coarsest.

    For :class:`torch.Tensor` inputs every downsampling step is performed with
    :func:`torch.nn.functional.interpolate`, which is fully differentiable —
    gradients propagate back through the entire pyramid into upstream tensors
    (e.g. rotation or FoV parameters).

    params:
    - equi (np.ndarray or torch.Tensor): equirectangular image with shape
      ``(C, H, W)`` (single image) or ``(B, C, H, W)`` (batch).
    - num_levels (int): number of pyramid levels (must be >= 1).
      ``num_levels=1`` returns a list containing only the original image.
    - scale_factor (float): spatial scale between consecutive levels.
      Must satisfy ``0 < scale_factor < 1``. Default: ``0.5`` (halving).
    - mode (str): interpolation method used for downsampling.
      For torch tensors any mode accepted by :func:`torch.nn.functional.interpolate`
      is valid (``"bilinear"``, ``"bicubic"``, ``"area"``, …).
      For numpy arrays only bilinear interpolation is applied regardless of
      this parameter. Default: ``"bilinear"``.

    returns:
    - List[ArrayLike]: list of ``num_levels`` images. ``result[0]`` is the
      original input; ``result[-1]`` is the coarsest level. The batch
      dimension (if present in the input) is preserved at every level.

    raises:
    - ValueError: if ``num_levels < 1`` or ``scale_factor`` is not in (0, 1).
    - TypeError: if ``equi`` is neither a numpy array nor a torch tensor.

    Example::

        >>> import torch
        >>> from equilib import make_equi_pyramid
        >>> equi = torch.rand(1, 3, 512, 1024)
        >>> pyr = make_equi_pyramid(equi, num_levels=4)
        >>> [tuple(p.shape) for p in pyr]
        [(1, 3, 512, 1024), (1, 3, 256, 512), (1, 3, 128, 256), (1, 3, 64, 128)]
    """
    if num_levels < 1:
        raise ValueError(
            f"num_levels must be >= 1, got {num_levels}"
        )
    if not (0.0 < scale_factor < 1.0):
        raise ValueError(
            f"scale_factor must be in the open interval (0, 1), got {scale_factor}"
        )

    is_single = False
    if isinstance(equi, np.ndarray):
        if equi.ndim == 3:
            equi = equi[np.newaxis, ...]
            is_single = True
        pyramid = _numpy_pyramid(equi, num_levels, scale_factor)
        if is_single:
            pyramid = [p[0] for p in pyramid]
        return pyramid
    elif torch.is_tensor(equi):
        if equi.ndim == 3:
            equi = equi.unsqueeze(0)
            is_single = True
        pyramid = _torch_pyramid(equi, num_levels, scale_factor, mode)
        if is_single:
            pyramid = [p.squeeze(0) for p in pyramid]
        return pyramid
    else:
        raise TypeError(
            f"equi must be a numpy.ndarray or torch.Tensor, got {type(equi)}"
        )
