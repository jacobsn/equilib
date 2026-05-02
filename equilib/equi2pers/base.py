#!/usr/bin/env python3

from typing import Dict, List, Union

import numpy as np

import torch

from .numpy import get_bounding_fov as get_bfov_numpy, run as run_numpy
from .torch import get_bounding_fov as get_bfov_torch, run as run_torch

__all__ = ["Equi2Pers", "equi2pers"]

ArrayLike = Union[np.ndarray, torch.Tensor]
Rot = Union[Dict[str, float], List[Dict[str, float]]]


class Equi2Pers(object):
    """
    params:
    - height, width (int): perspective size
    - fov_x (float): perspective image fov of x-axis in degrees
    - fov_y (float, optional): perspective image fov of y-axis in degrees;
      when provided, the vertical focal length is set independently of the
      horizontal focal length, enabling non-square pixels or direct vertical
      FOV control.  When omitted (default), ``fy`` is derived from ``fov_x``
      and the image aspect ratio (equivalent to square pixels).
    - skew (float): skew intrinsic parameter
    - sampling_method (str)
    - z_down (bool)
    - mode (str)
    - clip_output (bool)

    inputs:
    - equi (np.ndarray, torch.Tensor)
    - rot (dict, list): Dict[str, float]

    returns:
    - pers (np.ndarray, torch.Tensor)

    """

    def __init__(
        self,
        height: int,
        width: int,
        fov_x: float,
        fov_y: float = None,
        skew: float = 0.0,
        z_down: bool = False,
        mode: str = "bilinear",
        clip_output: bool = True,
    ) -> None:
        self.height = height
        self.width = width
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.skew = skew
        self.mode = mode
        self.z_down = z_down
        self.clip_output = clip_output
        # FIXME: maybe do useful stuff like precalculating the grid or something

    def __call__(self, equi: ArrayLike, rots: Rot, **kwargs) -> ArrayLike:
        # FIXME: should optimize since some parts of the code can be calculated
        # before hand.
        # 1. calculate grid
        # 2. grid sample
        return equi2pers(
            equi=equi,
            rots=rots,
            height=self.height,
            width=self.width,
            fov_x=self.fov_x,
            fov_y=self.fov_y,
            skew=self.skew,
            z_down=self.z_down,
            mode=self.mode,
            clip_output=self.clip_output,
            **kwargs,
        )

    def get_bounding_fov(self, equi: ArrayLike, rots: Rot) -> ArrayLike:
        return get_bounding_fov(
            equi=equi,
            rots=rots,
            height=self.height,
            width=self.width,
            fov_x=self.fov_x,
            fov_y=self.fov_y,
            skew=self.skew,
            z_down=self.z_down,
        )


def equi2pers(
    equi: ArrayLike,
    rots: Rot,
    height: int,
    width: int,
    fov_x: float,
    fov_y: float = None,
    skew: float = 0.0,
    mode: str = "bilinear",
    z_down: bool = False,
    clip_output: bool = True,
    **kwargs,
) -> ArrayLike:
    """
    params:
    - equi
    - rots
    - height, width (int): perspective size
    - fov_x (float): perspective image fov of x-axis in degrees
    - fov_y (float, optional): perspective image fov of y-axis in degrees;
      when provided, the vertical focal length is set independently of the
      horizontal focal length, enabling non-square pixels or direct vertical
      FOV control.  When omitted (default), ``fy`` is derived from ``fov_x``
      and the image aspect ratio (equivalent to square pixels).
    - z_down (bool)
    - skew (float): skew intrinsic parameter

    returns:
    - pers (np.ndarray, torch.Tensor)

    """

    _type = None
    if isinstance(equi, np.ndarray):
        _type = "numpy"
    elif torch.is_tensor(equi):
        _type = "torch"
    else:
        raise ValueError

    is_single = False
    if len(equi.shape) == 3 and isinstance(rots, dict):
        # probably the input was a single image
        equi = equi[None, ...]
        rots = [rots]
        is_single = True
    elif len(equi.shape) == 3:
        # probably a grayscale image
        equi = equi[:, None, ...]

    assert isinstance(rots, list), "ERR: rots is not a list"
    if _type == "numpy":
        out = run_numpy(
            equi=equi,
            rots=rots,
            height=height,
            width=width,
            fov_x=fov_x,
            fov_y=fov_y,
            skew=skew,
            z_down=z_down,
            clip_output=clip_output,
            mode=mode,
            **kwargs,
        )
    elif _type == "torch":
        out = run_torch(
            equi=equi,
            rots=rots,
            height=height,
            width=width,
            fov_x=fov_x,
            fov_y=fov_y,
            skew=skew,
            z_down=z_down,
            clip_output=clip_output,
            mode=mode,
            **kwargs,
        )
    else:
        raise ValueError

    # make sure that the output batch dim is removed if it's only a single image
    if is_single:
        out = out.squeeze(0)

    return out


def get_bounding_fov(
    equi: ArrayLike,
    rots: Rot,
    height: int,
    width: int,
    fov_x: float,
    fov_y: float = None,
    skew: float = 0.0,
    z_down: bool = False,
) -> np.ndarray:
    _type = None
    if isinstance(equi, np.ndarray):
        _type = "numpy"
    elif torch.is_tensor(equi):
        _type = "torch"
    else:
        raise ValueError

    is_single = False
    if len(equi.shape) == 3 and isinstance(rots, dict):
        # probably the input was a single image
        equi = equi[None, ...]
        rots = [rots]
        is_single = True
    elif len(equi.shape) == 3:
        # probably a grayscale image
        equi = equi[:, None, ...]

    assert isinstance(rots, list), "ERR: rots is not a list"
    if _type == "numpy":
        out = get_bfov_numpy(
            equi=equi,
            rots=rots,
            height=height,
            width=width,
            fov_x=fov_x,
            fov_y=fov_y,
            skew=skew,
            z_down=z_down,
        )
    elif _type == "torch":
        out = get_bfov_torch(
            equi=equi,
            rots=rots,
            height=height,
            width=width,
            fov_x=fov_x,
            fov_y=fov_y,
            skew=skew,
            z_down=z_down,
        )
    else:
        raise ValueError

    # make sure that the output batch dim is removed if it's only a single image
    if is_single:
        out = out.squeeze(0)

    return out
