#!/usr/bin/env python3

"""End-to-end gradient tests for all differentiable warp modules.

Each test verifies two things:
1. **Gradient flow**: a rotation / fov parameter with ``requires_grad=True``
   receives a non-None, non-zero gradient after ``backward()``.
2. **Gradient accuracy**: the analytical gradient matches a central
   finite-difference estimate to within 1 % relative error.

All tests run in ``float64`` on CPU so that numerical precision is not a
limiting factor.
"""

import pytest
import torch

from equilib.equi2pers.torch import run as equi2pers_run
from equilib.equi2equi.torch import run as equi2equi_run
from equilib.equi2cube.torch import run as equi2cube_run
from equilib.pers2equi.torch import run as pers2equi_run

# ---------------------------------------------------------------------------
# Small synthetic images (fixed seed for reproducibility)
# ---------------------------------------------------------------------------

_DTYPE = torch.float64
_EPS = 1e-5
_REL_TOL = 0.01  # 1 % relative tolerance for finite-difference comparison


def _equi(batch: int = 1, h: int = 32, w: int = 64, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(batch, 3, h, w, dtype=_DTYPE)


def _pers(batch: int = 1, h: int = 16, w: int = 16, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(batch, 3, h, w, dtype=_DTYPE)


def _make_params(base: dict, grad_key: str) -> dict:
    """Return a dict of float64 tensors; only *grad_key* has requires_grad=True."""
    return {
        k: torch.tensor(v, dtype=_DTYPE, requires_grad=(k == grad_key))
        for k, v in base.items()
    }


def _central_diff(fn, base: dict, param_name: str) -> float:
    """Estimate df/d(param_name) via central finite differences."""
    def _scalar(v):
        kw = {k: torch.tensor(val, dtype=_DTYPE) for k, val in base.items()}
        kw[param_name] = torch.tensor(v, dtype=_DTYPE)
        return fn(**kw).sum().item()

    return (
        _scalar(base[param_name] + _EPS) - _scalar(base[param_name] - _EPS)
    ) / (2.0 * _EPS)


# ---------------------------------------------------------------------------
# equi2pers
# ---------------------------------------------------------------------------

_E2P_BASE = {"roll": 0.1, "pitch": 0.2, "yaw": 0.3, "fov_x": 90.0}


def _e2p(roll, pitch, yaw, fov_x, equi=None):
    if equi is None:
        equi = _equi()
    return equi2pers_run(
        equi=equi,
        rots=[{"roll": roll, "pitch": pitch, "yaw": yaw}],
        height=16,
        width=16,
        fov_x=fov_x,
        skew=0.0,
        z_down=True,
        mode="bilinear",
    )


@pytest.mark.parametrize(
    "param_name",
    ["roll", "pitch", "yaw", "fov_x"],
)
def test_equi2pers_grad_flow(param_name: str) -> None:
    """equi2pers: each parameter receives a non-None, non-zero gradient."""
    params = _make_params(_E2P_BASE, param_name)
    _e2p(**params).sum().backward()

    grad = params[param_name].grad
    assert grad is not None, f"{param_name}.grad is None"
    assert grad.abs() > 0, f"{param_name}.grad is zero"


@pytest.mark.parametrize(
    "param_name",
    ["roll", "pitch", "yaw", "fov_x"],
)
def test_equi2pers_grad_accuracy(param_name: str) -> None:
    """equi2pers: analytical gradient matches central finite differences."""
    equi = _equi()

    # Analytical
    params = _make_params(_E2P_BASE, param_name)
    _e2p(**params, equi=equi.detach()).sum().backward()
    analytical = params[param_name].grad.item()

    # Numerical (central finite difference)
    numerical = _central_diff(
        lambda **kw: _e2p(**kw, equi=equi.detach()), _E2P_BASE, param_name
    )

    rel_err = abs(analytical - numerical) / (abs(numerical) + 1e-8)
    assert rel_err < _REL_TOL, (
        f"equi2pers param={param_name}: analytical={analytical:.6f}, "
        f"numerical={numerical:.6f}, rel_err={rel_err:.4f}"
    )


# ---------------------------------------------------------------------------
# equi2equi
# ---------------------------------------------------------------------------

_E2E_BASE = {"roll": 0.1, "pitch": 0.2, "yaw": 0.3}


def _e2e(roll, pitch, yaw, src=None):
    if src is None:
        src = _equi()
    return equi2equi_run(
        src=src,
        rots=[{"roll": roll, "pitch": pitch, "yaw": yaw}],
        z_down=True,
        mode="bilinear",
    )


@pytest.mark.parametrize("param_name", ["roll", "pitch", "yaw"])
def test_equi2equi_grad_flow(param_name: str) -> None:
    """equi2equi: rotation parameter receives a non-None, non-zero gradient."""
    params = _make_params(_E2E_BASE, param_name)
    _e2e(**params).sum().backward()

    grad = params[param_name].grad
    assert grad is not None, f"{param_name}.grad is None"
    assert grad.abs() > 0, f"{param_name}.grad is zero"


@pytest.mark.parametrize("param_name", ["roll", "pitch", "yaw"])
def test_equi2equi_grad_accuracy(param_name: str) -> None:
    """equi2equi: analytical gradient matches central finite differences."""
    src = _equi()

    # Analytical
    params = _make_params(_E2E_BASE, param_name)
    _e2e(**params, src=src.detach()).sum().backward()
    analytical = params[param_name].grad.item()

    # Numerical
    numerical = _central_diff(
        lambda **kw: _e2e(**kw, src=src.detach()), _E2E_BASE, param_name
    )

    rel_err = abs(analytical - numerical) / (abs(numerical) + 1e-8)
    assert rel_err < _REL_TOL, (
        f"equi2equi param={param_name}: analytical={analytical:.6f}, "
        f"numerical={numerical:.6f}, rel_err={rel_err:.4f}"
    )


# ---------------------------------------------------------------------------
# equi2cube
# ---------------------------------------------------------------------------

_E2C_BASE = {"roll": 0.1, "pitch": 0.2, "yaw": 0.3}


def _e2c(roll, pitch, yaw, equi=None):
    if equi is None:
        equi = _equi()
    return equi2cube_run(
        equi=equi,
        rots=[{"roll": roll, "pitch": pitch, "yaw": yaw}],
        w_face=8,
        cube_format="horizon",
        z_down=True,
        mode="bilinear",
    )


@pytest.mark.parametrize("param_name", ["roll", "pitch", "yaw"])
def test_equi2cube_grad_flow(param_name: str) -> None:
    """equi2cube: rotation parameter receives a non-None, non-zero gradient."""
    params = _make_params(_E2C_BASE, param_name)
    _e2c(**params).sum().backward()

    grad = params[param_name].grad
    assert grad is not None, f"{param_name}.grad is None"
    assert grad.abs() > 0, f"{param_name}.grad is zero"


@pytest.mark.parametrize("param_name", ["roll", "pitch", "yaw"])
def test_equi2cube_grad_accuracy(param_name: str) -> None:
    """equi2cube: analytical gradient matches central finite differences."""
    equi = _equi()

    # Analytical
    params = _make_params(_E2C_BASE, param_name)
    _e2c(**params, equi=equi.detach()).sum().backward()
    analytical = params[param_name].grad.item()

    # Numerical
    numerical = _central_diff(
        lambda **kw: _e2c(**kw, equi=equi.detach()), _E2C_BASE, param_name
    )

    rel_err = abs(analytical - numerical) / (abs(numerical) + 1e-8)
    assert rel_err < _REL_TOL, (
        f"equi2cube param={param_name}: analytical={analytical:.6f}, "
        f"numerical={numerical:.6f}, rel_err={rel_err:.4f}"
    )


# ---------------------------------------------------------------------------
# pers2equi
# ---------------------------------------------------------------------------

_P2E_BASE = {"roll": 0.1, "pitch": 0.2, "yaw": 0.3, "fov_x": 90.0}


def _p2e(roll, pitch, yaw, fov_x, pers=None):
    if pers is None:
        pers = _pers()
    return pers2equi_run(
        pers=pers,
        rots=[{"roll": roll, "pitch": pitch, "yaw": yaw}],
        height=32,
        width=64,
        fov_x=fov_x,
        skew=0.0,
        z_down=True,
        mode="bilinear",
    )


@pytest.mark.parametrize(
    "param_name",
    ["roll", "pitch", "yaw", "fov_x"],
)
def test_pers2equi_grad_flow(param_name: str) -> None:
    """pers2equi: each parameter receives a non-None, non-zero gradient."""
    params = _make_params(_P2E_BASE, param_name)
    _p2e(**params).sum().backward()

    grad = params[param_name].grad
    assert grad is not None, f"{param_name}.grad is None"
    assert grad.abs() > 0, f"{param_name}.grad is zero"


@pytest.mark.parametrize(
    "param_name",
    ["roll", "pitch", "yaw", "fov_x"],
)
def test_pers2equi_grad_accuracy(param_name: str) -> None:
    """pers2equi: analytical gradient matches central finite differences."""
    pers = _pers()

    # Analytical
    params = _make_params(_P2E_BASE, param_name)
    _p2e(**params, pers=pers.detach()).sum().backward()
    analytical = params[param_name].grad.item()

    # Numerical
    numerical = _central_diff(
        lambda **kw: _p2e(**kw, pers=pers.detach()), _P2E_BASE, param_name
    )

    rel_err = abs(analytical - numerical) / (abs(numerical) + 1e-8)
    assert rel_err < _REL_TOL, (
        f"pers2equi param={param_name}: analytical={analytical:.6f}, "
        f"numerical={numerical:.6f}, rel_err={rel_err:.4f}"
    )
