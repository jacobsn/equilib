## equi2pers

equirectangular to perspective transformation

### Parameters

- `equi` (`np.ndarray` or `torch.Tensor`): input equirectangular image, channel-first layout (`CxHxW` or `BxCxHxW`).
- `rots` (`dict` or `list[dict]`): rotation angles `{'roll': float, 'pitch': float, 'yaw': float}` in radians.
- `height`, `width` (`int`): output perspective image size.
- `fov_x` (`float`): horizontal field of view of the perspective image in **degrees**.
- `fov_y` (`float`, optional): vertical field of view in **degrees**.
  When provided, the vertical focal length `fy` is computed independently from `fx`, which supports non-square pixels and direct vertical-FOV specification.
  When omitted (default), `fy` is derived from `fov_x` and the image aspect ratio, equivalent to assuming square pixels.
- `skew` (`float`, default `0.0`): camera skew intrinsic parameter.
- `z_down` (`bool`, default `False`): use a coordinate system where the z-axis faces down.
- `mode` (`str`, default `"bilinear"`): interpolation mode — `"nearest"`, `"bilinear"`, or `"bicubic"`.
- `clip_output` (`bool`, default `True`): clip output values to the range of the input image.

### Intrinsic Matrix

The perspective projection uses a standard pinhole camera intrinsic matrix:

```
K = [[fx,   skew, cx],
     [0,    fy,   cy],
     [0,    0,    1 ]]
```

where `cx = width / 2`, `cy = height / 2`, and:

- `fx = width  / (2 * tan(fov_x / 2))`
- `fy = height / (2 * tan(fov_y / 2))` when `fov_y` is given explicitly, otherwise `fy` is derived from `fov_x` assuming square pixels: `fy = height / (2 * tan(atan(tan(fov_x/2) * height/width)))`

For square pixels the two formulations are equivalent — specifying `fov_y` is useful when the vertical FOV is known directly or when the pixel aspect ratio differs from 1.

### Example

```python
from equilib import equi2pers

# Default: fy is derived from fov_x + aspect ratio (square pixels assumed)
pers = equi2pers(equi, rots, height=480, width=640, fov_x=90.0)

# Explicit fov_y: set vertical FOV independently (e.g. from a camera datasheet)
pers = equi2pers(equi, rots, height=480, width=640, fov_x=90.0, fov_y=67.4)
```

### TODO:

- [x] Crop is slightly different `numpy` and `torch` (FIXED)
- [x] Equi2Pers outputs for `numpy` and `torch` differs a little bit. Need to figure out why this happens. The outputs are the same regardless of the sampling method, so it must be the preprocessing (where the rotation matrix is set). (FIXED)
