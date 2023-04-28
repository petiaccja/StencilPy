import numpy as np
from stencilpy.storage import Field
from stencilpy.concepts import Dimension
from stencilpy.run import func, stencil
from stencilpy.stdlib import *
from stencilpy.metalib import *
from ..config import use_jit

XDim = Dimension()
YDim = Dimension()


@func
def laplacian(u: Field) -> Field:
    left = u[XDim[0:-2], YDim[1:-1]]
    right = u[XDim[2::], YDim[1:-1]]
    bottom = u[XDim[1:-1], YDim[0:-2]]
    top = u[XDim[1:-1], YDim[2::]]
    center = u[XDim[1:-1], YDim[1:-1]]
    return 4.0 * center - (left + right + bottom + top)


@stencil
def _sn_inject_boundaries(central: Field, x_size: int, y_size: int) -> Field:
    idx = index()
    elem_t = element_type(typeof(central))
    if 0 < idx[XDim] <= x_size and 0 < idx[YDim] <= y_size:
        offset = idx[XDim[-1], YDim[-1]]
        return central[offset]
    elif idx[XDim] == 0:
        x = cast(idx[YDim], elem_t)
        scale = cast(0.4, elem_t)
        return sin(scale * x)
    return cast(0.0, elem_t)


@func
def inject_boundaries(central: Field) -> Field:
    x_size = shape(central, XDim)
    y_size = shape(central, YDim)
    return _sn_inject_boundaries[XDim[x_size + 2], YDim[y_size + 2]](central, x_size, y_size)


@func
def timestep(u: Field, step: float) -> Field:
    dudt = laplacian(u)
    central = u[XDim[1:-1], YDim[1:-1]]
    updated = central - step * dudt
    return inject_boundaries(updated)


def test_heat_eq():
    data = np.random.random(size=(20, 20))
    initial = Field([XDim, YDim], data)
    step = 0.05
    num_steps = 10
    u = initial
    for i in range(0, num_steps):
        u = timestep(u, step, jit=True)
    assert u.data.shape == data.shape
