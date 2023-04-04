import numpy as np
from stencilpy.storage import Field
from stencilpy.concepts import Dimension
from stencilpy.func import func, stencil
from stencilpy.lib import *
from stencilpy.meta import *
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
def apply_bcs_sn(central: Field, x_size: int, y_size: int) -> Field:
    idx = index()
    if 0 < idx[XDim] <= x_size and 0 < idx[YDim] <= y_size:
        offset = idx[XDim[-1], YDim[-1]]
        return central[offset]
    elem_t = element_type(typeof(central))
    return cast(0.0, elem_t)

@func
def apply_bcs(central: Field) -> Field:
    x_size = shape(central, XDim)
    y_size = shape(central, YDim)
    return apply_bcs_sn[XDim[x_size + 2], YDim[y_size + 2]](central, x_size, y_size)


@func
def advance(u: Field) -> Field:
    dudt = laplacian(u)
    central = u[XDim[1:-1], YDim[1:-1]]
    updated = central - 0.05 * dudt
    return apply_bcs(updated)


def test_heat_eq():
    data = np.random.random(size=(20, 20))
    u = Field([XDim, YDim], data)
    for i in range(0, 10):
        u = advance(u, jit=True)
    assert u.data.shape == data.shape


