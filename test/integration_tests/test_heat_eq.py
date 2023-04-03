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


@func
def advance(u: Field) -> Field:
    central = laplacian(u)
    boundary_left = central[XDim[0], :]
    boundary_right = central[XDim[-1], :]
    boundary_bottom = central[:, YDim[0]]
    boundary_top = central[:, YDim[-1]]
    boundary_bottomleft = central[XDim[0], YDim[0]]
    boundary_bottomright = central[XDim[-1], YDim[0]]
    boundary_topleft = central[XDim[0], YDim[-1]]
    boundary_topright = central[XDim[-1], YDim[-1]]


