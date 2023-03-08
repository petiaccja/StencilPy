import numpy as np
from stencilpy.storage import Field
from stencilpy.concepts import Dimension
from stencilpy.func import func, stencil
from stencilpy.lib import index
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


def np_laplacian(u: np.ndarray) -> np.ndarray:
    left = u[0:-2, 1:-1]
    right = u[2::, 1:-1]
    bottom = u[1:-1, 0:-2]
    top = u[1:-1, 2::]
    center = u[1:-1, 1:-1]
    return 4.0 * center - (left + right + bottom + top)


def test_laplacian(use_jit):
    u = Field([XDim, YDim], np.random.random(size=(7, 8)).astype(np.float64))
    
    r = laplacian(u, jit=use_jit)
    e = np_laplacian(u.data)

    assert np.allclose(r.data, e)