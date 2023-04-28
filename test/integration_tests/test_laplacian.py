import numpy as np
from stencilpy.storage import Field
from stencilpy.concepts import Dimension
from stencilpy.run import func, stencil
from stencilpy.stdlib import index, shape
from ..config import use_jit

XDim = Dimension()
YDim = Dimension()


@func
def f_laplacian(u: Field) -> Field:
    left = u[XDim[0:-2], YDim[1:-1]]
    right = u[XDim[2::], YDim[1:-1]]
    bottom = u[XDim[1:-1], YDim[0:-2]]
    top = u[XDim[1:-1], YDim[2::]]
    center = u[XDim[1:-1], YDim[1:-1]]
    return 4.0 * center - (left + right + bottom + top)


@stencil
def _sn_laplacian(u: Field):
    idx = index()
    center = idx[XDim[1], YDim[1]]
    left = center[XDim[-1]]
    right = center[XDim[1]]
    lower = center[YDim[-1]]
    upper = center[YDim[1]]
    return 4.0 * u[center] - (u[left] + u[right] + u[lower] + u[upper])


@func
def l_laplacian(u: Field) -> Field:
    return _sn_laplacian[XDim[shape(u, XDim) - 2], YDim[shape(u, YDim) - 2]](u)

def np_laplacian(u: np.ndarray) -> np.ndarray:
    left = u[0:-2, 1:-1]
    right = u[2::, 1:-1]
    bottom = u[1:-1, 0:-2]
    top = u[1:-1, 2::]
    center = u[1:-1, 1:-1]
    return 4.0 * center - (left + right + bottom + top)


@func
def f_laplacian_2(u: Field) -> Field:
    return f_laplacian(f_laplacian(u))\


@func
def l_laplacian_2(u: Field) -> Field:
    return l_laplacian(l_laplacian(u))


def np_laplacian_2(u: np.ndarray) -> np.ndarray:
    return np_laplacian(np_laplacian(u))


def test_laplacian_f(use_jit):
    u = Field([XDim, YDim], np.random.random(size=(7, 8)).astype(np.float64))
    
    r = f_laplacian(u, jit=use_jit)
    e = np_laplacian(u.data)

    assert np.allclose(r.data, e)


def test_laplacian_2_f(use_jit):
    u = Field([XDim, YDim], np.random.random(size=(7, 8)).astype(np.float64))

    r = f_laplacian_2(u, jit=use_jit)
    e = np_laplacian_2(u.data)

    assert np.allclose(r.data, e)


def test_laplacian_l(use_jit):
    u = Field([XDim, YDim], np.random.random(size=(7, 8)).astype(np.float64))

    r = l_laplacian(u, jit=use_jit)
    e = np_laplacian(u.data)

    assert np.allclose(r.data, e)


def test_laplacian_2_l(use_jit):
    u = Field([XDim, YDim], np.random.random(size=(7, 8)).astype(np.float64))

    r = l_laplacian_2(u, jit=use_jit)
    e = np_laplacian_2(u.data)

    assert np.allclose(r.data, e)