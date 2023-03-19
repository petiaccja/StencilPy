from stencilpy.lib import *
from .config import use_jit
from stencilpy.concepts import *
from stencilpy.storage import *
from stencilpy.func import func, stencil

TDim = concepts.Dimension()
UDim = concepts.Dimension()


def test_shape(use_jit):
    @func
    def fn(a):
        return shape(a, TDim)

    a = Field([TDim, UDim], np.zeros(shape=(3, 2)))

    assert fn(a, jit=use_jit) == 3


def test_index(use_jit):
    @stencil
    def sn():
        index()
        return 0.0

    @func
    def fn():
        return sn[TDim[3]]()

    assert np.allclose(fn(jit=use_jit).data, 0.0)


def test_cast_scalar(use_jit):
    @func
    def fn(a: float) -> float:
        return cast(a, ts.float32_t)

    a = np.float64(3.14159265358979)
    r = fn(a, jit=use_jit)
    assert a != r
    assert np.isclose(a, r)


