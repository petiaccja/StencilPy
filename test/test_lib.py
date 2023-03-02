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
        return sn[TDim][3]()

    assert np.allclose(fn(jit=use_jit).data, 0.0)


