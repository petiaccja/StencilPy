from stencilpy.lib import *
from .config import use_jit
from stencilpy.concepts import *
from stencilpy.storage import *
from stencilpy.func import func, stencil

TDim = concepts.Dimension()
UDim = concepts.Dimension()


def test_shape(use_jit):
    @func
    def f(a):
        return shape(a, TDim)

    a = Field([TDim, UDim], np.zeros(shape=(3, 2)))

    assert f(a, jit=use_jit) == 3

