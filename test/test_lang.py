import numpy as np
from stencilpy.storage import Field
from stencilpy.concepts import Dimension
from stencilpy.func import func, stencil
from .config import use_jit


TDim = Dimension()
UDim = Dimension()


def test_func_return_scalar(use_jit):
    @func
    def fn(a: int) -> int:
        return a

    assert fn(3, jit=use_jit) == 3


def test_func_return_field(use_jit):
    @func
    def fn(a: Field) -> Field:
        return a

    a = Field([TDim, UDim], np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    r = fn(a, jit=use_jit)
    assert np.allclose(r.data, a.data)


def test_apply(use_jit):
    @stencil
    def sn(v: int) -> int:
        return v

    @func
    def fn(v: int, w: int, h: int):
        return sn[TDim, UDim][w, h](v)

    r = fn(1, 4, 3, jit=use_jit)

    assert np.all(r.data == 1)
