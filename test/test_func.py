import numpy as np

from stencilpy.field import Field, Dimension, index
from stencilpy.func import func, stencil


TDim = Dimension()


def test_func():
    @func
    def add(a: Field, b: Field) -> Field:
        return a + b

    a = Field([TDim], np.array([1, 2, 3]))
    b = Field([TDim], np.array([4, 3, 6]))
    r = add(a, b)

    assert np.allclose(r.data, a.data + b.data)


def test_stencil():
    @stencil
    def add_stencil(a: Field, b: Field) -> float:
        return a[index()] + b[index()]

    a = Field([TDim], np.array([1, 2, 3]))
    b = Field([TDim], np.array([4, 3, 6]))
    r = add_stencil[TDim][3](a, b)

    assert np.allclose(r.data, a.data + b.data)
