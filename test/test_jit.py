import numpy as np

from stencilpy.field import Field, Dimension, index
from stencilpy.func import func, stencil


TDim = Dimension()


def test_single_func():
    @func
    def add(a: Field, b: Field) -> int:
        return 1

    a = Field([TDim], np.array([1, 2, 3], dtype=np.float32))
    b = Field([TDim], np.array([4, 3, 6], dtype=np.float32))
    r = add(a, b, jit=True)

    assert r == 1
    #assert np.allclose(r.data, a.data + b.data)