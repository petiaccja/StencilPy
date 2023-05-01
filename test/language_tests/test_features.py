import numpy as np
import pytest

from stencilpy.storage import Field, Connectivity
from stencilpy.concepts import Dimension
from stencilpy.run import func, stencil
from stencilpy.stdlib import index
from config import use_jit

TDim = Dimension()
UDim = Dimension()
LDim = Dimension()


def test_assign(use_jit):
    @func
    def fn(a: Field) -> Field:
        tmp = a
        return tmp

    a = Field([TDim, UDim], np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    r = fn(a, jit=use_jit)
    assert np.allclose(r.data, a.data)


def test_extract_slice(use_jit):
    @func
    def fn(a: Field) -> Field:
        return a[UDim[0:2], TDim[1]]

    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    a = Field([TDim, UDim], data)
    r = fn(a, jit=use_jit)
    expected = np.reshape(data[1, 0:2], newshape=(1, 2))

    assert np.all(r.data == expected)


def test_attribute(use_jit):
    from .. import helpers

    @func
    def fn():
        return helpers.CONSTANT

    assert fn(jit=use_jit) == helpers.CONSTANT


def test_tuple_create(use_jit):
    @func
    def fn(a, b, c):
        t = (a, (b, c))

    assert fn(1, 2, 3, jit=use_jit) is None


def test_tuple_get(use_jit):
    @func
    def fn(a, b, c):
        t = (a, (b, c))
        return t[1][0]

    assert fn(1, 2, 3, jit=use_jit) == 2
