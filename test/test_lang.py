import numpy as np
from stencilpy.storage import Field
from stencilpy.concepts import Dimension
from stencilpy.func import func, stencil
from stencilpy.lib import index
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


def test_assign(use_jit):
    @func
    def fn(a: Field) -> Field:
        tmp = a
        return tmp

    a = Field([TDim, UDim], np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    r = fn(a, jit=use_jit)
    assert np.allclose(r.data, a.data)


def test_sample(use_jit):
    @stencil
    def sn(a: Field) -> np.float32:
        return a[index()]

    @func
    def fn(a: Field, st: int, su: int) -> Field:
        return sn[UDim, TDim][su, st](a)

    a = Field([TDim, UDim], np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    r = fn(a, 2, 3, jit=use_jit)
    assert np.allclose(r.data, a.data)


def test_arithmetic_scalar(use_jit):
    @func
    def fn(a: int, b: int) -> int:
        return a + b

    r = fn(2, 3, jit=use_jit)
    assert r == 5


def test_arithmetic_field(use_jit):
    @func
    def fn(a: Field, b: Field) -> Field:
        return a + b

    a = Field([TDim, UDim], np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    b = Field([TDim, UDim], np.array([[6, 5, 4], [3, 2, 1]], dtype=np.float32))
    r = fn(a, b, jit=use_jit)
    assert np.allclose(r.data, a.data + b.data)

def test_arithmetic_broadcast(use_jit):
    @func
    def fn(a: Field, b: Field) -> Field:
        return a + b

    a = Field([TDim], np.array([1, 2, 3], dtype=np.float32))
    b = Field([UDim], np.array([4, 5, 6], dtype=np.float32))
    r = fn(a, b, jit=use_jit)
    e = np.reshape(a.data, (3, 1)) + np.reshape(b.data, (1, 3))
    assert np.allclose(r.data, e)