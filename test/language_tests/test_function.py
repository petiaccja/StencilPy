import numpy as np

from stencilpy.storage import Field
from stencilpy.concepts import Dimension
from stencilpy.run import func
from config import use_jit

TDim = Dimension()
UDim = Dimension()
LDim = Dimension()


@func
def passthrough(arg):
    return arg


def test_passthrough_void(use_jit):
    @func
    def fn():
        pass

    assert fn(jit=use_jit) is None


def test_passthrough_scalar(use_jit):
    @func
    def fn(a: int) -> int:
        return a

    assert fn(3, jit=use_jit) == 3


def test_passthrough_field(use_jit):
    a = Field([TDim, UDim], np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    r = passthrough(a, jit=use_jit)
    assert np.allclose(r.data, a.data)


def test_passthrough_tuple(use_jit):
    a = Field([TDim, UDim], np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    b = 3

    r = passthrough((a, b), jit=use_jit)
    assert isinstance(r, tuple)
    assert len(r) == 2
    assert np.all(r[0].data == a.data)
    assert r[1] == b


def test_passthrough_nested_tuple(use_jit):
    a = Field([TDim, UDim], np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    b = 3
    c = Field([TDim, UDim], np.array([[5, 7, 3], [9, 4, 7]], dtype=np.float32))

    r = passthrough((a, (b, c)), jit=use_jit)
    assert isinstance(r, tuple)
    assert len(r) == 2
    r1 = r[1]
    assert isinstance(r1, tuple)
    assert len(r1) == 2
    assert np.all(r[0].data == a.data)
    assert r1[0] == b
    assert np.all(r1[1].data == c.data)


@func
def caller(arg):
    return passthrough(arg)


def test_call_void(use_jit):
    @func
    def fn():
        pass

    @func
    def cfn():
        fn()

    assert cfn(jit=use_jit) is None


def test_call_scalar(use_jit):
    assert caller(3, jit=use_jit) == 3


def test_call_field(use_jit):
    a = Field([TDim, UDim], np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    r = caller(a, jit=use_jit)
    assert np.allclose(r.data, a.data)


def test_call_tuple(use_jit):
    a = Field([TDim, UDim], np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    b = 3

    r = caller((a, b), jit=use_jit)
    assert isinstance(r, tuple)
    assert len(r) == 2
    assert np.all(r[0].data == a.data)
    assert r[1] == b


def test_call_nested_tuple(use_jit):
    a = Field([TDim, UDim], np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    b = 3
    c = Field([TDim, UDim], np.array([[5, 7, 3], [9, 4, 7]], dtype=np.float32))

    r = caller((a, (b, c)), jit=use_jit)
    assert isinstance(r, tuple)
    assert len(r) == 2
    r1 = r[1]
    assert isinstance(r1, tuple)
    assert len(r1) == 2
    assert np.all(r[0].data == a.data)
    assert r1[0] == b
    assert np.all(r1[1].data == c.data)
