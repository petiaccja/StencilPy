import numpy as np
from stencilpy.storage import Field, Connectivity
from stencilpy.concepts import Dimension
from stencilpy.func import func, stencil
from stencilpy.lib import index
from .config import use_jit

TDim = Dimension()
UDim = Dimension()
LDim = Dimension()


def test_func_return_nothing(use_jit):
    @func
    def fn(a: int):
        a

    assert fn(3, jit=use_jit) is None


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
        return sn[TDim[w], UDim[h]](v)

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


def test_sample_field(use_jit):
    @stencil
    def sn(a: Field) -> np.float32:
        return a[index()]

    @func
    def fn(a: Field, st: int, su: int) -> Field:
        return sn[UDim[su], TDim[st]](a)

    a = Field([TDim, UDim], np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    r = fn(a, 2, 3, jit=use_jit)
    assert np.allclose(r.data, a.data)


def test_sample_connectivity(use_jit):
    @stencil
    def sn(a: Connectivity) -> np.int64:
        return a[index()]

    @func
    def fn(a: Connectivity, so: int, se: int) -> Field:
        return sn[TDim[so], LDim[se]](a)

    a = Connectivity(TDim, UDim, LDim, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64))
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


def test_arithmetic_broadcast_field(use_jit):
    @func
    def fn(a: Field, b: Field) -> Field:
        return a + b

    a = Field([TDim], np.array([1, 2, 3], dtype=np.float32))
    b = Field([UDim], np.array([4, 5, 6], dtype=np.float32))
    r = fn(a, b, jit=use_jit)
    e = np.reshape(a.data, (3, 1)) + np.reshape(b.data, (1, 3))
    assert np.allclose(r.data, e)


def test_arithmetic_broadcast_scalar(use_jit):
    @func
    def fn(a: Field, b: np.float32) -> Field:
        return a + b

    a = Field([TDim], np.array([1, 2, 3], dtype=np.float32))
    b = np.float32(3.2)
    r = fn(a, b, jit=use_jit)
    e = a.data + b
    assert np.allclose(r.data, e)


def test_comparison_scalar(use_jit):
    @func
    def fn(a: int, b: int, c: int) -> bool:
        return a < b < c

    r = fn(2, 3, 4, jit=use_jit)
    assert r
    r = fn(2, 3, 1, jit=use_jit)
    assert not r  # It is None when it should in fact be False


def test_extract_slice(use_jit):
    @func
    def fn(a: Field) -> Field:
        return a[UDim[0:2], TDim[1]]

    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    a = Field([TDim, UDim], data)
    r = fn(a, jit=use_jit)
    expected = np.reshape(data[1, 0:2], newshape=(1, 2))

    assert np.all(r.data == expected)


def test_call_scalar(use_jit):
    @func
    def callee(a):
        return a

    @func
    def caller(a):
        return callee(a)

    assert caller(42, jit=use_jit) == 42


def test_call_field(use_jit):
    @func
    def callee(a):
        return a

    @func
    def caller(a):
        return callee(a)

    a = Field([TDim], np.array([1, 2, 3]))
    r = caller(a, jit=use_jit)
    assert np.all(r.data == a.data)


def test_attribute(use_jit):
    from . import helpers

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


def test_tuple_arg(use_jit):
    @func
    def fn(t):
        pass

    assert fn((1, (2, 3)), jit=use_jit) is None


def test_tuple_return(use_jit):
    @func
    def fn(a, b, c):
        t = (a, (b, c))
        return t

    assert fn(1, 2, 3, jit=use_jit) == (1, (2, 3))


def test_tuple_return_mrv(use_jit):
    @func
    def fn(a, b):
        return a, b

    assert fn(1, 2, jit=use_jit) == (1, 2)


def test_tuple_arg_mixed(use_jit):
    @func
    def fn(t):
        pass

    a = Field([TDim], np.array([1, 2, 3]))
    b = Field([TDim], np.array([4, 5, 6]))
    c = 3.14
    assert fn((a, b, c), jit=use_jit) is None


def test_tuple_return_mixed(use_jit):
    @func
    def fn(a, b, c):
        t = (a, b, c)
        return t

    a = Field([TDim], np.array([1, 2, 3]))
    b = Field([TDim], np.array([4, 5, 6]))
    c = 3.14
    ra, rb, rc = fn(a, b, c, jit=use_jit)
    assert all(ra.data == a.data)
    assert all(rb.data == b.data)
    assert rc == c


def test_apply_mrv(use_jit):
    @stencil
    def sn(a: int, b: float) -> tuple[int, float]:
        return a, b

    @func
    def fn(a: int, b: int, w: int, h: int):
        return sn[TDim[w], UDim[h]](a, b)

    ra, rb = fn(2, 3.14, 4, 3, jit=use_jit)

    assert np.all(ra.data == 2)
    assert np.all(rb.data == 3.14)


def test_if_no_return(use_jit):
    @func
    def fn(c1: bool):
        if c1:
            a = 1
        else:
            a = 2
        return a

    assert fn(True, jit=use_jit) == 1
    assert fn(False, jit=use_jit) == 2


def test_if_then_return(use_jit):
    @func
    def fn(c1: bool):
        if c1:
            return 1
        else:
            a = 2
        return a

    assert fn(True, jit=use_jit) == 1
    assert fn(False, jit=use_jit) == 2


def test_if_else_return(use_jit):
    @func
    def fn(c1: bool):
        if c1:
            a = 1
        else:
            return 2
        return a

    assert fn(True, jit=use_jit) == 1
    assert fn(False, jit=use_jit) == 2


def test_if_both_return(use_jit):
    @func
    def fn(c1: bool):
        if c1:
            return 1
        else:
            return 2

    assert fn(True, jit=use_jit) == 1
    assert fn(False, jit=use_jit) == 2


def test_if_nested(use_jit):
    @func
    def fn(c1: bool, c2: bool):
        if c1:
            if c2:
                a = 1
            else:
                return 2
        else:
            return 3
        return a

    assert fn(True, True, jit=use_jit) == 1
    assert fn(True, False, jit=use_jit) == 2
    assert fn(False, True, jit=use_jit) == 3


def test_for(use_jit):
    @func
    def fn():
        a = 0
        for i in range(1, 6, 2):
            a = a + i
        return a

    assert fn(jit=use_jit) == 9


def test_for_changing_type(use_jit):
    @func
    def fn(init):
        for i in range(1, 6, 2):
            init = init + i
        return init

    assert fn(np.int32(0), jit=use_jit) == 9


def test_for_nested_if(use_jit):
    @func
    def fn(init, c1, c2):
        for i in range(0, 10):
            if c1:
                if c2:
                    init = i
                else:
                    init = init + i
            else:
                init = init - i
        return init

    assert fn(0, True, True, jit=use_jit) == 9
    assert fn(0, True, False, jit=use_jit) == 45
    assert fn(0, False, True, jit=use_jit) == -45