import math
import random

import stdlib
from stencilpy.stdlib import *
from config import use_jit
from stencilpy.concepts import *
from stencilpy.storage import *
from stencilpy.run import func, stencil
import pytest

TDim = concepts.Dimension()
UDim = concepts.Dimension()
VDim = concepts.Dimension()
WDim = concepts.Dimension()
LDim = concepts.Dimension()


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


def test_exchange(use_jit):
    @stencil
    def sn():
        exchange(index(), 0, TDim, UDim)
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


def test_select(use_jit):
    @func
    def fn(cond, a, b):
        return select(cond, a, b)

    assert fn(True, 1, 2, jit=use_jit) == 1
    assert fn(False, 1, 2, jit=use_jit) == 2


def test_remap(use_jit):
    @func
    def fn(a: Field, conn: Connectivity):
        return remap(a, conn)

    u = Field([TDim, WDim, UDim], np.array([
        [ # 0
            [9000, 9001], # 0
            [9010, 9011], # 1
        ],
        [ # 1
            [9100, 9101],  # 0
            [9110, 9111],  # 1
        ]
    ]))
    conn = Connectivity(VDim, UDim, LDim, np.array([
        [0, 1],
        [1, 0],
    ]))
    e = Field(
        [TDim, WDim, VDim, LDim],
        np.array([
            [  # 0
                [[9000, 9001], [9001, 9000]],  # 0
                [[9010, 9011], [9011, 9010]],  # 1
            ],
            [  # 1
                [[9100, 9101], [9101, 9100]],  # 0
                [[9110, 9111], [9111, 9110]],  # 1
            ]
        ])
    )

    r = fn(u, conn, jit=use_jit)
    assert r.sorted_dimensions == e.sorted_dimensions
    assert np.all(r.data == e.data)


def test_sparsity(use_jit):
    @func
    def fn(conn: Connectivity):
        return sparsity(conn)

    conn = Connectivity(VDim, UDim, LDim, np.array([
        [-1, 1],
        [1, 0],
    ]))
    expected = Field([VDim, LDim], np.array([
        [False, True],
        [True, True],
    ]))

    r = fn(conn, jit=use_jit)
    assert r.sorted_dimensions == expected.sorted_dimensions
    assert np.all(r.data == expected.data)


def test_reduce(use_jit):
    @func
    def fn(field: Field):
        return reduce(field, TDim)

    a = Field([TDim, UDim], np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
    ]))

    r = fn(a, jit=use_jit)
    expected = np.sum(a.data, axis=0)
    assert isinstance(r, Field)
    assert r.sorted_dimensions == [UDim]
    assert np.allclose(r.data, expected)


def math_fun_range(fun: concepts.Builtin):
    if fun == stdlib.acosh:
        return 1, 2
    return 0, 1


@pytest.mark.parametrize("math_func,verify_func", [
    # Exponential
    (stdlib.exp, np.exp),
    (stdlib.exp2, np.exp2),
    (stdlib.expm1, np.expm1),
    (stdlib.log, np.log),
    (stdlib.log10, np.log10),
    (stdlib.log2, np.log2),
    (stdlib.log1p, np.log1p),
    # Power
    (stdlib.sqrt, np.sqrt),
    (stdlib.cbrt, np.cbrt),
    # Trigonometric
    (stdlib.sin, np.sin),
    (stdlib.cos, np.cos),
    (stdlib.tan, np.tan),
    (stdlib.asin, np.arcsin),
    (stdlib.acos, np.arccos),
    (stdlib.atan, np.arctan),
    # Hyperbolic
    (stdlib.sinh, np.sinh),
    (stdlib.cosh, np.cosh),
    (stdlib.tanh, np.tanh),
    (stdlib.asinh, np.arcsinh),
    (stdlib.acosh, np.arccosh),
    (stdlib.atanh, np.arctanh),
])
def test_math_builtins_unary(use_jit, math_func, verify_func):
    @func
    def fn(arg):
        return math_func(arg)

    lr, ur = math_fun_range(math_func)
    s = random.uniform(lr, ur)
    f = Field([TDim, UDim], np.random.uniform(lr, ur, size=(5, 6)))

    assert math.isclose(fn(s, jit=use_jit), verify_func(s))
    assert np.allclose(fn(f, jit=use_jit).data, verify_func(f.data))\


@pytest.mark.parametrize("math_func,verify_func", [
    (stdlib.atan2, np.arctan2),
    (stdlib.pow, np.power),
    (stdlib.hypot, np.hypot),
])
def test_math_builtins_binary(use_jit, math_func, verify_func):
    @func
    def fn(arg0, arg1):
        return math_func(arg0, arg1)

    lr, ur = math_fun_range(math_func)
    s0 = random.uniform(lr, ur)
    s1 = random.uniform(lr, ur)
    f0 = Field([TDim, UDim], np.random.uniform(lr, ur, size=(5, 6)))
    f1 = Field([TDim, UDim], np.random.uniform(lr, ur, size=(5, 6)))

    assert math.isclose(fn(s0, s1, jit=use_jit), verify_func(s0, s1))
    assert np.allclose(fn(f0, f1, jit=use_jit).data, verify_func(f0.data, f1.data))