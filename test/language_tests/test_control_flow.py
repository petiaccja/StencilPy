import numpy as np
import pytest

from stencilpy.storage import Field
from stencilpy.concepts import Dimension
from stencilpy.run import func
from config import use_jit

TDim = Dimension()
UDim = Dimension()
LDim = Dimension()


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


def test_yield_for_field(use_jit):
    pytest.xfail()

    @func
    def fn(init: Field, stop: int) -> Field:
        for i in range(stop):
            init = 1*init
        return init

    a = Field([TDim], np.array([1, 2, 3]))
    r = fn(a, 10, jit=use_jit)
    assert np.all(r.data == a.data)


def test_yield_if_field(use_jit):
    pytest.xfail()

    @func
    def fn(c: bool, a: Field, b: Field) -> Field:
        if c:
            return 1 * a
        else:
            return 1 * b

    a = Field([TDim], np.array([1, 2, 3]))
    b = Field([TDim], np.array([1, 2, 3]))
    assert np.all(fn(True, a, b, jit=use_jit).data == a.data)
    assert np.all(fn(False, a, b, jit=use_jit).data == b.data)