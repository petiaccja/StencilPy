import math

import numpy as np
import pytest

from stencilpy.storage import Field
from stencilpy.concepts import Dimension
from stencilpy.run import func
from config import use_jit
from stencilpy.error import *

TDim = Dimension()

t_size = 3

field_t_f32 = Field([TDim], np.random.uniform(1, 2, size=(t_size,)).astype(np.float32))
scalar_f32 = np.float32(3.98)
scalar_i32 = np.int32(4)
scalar_i1 = np.bool_(True)


@func
def plus(a):
    return +a


@func
def minus(a):
    return -a


@func
def not_(a):
    return not a


@func
def invert(a):
    return ~a


def test_plus(use_jit):
    assert np.allclose(plus(field_t_f32, jit=use_jit).data, field_t_f32.data)
    assert plus(scalar_f32, jit=use_jit) == scalar_f32
    assert plus(scalar_i32, jit=use_jit) == scalar_i32
    assert plus(scalar_i1, jit=use_jit) == scalar_i1


def test_minus(use_jit):
    assert np.allclose(minus(field_t_f32, jit=use_jit).data, -field_t_f32.data)
    assert minus(scalar_f32, jit=use_jit) == -scalar_f32
    assert minus(scalar_i32, jit=use_jit) == -scalar_i32
    with pytest.raises(ArgumentCompatibilityError):
        minus(scalar_i1, jit=True)


def test_invert(use_jit):
    with pytest.raises(ArgumentCompatibilityError):
        invert(field_t_f32, jit=True)
    with pytest.raises(ArgumentCompatibilityError):
        invert(scalar_f32, jit=True)
    assert invert(scalar_i32, jit=use_jit) == ~scalar_i32
    assert invert(scalar_i1, jit=use_jit) == ~scalar_i1


def test_not(use_jit):
    # TODO: does not work with Python, should work with JIT
    # assert np.allclose(not_(field_t_f32, jit=use_jit).data, (not field_t_f32.data))
    assert not_(scalar_f32, jit=use_jit) == (not scalar_f32)
    assert not_(scalar_i32, jit=use_jit) == (not scalar_i32)
    assert not_(scalar_i1, jit=use_jit) == (not scalar_i1)
