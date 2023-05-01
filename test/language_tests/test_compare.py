import math

import numpy as np
import pytest

from stencilpy.storage import Field
from stencilpy.concepts import Dimension
from stencilpy.run import func
from config import use_jit
from stencilpy.error import *


TDim = Dimension()
UDim = Dimension()

t_size = 3
u_size = 4

field_t_f32 = Field([TDim], np.random.uniform(1, 2, size=(t_size,)).astype(np.float32))
field_u_f32 = Field([UDim], np.random.uniform(1, 2, size=(u_size,)).astype(np.float32))
field_tu_f32 = Field([TDim, UDim], np.random.uniform(1, 2, size=(t_size, u_size)).astype(np.float32))
scalar_f32 = np.float32(3.98)
scalar_i32 = np.int32(4)
scalar_f64 = np.float64(6.67)


@func
def lt(a, b):
    return a < b


@func
def gt(a, b):
    return a > b


@func
def lte(a, b):
    return a <= b


@func
def gte(a, b):
    return a >= b


@func
def eq(a, b):
    return a == b


@func
def neq(a, b):
    return a != b


def outer_product_result(lhs, rhs):
    lhs_ext = np.reshape(lhs.data, (t_size, 1))
    rhs_ext = np.reshape(rhs.data, (1, u_size))
    return lhs_ext < rhs_ext

def broadcast_result(lhs, rhs):
    lhs_ext = np.reshape(lhs.data, (t_size, 1))
    rhs_ext = rhs.data
    return lhs_ext < rhs_ext


@pytest.mark.parametrize("lhs,rhs,expected", [
    (scalar_f32, scalar_f32, scalar_f32 < scalar_f32), # Scalar-scalar float
    (scalar_i32, scalar_i32, scalar_i32 < scalar_i32), # Scalar-scalar int
    (scalar_i32, scalar_f64, scalar_i32 < scalar_f64), # Scalar-scalar promote
    (scalar_f32, field_t_f32, scalar_f32 < field_t_f32.data), # Scalar-field
    (field_t_f32, scalar_f32, field_t_f32.data < scalar_f32), # Field-scalar
    (field_t_f32, field_t_f32, field_t_f32.data < field_t_f32.data), # Field-field
    (field_t_f32, field_u_f32, outer_product_result(field_t_f32, field_u_f32)), # Field-field outer product
    (field_t_f32, field_tu_f32, broadcast_result(field_t_f32, field_tu_f32)), # Field-field broadcast
])
def test_arguments(use_jit, lhs, rhs, expected):
    if not isinstance(lhs, Field) and isinstance(rhs, Field):
        # Comparison operators don't work in the reverse 'cause Python tries to cast the result to bool...
        pytest.xfail()
    r = lt(lhs, rhs, jit=use_jit)
    if isinstance(lhs, Field) or isinstance(rhs, Field):
        assert isinstance(r, Field)
        assert np.allclose(r.data, expected.data)
    else:
        assert not isinstance(r, Field)
        assert math.isclose(r, expected)


@pytest.mark.parametrize("fn,verif_fn", [
    (lt, np.less),
    (gt, np.greater),
    (lte, np.less_equal),
    (gte, np.greater_equal),
    (eq, np.equal),
    (neq, np.not_equal),
])
def test_arithmetic_functions(use_jit, fn, verif_fn):
    r = fn(field_t_f32, field_t_f32, jit=use_jit)
    e = verif_fn(field_t_f32.data, field_t_f32.data)
    assert np.allclose(r.data, e)


def test_promotion_failure():
    with pytest.raises(ArgumentCompatibilityError):
        lt(np.float32(1), np.int32(2), jit=True)
