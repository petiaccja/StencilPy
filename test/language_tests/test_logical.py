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

field_bool = Field([TDim], np.random.uniform(1, 2, size=(t_size,)).astype(np.bool_))
scalar_bool = np.bool_(False)


@func
def and_(a, b):
    return a and b


@func
def or_(a, b):
    return a or b


def test_and_table(use_jit):
    assert and_(False, False, jit=use_jit) == False
    assert and_(False, True, jit=use_jit) == False
    assert and_(True, False, jit=use_jit) == False
    assert and_(True, True, jit=use_jit) == True


def test_or_table(use_jit):
    assert or_(False, False, jit=use_jit) == False
    assert or_(False, True, jit=use_jit) == True
    assert or_(True, False, jit=use_jit) == True
    assert or_(True, True, jit=use_jit) == True


def test_short_circuit(use_jit):
    @func
    def fn(a, b, c):
        return a or not not (b / c) # Division by zero never evaluated

    assert fn(True, True, 0, jit=use_jit) == True
