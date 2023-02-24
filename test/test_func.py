import numpy as np
import pytest

from stencilpy.storage import *
from stencilpy.concepts import Dimension
from stencilpy.lib import *
from stencilpy.func import func, stencil
from .config import use_jit


TDim = Dimension()


def test_func_passthrough_scalar(use_jit):
    @func
    def fn(a: int) -> int:
        return a

    assert fn(3, jit=use_jit) == 3


def test_stencil_passthrough_scalar(use_jit):
    if use_jit:
        pytest.skip("calling stencils directly via jit is not yet implemented")

    @stencil
    def sn(a: int) -> int:
        return a

    r = sn[TDim][3](3, jit=use_jit)

    assert isinstance(r, Field)
    assert shape(r, TDim) == 3
    assert np.allclose(r.data, 3)
