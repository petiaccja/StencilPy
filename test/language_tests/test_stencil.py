import numpy as np
import pytest

from stencilpy.storage import Field, Connectivity
from stencilpy.concepts import Dimension
from stencilpy.run import func, stencil
from stencilpy.stdlib import index, exchange, extend
from config import use_jit

TDim = Dimension()
UDim = Dimension()
LDim = Dimension()


def test_apply(use_jit):
    @stencil
    def sn(v):
        return v

    @func
    def fn(v, w: int, h: int):
        return sn[TDim[w], UDim[h]](v)

    r = fn(1, 4, 3, jit=use_jit)

    assert np.all(r.data == 1)
    assert r.data.shape[0] == 4
    assert r.data.shape[1] == 3


def test_apply_multiple(use_jit):
    @stencil
    def sn(u, v):
        return u, v

    @func
    def fn(u, v, w: int, h: int):
        return sn[TDim[w], UDim[h]](u, v)

    r = fn(1, 2, 4, 3, jit=use_jit)

    assert isinstance(r, tuple)
    assert len(r) == 2
    assert np.all(r[0].data == 1)
    assert np.all(r[1].data == 2)
    assert r[0].data.shape[0] == r[1].data.shape[0] == 4
    assert r[0].data.shape[1] == r[1].data.shape[1] == 3


@stencil
def sn_sample(source):
    return source[index()]


@func
def fn_sample(source, st, sl):
    return sn_sample[TDim[st], LDim[sl]](source)


def test_sample_field(use_jit):
    a = Field([TDim, LDim], np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    r = fn_sample(a, 2, 3, jit=use_jit)
    assert np.allclose(r.data, a.data)


def test_sample_connectivity(use_jit):
    a = Connectivity(TDim, UDim, LDim, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64))
    r = fn_sample(a, 2, 3, jit=use_jit)
    assert np.allclose(r.data, a.data)


def test_jump(use_jit):
    @stencil
    def sn(source):
        idx = index()
        jmp = idx[TDim[1]]
        return source[jmp]

    @func
    def fn(source, sz):
        return sn[TDim[sz]](source)

    a = Field([TDim], np.array([1, 2, 3, 4, 5], dtype=np.float32))
    e = Field([TDim], np.array([2, 3, 4, 5], dtype=np.float32))
    r = fn(a, 4, jit=use_jit)
    assert np.all(e.data == r.data)


def test_extract(use_jit):
    @stencil
    def sn():
        idx = index()
        return idx[TDim]

    @func
    def fn(sz):
        return sn[TDim[sz]]()

    e = Field([TDim], np.array([0, 1, 2, 3], dtype=np.float32))
    r = fn(4, jit=use_jit)
    assert np.all(e.data == r.data)


def test_exchange(use_jit):
    @stencil
    def sn(source, i):
        idx = index()
        ex = exchange(idx, i, TDim, UDim)
        return source[ex]

    @func
    def fn(source, sz):
        return sn[TDim[sz]](source, 2)

    a = Field([UDim], np.array([0, 1, 2, 3], dtype=np.float32))
    e = Field([TDim], np.array([2, 2, 2, 2], dtype=np.float32))
    r = fn(a, 4, jit=use_jit)
    assert np.all(e.data == r.data)


def test_extend(use_jit):
    @stencil
    def sn(source, i):
        idx = index()
        ex = extend(idx, i, UDim)
        return source[ex]

    @func
    def fn(source, sz):
        return sn[TDim[sz]](source, 1)

    a = Field([UDim, TDim], np.array([[0, 1, 2], [6, 7, 8]], dtype=np.float32))
    e = Field([TDim], np.array([6, 7, 8], dtype=np.float32))
    r = fn(a, 3, jit=use_jit)
    assert np.all(e.data == r.data)

