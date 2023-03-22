from stencilpy.lib import *
from .config import use_jit
from stencilpy.concepts import *
from stencilpy.storage import *
from stencilpy.func import func, stencil

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