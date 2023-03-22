from stencilpy.lib import *
from stencilpy.meta import *
from .config import use_jit
from stencilpy.concepts import *
from stencilpy.storage import *
from stencilpy.func import func, stencil


TDim = Dimension()


def test_typeof(use_jit):
    @func
    def fn(value: Any, forcing: Any):
        return cast(value, typeof(forcing))

    assert fn(3.1415, 3, jit=use_jit) == 3
    assert isinstance(fn(3, 3.14), float)


def test_element_type(use_jit):
    @func
    def fn(value: Any, forcing: Field):
        type_ = typeof(forcing)
        dtype = element_type(type_)
        return cast(value, dtype)

    forcing = Field([TDim], np.array([], np.int))
    assert fn(3.1415, forcing, jit=use_jit) == 3