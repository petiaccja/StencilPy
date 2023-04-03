from stencilpy import concepts
from stencilpy.compiler import types as ts, type_traits
from typing import Any
from stencilpy.compiler import hlast


@concepts.metafunc
def typeof(is_jit, value: Any):
    if is_jit:
        assert isinstance(value, hlast.Expr)
        return value.type_
    else:
        return type_traits.from_object(value)

@concepts.metafunc
def element_type(is_jit, value: ts.FieldLikeType):
    assert isinstance(value, ts.FieldLikeType)
    return value.element_type