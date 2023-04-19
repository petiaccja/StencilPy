from stencilpy import concepts
from stencilpy.compiler import types as ts, type_traits
from typing import Any
from stencilpy.compiler import hlast


@concepts.metafunc
def typeof(value: Any, transformer=None):
    if transformer:
        assert isinstance(value, hlast.Expr)
        return value.type_
    else:
        return type_traits.from_object(value)


@concepts.metafunc
def element_type(value: ts.FieldLikeType, **_):
    assert isinstance(value, ts.FieldLikeType)
    return value.element_type


@concepts.metafunc
def origin_dim(conn_type: ts.ConnectivityType, **_):
    return conn_type.origin_dimension


@concepts.metafunc
def neighbor_dim(conn_type: ts.ConnectivityType, **_):
    return conn_type.neighbor_dimension