from stencilpy import concepts
from typing import Optional, Any
from stencilpy import storage
from stencilpy.compiler import types as ts, hlast


_index: Optional[concepts.Index] = None


@concepts.builtin
def index():
    global _index
    return _index


@concepts.builtin
def shape(field: storage.Field, dimension: concepts.Dimension) -> int:
    assert dimension in field.sorted_dimensions
    idx = field.sorted_dimensions.index(dimension)
    return field.data.shape[idx]


@concepts.builtin
def cast(value: Any, type_: ts.Type):
    return ts.as_numpy_type(type_)(value)