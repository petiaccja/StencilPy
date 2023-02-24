from stencilpy import concepts
from typing import Optional
from stencilpy import storage


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
