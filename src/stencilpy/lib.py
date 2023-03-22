import copy

from stencilpy import concepts
from typing import Optional, Any
from stencilpy import storage
from stencilpy.compiler import types as ts, hlast
from stencilpy import func, meta


@concepts.builtin
def index():
    return func._index


@concepts.builtin
def shape(field: storage.Field, dimension: concepts.Dimension) -> int:
    assert dimension in field.sorted_dimensions
    idx = field.sorted_dimensions.index(dimension)
    return field.data.shape[idx]


@concepts.builtin
def cast(value: Any, type_: ts.Type):
    return ts.as_numpy_type(type_)(value)


@concepts.builtin
def select(cond: bool, true_value: Any, false_value: Any):
    return true_value if cond else false_value


@concepts.builtin
def exchange(
        index: concepts.Index,
        value: Any,
        old_dim: concepts.Dimension,
        new_dim: concepts.Dimension
) -> concepts.Index:
    exchanged = copy.deepcopy(index)
    del exchanged.values[old_dim]
    exchanged.values[new_dim] = value
    return exchanged


@concepts.metafunc
def remap_old_dim(is_jit: bool, conn_type: ts.ConnectivityType):
    return conn_type.origin_dimension


@concepts.metafunc
def remap_new_dim(is_jit: bool, conn_type: ts.ConnectivityType):
    return conn_type.neighbor_dimension


@func.stencil
def remap_stencil(source: storage.Field, conn: storage.Connectivity):
    idx = index()
    conn_value = conn[idx]
    invalid_value = cast(-1, meta.typeof(conn_value))
    fallback_value = cast(0, meta.typeof(conn_value))
    clamped_value = select(conn_value == invalid_value, fallback_value, conn_value)
    source_idx = exchange(idx, clamped_value, remap_old_dim(meta.typeof(conn)), remap_new_dim(meta.typeof(conn)))
    return source[source_idx]


@concepts.metafunc
def remap_domain(is_jit: bool, source: storage.Field | hlast.Expr, conn: storage.Connectivity | hlast.Expr):
    loc = concepts.Location("<remap_domain>", 1, 0)
    src_type = meta.typeof(source, is_jit=is_jit)
    conn_type = meta.typeof(conn, is_jit=is_jit)
    assert isinstance(src_type, ts.FieldType)
    assert isinstance(conn_type, ts.ConnectivityType)
    src_dims = list(set(src_type.dimensions) - {conn_type.neighbor_dimension})
    if is_jit:
        return (
            *[hlast.Size(dim, hlast.Shape(loc, ts.index_t, source, dim)) for dim in src_dims],
            *[hlast.Size(dim, hlast.Shape(loc, ts.index_t, conn, dim)) for dim in conn_type.dimensions],
        )
    else:
        return (
            *[concepts.Slice(dim, source.shape[dim]) for dim in src_dims],
            *[concepts.Slice(dim, conn.shape[dim]) for dim in conn_type.dimensions],
        )


@func.func
def remap(source: storage.Field, conn: storage.Connectivity):
    return remap_stencil[remap_domain(source, conn)](source, conn)


@func.stencil
def sparsity_stencil(conn: storage.Connectivity):
    invalid_value = cast(-1, meta.element_type(meta.typeof(conn)))
    value = conn[index()]
    return value != invalid_value


@concepts.metafunc
def sparsity_domain(is_jit: bool, conn: storage.Connectivity | hlast.Expr):
    loc = concepts.Location("<sparsity_domain>", 1, 0)
    conn_type = meta.typeof(conn, is_jit=is_jit)
    assert isinstance(conn_type, ts.ConnectivityType)
    if is_jit:
        return (
            *[hlast.Size(dim, hlast.Shape(loc, ts.index_t, conn, dim)) for dim in conn_type.dimensions],
        )
    else:
        return (
            *[concepts.Slice(dim, conn.shape[dim]) for dim in conn_type.dimensions],
        )

@func.func
def sparsity(conn: storage.Connectivity):
    return sparsity_stencil[sparsity_domain(conn)](conn)