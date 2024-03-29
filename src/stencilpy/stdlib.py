import copy
import numpy as np

from stencilpy import concepts
from typing import Optional, Any, Callable
from stencilpy import storage
from stencilpy.compiler import types as ts, hlast, type_traits
import stencilpy.run
import stencilpy.metalib
import stencilpy.utility


@concepts.builtin
def index():
    return stencilpy.run._index


@concepts.builtin
def shape(field: storage.Field, dimension: concepts.Dimension) -> int:
    assert dimension in field.sorted_dimensions
    idx = field.sorted_dimensions.index(dimension)
    return field.data.shape[idx]


@concepts.builtin
def cast(value: Any, type_: ts.Type):
    return type_traits.to_numpy_type(type_)(value)


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


@concepts.builtin
def extend(
        index: concepts.Index,
        value: Any,
        dim: concepts.Dimension,
) -> concepts.Index:
    exchanged = copy.deepcopy(index)
    exchanged.values[dim] = value
    return exchanged


#---------------------------------------
# Math
#---------------------------------------

# List of functions: https://en.cppreference.com/w/c/numeric/math

# Exponential
@concepts.builtin
def exp(x: Any):
    return storage.elementwise_op(np.exp, x)


@concepts.builtin
def exp2(x: Any):
    return storage.elementwise_op(np.exp2, x)


@concepts.builtin
def expm1(x: Any):
    return storage.elementwise_op(np.expm1, x)


@concepts.builtin
def log(x: Any):
    return storage.elementwise_op(np.log, x)


@concepts.builtin
def log10(x: Any):
    return storage.elementwise_op(np.log10, x)


@concepts.builtin
def log2(x: Any):
    return storage.elementwise_op(np.log2, x)


@concepts.builtin
def log1p(x: Any):
    return storage.elementwise_op(np.log1p, x)


# Power
@concepts.builtin
def pow(x: Any, y: Any):
    return storage.elementwise_op(np.power, x, y)


@concepts.builtin
def sqrt(x: Any):
    return storage.elementwise_op(np.sqrt, x)


@concepts.builtin
def cbrt(x: Any):
    return storage.elementwise_op(np.cbrt, x)


@concepts.builtin
def hypot(x: Any, y: Any):
    return storage.elementwise_op(np.hypot, x, y)


# Trigonometric
@concepts.builtin
def sin(x: Any):
    return storage.elementwise_op(np.sin, x)


@concepts.builtin
def cos(x: Any):
    return storage.elementwise_op(np.cos, x)


@concepts.builtin
def tan(x: Any):
    return storage.elementwise_op(np.tan, x)


@concepts.builtin
def asin(x: Any):
    return storage.elementwise_op(np.arcsin, x)


@concepts.builtin
def acos(x: Any):
    return storage.elementwise_op(np.arccos, x)


@concepts.builtin
def atan(x: Any):
    return storage.elementwise_op(np.arctan, x)


@concepts.builtin
def atan2(x: Any, y: Any):
    return storage.elementwise_op(np.arctan2, x, y)


# Hyperbolic
@concepts.builtin
def sinh(x: Any):
    return storage.elementwise_op(np.sinh, x)


@concepts.builtin
def cosh(x: Any):
    return storage.elementwise_op(np.cosh, x)


@concepts.builtin
def tanh(x: Any):
    return storage.elementwise_op(np.tanh, x)


@concepts.builtin
def asinh(x: Any):
    return storage.elementwise_op(np.arcsinh, x)


@concepts.builtin
def acosh(x: Any):
    return storage.elementwise_op(np.arccosh, x)


@concepts.builtin
def atanh(x: Any):
    return storage.elementwise_op(np.arctanh, x)



#---------------------------------------
# Remap
#---------------------------------------

@stencilpy.run.stencil
def _sn_remap(source: storage.Field, conn: storage.Connectivity):
    idx = index()
    conn_value = conn[idx]
    invalid_value = cast(-1, stencilpy.metalib.typeof(conn_value))
    fallback_value = cast(0, stencilpy.metalib.typeof(conn_value))
    clamped_value = select(conn_value == invalid_value, fallback_value, conn_value)
    conn_type = stencilpy.metalib.typeof(conn)
    source_idx = exchange(
        idx,
        clamped_value,
        stencilpy.metalib.origin_dim(conn_type),
        stencilpy.metalib.neighbor_dim(conn_type)
    )
    return source[source_idx]


@concepts.metafunc
def _remap_domain(source: storage.Field | hlast.Expr, conn: storage.Connectivity | hlast.Expr, transformer=None):
    loc = concepts.Location("<remap_domain>", 1, 0)
    src_type = stencilpy.metalib.typeof(source, transformer=transformer)
    conn_type = stencilpy.metalib.typeof(conn, transformer=transformer)
    assert isinstance(src_type, ts.FieldType)
    assert isinstance(conn_type, ts.ConnectivityType)
    src_dims = list(set(src_type.dimensions) - {conn_type.neighbor_dimension})
    if transformer:
        sizes = [
            *[hlast.Size(loc, ts.index_t, dim, hlast.Shape(loc, ts.index_t, source, dim)) for dim in src_dims],
            *[hlast.Size(loc, ts.index_t, dim, hlast.Shape(loc, ts.index_t, conn, dim)) for dim in conn_type.dimensions],
        ]
        return hlast.TupleCreate(loc, ts.TupleType([ts.index_t]*len(sizes)), sizes)
    else:
        return (
            *[concepts.Slice(dim, source.shape[dim]) for dim in src_dims],
            *[concepts.Slice(dim, conn.shape[dim]) for dim in conn_type.dimensions],
        )


@stencilpy.run.func
def remap(source: storage.Field, conn: storage.Connectivity):
    return _sn_remap[_remap_domain(source, conn)](source, conn)


#---------------------------------------
# Sparsity
#---------------------------------------

@stencilpy.run.stencil
def _sn_sparsity(conn: storage.Connectivity):
    invalid_value = cast(-1, stencilpy.metalib.element_type(stencilpy.metalib.typeof(conn)))
    value = conn[index()]
    return value != invalid_value


@concepts.metafunc
def _sparsity_domain(conn: storage.Connectivity | hlast.Expr, transformer=None):
    loc = concepts.Location("<sparsity_domain>", 1, 0)
    conn_type = stencilpy.metalib.typeof(conn, transformer=transformer)
    assert isinstance(conn_type, ts.ConnectivityType)
    if transformer:
        sizes = [
            hlast.Size(loc, ts.index_t, dim, hlast.Shape(loc, ts.index_t, conn, dim))
            for dim in conn_type.dimensions
        ]
        return hlast.TupleCreate(loc, ts.TupleType([ts.index_t] * len(sizes)), sizes)
    else:
        return (
            *[concepts.Slice(dim, conn.shape[dim]) for dim in conn_type.dimensions],
        )

@stencilpy.run.func
def sparsity(conn: storage.Connectivity):
    return _sn_sparsity[_sparsity_domain(conn)](conn)


#---------------------------------------
# Reduction
#---------------------------------------

_reduce_ctx_dim_var: Optional[concepts.Dimension] = None

@stencilpy.concepts.metafunc
def _reduce_ctx_dim(**_):
    return _reduce_ctx_dim_var


@stencilpy.concepts.metafunc
def _reduce_domain(field: storage.Field | hlast.Expr, transformer=None):
    loc = concepts.Location("<reduce_domain>", 1, 0)
    dim = _reduce_ctx_dim()
    if not transformer:
        assert isinstance(field, storage.Field)
        shape = field.shape
        del shape[dim]
        return tuple(concepts.Slice(dim, size) for dim, size in shape.items())
    else:
        assert isinstance(field, hlast.Expr)
        assert isinstance(field.type_, ts.FieldType)
        sizes = [
            hlast.Size(loc, ts.index_t, d, hlast.Shape(loc, ts.index_t, field, d))
            for d in field.type_.dimensions if d != dim
        ]
        return hlast.TupleCreate(loc, ts.TupleType([ts.index_t]*len(sizes)), sizes)

@stencilpy.run.stencil
def _sn_reduce(field: storage.Field):
    elem_t = stencilpy.metalib.element_type(stencilpy.metalib.typeof(field))
    dim = _reduce_ctx_dim()
    init = cast(0.0, elem_t)
    size = shape(field, dim)
    for i in range(size):
        idx = extend(index(), i, dim)
        value = field[idx]
        init = init + value
    return init


@stencilpy.run.func
def _fn_reduce(field: storage.Field):
    return _sn_reduce[_reduce_domain(field)](field)


@concepts.metafunc
def reduce(field: storage.Field | hlast.Expr, dim: concepts.Dimension, transformer=None):
    loc = concepts.Location("<reduce>", 1, 0)
    global _reduce_ctx_dim_var
    _reduce_ctx_dim_var = dim
    if not transformer:
        assert isinstance(field, storage.Field)
        return _fn_reduce(field)
    else:
        from stencilpy.compiler.parser.ast_transformer import AstTransformer
        assert isinstance(transformer, AstTransformer)
        assert isinstance(field, hlast.Expr)
        assert isinstance(field.type_, ts.FieldType)
        func: hlast.Function = transformer.instantiate(_fn_reduce.definition, [field.type_], False, None)
        type_ = ts.FieldType(field.type_.element_type, [d for d in field.type_.dimensions if d != dim])
        return hlast.Call(loc, type_, func.name, [field])

