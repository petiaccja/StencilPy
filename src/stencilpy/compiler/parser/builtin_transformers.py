from stencilpy.compiler import hlast
from stencilpy.compiler import types as ts
from stencilpy import concepts
from stencilpy.error import *
from typing import Optional, Sequence
from stencilpy.compiler.node_transformer import NodeTransformer
from .utility import FunctionSpecification



def builtin_shape(transformer: NodeTransformer, location: concepts.Location, args: Sequence[ast.AST]):
    assert len(args) == 2
    field = transformer.visit(args[0])
    dim = transformer.visit(args[1])
    if not isinstance(dim, concepts.Dimension):
        raise CompilationError(location, "the `shape` function expects a dimension for argument 2")

    return hlast.Shape(location, ts.IndexType(), field, dim)


def builtin_index(transformer: NodeTransformer, location: concepts.Location, args: Sequence[ast.AST]):
    function_def_info: Optional[FunctionSpecification] = None
    for info in transformer.symtable.infos():
        if isinstance(info, FunctionSpecification):
            function_def_info = info
            break
    if not function_def_info:
        raise CompilationError(location, "index expression must be used inside a stencil's body")
    type_ = ts.NDIndexType(function_def_info.dims)
    return hlast.Index(location, type_)


def builtin_exchange(transformer: NodeTransformer, location: concepts.Location, args: Sequence[ast.AST]):
    assert len(args) == 4
    index = transformer.visit(args[0])
    value = transformer.visit(args[1])
    old_dim = transformer.visit(args[2])
    new_dim = transformer.visit(args[3])
    type_ = index.type_
    if isinstance(index.type_, ts.NDIndexType):
        new_dims = list((set(index.type_.dims) - {old_dim}) | {new_dim})
        type_ = ts.NDIndexType(sorted(new_dims))
    return hlast.Exchange(location, type_, index, value, old_dim, new_dim)


def builtin_cast(transformer: NodeTransformer, location: concepts.Location, args: Sequence[ast.AST]):
    assert len(args) == 2
    value = transformer.visit(args[0])
    type_ = transformer.visit(args[1])
    return hlast.Cast(location, type_, value, type_)


def builtin_select(transformer: NodeTransformer, location: concepts.Location, args: Sequence[ast.AST]):
    assert len(args) == 3
    cond = transformer.visit(args[0])
    true_value = transformer.visit(args[1])
    false_value = transformer.visit(args[2])
    type_ = true_value.type_  # TODO: do type promotion, if possible
    return hlast.If(
        location,
        type_,
        cond,
        [hlast.Yield(location, ts.void_t, [true_value])],
        [hlast.Yield(location, ts.void_t, [false_value])],
    )


BUILTIN_MAPPING = {
    "shape": builtin_shape,
    "index": builtin_index,
    "exchange": builtin_exchange,
    "cast": builtin_cast,
    "select": builtin_select,
}
