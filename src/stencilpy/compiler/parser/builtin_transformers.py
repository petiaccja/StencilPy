import stencilpy.utility
from stencilpy.compiler import hlast
from stencilpy.compiler import types as ts, type_traits
from stencilpy import concepts
from stencilpy.error import *
from typing import Optional, Sequence
from stencilpy.compiler.node_transformer import NodeTransformer
from stencilpy.compiler.parser.utility import FunctionSpecification



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


def builtin_extend(transformer: NodeTransformer, location: concepts.Location, args: Sequence[ast.AST]):
    assert len(args) == 3
    index = transformer.visit(args[0])
    value = transformer.visit(args[1])
    dim = transformer.visit(args[2])
    type_ = index.type_
    if isinstance(index.type_, ts.NDIndexType):
        new_dims = list(set(index.type_.dims) | {dim})
        type_ = ts.NDIndexType(sorted(new_dims))
    return hlast.Extend(location, type_, index, value, dim)


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
        [hlast.Yield(location, ts.void_t, true_value)],
        [hlast.Yield(location, ts.void_t, false_value)],
    )


def _builtin_math_variant(name: str, common_ty: ts.Type, location: concepts.Location):
    if isinstance(common_ty, ts.IntegerType):
        return name, ts.FloatType(64)
    elif isinstance(common_ty, ts.IndexType):
        return name, ts.FloatType(64)
    elif isinstance(common_ty, ts.FloatType):
        if common_ty.width < 64:
            return f"{name}f", ts.FloatType(32)
        if common_ty.width >= 64:
            return name, ts.FloatType(64)
    else:
        raise CompilationError(
            location,
            f"no matching overloaded function found for {name}, argument type is {common_ty}"
        )


def _builtin_math(name: str, transformer: NodeTransformer, location: concepts.Location, args: Sequence[ast.AST]):
    from .ast_transformer import AstTransformer
    assert isinstance(transformer, AstTransformer)
    args = [transformer.visit(arg) for arg in args]
    arg_types = [arg.type_ for arg in args]
    common_ty = type_traits.common_type(*[type_traits.element_type(ty) for ty in arg_types])
    common_dims = type_traits.common_dims(*arg_types)

    if not common_ty:
        raise ArgumentCompatibilityError(location, "binop", arg_types)

    variant_name, variant_ty = _builtin_math_variant(name, common_ty, location)

    if variant_name not in transformer.instantiations:
        func_ty = ts.FunctionType([variant_ty] * len(args), variant_ty)
        fun_loc = concepts.Location.unknown()
        params = [hlast.Parameter(fun_loc, variant_ty, f"p{i}") for i in range(len(args))]
        variant_func = hlast.Function(fun_loc, func_ty, variant_name, params, func_ty.result, [], True)
        transformer.instantiations[variant_name] = variant_func

    def builder(args: list[hlast.Expr]) -> hlast.Expr:
        conv = [hlast.Cast(location, variant_ty, arg, variant_ty) for arg in args]
        result = hlast.Call(location, variant_ty, variant_name, conv)
        return hlast.Cast(location, common_ty, result, common_ty)

    if common_dims:
        type_ = ts.FieldType(common_ty, common_dims)
        return hlast.ElementwiseOperation(location, type_, args, builder)
    return builder(args)

def get_builtin_math(name: str):
    return lambda transformer, location, args: _builtin_math(name, transformer, location, args)


BUILTIN_MAPPING = {
    "shape": builtin_shape,
    "index": builtin_index,
    "exchange": builtin_exchange,
    "extend": builtin_extend,
    "cast": builtin_cast,
    "select": builtin_select,
    # Exponential
    "exp": get_builtin_math("exp"),
    "exp2": get_builtin_math("exp2"),
    "expm1": get_builtin_math("expm1"),
    "log": get_builtin_math("log"),
    "log10": get_builtin_math("log10"),
    "log2": get_builtin_math("log2"),
    "log1p": get_builtin_math("log1p"),
    # Power
    "pow": get_builtin_math("pow"),
    "sqrt": get_builtin_math("sqrt"),
    "cbrt": get_builtin_math("cbrt"),
    "hypot": get_builtin_math("hypot"),
    # Trigonometric
    "sin": get_builtin_math("sin"),
    "cos": get_builtin_math("cos"),
    "tan": get_builtin_math("tan"),
    "asin": get_builtin_math("asin"),
    "acos": get_builtin_math("acos"),
    "atan": get_builtin_math("atan"),
    "atan2": get_builtin_math("atan2"),
    # Hyperbolic
    "sinh": get_builtin_math("sinh"),
    "cosh": get_builtin_math("cosh"),
    "tanh": get_builtin_math("tanh"),
    "asinh": get_builtin_math("asinh"),
    "acosh": get_builtin_math("acosh"),
    "atanh": get_builtin_math("atanh"),
}
