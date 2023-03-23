import typing

from stencilpy.compiler import hlast, types as ts
from typing import Any, cast, Union
from stencilpy.error import *
from stencilpy import concepts


T = typing.TypeVar("T")


def _get_location(expr: Any, fallback: concepts.Location = None) -> hlast.Location:
    if isinstance(expr, hlast.Node):
        return expr.location
    if fallback:
        return fallback
    return concepts.Location.unknown()


def ensure_expr(
        node: Any,
        loc: concepts.Location=None,
        name: str=None,
        cls: type | tuple[type, ...] = hlast.Expr
) -> hlast.Expr:
    assert issubclass(cls, hlast.Expr)
    if isinstance(node, cls):
        return cast(hlast.Expr, node)

    loc = _get_location(node, loc)
    msg = (
        f"expected an expression for {name}, got `{type(node)}`"
        if name
        else f"expected an expression, got `{type(node)}`"
    )
    raise CompilationError(loc, msg)


def ensure_dimension(
        expr: Any,
        loc: concepts.Location=None,
        name: str=None
) -> concepts.Dimension:
    if isinstance(expr, concepts.Dimension):
        return expr

    loc = _get_location(expr, loc)
    msg = (
        f"expected a dimension for {name}, got `{expr.__name__}`"
        if name
        else f"expected a dimension, got `{expr.__name__}`"
    )
    raise CompilationError(loc, msg)


def ensure_function(
        expr: Any,
        loc: concepts.Location=None,
) -> concepts.Function:
    if isinstance(expr, concepts.Function):
        return expr

    loc = _get_location(expr, loc)
    msg = f"expected a function, got `{expr.__name__}`"
    raise CompilationError(loc, msg)


def ensure_stencil(
        expr: Any,
        loc: concepts.Location=None,
) -> concepts.Stencil:
    if isinstance(expr, concepts.Stencil):
        return expr

    loc = _get_location(expr, loc)
    msg = f"expected a function, got `{expr.__name__}`"
    raise CompilationError(loc, msg)


def ensure_builtin(
        expr: Any,
        loc: concepts.Location=None,
) -> concepts.Builtin:
    if isinstance(expr, concepts.Builtin):
        return expr

    loc = _get_location(expr, loc)
    msg = f"expected a function, got `{expr.__name__}`"
    raise CompilationError(loc, msg)


def ensure_type(
        expr: hlast.Expr,
        type_: type | tuple[type, ...]
) -> ts.Type:
    if isinstance(expr.type_, type_):
        return expr.type_

    if not isinstance(tuple, type_):
        type_ = type_,
    msg = f"expression must be of type(s) {', '.join([str(t) for t in type_])}"
    raise CompilationError(expr.location, msg)

