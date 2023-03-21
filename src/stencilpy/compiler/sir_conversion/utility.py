import itertools

from stencilpy import concepts
from stencilpy.compiler import hlast
from stencilpy.compiler import types as ts
from stencilir import ops
import stencilir as sir
from typing import Sequence, Iterable, Any, Optional


def as_sir_loc(loc: hlast.Location) -> ops.Location:
    return ops.Location(loc.file, loc.line, loc.column)


def as_sir_type(type_: ts.Type) -> sir.Type:
    if isinstance(type_, ts.IndexType):
        return sir.IndexType()
    elif isinstance(type_, ts.IntegerType):
        return sir.IntegerType(type_.width, type_.signed)
    elif isinstance(type_, ts.FloatType):
        return sir.FloatType(type_.width)
    elif isinstance(type_, ts.FieldLikeType):
        return sir.FieldType(as_sir_type(type_.element_type), len(type_.dimensions))
    elif isinstance(type_, ts.FunctionType):
        return sir.FunctionType(
            [as_sir_type(p) for p in type_.parameters],
            [as_sir_type(r) for r in type_.results],
        )
    else:
        raise ValueError(f"no SIR type equivalent for {type_.__class__}")


def as_sir_arithmetic(func: hlast.ArithmeticFunction) -> ops.ArithmeticFunction:
    _MAPPING = {
        hlast.ArithmeticFunction.ADD: ops.ArithmeticFunction.ADD,
        hlast.ArithmeticFunction.SUB: ops.ArithmeticFunction.SUB,
        hlast.ArithmeticFunction.MUL: ops.ArithmeticFunction.MUL,
        hlast.ArithmeticFunction.DIV: ops.ArithmeticFunction.DIV,
        hlast.ArithmeticFunction.MOD: ops.ArithmeticFunction.MOD,
        hlast.ArithmeticFunction.BIT_AND: ops.ArithmeticFunction.BIT_AND,
        hlast.ArithmeticFunction.BIT_OR: ops.ArithmeticFunction.BIT_OR,
        hlast.ArithmeticFunction.BIT_XOR: ops.ArithmeticFunction.BIT_XOR,
        hlast.ArithmeticFunction.BIT_SHL: ops.ArithmeticFunction.BIT_SHL,
        hlast.ArithmeticFunction.BIT_SHR: ops.ArithmeticFunction.BIT_SHR,
    }
    return _MAPPING[func]


def as_sir_comparison(func: hlast.ComparisonFunction) -> ops.ComparisonFunction:
    _MAPPING = {
        hlast.ComparisonFunction.EQ: ops.ComparisonFunction.EQ,
        hlast.ComparisonFunction.NEQ: ops.ComparisonFunction.NEQ,
        hlast.ComparisonFunction.LT: ops.ComparisonFunction.LT,
        hlast.ComparisonFunction.GT: ops.ComparisonFunction.GT,
        hlast.ComparisonFunction.LTE: ops.ComparisonFunction.LTE,
        hlast.ComparisonFunction.GTE: ops.ComparisonFunction.GTE,
    }
    return _MAPPING[func]


def map_elementwise_shape(
        dimensions: Sequence[concepts.Dimension],
        arg_types: Sequence[ts.Type]
) -> dict[concepts.Dimension, int]:
    dims_to_arg: dict[concepts.Dimension, int] = {}
    for dim in dimensions:
        for arg_idx, type_ in enumerate(arg_types):
            if not isinstance(type_, ts.FieldType):
                continue
            arg_dims = type_.dimensions
            if not dim in arg_dims:
                continue
            dims_to_arg[dim] = arg_idx
    return dims_to_arg


def is_slice_adjustment_trivial(
        start: Optional[hlast.Expr],
        step: Optional[hlast.Expr],
) -> bool:
    is_step_trivial = not step or (isinstance(step, hlast.Constant) and step.value > 0)
    is_start_trivial = (is_step_trivial and not start) or (isinstance(start, hlast.Constant) and start.value >= 0)
    return is_start_trivial and is_step_trivial


def shape_func_name(name: str):
    return f"__shapes_{name}"


def flatten(values: Iterable[Iterable[Any]]) -> list[Any]:
    return list(itertools.chain(*values))