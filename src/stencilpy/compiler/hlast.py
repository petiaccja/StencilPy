import dataclasses
from stencilpy.compiler import types as ts
from typing import Any, Callable, Optional
from stencilpy.concepts import Location, Dimension
import enum


#-------------------------------------------------------------------------------
# Basic nodes
#-------------------------------------------------------------------------------

@dataclasses.dataclass
class Node:
    location: Location
    type_: ts.Type


@dataclasses.dataclass
class Expr(Node):
    ...

@dataclasses.dataclass
class Statement(Node):
    ...


#-------------------------------------------------------------------------------
# Enums
#-------------------------------------------------------------------------------

class ArithmeticFunction(enum.Enum):
    ADD = enum.auto()
    SUB = enum.auto()
    MUL = enum.auto()
    DIV = enum.auto()
    MOD = enum.auto()
    BIT_AND = enum.auto()
    BIT_OR = enum.auto()
    BIT_XOR = enum.auto()
    BIT_SHL = enum.auto()
    BIT_SHR = enum.auto()


class ComparisonFunction(enum.Enum):
    EQ = enum.auto()
    NEQ = enum.auto()
    LT = enum.auto()
    GT = enum.auto()
    LTE = enum.auto()
    GTE = enum.auto()
    

#-------------------------------------------------------------------------------
# Primitive nodes & structures
#-------------------------------------------------------------------------------

@dataclasses.dataclass
class Parameter(Node):
    name: str
    
    
@dataclasses.dataclass
class Slice:
    dimension: Dimension
    lower: Optional[Expr]
    upper: Optional[Expr]
    step: Optional[Expr]
    single: bool = False


@dataclasses.dataclass
class Size:
    dimension: Dimension
    size: Expr


#-------------------------------------------------------------------------------
# Module structure
#-------------------------------------------------------------------------------

@dataclasses.dataclass
class Function(Node):
    name: str
    parameters: list[Parameter]
    result: ts.Type
    body: list[Statement]
    is_public: bool


@dataclasses.dataclass
class Stencil(Node):
    name: str
    parameters: list[Parameter]
    result: ts.Type
    body: list[Statement]
    dims: list[Dimension]


@dataclasses.dataclass
class Return(Statement):
    value: Optional[Expr]


@dataclasses.dataclass
class Call(Expr):
    name: str
    args: list[Expr]


@dataclasses.dataclass
class Apply(Expr):
    stencil: str
    shape: dict[Dimension, Expr]
    args: list[Expr]


@dataclasses.dataclass
class Module(Node):
    functions: list[Function]
    stencils: list[Stencil]


#-------------------------------------------------------------------------------
# Symbols
#-------------------------------------------------------------------------------

@dataclasses.dataclass
class SymbolRef(Expr):
    name: str


@dataclasses.dataclass
class Assign(Statement):
    names: list[str]
    values: list[Expr]


#-------------------------------------------------------------------------------
# Structured types
#-------------------------------------------------------------------------------

@dataclasses.dataclass
class TupleCreate(Expr):
    elements: list[Expr]


@dataclasses.dataclass
class TupleExtract(Expr):
    value: Expr
    item: int


#-------------------------------------------------------------------------------
# Control flow
#-------------------------------------------------------------------------------

@dataclasses.dataclass
class If(Expr):
    cond: Expr
    then_body: list[Statement]
    else_body: list[Statement]


@dataclasses.dataclass
class For(Expr):
    start: Expr
    stop: Expr
    step: Expr
    init: Optional[Expr]
    loop_index: str
    loop_carried: Optional[str]
    body: list[Statement]


@dataclasses.dataclass
class Yield(Statement):
    value: Optional[Expr]


#-------------------------------------------------------------------------------
# Arithmetic & logic
#-------------------------------------------------------------------------------

@dataclasses.dataclass
class Cast(Expr):
    value: Expr
    type: ts.Type
    
    
@dataclasses.dataclass
class Constant(Expr):
    value: Any


@dataclasses.dataclass
class ArithmeticOperation(Expr):
    lhs: Expr
    rhs: Expr
    func: ArithmeticFunction


@dataclasses.dataclass
class ComparisonOperation(Expr):
    lhs: Expr
    rhs: Expr
    func: ComparisonFunction


@dataclasses.dataclass
class Min(Expr):
    lhs: Expr
    rhs: Expr


@dataclasses.dataclass
class Max(Expr):
    lhs: Expr
    rhs: Expr


@dataclasses.dataclass
class ElementwiseOperation(Expr):
    args: list[Expr]
    element_expr: Callable[[list[Expr]], Expr]


#-------------------------------------------------------------------------------
# Tensor
#-------------------------------------------------------------------------------

@dataclasses.dataclass
class Shape(Expr):
    field: Expr
    dim: Dimension


@dataclasses.dataclass
class ExtractSlice(Expr):
    source: Expr
    slices: list[Slice]


@dataclasses.dataclass
class InsertSlice(Expr):
    source: Expr
    target: Expr
    slices: list[Slice]


@dataclasses.dataclass
class AllocEmpty(Expr):
    element_type: ts.Type
    shape: list[Size]


#-------------------------------------------------------------------------------
# Stencil intrinsics
#-------------------------------------------------------------------------------

@dataclasses.dataclass
class Index(Expr):
    pass


@dataclasses.dataclass
class Exchange(Expr):
    index: Expr
    value: Expr
    old_dim: Dimension
    new_dim: Dimension


@dataclasses.dataclass
class Sample(Expr):
    field: Expr
    index: Expr


#-------------------------------------------------------------------------------
# Special nodes
#-------------------------------------------------------------------------------

@dataclasses.dataclass
class ClosureVariable(Node):
    name: str
    value: Any


@dataclasses.dataclass
class Noop(Statement):
    pass