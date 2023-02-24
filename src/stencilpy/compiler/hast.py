import dataclasses
from stencilpy.compiler import types as ts
from typing import Any
from stencilpy.concepts import Location, Dimension


@dataclasses.dataclass
class Parameter:
    name: str
    type_: ts.Type


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


@dataclasses.dataclass
class Function(Node):
    name: str
    parameters: list[Parameter]
    results: list[ts.Type]
    body: list[Statement]


@dataclasses.dataclass
class Stencil(Node):
    name: str
    parameters: list[Parameter]
    results: list[ts.Type]
    body: list[Statement]
    ndim: int


@dataclasses.dataclass
class Return(Statement):
    values: list[Expr]


@dataclasses.dataclass
class Constant(Expr):
    value: Any


@dataclasses.dataclass
class SymbolRef(Expr):
    name: str


@dataclasses.dataclass
class Shape(Expr):
    field: Expr
    dim: Dimension


@dataclasses.dataclass
class ExternalSymbol(Node):
    name: str
    value: Any


@dataclasses.dataclass
class Module(Node):
    functions: list[Function]
    stencils: list[Stencil]
