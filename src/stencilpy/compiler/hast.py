import dataclasses
from stencilpy.compiler import types as ts
from typing import Any


@dataclasses.dataclass
class Location:
    file: str
    line: int
    column: int

    @staticmethod
    def unknown():
        return Location("unknown", 0, 0)


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
class Module(Node):
    functions: list[Function]
    stencils: list[Stencil]
