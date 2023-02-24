import dataclasses

from stencilpy import concepts
from stencilpy import storage
from typing import Any
import numpy as np


@dataclasses.dataclass
class Type:
    def __str__(self):
        return "<TYPE>"


@dataclasses.dataclass
class IntegerType(Type):
    width: int
    signed: bool

    def __str__(self):
        return f"{'' if self.signed else 'u'}int{self.width}"


@dataclasses.dataclass
class FloatType(Type):
    width: int

    def __str__(self):
        return f"float{self.width}"


@dataclasses.dataclass
class IndexType(Type):
    def __str__(self):
        return "index"


@dataclasses.dataclass
class FieldType(Type):
    element_type: Type
    dimensions: list[concepts.Dimension]

    def __str__(self):
        spec = "x".join(str(v) for v in [self.element_type, *self.dimensions])
        return f"field<{spec}>"


@dataclasses.dataclass
class VoidType(Type):
    def __str__(self):
        return "void"


@dataclasses.dataclass
class FunctionType(Type):
    parameters: list[Type]
    results: list[Type]

    def __str__(self):
        params = ', '.join(str(p) for p in self.parameters)
        results = ', '.join(str(r) for r in self.results)
        return f"({params}) -> {results}"


def infer_object_type(arg: Any) -> Type:
    if isinstance(arg, storage.Field):
        dtype = infer_object_type(arg.data[tuple([0] * arg.data.ndim)])
        dims = arg.sorted_dimensions
        return FieldType(dtype, dims)
    else:
        dtype = np.dtype(type(arg))
        if dtype.kind == 'i':
            return IntegerType(8 * dtype.itemsize, True)
        if dtype.kind == 'u':
            return IntegerType(8 * dtype.itemsize, False)
        if dtype.kind == 'f':
            return FloatType(8 * dtype.itemsize)
    raise ValueError(f"cannot infer type for object `{arg}`")