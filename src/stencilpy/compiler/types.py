import dataclasses

from stencilpy import field
from typing import Any
import numpy as np


@dataclasses.dataclass
class Type:
    pass


@dataclasses.dataclass
class IntegerType(Type):
    width: int
    signed: bool


@dataclasses.dataclass
class FloatType(Type):
    width: int


@dataclasses.dataclass
class IndexType(Type):
    pass


@dataclasses.dataclass
class FieldType(Type):
    element_type: Type
    dimensions: list[field.Dimension]


@dataclasses.dataclass
class VoidType(Type):
    pass


@dataclasses.dataclass
class FunctionType(Type):
    pass


def infer_object_type(arg: Any) -> Type:
    if isinstance(arg, field.Field):
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