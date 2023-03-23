import abc
import ctypes
import dataclasses

from stencilpy import concepts
from stencilpy import storage
from typing import Any, Sequence, Iterable
import numpy as np
import numpy.typing as np_typing
from stencilpy import utility


@dataclasses.dataclass
class Type:
    def __str__(self):
        return "<TYPE>"


@dataclasses.dataclass
class NumberType(Type):
    def __str__(self):
        return "<NUMBER>"


@dataclasses.dataclass
class IntegerType(NumberType):
    width: int
    signed: bool

    def __str__(self):
        return f"{'' if self.signed else 'u'}int{self.width}"


@dataclasses.dataclass
class FloatType(NumberType):
    width: int

    def __str__(self):
        return f"float{self.width}"


@dataclasses.dataclass
class IndexType(NumberType):
    def __str__(self):
        return "index"


@dataclasses.dataclass
class NDIndexType(Type):
    dims: list[concepts.Dimension]

    def __str__(self):
        spec = "x".join(str(v) for v in [*self.dims])
        return f"index<{spec}>"


class FieldLikeType(Type):
    @property
    @abc.abstractmethod
    def element_type(self) -> Type:
        ...

    @property
    @abc.abstractmethod
    def dimensions(self) -> Sequence[concepts.Dimension]:
        ...


@dataclasses.dataclass
class FieldType(FieldLikeType):
    _element_type: Type
    _dimensions: list[concepts.Dimension]

    @property
    def element_type(self) -> Type:
        return self._element_type

    @property
    def dimensions(self) -> Sequence[concepts.Dimension]:
        return self._dimensions

    def __str__(self):
        spec = "x".join(str(v) for v in [self.element_type, *self.dimensions])
        return f"field<{spec}>"


@dataclasses.dataclass
class ConnectivityType(FieldLikeType):
    _element_type: Type
    origin_dimension: concepts.Dimension
    neighbor_dimension: concepts.Dimension
    element_dimension: concepts.Dimension

    @property
    def element_type(self) -> Type:
        return self._element_type

    @property
    def dimensions(self) -> Sequence[concepts.Dimension]:
        return sorted([self.origin_dimension, self.element_dimension])

    def __str__(self):
        return f"connectivity<{str(self.element_type)}x{str(self.origin_dimension)}" \
               f"to{str(self.neighbor_dimension)}x{str(self.element_dimension)}>"


@dataclasses.dataclass
class VoidType(Type):
    def __str__(self):
        return "void"


@dataclasses.dataclass
class FunctionType(Type):
    parameters: list[Type]
    result: Type

    def __str__(self):
        params = ', '.join(str(p) for p in self.parameters)
        results = str(self.result)
        return f"({params}) -> {results}"


@dataclasses.dataclass
class StencilType(Type):
    parameters: list[Type]
    result: Type
    dims: list[concepts.Dimension]

    def __str__(self):
        params = ', '.join(str(p) for p in self.parameters)
        results = str(self.result)
        dims = 'x'.join(str(dim) for dim in self.dims)
        return f"({params}) <{dims}> -> {results}"


@dataclasses.dataclass
class TupleType(Type):
    elements: list[Type]

    def __str__(self):
        return f"({', '.join([str(t) for t in self.elements])})"


def infer_object_type(arg: Any) -> Type:
    def translate_dtype(dtype: np.typing.DTypeLike):
        if dtype.kind == 'i':
            return IntegerType(8 * dtype.itemsize, True)
        if dtype.kind == 'u':
            return IntegerType(8 * dtype.itemsize, False)
        if dtype.kind == 'f':
            return FloatType(8 * dtype.itemsize)
        if dtype.kind == 'b':
            return IntegerType(1, True)
        raise ValueError("unknown dtype")

    if isinstance(arg, storage.Field):
        element_type = translate_dtype(arg.data.dtype)
        dims = arg.sorted_dimensions
        return FieldType(element_type, dims)
    elif isinstance(arg, storage.Connectivity):
        element_type = translate_dtype(arg.data.dtype)
        return ConnectivityType(element_type, arg.origin_dimension, arg.neighbor_dimension, arg.element_dimension)
    elif isinstance(arg, tuple):
        elements = [infer_object_type(e) for e in arg]
        return TupleType(elements)
    else:
        dtype = np.dtype(type(arg))
        return translate_dtype(dtype)


def as_numpy_type(type_: Type) -> np_typing.DTypeLike:
    if isinstance(type_, IntegerType):
        if type_.signed:
            if type_.width == 1: return np.bool_
            if type_.width == 8: return np.int8
            if type_.width == 16: return np.int16
            if type_.width == 32: return np.int32
            if type_.width == 64: return np.int64
        else:
            if type_.width == 1: return np.bool_
            if type_.width == 8: return np.uint8
            if type_.width == 16: return np.uint16
            if type_.width == 32: return np.uint32
            if type_.width == 64: return np.uint64
    if isinstance(type_, FloatType):
        if type_.width:
            if type_.width == 16: return np.float16
            if type_.width == 32: return np.float32
            if type_.width == 64: return np.float64
    if isinstance(type_, IndexType):
        if ctypes.sizeof(ctypes.c_void_p) == 4: return np.int32
        if ctypes.sizeof(ctypes.c_void_p) == 8: return np.int64
    raise ValueError(f"cannot convert type {type_} to numpy dtype-like")


def flatten_type(type_: Type) -> list[Type]:
    if isinstance(type_, TupleType):
        return utility.flatten(flatten_type(elem_type) for elem_type in type_.elements)
    return [type_]


def _unflatten_helper(values: Sequence, type_: Type):
    if not isinstance(type_, TupleType):
        return values[0], values[1:]

    elements = []
    for elem_type in type_.elements:
        element, values = _unflatten_helper(values, elem_type)
        elements.append(element)
    return tuple(elements), values

def unflatten(values: Sequence, type_: Type):
    return _unflatten_helper(values, type_)[0]


void_t = VoidType()
index_t = IndexType()
bool_t = IntegerType(1, True)
int8_t = IntegerType(8, True)
int16_t = IntegerType(16, True)
int32_t = IntegerType(32, True)
int64_t = IntegerType(64, True)
uint8_t = IntegerType(8, False)
uint16_t = IntegerType(16, False)
uint32_t = IntegerType(32, False)
uint64_t = IntegerType(64, False)
float32_t = FloatType(32)
float64_t = FloatType(64)