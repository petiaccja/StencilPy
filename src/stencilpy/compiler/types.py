import abc
import dataclasses

from stencilpy import concepts
from typing import Sequence


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