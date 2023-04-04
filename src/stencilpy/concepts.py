import copy
import dataclasses
import typing
from typing import Optional, Callable, Any
from stencilpy import utility
import abc


@dataclasses.dataclass
class Location:
    file: str
    line: int
    column: int

    @staticmethod
    def unknown():
        return Location("unknown", -1, -1)

    def is_unknown(self):
        return self == Location.unknown()

    def __str__(self):
        return f"{self.file}:{self.line}:{self.column}"


@dataclasses.dataclass(unsafe_hash=True)
class Dimension:
    id: int
    name: Optional[str] = dataclasses.field(default=None, hash=False)

    def __init__(self, name: Optional[str] = None):
        self.id = utility.unique_id()
        self.name = name

    def __lt__(self, other):
        return self.id < other.id

    def __str__(self):
        return f"'{self.id}"

    def __getitem__(self, item: int | slice) -> "Slice":
        return Slice(self, item)


@dataclasses.dataclass
class Slice:
    dimension: Dimension
    slice: int | slice


@dataclasses.dataclass
class Index:
    values: dict[Dimension, int]

    def __getitem__(self, item: Dimension | tuple[Slice, ...]):
        if isinstance(item, Dimension):
            return self.values[item]
        elif isinstance(item, Slice):
            return self[item,]
        elif isinstance(item, tuple):
            new_values = copy.deepcopy(self.values)
            for slc in item:
                new_values[slc.dimension] = new_values[slc.dimension] + slc.slice
            return Index(new_values)


@dataclasses.dataclass
class Builtin:
    definition: Callable

    @property
    def name(self):
        return self.definition.__name__

    def __call__(self, *args, **kwargs):
        return self.definition(*args, **kwargs)


@typing.runtime_checkable
class Function(typing.Protocol):
    @property
    @abc.abstractmethod
    def definition(self) -> Callable:
        ...

    def __call__(self, *args, **kwargs):
        ...


@typing.runtime_checkable
class Stencil(typing.Protocol):
    @property
    @abc.abstractmethod
    def definition(self) -> Callable:
        ...

    def __getitem__(self, dimensions: Dimension | tuple[Dimension, ...]):
        ...


@dataclasses.dataclass
class MetaFunc:
    definition: Callable[[bool, Any, ...], Any]

    def __call__(self, *args, is_jit=False, **kwargs):
        return self.definition(is_jit, *args, **kwargs)


def builtin(definition: Callable):
    return Builtin(definition)


def metafunc(definition: Callable[[bool, Any, ...], Any]):
    return MetaFunc(definition)