import dataclasses
from typing import Optional
from stencilpy import utility


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
        self.id = next(utility.unique_id)
        self.name = name

    def __lt__(self, other):
        return self.id < other.id


@dataclasses.dataclass
class Index:
    values: dict[Dimension, int]


@dataclasses.dataclass
class Builtin:
    definition: callable

    @property
    def name(self):
        return self.definition.__name__

    def __call__(self, *args, **kwargs):
        return self.definition(*args, **kwargs)


def builtin(definition: callable):
    return Builtin(definition)