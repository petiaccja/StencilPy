import dataclasses
from typing import Optional, Sequence

import numpy as np

from stencilpy import utility


@dataclasses.dataclass(unsafe_hash=True)
class Dimension:
    id: int
    name: Optional[str] = dataclasses.field(default=None, hash=False)

    def __init__(self, name: Optional[str] = None):
        self.id = next(utility.unique_id)
        self.name = name


@dataclasses.dataclass
class Index:
    values: dict[Dimension, int]


_index: Optional[Index] = None


def set_index(idx: Index):
    global _index
    _index = idx


def index():
    global _index
    return _index


@dataclasses.dataclass
class Connectivity:
    source_dimension: Dimension
    neighbor_dimension: Dimension
    element_dimension: Dimension
    indices: np.ndarray


class Field:
    sorted_dimensions: list[Dimension]
    data: np.ndarray

    def __init__(self, dimensions: Sequence[Dimension], data: np.ndarray):
        assert len(dimensions) == data.ndim
        indexed_dimensions = [(dim, index) for index, dim in enumerate(dimensions)]
        sorted_dimensions = sorted(indexed_dimensions, key=lambda d: d[0].id)
        self.sorted_dimensions = [dim for dim, _ in sorted_dimensions]
        self.data = np.moveaxis(data, range(data.ndim), [index for _, index in sorted_dimensions])

    def __add__(self, other: "Field") -> "Field":
        assert self.sorted_dimensions == other.sorted_dimensions
        new_data = self.data + other.data
        return Field(self.sorted_dimensions, new_data)

    def __sub__(self, other: "Field") -> "Field":
        assert self.sorted_dimensions == other.sorted_dimensions
        new_data = self.data - other.data
        return Field(self.sorted_dimensions, new_data)

    def __mul__(self, other: "Field") -> "Field":
        assert self.sorted_dimensions == other.sorted_dimensions
        new_data = self.data * other.data
        return Field(self.sorted_dimensions, new_data)

    def __truediv__(self, other: "Field") -> "Field":
        assert self.sorted_dimensions == other.sorted_dimensions
        new_data = self.data / other.data
        return Field(self.sorted_dimensions, new_data)

    def __getitem__(self, dimensions: Index | tuple[Dimension, ...]):
        if isinstance(dimensions, Index):
            raw_idx = tuple(dimensions.values[dim] for dim in self.sorted_dimensions)
            return self.data[raw_idx]

        @dataclasses.dataclass
        class Slicer:
            field: Field
            dimensions: tuple[Dimension]
            def __getitem__(self, slices: tuple[int, ...] | tuple[slice, ...]):
                mapping = {dim: slc for dim, slc in zip(dimensions, slices)}
                slices = tuple(mapping[dim] for dim in self.field.sorted_dimensions)
                new_data = self.field.data[slices]
                return Field(self.field.sorted_dimensions, new_data)
        return Slicer(self, dimensions)

    def __matmul__(self, mapping: tuple[Connectivity, "Field"]):
        conn = mapping[0]
        source = self
        target = mapping[1]

        raise NotImplementedError()
