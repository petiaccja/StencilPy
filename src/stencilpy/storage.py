import dataclasses
from typing import Sequence

import numpy as np

from stencilpy import concepts


@dataclasses.dataclass
class Connectivity:
    source_dimension: concepts.Dimension
    neighbor_dimension: concepts.Dimension
    element_dimension: concepts.Dimension
    indices: np.ndarray


class Field:
    sorted_dimensions: list[concepts.Dimension]
    data: np.ndarray

    def __init__(self, dimensions: Sequence[concepts.Dimension], data: np.ndarray):
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

    def __getitem__(self, dimensions: concepts.Index | tuple[concepts.Dimension, ...]):
        if isinstance(dimensions, concepts.Index):
            raw_idx = tuple(dimensions.values[dim] for dim in self.sorted_dimensions)
            return self.data[raw_idx]

        @dataclasses.dataclass
        class Slicer:
            field: Field
            dimensions: tuple[concepts.Dimension]
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
