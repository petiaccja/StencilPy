import dataclasses
from collections.abc import Sequence
from typing import Callable, Any

import numpy as np

from stencilpy import concepts


def merge_dims(
        dims1: Sequence[concepts.Dimension],
        dims2: Sequence[concepts.Dimension]
) -> list[concepts.Dimension]:
    return sorted(list(set(dims1) | set(dims2)))


def broadcast_shape(
        dims: Sequence[concepts.Dimension],
        broadcast_dims: Sequence[concepts.Dimension],
        shape: Sequence[int]
) -> tuple[int]:
    return tuple(
        shape[dims.index(dim)] if dim in dims else 1
        for dim in broadcast_dims
    )


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

    def _elementwise_op(self, other: Any, op: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> "Field":
        if isinstance(other, Field):
            bcast_dims = merge_dims(self.sorted_dimensions, other.sorted_dimensions)
            self_new_shape = broadcast_shape(self.sorted_dimensions, bcast_dims, self.data.shape)
            other_new_shape = broadcast_shape(other.sorted_dimensions, bcast_dims, other.data.shape)
            self_reshaped = np.reshape(self.data, self_new_shape)
            other_reshaped = np.reshape(other.data, other_new_shape)
            return Field(bcast_dims, op(self_reshaped, other_reshaped))
        else:
            return Field(self.sorted_dimensions, op(self.data, other))

    def __add__(self, other: "Field") -> "Field":
        return self._elementwise_op(other, lambda x, y: x + y)

    def __sub__(self, other: "Field") -> "Field":
        return self._elementwise_op(other, lambda x, y: x - y)

    def __mul__(self, other: "Field") -> "Field":
        return self._elementwise_op(other, lambda x, y: x * y)

    def __truediv__(self, other: "Field") -> "Field":
        return self._elementwise_op(other, lambda x, y: x / y)

    def __getitem__(self, slices: concepts.Index | concepts.Slice | tuple[concepts.Slice, ...]):
        if isinstance(slices, concepts.Index):
            raw_idx = tuple(slices.values[dim] for dim in self.sorted_dimensions)
            return self.data[raw_idx]
        elif isinstance(slices, concepts.Slice):
            return self.__getitem__((slices,))
        elif isinstance(slices, tuple):
            mapping = {slc.dimension: slc.slice for slc in slices}
            raw_slices = tuple(mapping[dim] for dim in self.sorted_dimensions)
            raw_shape = tuple(
                len(range(*slc.indices(shp))) if isinstance(slc, slice) else 1
                for slc, shp in zip(raw_slices, self.data.shape)
            )
            new_data = np.reshape(self.data[raw_slices], raw_shape)
            return Field(self.sorted_dimensions, new_data)
        raise TypeError(f"cannot subscript field with object of type {type(slices)}")

    def __matmul__(self, mapping: tuple[Connectivity, "Field"]):
        conn = mapping[0]
        source = self
        target = mapping[1]

        raise NotImplementedError()
