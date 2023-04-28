import dataclasses
from collections.abc import Sequence
from typing import Callable, Any

import numpy as np

from stencilpy import concepts
from stencilpy import utility


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


def elementwise_op(op: Callable, *args) -> "Field":
    all_dims = [arg.sorted_dimensions for arg in args if isinstance(arg, Field)]
    common_dims = list(sorted(set(utility.flatten(all_dims))))
    reshaped = [
        (
            np.reshape(arg.data, broadcast_shape(arg.sorted_dimensions, common_dims, arg.data.shape))
            if isinstance(arg, Field)
            else arg
        )
        for arg in args
    ]
    if common_dims:
        return Field(common_dims, op(*reshaped))
    return op(*reshaped)


@dataclasses.dataclass
class FieldLike:
    sorted_dimensions: list[concepts.Dimension]
    data: np.ndarray

    def __init__(self, dimensions: Sequence[concepts.Dimension], data: np.ndarray):
        assert len(dimensions) == data.ndim
        indexed_dimensions = [(dim, index) for index, dim in enumerate(dimensions)]
        sorted_dimensions = sorted(indexed_dimensions, key=lambda d: d[0].id)
        self.sorted_dimensions = [dim for dim, _ in sorted_dimensions]
        self.data = np.moveaxis(data, range(data.ndim), [index for _, index in sorted_dimensions])

    @property
    def shape(self):
        return {dim: size for dim, size in zip(self.sorted_dimensions, self.data.shape)}

    def _sample(self, slices: concepts.Index):
        raw_idx = tuple(slices.values[dim] for dim in self.sorted_dimensions)
        return self.data[raw_idx]

    def __getitem__(self, index: concepts.Index):
        return self._sample(index)


class Field(FieldLike):
    def __init__(self, dimensions: Sequence[concepts.Dimension], data: np.ndarray):
        super().__init__(dimensions, data)

    def _extract_slice(self, slices: tuple[concepts.Slice, ...]):
        mapping = {slc.dimension: slc.slice for slc in slices}
        raw_slices = tuple(mapping[dim] for dim in self.sorted_dimensions)
        raw_shape = tuple(
            len(range(*slc.indices(shp))) if isinstance(slc, slice) else 1
            for slc, shp in zip(raw_slices, self.data.shape)
        )
        new_data = np.reshape(self.data[raw_slices], raw_shape)
        return Field(self.sorted_dimensions, new_data)

    def __getitem__(self, slices: concepts.Index | concepts.Slice | tuple[concepts.Slice, ...]):
        if isinstance(slices, concepts.Index):
            return self._sample(slices)
        elif isinstance(slices, concepts.Slice):
            return self._extract_slice((slices,))
        elif isinstance(slices, tuple):
            return self._extract_slice(slices)
        raise TypeError(f"cannot subscript field with object of type {type(slices)}")

    def __add__(self, other: "Field") -> "Field":
        return elementwise_op(lambda x, y: x + y, self, other)

    def __sub__(self, other: "Field") -> "Field":
        return elementwise_op(lambda x, y: x - y, self, other)

    def __mul__(self, other: "Field") -> "Field":
        return elementwise_op(lambda x, y: x * y, self, other)

    def __truediv__(self, other: "Field") -> "Field":
        return elementwise_op(lambda x, y: x / y, self, other)

    def __radd__(self, other: "Field") -> "Field":
        return elementwise_op(lambda x, y: x + y, other, self)

    def __rsub__(self, other: "Field") -> "Field":
        return elementwise_op(lambda x, y: x - y, other, self)

    def __rmul__(self, other: "Field") -> "Field":
        return elementwise_op(lambda x, y: x * y, other, self)

    def __rtruediv__(self, other: "Field") -> "Field":
        return elementwise_op(lambda x, y: x / y, other, self)


class Connectivity(FieldLike):
    origin_dimension: concepts.Dimension
    neighbor_dimension: concepts.Dimension
    element_dimension: concepts.Dimension

    def __init__(
            self,
             origin_dimension: concepts.Dimension,
             neighbor_dimension: concepts.Dimension,
             element_dimension: concepts.Dimension,
             indices: np.ndarray
    ):
        self.origin_dimension = origin_dimension
        self.neighbor_dimension = neighbor_dimension
        self.element_dimension = element_dimension
        super().__init__([origin_dimension, element_dimension], indices)

