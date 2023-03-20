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

    @staticmethod
    def _elementwise_op(lhs: Any, rhs: Any, op: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> "Field":
        if isinstance(lhs, Field) and isinstance(rhs, Field):
            bcast_dims = merge_dims(lhs.sorted_dimensions, rhs.sorted_dimensions)
            self_new_shape = broadcast_shape(lhs.sorted_dimensions, bcast_dims, lhs.data.shape)
            other_new_shape = broadcast_shape(rhs.sorted_dimensions, bcast_dims, rhs.data.shape)
            self_reshaped = np.reshape(lhs.data, self_new_shape)
            other_reshaped = np.reshape(rhs.data, other_new_shape)
            return Field(bcast_dims, op(self_reshaped, other_reshaped))
        elif isinstance(lhs, Field):
            return Field(lhs.sorted_dimensions, op(lhs.data, rhs))
        elif isinstance(rhs, Field):
            return Field(rhs.sorted_dimensions, op(lhs, rhs.data))
        else:
            raise TypeError("expected field type for either lhs or rhs")

    def __add__(self, other: "Field") -> "Field":
        return self._elementwise_op(self, other, lambda x, y: x + y)

    def __sub__(self, other: "Field") -> "Field":
        return self._elementwise_op(self, other, lambda x, y: x - y)

    def __mul__(self, other: "Field") -> "Field":
        return self._elementwise_op(self, other, lambda x, y: x * y)

    def __truediv__(self, other: "Field") -> "Field":
        return self._elementwise_op(self, other, lambda x, y: x / y)

    def __radd__(self, other: "Field") -> "Field":
        return self._elementwise_op(other, self, lambda x, y: x + y)

    def __rsub__(self, other: "Field") -> "Field":
        return self._elementwise_op(other, self, lambda x, y: x - y)

    def __rmul__(self, other: "Field") -> "Field":
        return self._elementwise_op(other, self, lambda x, y: x * y)

    def __rtruediv__(self, other: "Field") -> "Field":
        return self._elementwise_op(other, self, lambda x, y: x / y)


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

