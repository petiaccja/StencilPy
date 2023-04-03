import ctypes
from typing import Any, Sequence, Optional

import numpy as np
from numpy import typing as np_typing

from stencilpy import storage, utility
from stencilpy.compiler import types as ts


def from_object(arg: Any) -> ts.Type:
    def translate_dtype(dtype: np.typing.DTypeLike):
        if dtype.kind == 'i':
            return ts.IntegerType(8 * dtype.itemsize, True)
        if dtype.kind == 'u':
            return ts.IntegerType(8 * dtype.itemsize, False)
        if dtype.kind == 'f':
            return ts.FloatType(8 * dtype.itemsize)
        if dtype.kind == 'b':
            return ts.IntegerType(1, True)
        raise ValueError("unknown dtype")

    if isinstance(arg, storage.Field):
        element_type = translate_dtype(arg.data.dtype)
        dims = arg.sorted_dimensions
        return ts.FieldType(element_type, dims)
    elif isinstance(arg, storage.Connectivity):
        element_type = translate_dtype(arg.data.dtype)
        return ts.ConnectivityType(element_type, arg.origin_dimension, arg.neighbor_dimension, arg.element_dimension)
    elif isinstance(arg, tuple):
        elements = [from_object(e) for e in arg]
        return ts.TupleType(elements)
    else:
        dtype = np.dtype(type(arg))
        return translate_dtype(dtype)


def to_numpy_type(type_: ts.Type) -> np_typing.DTypeLike:
    if isinstance(type_, ts.IntegerType):
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
    if isinstance(type_, ts.FloatType):
        if type_.width:
            if type_.width == 16: return np.float16
            if type_.width == 32: return np.float32
            if type_.width == 64: return np.float64
    if isinstance(type_, ts.IndexType):
        if ctypes.sizeof(ctypes.c_void_p) == 4: return np.int32
        if ctypes.sizeof(ctypes.c_void_p) == 8: return np.int64
    raise ValueError(f"cannot convert type {type_} to numpy dtype-like")


def flatten(type_: ts.Type) -> list[ts.Type]:
    if isinstance(type_, ts.TupleType):
        return utility.flatten(flatten(elem_type) for elem_type in type_.elements)
    return [type_]


def _unflatten_helper(values: Sequence, type_: ts.Type):
    if not isinstance(type_, ts.TupleType):
        return values[0], values[1:]

    elements = []
    for elem_type in type_.elements:
        element, values = _unflatten_helper(values, elem_type)
        elements.append(element)
    return tuple(elements), values


def unflatten(values: Sequence, type_: ts.Type):
    return _unflatten_helper(values, type_)[0]


def pointer_width():
    return ctypes.sizeof(ctypes.c_void_p) * 8


def is_convertible(source: ts.Type, target: ts.Type):
    if isinstance(target, ts.IntegerType):
        if isinstance(source, ts.IntegerType):
            return source.width <= target.width
        if isinstance(source, ts.IndexType):
            return pointer_width() <= target.width
    if isinstance(target, ts.IndexType):
        if isinstance(source, ts.IntegerType):
            return source.width <= pointer_width()
        if isinstance(source, ts.IndexType):
            return True
    if isinstance(target, ts.FloatType):
        if isinstance(source, ts.IntegerType):
            return source.width < target.width
        if isinstance(source, ts.IndexType):
            return pointer_width() < target.width
        if isinstance(source, ts.FloatType):
            return source.width <= target.width
    return None


def common_type(*types: ts.Type) -> Optional[ts.Type]:
    for target in types:
        if all(is_convertible(source, target) for source in types):
            return target
    return None