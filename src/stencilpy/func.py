import dataclasses
import itertools
from typing import Sequence, Optional, Any, Callable

import numpy as np

from stencilpy import storage
from stencilpy import concepts
from stencilpy.compiler import hlast
from stencilpy.compiler import parser
from stencilpy.compiler import utility as cutil
from stencilpy import utility as gutil

import stencilir as sir
from stencilpy.compiler import types as ts, type_traits
from stencilpy.compiler import sir_conversion


#-------------------------------------------------------------------------------
# Stencils and indices
#-------------------------------------------------------------------------------
_index: Optional[concepts.Index] = None


def _set_index(idx: concepts.Index):
    global _index
    _index = idx


def _generate_indices(slices: Sequence[slice]) -> tuple[int, ...]:
    if len(slices) == 1:
        for i in range(slices[0].start, slices[0].stop, slices[0].step):
            yield i,
    else:
        for i in range(slices[0].start, slices[0].stop, slices[0].step):
            for rest in _generate_indices(slices[1:]):
                yield i, *rest


def _set_field_item(field_: storage.Field, idx: concepts.Index, value: Any):
    raw_idx = tuple(idx.values[dim] for dim in field_.sorted_dimensions)
    field_.data[raw_idx] = value


#-------------------------------------------------------------------------------
# Argument and return value ABI translation
#-------------------------------------------------------------------------------
def _translate_regular_arg(arg: Any) -> Any:
    if isinstance(arg, storage.FieldLike):
        return memoryview(arg.data)
    return arg


def _translate_regular_args(args: Any) -> list[Any]:
    flat_args = gutil.flatten_recursive(args)
    return [_translate_regular_arg(arg) for arg in flat_args]


def _translate_shape_arg(arg: Any) -> Any:
    if isinstance(arg, storage.FieldLike):
        return memoryview(arg.data).shape
    return arg


def _translate_shape_args(args: Any) -> Any:
    flat_args = gutil.flatten_recursive(args)
    return gutil.flatten_recursive(_translate_shape_arg(arg) for arg in flat_args)


def _get_result_shapes(
        mangled_name: str,
        result_types: list[ts.Type],
        module: sir.CompiledModule,
        args: Sequence[Any]
) -> list[tuple]:
    translated_args = _translate_shape_args(args)
    flat_shapes = module.invoke(sir_conversion.shape_func_name(mangled_name), *translated_args)
    if flat_shapes and not isinstance(flat_shapes, Sequence):
        flat_shapes = flat_shapes,
    ndims = [len(t.dimensions) for t in result_types if isinstance(t, ts.FieldLikeType)]
    offsets = [v - ndims[0] for v in itertools.accumulate(ndims)]
    shapes = [tuple(flat_shapes[off:off+nd]) for off, nd in zip(offsets, ndims)]
    return shapes


def _allocate_results(result_types: list[ts.Type], result_shapes: list[tuple]) -> tuple[storage.Field, ...]:
    field_results = [type_ for type_ in result_types if isinstance(type_, ts.FieldLikeType)]
    assert len(field_results) == len(result_shapes)
    outs: list[storage.Field | storage.Connectivity] = []
    for type_, shape in zip(field_results, result_shapes):
        element_type = type_.element_type
        dtype = type_traits.to_numpy_type(element_type)
        buffer = np.empty(shape, dtype)
        if isinstance(type_, ts.FieldType):
            out = storage.Field(type_.dimensions, buffer)
        elif isinstance(type_, ts.ConnectivityType):
            out = storage.Connectivity(type_.origin_dimension, type_.neighbor_dimension, type_.element_dimension, buffer)
        else:
            assert False
        outs.append(out)
    return tuple(outs)


def _match_results_to_outs(results: Any, out_args: tuple) -> Any:
    if not results:
        return None
    if not isinstance(results, tuple):
        results = (results,)
    matched = []
    out_idx = 0
    for result in results:
        if isinstance(result, memoryview):
            matched.append(out_args[out_idx])
            out_idx += 1
        else:
            matched.append(result)
    return matched[0] if len(matched) == 1 else tuple(matched)


#-------------------------------------------------------------------------------
# Parsing
#-------------------------------------------------------------------------------
def _get_signature(hast_module: hlast.Module, func_name: str) -> ts.FunctionType:
    for func in hast_module.functions:
        if func.name == func_name:
            assert isinstance(func.type_, ts.FunctionType)
            return func.type_
    raise KeyError(f"function {func_name} not found in module")


@dataclasses.dataclass
class JitFunction:
    definition: Callable

    def __call__(self, *args, **kwargs):
        use_jit = False
        if "jit" in kwargs:
            use_jit = kwargs["jit"]
            del kwargs["jit"]
        return self.definition(*args, **kwargs) if not use_jit else self.call_jit(*args, **kwargs)

    def call_jit(self, *args, **kwargs):
        arg_types = [type_traits.from_object(arg) for arg in args]
        func_name = cutil.mangle_name(f"{self.definition.__module__}.{self.definition.__name__}", arg_types)
        kwarg_types = {name: type_traits.from_object(value) for name, value in kwargs.items()}
        hast_module = self.parse(arg_types, kwarg_types)
        signature = _get_signature(hast_module, func_name)
        sir_module = sir_conversion.hlast_to_sir(hast_module)
        opt = sir.OptimizationOptions(True, True, True, True)
        options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O3, opt)
        compiled_module = sir.CompiledModule(sir_module, options)
        try:
            compiled_module.compile(True)
        except:
            ir = compiled_module.get_stage_results()
            raise
        ir = compiled_module.get_stage_results()
        translated_args = _translate_regular_args(args)
        out_shapes = _get_result_shapes(func_name, type_traits.flatten(signature.result), compiled_module, args)
        out_args = _allocate_results(type_traits.flatten(signature.result), out_shapes)
        translated_out_args = [_translate_regular_arg(arg) for arg in out_args]
        results = compiled_module.invoke(func_name, *translated_args, *translated_out_args)
        matched = _match_results_to_outs(results, out_args)
        if not isinstance(matched, Sequence):
            return matched
        return type_traits.unflatten(matched, signature.result)

    def parse(self, arg_types: list[sir.Type], kwarg_types: dict[str, sir.Type]) -> hlast.Module:
        return parser.function_to_hlast(self.definition, arg_types, kwarg_types)


@dataclasses.dataclass
class JitStencil:
    definition: Callable

    def __getitem__(self, slices: concepts.Slice | tuple[concepts.Slice]):
        if not isinstance(slices, tuple):
            slices = slices,
        dimensions = tuple(slc.dimension for slc in slices)
        sizes = tuple(slc.slice for slc in slices)
        return JitStencil.Executor(self.definition, dimensions, sizes)


    @dataclasses.dataclass
    class Executor:
        definition: Callable
        dimensions: tuple[concepts.Dimension, ...]
        sizes: tuple[int, ...]

        def __call__(self, *args, **kwargs):
            use_jit = False
            if "jit" in kwargs:
                use_jit = kwargs["jit"]
                del kwargs["jit"]
            if use_jit:
                raise NotImplementedError()

            outs: Optional[tuple[storage.Field]] = None
            index_gen = _generate_indices(tuple(slice(0, ub, 1) for ub in self.sizes))
            for raw_idx in index_gen:
                idx = concepts.Index({dim: value for dim, value in zip(self.dimensions, raw_idx)})
                _set_index(idx)
                out_items = self.definition(*args, **kwargs)
                if not isinstance(out_items, tuple):
                    out_items = out_items,
                if not outs:
                    outs = tuple(
                        storage.Field(self.dimensions, np.zeros(shape=self.sizes, dtype=type(item)))
                        for item in out_items
                    )
                for o, i in zip(outs, out_items):
                    _set_field_item(o, idx, i)
            return outs if len(outs) > 1 else outs[0]


def func(definition: Callable):
    return JitFunction(definition)


def stencil(definition: Callable):
    return JitStencil(definition)
