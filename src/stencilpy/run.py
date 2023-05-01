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
    if all(not isinstance(ty, ts.FieldLikeType) for ty in result_types):
        return []
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


def _cast_results(results: Any, result_types: Sequence) -> Any:
    def do_cast(value, type_):
        if not isinstance(value, storage.FieldLike):
            return type_traits.to_numpy_type(type_)(value)
        return value
    if isinstance(result_types[0], ts.VoidType):
        return None
    if not isinstance(results, Sequence):
        return do_cast(results, *result_types)
    return [do_cast(result, type_) for result, type_ in zip(results, result_types)]



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
    runtime_cache: dict[tuple[str, sir.CompileOptions], tuple[ts.FunctionType, sir.CompiledModule]]\
        = dataclasses.field(init=False, repr=False, default_factory=lambda: {})

    def __call__(self, *args, **kwargs):
        use_jit = False
        if "jit" in kwargs:
            use_jit = kwargs["jit"]
            del kwargs["jit"]
        return self.definition(*args, **kwargs) if not use_jit else self.call_jit(*args, **kwargs)

    def call_jit(self, *args, **kwargs):
        optimizations = sir.OptimizationOptions(
            inline_functions=True,
            fuse_extract_slice_ops=True,
            fuse_apply_ops=True,
            eliminate_alloc_buffers=True,
            enable_runtime_verification=True
        )
        compile_options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O3, optimizations)

        arg_types = [type_traits.from_object(arg) for arg in args]
        function_type, compiled_module = self.get_compiled_module(arg_types, compile_options)
        mangled_name = cutil.mangle_name(cutil.get_qualified_name(self.definition), arg_types)

        translated_args = _translate_regular_args(args)
        out_shapes = _get_result_shapes(mangled_name, type_traits.flatten(function_type.result), compiled_module, args)
        out_args = _allocate_results(type_traits.flatten(function_type.result), out_shapes)
        translated_out_args = [_translate_regular_arg(arg) for arg in out_args]
        results = compiled_module.invoke(mangled_name, *translated_args, *translated_out_args)
        matched = _match_results_to_outs(results, out_args)
        casted = _cast_results(matched, type_traits.flatten(function_type.result))
        if not isinstance(casted, Sequence):
            return casted
        return type_traits.unflatten(casted, function_type.result)

    def get_compiled_module(self, arg_types: Sequence[ts.Type], options: sir.CompileOptions):
        mangled_name = cutil.mangle_name(cutil.get_qualified_name(self.definition), arg_types)
        key = (mangled_name, options)
        if key not in self.runtime_cache:
            function_type, compiled_module = JitFunction.compile(self.definition, arg_types, options)
            self.runtime_cache[key] = (function_type, compiled_module)
        return self.runtime_cache[key]

    @staticmethod
    def compile(definition: Callable, arg_types: Sequence[ts.Type], options: sir.CompileOptions):
        mangled_name = cutil.mangle_name(cutil.get_qualified_name(definition), arg_types)
        hast_module = parser.function_to_hlast(definition, arg_types, {})
        function_type = _get_signature(hast_module, mangled_name)
        sir_module = sir_conversion.hlast_to_sir(hast_module)
        compiled_module = sir.CompiledModule(sir_module, options)
        try:
            compiled_module.compile(True)
        except:
            ir = compiled_module.get_stage_results()
            raise
        ir = compiled_module.get_stage_results()
        return function_type, compiled_module


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
