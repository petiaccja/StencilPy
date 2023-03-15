import dataclasses
import itertools
from typing import Sequence, Optional, Any, Callable

import numpy as np

from stencilpy import storage
from stencilpy import concepts
from stencilpy import lib
from stencilpy.compiler import hlast
from stencilpy.compiler import parser
from stencilpy.compiler import utility

import stencilir as sir
from stencilpy.compiler import types as ts
from stencilpy.compiler import sir_conversion


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


def _translate_arg(arg: Any) -> Any:
    if isinstance(arg, storage.Field):
        return memoryview(arg.data)
    return arg


def _set_index(idx: concepts.Index):
    lib._index = idx


def _get_signature(hast_module: hlast.Module, func_name: str) -> ts.FunctionType:
    for func in hast_module.functions:
        if func.name == func_name:
            assert isinstance(func.type_, ts.FunctionType)
            return func.type_
    raise KeyError(f"function {func_name} not found in module")


def _allocate_results(
        func_name: str,
        func_signature: ts.FunctionType,
        module: sir.CompiledModule,
        translated_args: list[Any]
):
    shape_func_name = sir_conversion.shape_func_name(func_name)
    shape_args = [arg.shape if isinstance(arg, memoryview) else [arg] for arg in translated_args]
    shape_args = list(itertools.chain(*shape_args))
    shapes = module.invoke(shape_func_name, *shape_args)
    if not isinstance(shapes, tuple):
        shapes = (shapes,)
    fields = []
    start = 0
    for result in func_signature.results:
        if isinstance(result, ts.FieldType):
            end = start + len(result.dimensions)
            shape = shapes[start:end]
            dtype = ts.as_numpy_type(result.element_type)
            fields.append(storage.Field(result.dimensions, np.empty(shape, dtype)))
            start = end
    return tuple(fields)


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
        arg_types = [ts.infer_object_type(arg) for arg in args]
        func_name = utility.mangle_name(self.definition.__name__, arg_types)
        kwarg_types = {name: ts.infer_object_type(value) for name, value in kwargs.items()}
        hast_module = self.parse(arg_types, kwarg_types)
        signature = _get_signature(hast_module, func_name)
        sir_module = sir_conversion.hlast_to_sir(hast_module)
        opt = sir.OptimizationOptions(True, True, True, True)
        options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O3, opt)
        compiled_module = sir.CompiledModule(sir_module, options)
        compiled_module.compile()
        ir = compiled_module.get_stage_results()
        translated_args = [_translate_arg(arg) for arg in args]
        out_args = _allocate_results(func_name, signature, compiled_module, translated_args)
        translated_out_args = [_translate_arg(arg) for arg in out_args]
        results = compiled_module.invoke(func_name, *translated_args, *translated_out_args)
        return _match_results_to_outs(results, out_args)

    def parse(self, arg_types: list[sir.Type], kwarg_types: dict[str, sir.Type]) -> hlast.Module:
        return parser.parse_as_function(self.definition, arg_types, kwarg_types)


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
