import dataclasses
from typing import Sequence, Optional, Any

import numpy as np

from stencilpy import storage
from stencilpy import concepts
from stencilpy import lib
from stencilpy.compiler import hast
from stencilpy.compiler import parser

import stencilir as sir
from stencilpy.compiler import types as ts
from stencilpy.compiler import lowering


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
        return arg.data
    return arg


def _set_index(idx: concepts.Index):
    lib._index = idx

@dataclasses.dataclass
class Func:
    definition: callable

    def __call__(self, *args, **kwargs):
        use_jit = False
        if "jit" in kwargs:
            use_jit = kwargs["jit"]
            del kwargs["jit"]
        return self.definition(*args, **kwargs) if not use_jit else self.call_jit(*args, **kwargs)

    def call_jit(self, *args, **kwargs):
        arg_types = [ts.infer_object_type(arg) for arg in args]
        kwarg_types = {name: ts.infer_object_type(value) for name, value in kwargs.items()}
        hast_module = self.parse(arg_types, kwarg_types)
        sir_module = lowering.lower(hast_module)
        options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O3)
        compiled_module: sir.CompiledModule = sir.compile(sir_module, options, True)
        ir = compiled_module.get_ir()
        translated_args = [_translate_arg(arg) for arg in args]
        return compiled_module.invoke(self.definition.__name__, *translated_args)

    def parse(self, arg_types: list[sir.Type], kwarg_types: dict[str, sir.Type]) -> hast.Module:
        return parser.parse_as_function(self.definition, arg_types, kwarg_types)


@dataclasses.dataclass
class Stencil:
    definition: callable

    def __getitem__(self, dimensions: concepts.Dimension | tuple[concepts.Dimension, ...]):
        if not isinstance(dimensions, tuple):
            dimensions = dimensions,
        return Stencil.Slicer(self.definition, dimensions)

    @dataclasses.dataclass
    class Slicer:
        definition: callable
        dimensions: tuple[concepts.Dimension, ...]

        def __getitem__(self, sizes: int | tuple[int, ...]):
            if not isinstance(sizes, tuple):
                sizes = sizes,
            return Stencil.Executor(self.definition, self.dimensions, sizes)

    @dataclasses.dataclass
    class Executor:
        definition: callable
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


def func(definition: callable):
    return Func(definition)


def stencil(definition: callable):
    return Stencil(definition)
