from dataclasses import dataclass
from typing import Any
import frontendcf.api as api
import frontendcf.compiler as compiler
import stencilir as sir


def _generate_indices(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) == 1:
        for i in range(shape[0]):
            yield (i,)
    else:
        for i in range(shape[0]):
            for rest in _generate_indices(shape[1:]):
                yield i, *rest



@dataclass
class _Stencil:
    stencil: callable

    def __call__(self, *args, **kwargs):
        @dataclass
        class Helper:
            parent: _Stencil
            inputs: tuple[Any, ...]

            def __call__(self, *args, **kwargs):
                outputs = args
                return self.parent._execute(self.inputs, outputs)
        inputs = args
        return Helper(self, inputs)

    def _execute(self, inputs, outputs):
        shape = outputs[0].shape
        for index in _generate_indices(shape):
            api.set_index(index)
            results = self.stencil(*inputs)
            if not isinstance(results, tuple):
                results = (results,)
            for output, result in zip(outputs, results):
                output[index] = result


@dataclass
class _JitStencil:
    stencil_ast: sir.Module

    def __call__(self, *args, **kwargs):
        @dataclass
        class Helper:
            parent: _Stencil
            inputs: tuple[Any, ...]

            def __call__(self, *args, **kwargs):
                outputs = args
                return self.parent._execute(self.inputs, outputs)
        inputs = args
        return Helper(self, inputs)

    def _execute(self, inputs, outputs):
        print("JIT stencil is WIP")


def stencil(jit: bool = False) -> _Stencil | _JitStencil:
    def stencil_chain(stencil_fun: callable):
        return _Stencil(stencil_fun) if not jit else _JitStencil(compiler.translate_to_stencilir(stencil_fun))
    return stencil_chain
