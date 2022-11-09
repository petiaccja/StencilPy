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


def _get_type(obj: Any) -> type:
    try:
        view = memoryview(obj)
        if view.format == "f":
            element_type = sir.ScalarType.FLOAT32
        elif view.format == "d":
            element_type = sir.ScalarType.FLOAT64
        elif view.format == "?":
            element_type = sir.ScalarType.SINT8
        elif view.format == "i":
            element_type = sir.ScalarType.SINT32
        elif view.format == "l":
            element_type = sir.ScalarType.SINT64
        else:
            raise NotImplementedError()
        return sir.FieldType(element_type, view.ndim)
    except:
        ...

    if isinstance(obj, bool):
        return sir.ScalarType.BOOL
    elif isinstance(obj, int):
        return sir.ScalarType.SINT64
    elif isinstance(obj, float):
        return sir.ScalarType.FLOAT64
    else:
        raise NotImplementedError()


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
        input_types = [_get_type(inp) for inp in inputs]
        output_types = [_get_type(outp) for outp in outputs]
        stencil_ast = compiler.parse_function(self.stencil, input_types, output_types)
        options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O3)
        compiled_module: sir.CompiledModule = sir.compile(stencil_ast, options, True)
        ir = compiled_module.get_ir()
        compiled_module.invoke("main", *inputs, *outputs)


def stencil(jit: bool = False) -> _Stencil | _JitStencil:
    def stencil_chain(stencil_fun: callable):
        return _Stencil(stencil_fun) if not jit else _JitStencil(stencil_fun)
    return stencil_chain
