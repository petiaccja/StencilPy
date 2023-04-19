from stencilpy.run import func
import pathlib, tempfile
from stencilpy.compiler import types as ts
import stencilir as sir


@func
def mad(a, b, c):
    return a * b + c


@func
def foo(a, b, c):
    return mad(a, b, c)


def test_object_file():
    arg_types = [ts.float32_t, ts.float32_t, ts.int16_t]
    optimizations = sir.OptimizationOptions(True, False, False, False)
    compile_options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O0, optimizations)
    _, compiled_module = foo.get_compiled_module(arg_types, compile_options)
    object_file_buffer = compiled_module.get_object_file()
    object_file_path = pathlib.Path(tempfile.gettempdir()) / "test_py_object_file.obj"
    object_file_path.write_bytes(object_file_buffer)
    assert object_file_path.exists()