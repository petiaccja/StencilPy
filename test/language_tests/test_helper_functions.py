import pytest

from stencilpy.compiler.sir_conversion import internal_functions
import stencilir as sir
from stencilir import ops


@pytest.fixture
def slice_size_module():
    module = ops.ModuleOp()
    module.add(internal_functions.slice_size_function(True))

    opt_options = sir.OptimizationOptions(True, True, True, True)
    compile_options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O3, opt_options)
    compiled_module = sir.CompiledModule(module, compile_options)
    compiled_module.compile()
    return compiled_module


@pytest.fixture
def adjust_slice_module():
    module = ops.ModuleOp()
    module.add(internal_functions.adjust_slice_function(True))

    opt_options = sir.OptimizationOptions(True, True, True, True)
    compile_options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O3, opt_options)
    compiled_module = sir.CompiledModule(module, compile_options)
    compiled_module.compile()
    return compiled_module\


@pytest.fixture
def adjust_slice_trivial_module():
    module = ops.ModuleOp()
    module.add(internal_functions.adjust_slice_trivial_function(True))

    opt_options = sir.OptimizationOptions(True, True, True, True)
    compile_options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O3, opt_options)
    compiled_module = sir.CompiledModule(module, compile_options)
    compiled_module.compile()
    return compiled_module


#-------------------------------------------------------------------------------
# slice size
#-------------------------------------------------------------------------------

def test_slice_size_unit(slice_size_module):
    assert slice_size_module.invoke("__slice_size", 3, 7, 1) == 4
    assert slice_size_module.invoke("__slice_size", -3, -7, -1) == 4


def test_slice_size_sparse(slice_size_module):
    assert slice_size_module.invoke("__slice_size", 3, 3, 3) == 0
    assert slice_size_module.invoke("__slice_size", 3, 4, 3) == 1
    assert slice_size_module.invoke("__slice_size", 3, 5, 3) == 1
    assert slice_size_module.invoke("__slice_size", 3, 6, 3) == 1
    assert slice_size_module.invoke("__slice_size", 3, 7, 3) == 2
    assert slice_size_module.invoke("__slice_size", 3, 8, 3) == 2


def test_slice_size_negative(slice_size_module):
    assert slice_size_module.invoke("__slice_size", -3, -8, -3) == 2


def test_slice_size_underflow(slice_size_module):
    assert slice_size_module.invoke("__slice_size", 3, 8, -3) == 0


#-------------------------------------------------------------------------------
# adjust slice
#-------------------------------------------------------------------------------

def test_adjust_slice_regular(adjust_slice_module):
    assert adjust_slice_module.invoke("__adjust_slice", 3, 7, 1, 9) == (3, 7)


def test_adjust_slice_empty(adjust_slice_module):
    assert adjust_slice_module.invoke("__adjust_slice", 7, 3, 1, 9) == (7, 3)


def test_adjust_slice_stop_overflow(adjust_slice_module):
    assert adjust_slice_module.invoke("__adjust_slice", 3, 16, 1, 9) == (3, 9)


def test_adjust_slice_stop_negative(adjust_slice_module):
    assert adjust_slice_module.invoke("__adjust_slice", 3, -2, 1, 9) == (3, 7)


def test_adjust_slice_stop_underflow(adjust_slice_module):
    assert adjust_slice_module.invoke("__adjust_slice", 3, -11, 1, 9) == (3, 0)


def test_adjust_slice_start_overflow(adjust_slice_module):
    assert adjust_slice_module.invoke("__adjust_slice", 16, 7, 1, 9) == (9, 7)


def test_adjust_slice_start_negative(adjust_slice_module):
    assert adjust_slice_module.invoke("__adjust_slice", -6, 7, 1, 9) == (3, 7)


def test_adjust_slice_start_underflow(adjust_slice_module):
    assert adjust_slice_module.invoke("__adjust_slice", -11, 7, 1, 9) == (0, 7)


#-------------------------------------------------------------------------------
# adjust slice trivial
#-------------------------------------------------------------------------------

def test_adjust_slice_trivial(adjust_slice_trivial_module):
    assert adjust_slice_trivial_module.invoke("__adjust_slice_trivial", 0, 7, 1, 9) == (0, 7)
    assert adjust_slice_trivial_module.invoke("__adjust_slice_trivial", 3, 7, 2, 9) == (3, 7)
    assert adjust_slice_trivial_module.invoke("__adjust_slice_trivial", 7, 3, 2, 9) == (7, 3)


def test_adjust_slice_trivial_overflow(adjust_slice_trivial_module):
    assert adjust_slice_trivial_module.invoke("__adjust_slice_trivial", 16, 7, 1, 9) == (16, 7)
    assert adjust_slice_trivial_module.invoke("__adjust_slice_trivial", 3, 16, 1, 9) == (3, 9)
