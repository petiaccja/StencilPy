import numpy as np

from stencilpy.storage import Field, Connectivity
from stencilpy.concepts import Dimension
from stencilpy.run import func, stencil, JitFunction
from stencilpy.stdlib import *
from stencilpy.metalib import *
from stencilpy.compiler import types as ts
from typing import Sequence
import stencilir as sir

XDim = Dimension()
YDim = Dimension()

opt_flags = sir.OptimizationOptions(
    inline_functions=True,
    fuse_extract_slice_ops=True,
    fuse_apply_ops=True,
    eliminate_alloc_buffers=True,
    enable_runtime_verification=False
)
compile_options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O3, opt_flags)

def get_ir(f: JitFunction, arg_types: Sequence[ts.Type]):
    _, mod_copy = f.get_compiled_module(arg_types, compile_options)
    mod_copy.compile(record_stages=True)
    return mod_copy.get_stage_results()[2].ir


#-------------------------------------------------------------------------------
# Dimensions, fields
#-------------------------------------------------------------------------------

ADim = Dimension()
BDim = Dimension()
LDim = Dimension()

#-------------------------------------------------------------------------------
# Shifts/offsets/remaps
#-------------------------------------------------------------------------------

ab_field = Field([ADim, BDim], np.ones((3, 4), dtype=np.float32))

@func
def structured_shift(arg: Field) -> Field:
    return arg[ADim[0:-2], BDim[:]]

result_structured = structured_shift(ab_field, jit=True)

b_field = Field([BDim], np.array([5, 6], dtype=np.float32))
connectivity = Connectivity(ADim, BDim, LDim, np.array([[1, 0], [0, 1]]))

@func
def unstructured_shift(neighbors: Field, connectivity: Connectivity) -> Field:
    return remap(neighbors, connectivity)

result_unstructured = unstructured_shift(b_field, connectivity, jit=True)

#-------------------------------------------------------------------------------
# Module structure, ABI
#-------------------------------------------------------------------------------
@func
def copy(arg):
    return arg

@func
def copycopy(arg):
    return copy(arg)

print(get_ir(copycopy, [ts.float32_t]))
print(get_ir(copy, [ts.FieldType(ts.float32_t, [XDim, YDim])]))


#-------------------------------------------------------------------------------
# Elementwise ops
#-------------------------------------------------------------------------------

@func
def mul(a, b):
    return a * b

print(get_ir(mul, [ts.float32_t, ts.FieldType(ts.float32_t, [XDim])]))





