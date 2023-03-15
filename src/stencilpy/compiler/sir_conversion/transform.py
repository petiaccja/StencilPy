from stencilpy.compiler import hlast
from stencilir import ops
from .shape_transformer import ShapeFunctionTransformer
from .core_transformer import HlastToSirTransformer
from .internal_functions import adjust_slice_function, slice_size_function, adjust_slice_trivial_function


def hlast_to_sir(hlast_module: hlast.Module) -> ops.ModuleOp:
    shape_module: ops.ModuleOp = ShapeFunctionTransformer().visit(hlast_module)
    code_module: ops.ModuleOp = HlastToSirTransformer().visit(hlast_module)
    merged = ops.ModuleOp()
    merged.add(slice_size_function())
    merged.add(adjust_slice_function())
    merged.add(adjust_slice_trivial_function())
    for op in shape_module.get_body().get_operations():
        merged.add(op)
    for op in code_module.get_body().get_operations():
        merged.add(op)
    return merged
