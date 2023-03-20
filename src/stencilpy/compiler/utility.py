from stencilpy.compiler import types as ts
from typing import Optional
from stencilpy import concepts


def format_type_short(type_: ts.Type):
    if isinstance(type_, ts.IndexType):
        return "I"
    if isinstance(type_, ts.IntegerType):
        return f"{'s' if type_.signed else ''}i{type_.width}"
    if isinstance(type_, ts.FloatType):
        return f"f{type_.width}"
    if isinstance(type_, ts.FieldType):
        element_type = format_type_short(type_.element_type)
        dims = [str(dim.id) for dim in type_.dimensions]
        return f"F{element_type}x{'x'.join(dims)}"
    if isinstance(type_, ts.ConnectivityType):
        element_type = format_type_short(type_.element_type)
        dims = [str(dim.id) for dim in type_.dimensions]
        return f"C{element_type}x{'x'.join(dims)}"
    raise NotImplementedError()


def mangle_name(name: str, param_types: list[ts.Type], dims: Optional[list[concepts.Dimension]] = None):
    terms = name.split(".")
    base_name = "".join([f"{len(term)}{term}" for term in terms])
    param_codes = [format_type_short(type_) for type_ in param_types]
    dim_codes = f"_{'x'.join([str(dim.id) for dim in dims])}" if dims else ""
    return f"__{base_name}__{'_'.join(param_codes)}{dim_codes}"