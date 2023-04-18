from stencilpy.compiler import types as ts, type_traits
from typing import Optional, Sequence
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


def get_qualified_name(obj: object):
    module = getattr(obj, "__module__", None)
    name = getattr(obj, "__qualname__", None)
    assert name is not None
    qual_name = f"{module}.{name}" if module else name
    qual_name = qual_name.replace("<", "_")
    qual_name = qual_name.replace(">", "_")
    return qual_name


def mangle_name(name: str, param_types: Sequence[ts.Type], dims: Optional[list[concepts.Dimension]] = None):
    flat_types = type_traits.flatten(ts.TupleType(param_types))
    terms = name.split(".")
    base_name = "".join([f"{len(term)}{term}" for term in terms])
    param_codes = [format_type_short(type_) for type_ in flat_types]
    dim_codes = f"_{'x'.join([str(dim.id) for dim in dims])}" if dims else ""
    return f"__{base_name}__{'_'.join(param_codes)}{dim_codes}"
