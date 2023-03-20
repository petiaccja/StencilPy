from stencilpy.compiler import types as ts, hlast
from stencilpy import concepts
from stencilpy.compiler.symbol_table import SymbolTable

from typing import Optional, Callable, Mapping, Any
import dataclasses
import inspect
import textwrap


@dataclasses.dataclass
class FunctionSpecification:
    mangled_name: str
    param_types: list[ts.Type]
    kwparam_types: dict[str, ts.Type]
    is_public: bool
    dims: Optional[list[concepts.Dimension]]


def get_source_code(definition: Callable):
    file = inspect.getsourcefile(definition)
    source_lines, start_line = inspect.getsourcelines(definition)
    start_col = min((len(line) - len(line.lstrip()) for line in source_lines))
    source_code = ''.join(source_lines)
    return textwrap.dedent(source_code), file, start_line - 1, start_col


def add_closure_vars_to_symtable(symtable: SymbolTable, closure_vars: Mapping[str, Any]):
    for name, value in closure_vars.items():
        loc = concepts.Location.unknown()
        try:
            type_ = ts.infer_object_type(value)
        except Exception:
            type_ = ts.VoidType()
        symtable.assign(name, hlast.ClosureVariable(loc, type_, name, value))