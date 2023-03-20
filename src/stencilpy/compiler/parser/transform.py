import inspect
import ast
from typing import Optional, Callable

from stencilpy.compiler import types as ts
from stencilpy import concepts
from stencilpy.compiler.symbol_table import SymbolTable
from stencilpy.compiler import hlast
from .ast_transformer import AstTransformer
from .utility import get_source_code, add_closure_vars_to_symtable


def function_to_hlast(
        definition: Callable,
        param_types: list[ts.Type],
        kwparam_types: dict[str, ts.Type],
        dims: Optional[list[concepts.Dimension]] = None
) -> hlast.Module:
    source_code, file, start_line, start_col = get_source_code(definition)
    python_ast = ast.parse(source_code)

    assert isinstance(python_ast, ast.Module)
    assert len(python_ast.body) == 1
    assert isinstance(python_ast.body[0], ast.FunctionDef)

    symtable = SymbolTable()
    closure_vars = inspect.getclosurevars(definition)
    add_closure_vars_to_symtable(symtable, closure_vars.globals)
    add_closure_vars_to_symtable(symtable, closure_vars.nonlocals)

    transformer = AstTransformer(file, start_line, start_col, symtable)
    transformer.instantiate(definition, param_types, True, dims)

    callables = [*transformer.instantiations.values()]
    functions = [f for f in callables if isinstance(f, hlast.Function)]
    stencils = [f for f in callables if isinstance(f, hlast.Stencil)]
    module = hlast.Module(hlast.Location.unknown(), ts.VoidType(), functions, stencils)

    return module
