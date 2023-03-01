import textwrap

from stencilpy.compiler import types as ts
from stencilpy.error import *
from stencilpy import concepts
from .symbol_table import SymbolTable

import inspect
import ast
from stencilpy.compiler import hast

from typing import Any, Optional, cast
from collections.abc import Sequence, Mapping


def builtin_shape(transformer: Any, location: concepts.Location, args: Sequence[ast.AST]):
    assert len(args) == 2
    field = transformer.visit(args[0])
    dim = transformer.visit(args[1])
    if not isinstance(dim, concepts.Dimension):
        raise CompilationError(location, "the `shape` function expects a dimension for argument 2")

    return hast.Shape(location, ts.IndexType(), field, dim)


_BUILTIN_MAPPING = {
    "shape": builtin_shape
}


class PythonToHAST(ast.NodeTransformer):
    symtable: SymbolTable
    param_types: list[ts.Type]
    kwparam_types: dict[str, ts.Type]
    ndims: Optional[int] = None
    file: str
    start_line: int

    def __init__(
            self,
            file: str,
            start_line: int,
            start_col: int,
            symtable: SymbolTable,
            param_types: list[ts.Type],
            kwparam_types: dict[str, ts.Type],
            ndims: Optional[int] = None,
    ):
        super().__init__()
        self.file = file
        self.start_line = start_line
        self.start_col = start_col
        self.symtable = symtable
        self.param_types = param_types
        self.kwparam_types = kwparam_types
        self.ndims = ndims

    def get_ast_loc(self, node: ast.AST):
        return hast.Location(self.file, self.start_line + node.lineno, self.start_col + node.col_offset)

    def generic_visit(self, node: ast.AST, check=True) -> hast.Node:
        if not check:
            return cast(super().generic_visit(node), hast.Node)
        loc = self.get_ast_loc(node)
        raise UnsupportedLanguageError(loc, node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> hast.Function | hast.Stencil:
        def sc():
            parameters = [
                hast.Parameter(name.arg, type_)
                for name, type_ in zip(node.args.args, self.param_types)
            ]
            for param in parameters:
                self.symtable.assign(param.name, param.type_)

            statements = [self.visit(statement) for statement in node.body]

            results: list[ts.Type] = []
            for statement in statements:
                if isinstance(statement, hast.Return):
                    results = [expr.type_ for expr in statement.values]

            loc = self.get_ast_loc(node)
            type_ = ts.FunctionType(self.param_types, results)
            name = node.name

            if self.ndims is None:
                return hast.Function(loc, type_, name, parameters, results, statements)
            else:
                return hast.Stencil(loc, type_, name, parameters, results, statements, self.ndims)

        return self.symtable.scope(sc)

    def visit_Return(self, node: ast.Return) -> hast.Return:
        if isinstance(node.value, ast.Tuple):
            values = [self.visit(value) for value in node.value.elts]
        else:
            values = [self.visit(node.value)] if node.value else []
        loc = self.get_ast_loc(node)
        type_ = ts.VoidType()
        return hast.Return(loc, type_, values)

    def visit_Constant(self, node: ast.Constant) -> hast.Constant:
        loc = self.get_ast_loc(node)
        type_ = ts.infer_object_type(node.value)
        value = node.value
        return hast.Constant(loc, type_, value)

    def visit_Name(self, node: ast.Name) -> hast.SymbolRef:
        assert isinstance(node.ctx, ast.Load) # Store contexts are handled explicitly in the parent node.

        loc = self.get_ast_loc(node)
        name = node.id
        symbol_entry = self.symtable.lookup(name)
        if not symbol_entry:
            raise UndefinedSymbolError(loc, name)
        if isinstance(symbol_entry, hast.ExternalSymbol):
            return self._process_external_symbol(symbol_entry, loc)
        return hast.SymbolRef(loc, symbol_entry, name)

    def visit_Call(self, node: ast.Call) -> hast.Expr:
        loc = self.get_ast_loc(node)
        if not isinstance(node.func, ast.Name):
            raise CompilationError(loc, "function call are only allowed on symbols")

        callee = node.func.id
        callee_entry = self.symtable.lookup(callee)
        if not callee_entry:
            raise UndefinedSymbolError(loc, callee)
        if isinstance(callee_entry, hast.ExternalSymbol):
            if callee_entry.name not in _BUILTIN_MAPPING:
                raise InternalCompilerError(loc, f"builtin function `{callee_entry.name}` is not implemented")
            return _BUILTIN_MAPPING[callee](self, loc, node.args)
        raise NotImplementedError()

    def _process_external_symbol(self, node: hast.ExternalSymbol, location: concepts.Location) -> Any:
        if isinstance(node.type_, (ts.IndexType, ts.IntegerType, ts.FloatType)):
            return hast.Constant(location, node.type_, node.value)
        elif isinstance(node.value, concepts.Builtin):
            return node.value
        elif isinstance(node.value, concepts.Dimension):
            return node.value
        else:
            raise CompilationError(location, f"external symbol of type `{type(node.value)}` is not understood")


def get_source_code(definition: callable):
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
        symtable.assign(name, hast.ExternalSymbol(loc, type_, name, value))


def parse_as_function(definition: callable, param_types: list[ts.Type], kwparam_types: dict[str, ts.Type]) -> hast.Module:
    source_code, file, start_line, start_col = get_source_code(definition)
    python_ast = ast.parse(source_code)

    assert isinstance(python_ast, ast.Module)
    assert len(python_ast.body) == 1
    assert isinstance(python_ast.body[0], ast.FunctionDef)

    symtable = SymbolTable()
    closure_vars = inspect.getclosurevars(definition)
    add_closure_vars_to_symtable(symtable, closure_vars.globals)
    add_closure_vars_to_symtable(symtable, closure_vars.nonlocals)

    transformer = PythonToHAST(file, start_line, start_col, symtable, param_types, kwparam_types)
    func = transformer.visit(python_ast.body[0])
    assert isinstance(func, hast.Function)

    module = hast.Module(hast.Location.unknown(), ts.VoidType(), [func], [])
    return module


def parse_as_stencil(definition: callable, param_types: list[ts.Type], kwparam_types: dict[str, ts.Type]) -> hast.Module:
    source_code, file, start_line, start_col = get_source_code(definition)
    python_ast = ast.parse(source_code)

    assert isinstance(python_ast, ast.Module)
    assert len(python_ast.body) == 1
    assert isinstance(python_ast.body[0], ast.FunctionDef)

    symtable = SymbolTable()
    add_closure_vars_to_symtable(symtable)
    transformer = PythonToHAST(file, start_line, start_col, symtable, param_types, kwparam_types)

    stencil = transformer.visit(python_ast.body[0])
    assert isinstance(stencil, hast.Stencil)

    module = hast.Module(hast.Location.unknown(), ts.VoidType(), [], [stencil])
    return module
