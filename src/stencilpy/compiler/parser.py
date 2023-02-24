import textwrap

from stencilpy.compiler import types as ts

import inspect
import ast
from stencilpy.compiler import hast

from typing import Any, Optional, cast


def get_ast_loc(node: ast.AST):
    return hast.Location("", node.lineno, node.col_offset)


class SymbolTable:
    tables: list[dict[str, Any]]

    def __init__(self):
        self.tables = [{}]

    def assign(self, name: str, value: Any):
        self.tables[-1][name] = value

    def lookup(self, name: str) -> Any:
        for table in reversed(self.tables):
            if name in table:
                return table[name]
        return None

    def scope(self, callback: callable) -> Any:
        self.tables.append({})
        result = callback()
        self.tables.pop(-1)
        return result


class PythonToHAST(ast.NodeTransformer):
    symtable: SymbolTable
    param_types: list[ts.Type]
    kwparam_types: dict[str, ts.Type]
    ndims: Optional[int] = None

    def __init__(
            self,
            symtable: SymbolTable,
            param_types: list[ts.Type],
            kwparam_types: dict[str, ts.Type],
            ndims: Optional[int] = None):
        super().__init__()
        self.symtable = symtable
        self.param_types = param_types
        self.kwparam_types = kwparam_types
        self.ndims = ndims

    def generic_visit(self, node: ast.AST, check=True) -> hast.Node:
        if not check:
            return cast(super().generic_visit(node), hast.Node)
        raise ValueError(f"python construct `{node.__class__.__name__}` not supported")

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

            loc = get_ast_loc(node)
            type_ = ts.FunctionType()
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
        loc = get_ast_loc(node)
        type_ = ts.VoidType()
        return hast.Return(loc, type_, values)

    def visit_Constant(self, node: ast.Constant) -> hast.Constant:
        loc = get_ast_loc(node)
        type_ = ts.infer_object_type(node.value)
        value = node.value
        return hast.Constant(loc, type_, value)


def get_source_code(definition: callable):
    source_code = inspect.getsource(definition)
    return textwrap.dedent(source_code)


def parse_as_function(source_code: str, param_types: list[ts.Type], kwparam_types: dict[str, ts.Type]) -> hast.Module:
    symtable = SymbolTable()
    transformer = PythonToHAST(symtable, param_types, kwparam_types)
    python_ast = ast.parse(source_code)

    assert len(python_ast.body) == 1
    assert isinstance(python_ast.body[0], ast.FunctionDef)

    func = transformer.visit(python_ast.body[0])
    assert isinstance(func, hast.Function)

    module = hast.Module(hast.Location.unknown(), ts.VoidType(), [func], [])
    return module


def parse_as_stencil(source_code: str, param_types: list[ts.Type], kwparam_types: dict[str, ts.Type]) -> hast.Module:
    symtable = SymbolTable()
    transformer = PythonToHAST(symtable, param_types, kwparam_types)
    python_ast = ast.parse(source_code)

    assert len(python_ast.body) == 1
    assert isinstance(python_ast.body[0], ast.FunctionDef)

    stencil = transformer.visit(python_ast.body[0])
    assert isinstance(stencil, hast.Stencil)

    module = hast.Module(hast.Location.unknown(), ts.VoidType(), [], [stencil])
    return module
