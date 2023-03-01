import textwrap

from stencilpy.compiler import types as ts
from stencilpy.error import *
from stencilpy import concepts
from .symbol_table import SymbolTable

import inspect
import ast
from stencilpy.compiler import hlast

from typing import Any, Optional, cast
from collections.abc import Sequence, Mapping


def builtin_shape(transformer: Any, location: concepts.Location, args: Sequence[ast.AST]):
    assert len(args) == 2
    field = transformer.visit(args[0])
    dim = transformer.visit(args[1])
    if not isinstance(dim, concepts.Dimension):
        raise CompilationError(location, "the `shape` function expects a dimension for argument 2")

    return hlast.Shape(location, ts.IndexType(), field, dim)


_BUILTIN_MAPPING = {
    "shape": builtin_shape
}


def type_code(type_: ts.Type):
    if isinstance(type_, ts.IndexType):
        return "I"
    if isinstance(type_, ts.IntegerType):
        return f"{'s' if type_.signed else ''}i{type_.width}"
    if isinstance(type_, ts.FloatType):
        return f"f{type_.width}"
    if isinstance(type_, ts.FieldType):
        element_type = type_code(type_.element_type)
        dims = [str(dim.id) for dim in type_.dimensions]
        return f"F{element_type}x{'x'.join(dims)}"
    raise NotImplementedError()


def mangle_name(name: str, param_types: list[ts.Type], dims: Optional[list[concepts.Dimension]] = None):
    param_codes = [type_code(type_) for type_ in param_types]
    dim_codes = f"_{'x'.join([str(dim.id) for dim in dims])}" if dims else ""
    return f"__{name}_{'_'.join(param_codes)}{dim_codes}"


class PythonToHlast(ast.NodeTransformer):
    file: str
    start_line: int
    start_col: int
    symtable: SymbolTable
    instantiations: dict[str, hlast.Function | hlast.Stencil]

    def __init__(
            self,
            file: str,
            start_line: int,
            start_col: int,
            symtable: SymbolTable,
    ):
        super().__init__()
        self.file = file
        self.start_line = start_line
        self.start_col = start_col
        self.symtable = symtable
        self.instantiations = {}

    def get_ast_loc(self, node: ast.AST):
        return hlast.Location(self.file, self.start_line + node.lineno, self.start_col + node.col_offset)

    def generic_visit(self, node: ast.AST, check=True) -> hlast.Node:
        if not check:
            return cast(super().generic_visit(node), hlast.Node)
        loc = self.get_ast_loc(node)
        raise UnsupportedLanguageError(loc, node)

    def visit_FunctionDef(
            self,
            node: ast.FunctionDef,
            *,
            param_types: list[ts.Type] = None,
            kwparam_types: dict[str, ts.Type] = None,
            dims: Optional[list[concepts.Dimension]] = None,
    ) -> hlast.Function | hlast.Stencil:
        def sc():
            parameters = [
                hlast.Parameter(name.arg, type_)
                for name, type_ in zip(node.args.args, param_types)
            ]
            for param in parameters:
                self.symtable.assign(param.name, param.type_)

            statements = [self.visit(statement) for statement in node.body]

            results: list[ts.Type] = []
            for statement in statements:
                if isinstance(statement, hlast.Return):
                    results = [expr.type_ for expr in statement.values]

            loc = self.get_ast_loc(node)
            name = mangle_name(node.name, param_types, dims)

            if dims is None:
                type_ = ts.FunctionType(param_types, results)
                return hlast.Function(loc, type_, name, parameters, results, statements)
            else:
                type_ = ts.StencilType(param_types, results, dims)
                return hlast.Stencil(loc, type_, name, parameters, results, statements, dims)

        return self.symtable.scope(sc)

    def visit_Return(self, node: ast.Return) -> hlast.Return:
        if isinstance(node.value, ast.Tuple):
            values = [self.visit(value) for value in node.value.elts]
        else:
            values = [self.visit(node.value)] if node.value else []
        loc = self.get_ast_loc(node)
        type_ = ts.VoidType()
        return hlast.Return(loc, type_, values)

    def visit_Constant(self, node: ast.Constant) -> hlast.Constant:
        loc = self.get_ast_loc(node)
        type_ = ts.infer_object_type(node.value)
        value = node.value
        return hlast.Constant(loc, type_, value)

    def visit_Name(self, node: ast.Name) -> hlast.SymbolRef:
        assert isinstance(node.ctx, ast.Load) # Store contexts are handled explicitly in the parent node.

        loc = self.get_ast_loc(node)
        name = node.id
        symbol_entry = self.symtable.lookup(name)
        if not symbol_entry:
            raise UndefinedSymbolError(loc, name)
        if isinstance(symbol_entry, hlast.ClosureVariable):
            return self._process_external_symbol(symbol_entry, loc)
        return hlast.SymbolRef(loc, symbol_entry, name)

    def visit_Call(self, node: ast.Call) -> hlast.Expr:
        loc = self.get_ast_loc(node)
        builtin = self._visit_call_builtin(node)
        if builtin: return builtin
        apply = self._visit_apply(node)
        if apply: return apply
        raise CompilationError(loc, "object not callable")

    def _visit_call_builtin(self, node: ast.Call) -> Optional[hlast.Expr]:
        loc = self.get_ast_loc(node)
        if not isinstance(node.func, ast.Name):
            return None
        callee = self.symtable.lookup(node.func.id)
        if not callee:
            raise UndefinedSymbolError(loc, callee)
        if not isinstance(callee, hlast.ClosureVariable):
            return None
        if not isinstance(callee.value, concepts.Builtin):
            return None
        if callee.value.name not in _BUILTIN_MAPPING:
                raise InternalCompilerError(loc, f"builtin function `{callee.value.name}` is not implemented")
        return _BUILTIN_MAPPING[callee.value.name](self, loc, node.args)

    def _visit_apply(self, node: ast.Call) -> Optional[hlast.Apply]:
        loc = self.get_ast_loc(node)
        node_shape = node.func
        if not isinstance(node_shape, ast.Subscript):
            return None
        node_dims = node_shape.value
        if not isinstance(node_dims, ast.Subscript):
            return None
        node_stencil = node_dims.value
        if not isinstance(node_stencil, ast.Name):
            return None
        sizes = self._visit_slice(node_shape.slice)
        dims = self._visit_slice(node_dims.slice)
        if len(dims) != len(sizes):
            raise CompilationError(loc, "number of dimensions and number of sizes must match in stencil invocation")
        shape = {dim: size for dim, size in zip(dims, sizes)}
        args = [self.visit(arg) for arg in node.args]
        stencil = self._instantiate(node_stencil, [arg.type_ for arg in args], dims)
        assert isinstance(stencil.type_, ts.StencilType)
        assert len(stencil.type_.results) == 1
        type_ = ts.FieldType(stencil.type_.results[0], stencil.type_.dims)
        return hlast.Apply(loc, type_, stencil, shape, args)

    def _visit_slice(self, node: ast.AST):
        if isinstance(node, ast.Tuple):
            return [self.visit(element) for element in node.elts]
        return [self.visit(node)]

    def _instantiate(self, name: ast.Name, arg_types: list[ts.Type], dims: Optional[list[concepts.Dimension]] = None):
        mangled_name = mangle_name(name.id, arg_types, dims)
        if mangled_name in self.instantiations:
            return self.instantiations[mangled_name]

        closure_var = self.symtable.lookup(name.id)
        loc = self.get_ast_loc(name)
        if (
                not isinstance(closure_var, hlast.ClosureVariable)
                or not isinstance(closure_var.value, (concepts.Stencil, concepts.Function))
        ):
            raise CompilationError(loc, f"object `{name.id}` being called is not a stencil")

        def sc():
            definition = closure_var.value.definition
            source_code, file, start_line, start_col = get_source_code(definition)
            python_ast = ast.parse(source_code)

            assert isinstance(python_ast, ast.Module)
            assert len(python_ast.body) == 1
            assert isinstance(python_ast.body[0], ast.FunctionDef)

            symtable = SymbolTable()
            closure_vars = inspect.getclosurevars(definition)
            add_closure_vars_to_symtable(symtable, closure_vars.globals)
            add_closure_vars_to_symtable(symtable, closure_vars.nonlocals)

            instantiation = self.visit_FunctionDef(
                python_ast.body[0],
                param_types=arg_types,
                kwparam_types={},
                dims=dims
            )
            return instantiation

        instantiation = self.symtable.scope(sc)
        self.instantiations[instantiation.name] = self.symtable.scope(sc)
        return hlast.SymbolRef(loc, instantiation.type_, instantiation.name)

    def _process_external_symbol(self, node: hlast.ClosureVariable, location: concepts.Location) -> Any:
        if isinstance(node.type_, (ts.IndexType, ts.IntegerType, ts.FloatType)):
            return hlast.Constant(location, node.type_, node.value)
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
        symtable.assign(name, hlast.ClosureVariable(loc, type_, name, value))


def parse_as_function(definition: callable, param_types: list[ts.Type], kwparam_types: dict[str, ts.Type]) -> hlast.Module:
    source_code, file, start_line, start_col = get_source_code(definition)
    python_ast = ast.parse(source_code)

    assert isinstance(python_ast, ast.Module)
    assert len(python_ast.body) == 1
    assert isinstance(python_ast.body[0], ast.FunctionDef)

    symtable = SymbolTable()
    closure_vars = inspect.getclosurevars(definition)
    add_closure_vars_to_symtable(symtable, closure_vars.globals)
    add_closure_vars_to_symtable(symtable, closure_vars.nonlocals)

    transformer = PythonToHlast(file, start_line, start_col, symtable)
    func = transformer.visit_FunctionDef(python_ast.body[0], param_types=param_types, kwparam_types=kwparam_types)
    assert isinstance(func, hlast.Function)

    funcs = [f for f in transformer.instantiations.values() if isinstance(f, hlast.Function)]
    stencils = [f for f in transformer.instantiations.values() if isinstance(f, hlast.Stencil)]
    module = hlast.Module(hlast.Location.unknown(), ts.VoidType(), [func, *funcs], stencils)

    return module


def parse_as_stencil(definition: callable, param_types: list[ts.Type], kwparam_types: dict[str, ts.Type]) -> hlast.Module:
    source_code, file, start_line, start_col = get_source_code(definition)
    python_ast = ast.parse(source_code)

    assert isinstance(python_ast, ast.Module)
    assert len(python_ast.body) == 1
    assert isinstance(python_ast.body[0], ast.FunctionDef)

    symtable = SymbolTable()
    closure_vars = inspect.getclosurevars(definition)
    add_closure_vars_to_symtable(symtable, closure_vars.globals)
    add_closure_vars_to_symtable(symtable, closure_vars.nonlocals)

    transformer = PythonToHlast(file, start_line, start_col, symtable)
    stencil = transformer.visit_FunctionDef(python_ast.body[0], param_types=param_types, kwparam_types=kwparam_types)
    assert isinstance(stencil, hlast.Stencil)

    funcs = [f for f in transformer.instantiations.values() if isinstance(f, hlast.Function)]
    stencils = [f for f in transformer.instantiations.values() if isinstance(f, hlast.Stencil)]
    module = hlast.Module(hlast.Location.unknown(), ts.VoidType(), funcs, [stencil, *stencils])

    return module
