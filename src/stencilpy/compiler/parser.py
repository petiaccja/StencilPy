import textwrap

from stencilpy.compiler import types as ts
from stencilpy.error import *
from stencilpy import concepts
from .symbol_table import SymbolTable

import inspect
import ast
from stencilpy.compiler import hlast
from stencilpy.compiler import utility

from typing import Any, Optional, cast, Callable
from collections.abc import Sequence, Mapping


@dataclasses.dataclass
class FunctionSpecification:
    param_types: list[ts.Type]
    kwparam_types: dict[str, ts.Type]
    dims: Optional[list[concepts.Dimension]]


def builtin_shape(transformer: Any, location: concepts.Location, args: Sequence[ast.AST]):
    assert len(args) == 2
    field = transformer.visit(args[0])
    dim = transformer.visit(args[1])
    if not isinstance(dim, concepts.Dimension):
        raise CompilationError(location, "the `shape` function expects a dimension for argument 2")

    return hlast.Shape(location, ts.IndexType(), field, dim)


def builtin_index(transformer: "PythonToHlast", location: concepts.Location, *args):
    function_def_info: Optional[FunctionSpecification] = None
    for info in transformer.symtable.infos():
        if isinstance(info, FunctionSpecification):
            function_def_info = info
            break
    if not function_def_info:
        raise CompilationError(location, "index expression must be used inside a stencil's body")
    type_ = ts.NDIndexType(function_def_info.dims)
    return hlast.Index(location, type_)


_BUILTIN_MAPPING = {
    "shape": builtin_shape,
    "index": builtin_index,
}


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
            spec: FunctionSpecification = None,
    ) -> hlast.Function | hlast.Stencil:
        assert spec is not None
        def sc():
            loc = self.get_ast_loc(node)
            name = utility.mangle_name(node.name, spec.param_types, spec.dims)
            if len(node.args.args) != len(spec.param_types):
                raise ArgumentCountError(loc, len(node.args.args), len(spec.param_types))
            parameters = [
                hlast.Parameter(name.arg, type_)
                for name, type_ in zip(node.args.args, spec.param_types)
            ]
            for param in parameters:
                self.symtable.assign(param.name, param.type_)

            statements = [self.visit(statement) for statement in node.body]

            results: list[ts.Type] = []
            for statement in statements:
                if isinstance(statement, hlast.Return):
                    results = [expr.type_ for expr in statement.values]

            if spec.dims is None:
                type_ = ts.FunctionType(spec.param_types, results)
                return hlast.Function(loc, type_, name, parameters, results, statements)
            else:
                type_ = ts.StencilType(spec.param_types, results, sorted(spec.dims))
                return hlast.Stencil(loc, type_, name, parameters, results, statements, sorted(spec.dims))

        return self.symtable.scope(sc, spec)

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

    def visit_Assign(self, node: ast.Assign) -> hlast.Assign:
        loc = self.get_ast_loc(node)
        values = [self.visit(node.value)]
        if not all(isinstance(target, ast.Name) for target in node.targets):
            raise CompilationError(loc, "only assigning to simple variables is supported")
        names = [target.id for target in node.targets]
        for name, value in zip(names, values):
            self.symtable.assign(name, value.type_)
        return hlast.Assign(loc, ts.VoidType(), names, values)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> hlast.Assign:
        loc = self.get_ast_loc(node)
        values = [self.visit(node.value)]
        if not isinstance(node.target, ast.Name):
            raise CompilationError(loc, "only assigning to simple variables is supported")
        names = [node.target.id]
        for name, value in zip(names, values):
            self.symtable.assign(name, value.type_)
        return hlast.Assign(loc, ts.VoidType(), names, values)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        loc = self.get_ast_loc(node)
        value = self.visit(node.value)
        slice = self.visit(node.slice)
        if isinstance(value.type_, ts.FieldType) and isinstance(slice.type_, ts.NDIndexType):
            return hlast.Sample(loc, value.type_.element_type, value, slice)
        raise CompilationError(
            loc,
            f"object of type {value.type_} cannot be subscripted"
            f" with object of type {slice.type_}"
        )

    def visit_BinOp(self, node: ast.BinOp) -> hlast.Expr:
        loc = self.get_ast_loc(node)
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        func = self.visit(node.op)

        def builder(args: list[hlast.Expr]) -> hlast.Expr:
            type_ = args[0].type_  # TODO: promote to common type
            return hlast.ArithmeticOperation(loc, type_, args[0], args[1], func)

        is_lhs_field = isinstance(lhs.type_, ts.FieldType)
        is_rhs_field = isinstance(rhs.type_, ts.FieldType)
        if is_lhs_field and is_rhs_field:
            if lhs.type_.dimensions != rhs.type_.dimensions:
                raise ArgumentCompatibilityError(loc, f"operator {func}", [lhs.type_, rhs.type_])
            type_ = lhs.type_  # TODO: promote to common type
            return hlast.ElementwiseOperation(loc, type_, [lhs, rhs], builder)
        elif is_lhs_field or is_rhs_field:
            raise ArgumentCompatibilityError(loc, f"operator {func}", [lhs.type_, rhs.type_])
        else:
            return builder([lhs, rhs])

    def visit_Compare(self, node: ast.Compare) -> Any:
        raise NotImplementedError()

    def visit_Expr(self, node: ast.Expr) -> Any:
        return self.visit(node.value)

    def visit_Add(self, node: ast.Add) -> Any: return hlast.ArithmeticFunction.ADD
    def visit_Sub(self, node: ast.Add) -> Any: return hlast.ArithmeticFunction.SUB
    def visit_Mult(self, node: ast.Add) -> Any: return hlast.ArithmeticFunction.MUL
    def visit_Div(self, node: ast.Add) -> Any: return hlast.ArithmeticFunction.DIV
    def visit_Mod(self, node: ast.Add) -> Any: return hlast.ArithmeticFunction.MOD
    def visit_LShift(self, node: ast.Add) -> Any: return hlast.ArithmeticFunction.BIT_SHL
    def visit_RShift(self, node: ast.Add) -> Any: return hlast.ArithmeticFunction.BIT_SHR
    def visit_BitOr(self, node: ast.Add) -> Any: return hlast.ArithmeticFunction.BIT_OR
    def visit_BitXor(self, node: ast.Add) -> Any: return hlast.ArithmeticFunction.BIT_XOR
    def visit_BitAnd(self, node: ast.Add) -> Any: return hlast.ArithmeticFunction.BIT_AND

    def visit_Eq(self, node: ast.Eq): return hlast.ComparisonFunction.EQ
    def visit_NotEq(self, node: ast.NotEq): return hlast.ComparisonFunction.NEQ
    def visit_Lt(self, node: ast.Lt): return hlast.ComparisonFunction.LT
    def visit_LtE(self, node: ast.LtE): return hlast.ComparisonFunction.LTE
    def visit_Gt(self, node: ast.Gt): return hlast.ComparisonFunction.GT
    def visit_GtE(self, node: ast.GtE): return hlast.ComparisonFunction.GTE

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
        stencil = self._instantiate(node_stencil, [arg.type_ for arg in args], sorted(dims))
        assert isinstance(stencil.type_, ts.StencilType)
        assert len(stencil.type_.results) == 1
        type_ = ts.FieldType(stencil.type_.results[0], stencil.type_.dims)
        return hlast.Apply(loc, type_, stencil, shape, args)

    def _visit_slice(self, node: ast.AST):
        if isinstance(node, ast.Tuple):
            return [self.visit(element) for element in node.elts]
        return [self.visit(node)]

    def _instantiate(
            self,
            name: ast.Name,
            arg_types: list[ts.Type],
            dims: Optional[list[concepts.Dimension]] = None
    ):
        mangled_name = utility.mangle_name(name.id, arg_types, dims)
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

            closure_vars = inspect.getclosurevars(definition)
            add_closure_vars_to_symtable(self.symtable, closure_vars.globals)
            add_closure_vars_to_symtable(self.symtable, closure_vars.nonlocals)

            spec = FunctionSpecification(arg_types, {}, dims)
            instantiation = self.visit_FunctionDef(python_ast.body[0], spec=spec)
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


def parse_as_function(
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

    transformer = PythonToHlast(file, start_line, start_col, symtable)
    spec = FunctionSpecification(param_types, kwparam_types, dims)
    entry_point = transformer.visit_FunctionDef(python_ast.body[0], spec=spec)
    callables = [entry_point, *transformer.instantiations.values()]

    functions = [f for f in callables if isinstance(f, hlast.Function)]
    stencils = [f for f in callables if isinstance(f, hlast.Stencil)]
    module = hlast.Module(hlast.Location.unknown(), ts.VoidType(), functions, stencils)

    return module
