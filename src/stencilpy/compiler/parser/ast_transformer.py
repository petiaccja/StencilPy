import ctypes
import inspect
import ast
from typing import Optional, cast, Callable
from collections.abc import Sequence

from stencilpy.compiler import types as ts
from stencilpy.error import *
from stencilpy import concepts
from stencilpy.compiler.symbol_table import SymbolTable
from stencilpy.compiler import hlast
from stencilpy.compiler import utility
from .utility import FunctionSpecification, get_source_code, add_closure_vars_to_symtable
from .builtin_transformers import BUILTIN_MAPPING


class AstTransformer(ast.NodeTransformer):
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

    #-----------------------------------
    # Module structure
    #-----------------------------------

    def visit_FunctionDef(
            self,
            node: ast.FunctionDef,
            *,
            spec: FunctionSpecification = None,
    ) -> hlast.Function | hlast.Stencil:
        assert spec is not None
        def sc():
            loc = self.get_ast_loc(node)
            name = spec.mangled_name
            if len(node.args.args) != len(spec.param_types):
                raise ArgumentCountError(loc, len(node.args.args), len(spec.param_types))
            parameters = [
                hlast.Parameter(name.arg, type_)
                for name, type_ in zip(node.args.args, spec.param_types)
            ]
            for param in parameters:
                self.symtable.assign(param.name, param.type_)

            statements = [self.visit(statement) for statement in node.body]

            result: ts.Type = ts.VoidType()
            for statement in statements:
                if isinstance(statement, hlast.Return):
                    result = statement.value.type_

            if spec.dims is None:
                type_ = ts.FunctionType(spec.param_types, result)
                return hlast.Function(loc, type_, name, parameters, result, statements, spec.is_public)
            else:
                type_ = ts.StencilType(spec.param_types, result, sorted(spec.dims))
                return hlast.Stencil(loc, type_, name, parameters, result, statements, sorted(spec.dims))

        return self.symtable.scope(sc, spec)

    def visit_Return(self, node: ast.Return) -> hlast.Return:
        value = self.visit(node.value)
        loc = self.get_ast_loc(node)
        type_ = ts.VoidType()
        return hlast.Return(loc, type_, value)

    def visit_Call(self, node: ast.Call) -> hlast.Expr:
        loc = self.get_ast_loc(node)
        builtin = self._visit_call_builtin(node)
        if builtin: return builtin
        meta = self._visit_call_meta(node)
        if meta: return meta
        apply = self._visit_call_apply(node)
        if apply: return apply
        call = self._visit_call_call(node)
        if call: return call
        raise CompilationError(loc, "object is not callable")

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
        if callee.value.name not in BUILTIN_MAPPING:
                raise InternalCompilerError(loc, f"builtin function `{callee.value.name}` is not implemented")
        return BUILTIN_MAPPING[callee.value.name](self, loc, node.args)

    def _visit_call_apply(self, node: ast.Call) -> Optional[hlast.Apply]:
        loc = self.get_ast_loc(node)
        node_shape = node.func
        if not isinstance(node_shape, ast.Subscript):
            return None
        callable_ = self.visit(node_shape.value)
        if not isinstance(callable_, concepts.Stencil):
            return None
        sizes: Sequence[hlast.Size] = self._visit_getitem_expr(node_shape.slice)
        shape = {sz.dimension: sz.size for sz in sizes}
        dims = sorted(sz.dimension for sz in sizes)
        args = [self.visit(arg) for arg in node.args]
        stencil = self.instantiate(callable_.definition, [arg.type_ for arg in args], False, dims)
        assert isinstance(stencil.type_, ts.StencilType)
        type_ = ts.FieldType(stencil.type_.result, stencil.type_.dims)
        return hlast.Apply(loc, type_, stencil.name, shape, args)

    def _visit_call_call(self, node: ast.Call) -> Optional[hlast.Call]:
        loc = self.get_ast_loc(node)
        callable_ = self.visit(node.func)
        if not isinstance(callable_, concepts.Function):
            return None
        args = [self.visit(arg) for arg in node.args]
        func = self.instantiate(callable_.definition, [arg.type_ for arg in args], False)
        assert isinstance(func.type_, ts.FunctionType)
        type_ = func.type_.result
        return hlast.Call(loc, type_, func.name, args)

    def _visit_call_meta(self, node: ast.Call) -> Optional[Any]:
        try:
            func = self.visit(node.func)
            assert isinstance(func, concepts.MetaFunc)
        except:
            return None
        args = [self.visit(arg) for arg in node.args]
        return func(*args, is_jit=True)

    #-----------------------------------
    # Symbols
    #-----------------------------------

    def visit_Name(self, node: ast.Name) -> hlast.SymbolRef:
        assert isinstance(node.ctx, ast.Load) # Store contexts are handled explicitly in the parent node.

        loc = self.get_ast_loc(node)
        name = node.id
        symbol_entry = self.symtable.lookup(name)
        if not symbol_entry:
            raise UndefinedSymbolError(loc, name)
        if isinstance(symbol_entry, hlast.ClosureVariable):
            return self._process_closure_value(symbol_entry.value, loc)
        return hlast.SymbolRef(loc, symbol_entry, name)

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

    #-----------------------------------
    # Arithmetic & logic
    #-----------------------------------

    def visit_Constant(self, node: ast.Constant) -> hlast.Constant:
        loc = self.get_ast_loc(node)
        type_ = ts.infer_object_type(node.value)
        value = node.value
        return hlast.Constant(loc, type_, value)

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
        lhs_element_type = lhs.type_.element_type if is_lhs_field else lhs.type_
        rhs_element_type = rhs.type_.element_type if is_rhs_field else rhs.type_
        lhs_dimensions = lhs.type_.dimensions if is_lhs_field else []
        rhs_dimensions = rhs.type_.dimensions if is_rhs_field else []
        if is_lhs_field or is_rhs_field:
            element_type = lhs_element_type  # TODO: promote to common type
            dimensions = sorted(list(set(lhs_dimensions) | set(rhs_dimensions)))
            type_ = ts.FieldType(element_type, dimensions)
            return hlast.ElementwiseOperation(loc, type_, [lhs, rhs], builder)
        else:
            return builder([lhs, rhs])

    def visit_Compare(self, node: ast.Compare) -> Any:
        loc = self.get_ast_loc(node)
        operands = [self.visit(operand) for operand in [node.left, *node.comparators]]
        funcs = [self.visit(op) for op in node.ops]

        bool_type = ts.IntegerType(1, True)
        def builder(args: list[hlast.Expr]) -> hlast.Expr:
            args_names = [f"__cmp_operand{i}" for i in range(len(args))]
            args_assign = hlast.Assign(loc, ts.VoidType(), args_names, args)
            args_refs = [hlast.SymbolRef(loc, v.type_, name) for v, name in zip(args, args_names)]

            c_true = hlast.Constant(loc, bool_type, 1)
            c_false = hlast.Constant(loc, bool_type, 0)
            expr = c_true
            for i in range(len(funcs) - 1, -1, -1):
                func = funcs[i]
                lhs = args_refs[i]
                rhs = args_refs[i + 1]
                cmp = hlast.ComparisonOperation(loc, bool_type, lhs, rhs, func)
                expr = hlast.If(
                    loc, bool_type, cmp,
                    [hlast.Yield(loc, bool_type, [expr])],
                    [hlast.Yield(loc, bool_type, [c_false])]
                )
            return hlast.If(
                loc, bool_type, c_true,
                [args_assign, hlast.Yield(loc, bool_type, [expr])],
                [hlast.Yield(loc, bool_type, [c_false])]
            )

        is_operand_field = [isinstance(arg.type_, ts.FieldType) for arg in operands]
        operand_dimensions = [
            arg.type_.dimensions if is_field else []
            for is_field, arg in zip(is_operand_field, operands)
        ]
        if any(is_operand_field):
            element_type = ts.IntegerType(1, False)
            merged_dims = set()
            for dims in operand_dimensions:
                merged_dims = merged_dims | dims
            dimensions = sorted(list(merged_dims))
            type_ = ts.FieldType(element_type, dimensions)
            return hlast.ElementwiseOperation(loc, type_, operands, builder)
        else:
            return builder(operands)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> hlast.Expr:
        loc = self.get_ast_loc(node)
        value = self.visit(node.operand)
        type_ = value.type_.element_type if isinstance(value.type_, ts.FieldType) else value.type_
        if isinstance(node.op, ast.UAdd):
            return value
        elif isinstance(node.op, ast.USub):
            sign_val = -1.0 if isinstance(type_, ts.FloatType) else -1
            c_sign = hlast.Constant(loc, type_, sign_val)
            return hlast.ArithmeticOperation(loc, value.type_, c_sign, value, hlast.ArithmeticFunction.MUL)
        elif isinstance(node.op, ast.Invert):
            if not isinstance(type_, (ts.IndexType, ts.IntegerType)):
                raise ArgumentCompatibilityError(loc, "bit inversion", [type_])
            c_bitmask = hlast.Constant(loc, type_, -1)
            return hlast.ArithmeticOperation(loc, value.type_, c_bitmask, value, hlast.ArithmeticFunction.BIT_XOR)
        elif isinstance(node.op, ast.Not):
            bool_t = ts.IntegerType(1, True)
            compare_val = 0.0 if isinstance(type_, ts.FloatType) else 0
            c_compare = hlast.Constant(loc, type_, compare_val)
            c_bitmask = hlast.Constant(loc, type_, 1)
            boolified = hlast.ComparisonOperation(loc, bool_t, value, c_compare, hlast.ComparisonFunction.NEQ)
            return hlast.ArithmeticOperation(loc, bool_t, c_bitmask, boolified, hlast.ArithmeticFunction.BIT_XOR)
        raise CompilationError(loc, f"unary operator {type(node.op)} not implemented")

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

    #-----------------------------------
    # Misc
    #-----------------------------------

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        loc = self.get_ast_loc(node)
        value = self.visit(node.value)

        if isinstance(value, concepts.Dimension):
            slice_expr = self.visit(node.slice)
            if isinstance(slice_expr, hlast.Expr):
                return hlast.Size(value, slice_expr)
            elif isinstance(slice_expr, hlast.Slice):
                return hlast.Slice(value, slice_expr.lower, slice_expr.upper, slice_expr.step)
            raise CompilationError(loc, f"dimension cannot be subscripted by object of type `{slice_expr.type_}`")
        elif isinstance(value.type_, ts.FieldLikeType):
            slice_exprs: Sequence[hlast.Expr] = self._visit_getitem_expr(node.slice)
            if all(isinstance(expr, (hlast.Slice, hlast.Size)) for expr in slice_exprs):
                slices: list[hlast.Slice] = [
                    self._promote_to_slice(expr) if isinstance(expr, hlast.Size) else expr
                    for expr in slice_exprs
                ]
                return hlast.ExtractSlice(loc, value.type_, value, slices)
            elif isinstance(slice_exprs[0].type_, ts.NDIndexType):
                return hlast.Sample(loc, value.type_.element_type, value, slice_exprs[0])
            expr_types = [str(expr.type_) if hasattr(expr, "type_") else str(type(expr)) for expr in slice_exprs]
            str_types = ", ".join(expr_types)
            raise CompilationError(loc, f"field cannot be subscripted by object of type `({str_types})`")
        raise CompilationError(loc, f"object of type {value.type_} be subscripted")

    def visit_Slice(self, node: ast.Slice) -> hlast.Slice:
        loc = self.get_ast_loc(node)
        global invalid_dim
        if not "invalid_dim" in globals():
            invalid_dim = concepts.Dimension("INVALID_DIM")
        index_max = 2**(8*ctypes.sizeof(ctypes.c_void_p) - 1) - 1
        assert isinstance(index_max, int)
        lower = self.visit(node.lower) if node.lower else None
        upper = self.visit(node.upper) if node.upper else None
        step = self.visit(node.step) if node.step else None
        return hlast.Slice(invalid_dim, lower, upper, step)

    def visit_Expr(self, node: ast.Expr) -> Any:
        return self.visit(node.value)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        loc = self.get_ast_loc(node)
        if isinstance(node.value, ast.Attribute):
            value = self.visit(node.value)
            if hasattr(value, node.attr):
                return getattr(value, node.attr)
            raise CompilationError(loc, f"object has no attribute {node.attr}")
        if isinstance(node.value, ast.Name):
            closure_var = self.symtable.lookup(node.value.id)
            if not isinstance(closure_var, hlast.ClosureVariable):
                raise CompilationError(loc, "expected a closure variable")
            value = closure_var.value
            if hasattr(value, node.attr):
                return self._process_closure_value(getattr(value, node.attr), loc)
            raise CompilationError(loc, f"object has no attribute {node.attr}")
        raise CompilationError(loc, "expected a name or an attribute of a name")

    #-----------------------------------
    # Utilities
    #-----------------------------------
    def _visit_getitem_expr(self, node: ast.AST):
        if isinstance(node, ast.Tuple):
            return tuple(self.visit(element) for element in node.elts)
        slices = self.visit(node)
        return slices if isinstance(slices, tuple) else (slices,)

    def instantiate(
            self,
            definition: Callable,
            arg_types: list[ts.Type],
            is_public: bool,
            dims: Optional[list[concepts.Dimension]] = None
    ) -> hlast.Function | hlast.Stencil:
        name = definition.__name__
        module = definition.__module__
        mangled_name = utility.mangle_name(f"{module}.{name}", arg_types, dims)
        if mangled_name in self.instantiations:
            return self.instantiations[mangled_name]

        def sc():
            source_code, file, start_line, start_col = get_source_code(definition)
            python_ast = ast.parse(source_code)

            assert isinstance(python_ast, ast.Module)
            assert len(python_ast.body) == 1
            assert isinstance(python_ast.body[0], ast.FunctionDef)

            closure_vars = inspect.getclosurevars(definition)
            add_closure_vars_to_symtable(self.symtable, closure_vars.globals)
            add_closure_vars_to_symtable(self.symtable, closure_vars.nonlocals)

            spec = FunctionSpecification(mangled_name, arg_types, {}, is_public, dims)
            instantiation = self.visit_FunctionDef(python_ast.body[0], spec=spec)
            return instantiation

        instantiation = self.symtable.scope(sc)
        self.instantiations[instantiation.name] = self.symtable.scope(sc)
        return instantiation

    def _process_closure_value(self, value: Any, location: concepts.Location) -> Any:
        # Stencils, functions
        if isinstance(value, (concepts.Function, concepts.Stencil)):
            return value
        # Builtins
        elif isinstance(value, concepts.Builtin):
            return value
        # Dimensions
        elif isinstance(value, concepts.Dimension):
            return value
        # Modules
        elif inspect.ismodule(value):
            return value
        # Types
        elif isinstance(value, ts.Type):
            return value
        # Constant expressions
        try:
            type_ = ts.infer_object_type(value)
            if isinstance(type_, (ts.IndexType, ts.IntegerType, ts.FloatType)):
                return hlast.Constant(location, type_, value)
        except Exception:
            pass
        raise CompilationError(location, f"closure variable of type `{type(value)}` is not understood")

    def _promote_to_slice(self, size: hlast.Size) -> hlast.Slice:
        lower = size.size
        return hlast.Slice(size.dimension, lower, None, None, True)