import inspect
import ast
import copy
from typing import Optional, cast, Callable
from collections.abc import Sequence

from stencilpy.error import *
from stencilpy import concepts
from stencilpy.compiler.symbol_table import SymbolTable
from stencilpy.compiler import types as ts, hlast, utility, type_traits
from .utility import FunctionSpecification, get_source_code, add_closure_vars_to_symtable
from .builtin_transformers import BUILTIN_MAPPING
from .verification import ensure_expr, ensure_type, ensure_dimension, ensure_function, ensure_stencil, ensure_builtin


class AstTransformer(ast.NodeTransformer):
    base_location: concepts.Location
    symtable: SymbolTable
    instantiations: dict[str, hlast.Function | hlast.Stencil]

    def __init__(
            self,
            base_location: concepts.Location,
            symtable: SymbolTable,
    ):
        super().__init__()
        self.base_location = base_location
        self.symtable = symtable
        self.instantiations = {}

    def get_ast_loc(self, node: ast.AST):
        return hlast.Location(
            self.base_location.file,
            self.base_location.line + node.lineno,
            self.base_location.column + node.col_offset
        )

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
                hlast.Parameter(self.get_ast_loc(name), type_, name.arg)
                for name, type_ in zip(node.args.args, spec.param_types)
            ]
            for param in parameters:
                self.symtable.assign(param.name, param)

            statements = [self.visit(statement) for statement in node.body]

            result = spec.result
            if result is None:
                statements.append(hlast.Return(loc, ts.void_t, None))
                result = ts.void_t

            if spec.dims is None:
                type_ = ts.FunctionType(spec.param_types, result)
                return hlast.Function(loc, type_, name, parameters, result, statements, spec.is_public)
            else:
                type_ = ts.StencilType(spec.param_types, result, sorted(spec.dims))
                return hlast.Stencil(loc, type_, name, parameters, result, statements, sorted(spec.dims))

        return self.symtable.scope(sc, spec)

    def visit_Return(self, node: ast.Return) -> hlast.Return:
        loc = self.get_ast_loc(node)
        value = ensure_expr(self.visit(node.value), loc=loc)
        type_ = ts.VoidType()

        spec = self._get_enclosing_function()
        if not spec:
            raise CompilationError(loc, "return statement outside of function")
        if not spec.result:
            spec.result = value.type_

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
        try:
            callee = ensure_builtin(self.visit(node.func), loc=loc)
        except:
            return None
        if callee.name not in BUILTIN_MAPPING:
            raise InternalCompilerError(loc, f"builtin function `{callee.name}` is not implemented")
        return BUILTIN_MAPPING[callee.name](self, loc, node.args)

    def _visit_call_apply(self, node: ast.Call) -> Optional[hlast.Apply]:
        loc = self.get_ast_loc(node)
        try:
            node_shape = node.func
            assert isinstance(node_shape, ast.Subscript)
            callable_ = ensure_stencil(self.visit(node_shape.value), loc=loc)
        except:
            return None
        domain = self.visit(node_shape.slice)

        if isinstance(domain, hlast.TupleCreate):
            sizes = domain.elements
        else:
            sizes = [domain]
        if not all(isinstance(sz, hlast.Size) for sz in sizes):
            return None
        sizes = cast(list[hlast.Size], sizes)
        shape = {sz.dimension: sz.size for sz in sizes}
        dims = sorted(sz.dimension for sz in sizes)
        args = [ensure_expr(self.visit(arg)) for arg in node.args]
        stencil = self.instantiate(callable_.definition, [arg.type_ for arg in args], False, dims)
        assert isinstance(stencil.type_, ts.StencilType)
        result_types = type_traits.flatten(stencil.type_.result)
        field_types = [ts.FieldType(t, stencil.type_.dims) for t in result_types]
        if len(field_types) > 1:
            type_ = ts.TupleType(field_types)
        else:
            type_ = field_types[0]
        return hlast.Apply(loc, type_, stencil.name, shape, args)

    def _visit_call_call(self, node: ast.Call) -> Optional[hlast.Call]:
        loc = self.get_ast_loc(node)
        try:
            callable_ = ensure_function(self.visit(node.func))
        except:
            return None
        args = [ensure_expr(self.visit(arg)) for arg in node.args]

        arg_types = [arg.type_ for arg in args]
        mangled_name = utility.mangle_name(utility.get_qualified_name(callable_.definition), arg_types, None)
        spec = self._get_enclosing_function(mangled_name)
        if spec:
            callee = spec.mangled_name
            type_ = spec.result
            if not type_:
                raise CompilationError(loc, "recursive function must have a non-recursive return statement first")
        else:
            func = self.instantiate(callable_.definition, arg_types, None)
            callee = func.name
            assert isinstance(func.type_, ts.FunctionType)
            type_ = func.type_.result
        return hlast.Call(loc, type_, callee, args)

    def _visit_call_meta(self, node: ast.Call) -> Optional[Any]:
        try:
            func = self.visit(node.func)
            assert isinstance(func, concepts.MetaFunc)
        except:
            return None
        args = [self.visit(arg) for arg in node.args]
        return func(*args, transformer=self)

    #-----------------------------------
    # Control flow
    #-----------------------------------

    def visit_If(self, node: ast.If) -> hlast.IfStatement:
        loc = self.get_ast_loc(node)
        type_ = ts.void_t
        cond = self.visit(node.test)
        then_body = [self.visit(stmt) for stmt in node.body]
        else_body = [self.visit(stmt) for stmt in node.orelse]
        return hlast.IfStatement(loc, type_, cond, then_body, else_body)

    def visit_For(self, node: ast.For) -> hlast.ForStatement:
        loc = self.get_ast_loc(node)

        def as_index(value: hlast.Expr):
            return hlast.Cast(loc, ts.index_t, value, ts.index_t)

        if node.orelse:
            raise CompilationError(loc, "else body for loops is not supported")
        if not isinstance(node.target, ast.Name):
            raise CompilationError(self.get_ast_loc(node.target), "expected an identifier")
        if (
                not isinstance(node.iter, ast.Call)
                or not isinstance(node.iter.func, ast.Name)
                or not node.iter.func.id == "range"
        ):
            raise CompilationError(self.get_ast_loc(node.iter), "expected a call to `range`")

        type_ = ts.void_t
        range = node.iter.args
        c0 = hlast.Constant(loc, ts.index_t, 0)
        c1 = hlast.Constant(loc, ts.index_t, 1)
        if len(range) == 1:
            start, stop, step = c0, self.visit(range[0]), c1
        elif len(range) == 2:
            start, stop, step = self.visit(range[0]), self.visit(range[1]), c1
        elif len(range) == 3:
            start, stop, step = self.visit(range[0]), self.visit(range[1]), self.visit(range[2])
        else:
            raise CompilationError(self.get_ast_loc(node.iter), "range function expects 1, 2, or 3 arguments")
        loop_var = hlast.Parameter(self.get_ast_loc(node.target), ts.index_t, node.target.id)
        self.symtable.assign(loop_var.name, loop_var)
        body = [self.visit(stmt) for stmt in node.body]
        return hlast.ForStatement(loc, type_, as_index(start), as_index(stop), as_index(step), loop_var, body)

    #-----------------------------------
    # Symbols
    #-----------------------------------

    def visit_Name(self, node: ast.Name) -> hlast.SymbolRef:
        assert isinstance(node.ctx, ast.Load) # Store contexts are handled explicitly in the parent node.

        loc = self.get_ast_loc(node)
        name = node.id
        entry = self.symtable.lookup(name)
        if not entry:
            raise UndefinedSymbolError(loc, name)
        if isinstance(entry, hlast.ClosureVariable):
            return self._process_closure_value(entry.value, loc)
        if isinstance(entry, hlast.Node):
            return hlast.SymbolRef(loc, entry.type_, name)
        return entry

    def visit_Assign(self, node: ast.Assign) -> hlast.Statement:
        loc = self.get_ast_loc(node)
        value = self.visit(node.value)
        if not all(isinstance(target, ast.Name) for target in node.targets):
            raise CompilationError(loc, "only assigning to simple variables is supported")

        names = [target.id for target in node.targets]
        self.symtable.assign(names[0], value)
        if isinstance(value, hlast.Node):
            return hlast.Assign(loc, ts.void_t, names, value)
        else:
            return hlast.Noop(loc, ts.void_t)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> hlast.Statement:
        loc = self.get_ast_loc(node)
        value = self.visit(node.value)
        if not isinstance(node.target, ast.Name):
            raise CompilationError(loc, "only assigning to simple variables is supported")

        name = node.target.id
        self.symtable.assign(name, value.type_)
        if isinstance(value, hlast.Node):
            return hlast.Assign(loc, ts.VoidType(), [name], value)
        else:
            return hlast.Noop(loc, ts.void_t)

    #-----------------------------------
    # Arithmetic & logic
    #-----------------------------------

    def visit_Constant(self, node: ast.Constant) -> hlast.Constant:
        loc = self.get_ast_loc(node)
        try:
            type_ = type_traits.from_object(node.value)
            value = node.value
            return hlast.Constant(loc, type_, value)
        except Exception as ex:
            raise CompilationError(loc, f"invalid constant expression: {node.value}") from ex

    def visit_BinOp(self, node: ast.BinOp) -> hlast.Expr:
        loc = self.get_ast_loc(node)
        lhs = ensure_expr(self.visit(node.left), loc=loc)
        rhs = ensure_expr(self.visit(node.right), loc=loc)
        func = self.visit(node.op)

        def builder(args: list[hlast.Expr]) -> hlast.Expr:
            arg_types = [arg.type_ for arg in args]
            type_ = type_traits.common_type(*arg_types)
            if not type_:
                raise ArgumentCompatibilityError(loc, f"arithmetic:{str(func)}", arg_types)
            lhs = hlast.Cast(loc, type_, args[0], type_)
            rhs = hlast.Cast(loc, type_, args[1], type_)
            return hlast.ArithmeticOperation(loc, type_, lhs, rhs, func)

        arg_types = [lhs.type_, rhs.type_]
        element_types = [type_traits.element_type(ty) for ty in arg_types]
        common_ty = type_traits.common_type(*element_types)
        common_dims = type_traits.common_dims(*arg_types)

        if not common_ty:
            raise ArgumentCompatibilityError(loc, f"arithmetic:{str(func)}", arg_types)

        if common_dims:
            type_ = ts.FieldType(common_ty, common_dims)
            return hlast.ElementwiseOperation(loc, type_, [lhs, rhs], builder)
        return builder([lhs, rhs])

    def visit_Compare(self, node: ast.Compare) -> Any:
        loc = self.get_ast_loc(node)
        operands = [ensure_expr(self.visit(operand), loc=loc) for operand in [node.left, *node.comparators]]
        funcs = [self.visit(op) for op in node.ops]

        arg_types = [ensure_type(arg, (ts.FieldType, ts.NumberType)) for arg in operands]
        common_dims = type_traits.common_dims(*arg_types)
        result_ty = ts.bool_t

        def builder(args: list[hlast.Expr]) -> hlast.Expr:
            args_names = [f"__cmp_operand{i}" for i in range(len(args))]
            args_tuple = hlast.TupleCreate(loc, ts.TupleType([arg.type_ for arg in args]), args)
            args_assign = hlast.Assign(loc, ts.VoidType(), args_names, args_tuple)
            args_refs = [hlast.SymbolRef(loc, v.type_, name) for v, name in zip(args, args_names)]

            c_true = hlast.Constant(loc, ts.bool_t, 1)
            c_false = hlast.Constant(loc, ts.bool_t, 0)
            expr = c_true
            for i in range(len(funcs) - 1, -1, -1):
                func = funcs[i]
                lhs = args_refs[i]
                rhs = args_refs[i + 1]
                common_ty = type_traits.common_type(lhs.type_, rhs.type_)
                if not common_ty:
                    raise ArgumentCompatibilityError(loc, f"comparison.{str(func)}", [lhs.type_, rhs.type_])
                lhs = hlast.Cast(loc, common_ty, lhs, common_ty)
                rhs = hlast.Cast(loc, common_ty, rhs, common_ty)
                cmp = hlast.ComparisonOperation(loc, ts.bool_t, lhs, rhs, func)
                expr = hlast.If(
                    loc, ts.bool_t, cmp,
                    [hlast.Yield(loc, ts.bool_t, expr)],
                    [hlast.Yield(loc, ts.bool_t, c_false)]
                )
            return hlast.If(
                loc, ts.bool_t, c_true,
                [args_assign, hlast.Yield(loc, ts.bool_t, expr)],
                [hlast.Yield(loc, ts.bool_t, c_false)]
            )

        if common_dims:
            type_ = ts.FieldType(result_ty, common_dims)
            return hlast.ElementwiseOperation(loc, type_, operands, builder)
        else:
            return builder(operands)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> hlast.Expr:
        loc = self.get_ast_loc(node)
        value = ensure_expr(self.visit(node.operand), loc=loc)
        value_type = ensure_type(value, (ts.FieldType, ts.NumberType))
        element_type = type_traits.element_type(value_type)

        def builder_plus(values: list[hlast.Expr]):
            return values[0]

        def builder_minus(values: list[hlast.Expr]):
            value = values[0]
            if isinstance(element_type, ts.IntegerType) and element_type.width == 1:
                raise ArgumentCompatibilityError(loc, "unary minus", [element_type])
            if isinstance(value, hlast.Constant):
                return hlast.Constant(value.location, value.type_, -value.value)
            sign_value = -1.0 if isinstance(element_type, ts.FloatType) else -1
            sign_expr = hlast.Constant(loc, element_type, sign_value)
            return hlast.ArithmeticOperation(loc, element_type, sign_expr, value, hlast.ArithmeticFunction.MUL)

        def builder_invert(values: list[hlast.Expr]):
            value = values[0]
            if not isinstance(element_type, (ts.IndexType, ts.IntegerType)):
                raise ArgumentCompatibilityError(loc, "bit inversion", [element_type])
            if isinstance(value, hlast.Constant):
                return hlast.Constant(value.location, value.type_, ~value.value)
            c_bitmask = hlast.Constant(loc, element_type, -1)
            return hlast.ArithmeticOperation(loc, element_type, c_bitmask, value, hlast.ArithmeticFunction.BIT_XOR)

        def builder_not(values: list[hlast.Expr]):
            value = values[0]
            if isinstance(value, hlast.Constant):
                return hlast.Constant(value.location, ts.bool_t, not value.value)
            compare_value = 0.0 if isinstance(element_type, ts.FloatType) else 0
            compare_expr = hlast.Constant(loc, element_type, compare_value)
            return hlast.ComparisonOperation(loc, ts.bool_t, value, compare_expr, hlast.ComparisonFunction.EQ)

        if isinstance(node.op, ast.UAdd):
            builder = builder_plus
            result_ty = element_type
        elif isinstance(node.op, ast.USub):
            builder = builder_minus
            result_ty = element_type
        elif isinstance(node.op, ast.Invert):
            builder = builder_invert
            result_ty = element_type
        elif isinstance(node.op, ast.Not):
            builder = builder_not
            result_ty = ts.bool_t
        else:
            raise CompilationError(loc, f"unary operator {type(node.op)} not implemented")

        if isinstance(value_type, ts.FieldType):
            ty = ts.FieldType(result_ty, list(value_type.dimensions))
            return hlast.ElementwiseOperation(loc, ty, [value], builder)
        return builder([value])

    def visit_BoolOp(self, node: ast.BoolOp) -> hlast.Expr:
        loc = self.get_ast_loc(node)
        values = [ensure_expr(self.visit(value)) for value in node.values]
        is_and = isinstance(node.op, ast.And)
        c_false = hlast.Constant(loc, ts.bool_t, False)
        c_true = hlast.Constant(loc, ts.bool_t, True)
        expr = c_true if is_and else c_false

        for value in reversed(values):
            if value.type_ != ts.bool_t:
                raise CompilationError(loc, "operands to boolean logical operators must be booleans")
            expr = hlast.If(
                loc,
                ts.bool_t,
                value,
                [hlast.Yield(loc, ts.void_t, expr) if is_and else hlast.Yield(loc, ts.void_t, c_true)],
                [hlast.Yield(loc, ts.void_t, c_false) if is_and else hlast.Yield(loc, ts.void_t, expr)]
            )
        return expr

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

    def _visit_subscript_size(self, node: ast.Subscript) -> Optional[hlast.Expr]:
        loc = self.get_ast_loc(node)
        value = self.visit(node.value)
        if not isinstance(value, concepts.Dimension):
            return None
        if isinstance(node.slice, ast.Slice):
            return None
        size = self.visit(node.slice)
        return hlast.Size(loc, size.type_, value, size)

    def _visit_subscript_slice(self, node: ast.Subscript) -> Optional[hlast.Expr]:
        loc = self.get_ast_loc(node)
        value = self.visit(node.value)
        if not isinstance(value, concepts.Dimension):
            return None
        if not isinstance(node.slice, ast.Slice):
            return None
        start = ensure_expr(self.visit(node.slice.lower), loc=loc) if node.slice.lower else None
        stop = ensure_expr(self.visit(node.slice.upper), loc=loc) if node.slice.upper else None
        step = ensure_expr(self.visit(node.slice.step), loc=loc) if node.slice.step else None
        return hlast.Slice(loc, ts.index_t, value, start, stop, step)

    def _visit_subscript_tuple_extract(self, node: ast.Subscript):
        loc = self.get_ast_loc(node)
        value = self.visit(node.value)
        if not isinstance(value, hlast.Expr):
            return None
        if not isinstance(value.type_, ts.TupleType):
            return None
        index_node = self.visit(node.slice)
        if not isinstance(index_node, hlast.Constant) or not isinstance(index_node.type_, (ts.IndexType, ts.IntegerType)):
            raise CompilationError(loc, "tuples can only be subscripted with a constant integral")
        index_value = index_node.value
        if not 0 <= index_value < len(value.type_.elements):
            raise CompilationError(loc, "index out of bounds")
        return hlast.TupleExtract(loc, value.type_.elements[index_value], value, index_value)

    def _visit_subscript_sample(self, node: ast.Subscript):
        loc = self.get_ast_loc(node)
        source = self.visit(node.value)
        if not isinstance(source, hlast.Expr):
            return None
        if not isinstance(source.type_, ts.FieldLikeType):
            return None
        index = ensure_expr(self.visit(node.slice))
        if not isinstance(index.type_, ts.NDIndexType):
            return None
        for dim in source.type_.dimensions:
            if dim not in index.type_.dims:
                raise CompilationError(loc, f"dimension `{dim}` is present in the source, but not in the index")
        return hlast.Sample(loc, source.type_.element_type, source, index)

    def _visit_subscript_extract_slice(self, node: ast.Subscript):
        loc = self.get_ast_loc(node)
        source = self.visit(node.value)
        if not isinstance(source, hlast.Expr):
            return None
        if not isinstance(source.type_, ts.FieldLikeType):
            return None
        slices = ensure_expr(self.visit(node.slice))
        if isinstance(slices, hlast.TupleCreate):
            slice_list = slices.elements
        else:
            slice_list = [slices]
        promoted_slice_list: list[hlast.Slice] = []
        for slc in slice_list:
            if isinstance(slc, hlast.Size):
                slc = self._promote_to_slice(slc)
            if not isinstance(slc, hlast.Slice):
                return None
            promoted_slice_list.append(slc)
        return hlast.ExtractSlice(loc, source.type_, source, promoted_slice_list)

    def _visit_subscript_extract_index(self, node: ast.Subscript):
        loc = self.get_ast_loc(node)
        index = self.visit(node.value)
        if not isinstance(index, hlast.Expr):
            return None
        if not isinstance(index.type_, ts.NDIndexType):
            return None
        dim = self.visit(node.slice)
        if not isinstance(dim, concepts.Dimension):
            return None
        return hlast.Extract(loc, ts.index_t, index, dim)

    def _visit_subscript_jump(self, node: ast.Subscript):
        loc = self.get_ast_loc(node)
        index = self.visit(node.value)
        if not isinstance(index, hlast.Expr):
            return None
        if not isinstance(index.type_, ts.NDIndexType):
            return None
        offset = self.visit(node.slice)
        if isinstance(offset, hlast.TupleCreate):
            offset_list = offset.elements
        else:
            offset_list = [offset]
        if not all(isinstance(o, hlast.Size) for o in offset_list):
            return None
        offset_dict: dict[concepts.Dimension, int] = {}
        for o in offset_list:
            assert isinstance(o, hlast.Size)
            if not isinstance(o.size, hlast.Constant) or not isinstance(o.size.type_, (ts.IndexType, ts.IntegerType)):
                raise CompilationError(o.size.location, "expected integer literal expression")
            offset_dict[o.dimension] = o.size.value
        return hlast.Jump(loc, index.type_, index, offset_dict)

    def _visit_subscript_connect(self, node: ast.Subscript):
        loc = self.get_ast_loc(node)
        index = self.visit(node.value)
        if not isinstance(index, hlast.Expr):
            return None
        if not isinstance(index.type_, ts.NDIndexType):
            return None
        slc = self.visit(node.slice)
        if not isinstance(slc, hlast.TupleCreate):
            return None
        connectivity = slc.elements[0]
        element = slc.elements[1]
        if not isinstance(connectivity, hlast.Expr):
            return None
        if not isinstance(connectivity.type_, ts.ConnectivityType):
            return None
        origin = connectivity.type_.origin_dimension
        neighbor = connectivity.type_.neighbor_dimension
        local = connectivity.type_.element_dimension

        local_dims = sorted([*copy.deepcopy(index.type_.dims), local])
        local_type = ts.NDIndexType(local_dims)

        new_dims = copy.deepcopy(index.type_.dims)
        if not origin in new_dims:
            raise CompilationError(loc, f"origin dimension {origin} not found in ND index `{index.type_}`")
        new_dims.remove(origin)
        new_dims.append(neighbor)
        new_dims = sorted(new_dims)
        type_ = ts.NDIndexType(new_dims)

        extended = hlast.Extend(loc, local_type, index, element, local)
        value = hlast.Sample(loc, connectivity.type_.element_type, connectivity, extended)
        return hlast.Exchange(loc, type_, index, value, origin, neighbor)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        expr = self._visit_subscript_size(node)
        if expr: return expr
        expr = self._visit_subscript_slice(node)
        if expr: return expr
        expr = self._visit_subscript_tuple_extract(node)
        if expr: return expr
        expr = self._visit_subscript_sample(node)
        if expr: return expr
        expr = self._visit_subscript_extract_slice(node)
        if expr: return expr
        expr = self._visit_subscript_extract_index(node)
        if expr: return expr
        expr = self._visit_subscript_jump(node)
        if expr: return expr
        expr = self._visit_subscript_connect(node)
        if expr: return expr

        loc = self.get_ast_loc(node)
        raise CompilationError(loc, "invalid subscript")

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

    def visit_Pass(self, node: ast.Pass) -> hlast.Noop:
        loc = self.get_ast_loc(node)
        return hlast.Noop(loc, ts.void_t)

    def visit_Tuple(self, node: ast.Tuple) -> hlast.TupleCreate:
        loc = self.get_ast_loc(node)
        elements = [ensure_expr(self.visit(e), loc=loc) for e in node.elts]
        type_ = ts.TupleType([e.type_ for e in elements])
        return hlast.TupleCreate(loc, type_, elements)

    #-----------------------------------
    # Utilities
    #-----------------------------------
    def instantiate(
            self,
            definition: Callable,
            arg_types: Sequence[ts.Type],
            is_public: bool,
            dims: Optional[Sequence[concepts.Dimension]] = None
    ) -> hlast.Function | hlast.Stencil:
        mangled_name = utility.mangle_name(utility.get_qualified_name(definition), arg_types, dims)
        func = self._get_instantiated_function(mangled_name)
        if func:
            return func

        def sc():
            source_code, file, start_line, start_col = get_source_code(definition)
            self.base_location = concepts.Location(file, start_line, start_col)
            python_ast = ast.parse(source_code)

            assert isinstance(python_ast, ast.Module)
            assert len(python_ast.body) == 1
            assert isinstance(python_ast.body[0], ast.FunctionDef)

            closure_vars = inspect.getclosurevars(definition)
            add_closure_vars_to_symtable(self.symtable, closure_vars.globals)
            add_closure_vars_to_symtable(self.symtable, closure_vars.nonlocals)

            spec = FunctionSpecification(mangled_name, list(arg_types), {}, is_public, dims)
            instantiation = self.visit_FunctionDef(python_ast.body[0], spec=spec)
            return instantiation

        enclosing_symtable = self.symtable
        self.symtable = SymbolTable()
        instantiation = self.symtable.scope(sc)
        self.instantiations[instantiation.name] = instantiation
        self.symtable = enclosing_symtable
        return instantiation
    
    def _get_enclosing_function(self, mangled_name: Optional[str] = None) -> Optional[FunctionSpecification]:
        for info in self.symtable.infos():
            if isinstance(info, FunctionSpecification):
                if mangled_name and info.mangled_name == mangled_name:
                    return info
                if not mangled_name:
                    return info
        return None

    def _get_instantiated_function(self, mangled_name: str) -> Optional[hlast.Function | hlast.Stencil]:
        if mangled_name in self.instantiations:
            return self.instantiations[mangled_name]
        return None

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
            type_ = type_traits.from_object(value)
            if isinstance(type_, (ts.IndexType, ts.IntegerType, ts.FloatType)):
                return hlast.Constant(location, type_, value)
        except Exception:
            pass
        raise CompilationError(location, f"closure variable of type `{type(value)}` is not understood")

    def _promote_to_slice(self, size: hlast.Size) -> hlast.Slice:
        lower = size.size
        return hlast.Slice(size.location, ts.index_t, size.dimension, lower, None, None, True)