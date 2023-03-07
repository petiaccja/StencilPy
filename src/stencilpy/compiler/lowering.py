import itertools

import stencilir as sir
from stencilpy.compiler import hlast
from stencilpy import utility
from .node_transformer import NodeTransformer
from stencilpy.compiler import types as ts
from stencilpy.error import *
from stencilpy import concepts
from collections.abc import Mapping


def as_sir_loc(loc: hlast.Location) -> sir.Location:
    return sir.Location(loc.file, loc.line, loc.column)


def as_sir_type(type_: ts.Type) -> sir.Type:
    if isinstance(type_, ts.IndexType):
        return sir.IndexType()
    elif isinstance(type_, ts.IntegerType):
        return sir.IntegerType(type_.width, type_.signed)
    elif isinstance(type_, ts.FloatType):
        return sir.FloatType(type_.width)
    elif isinstance(type_, ts.FieldType):
        return sir.FieldType(as_sir_type(type_.element_type), len(type_.dimensions))
    else:
        raise ValueError(f"no SIR type equivalent for {type_.__class__}")


def as_sir_arithmetic(func: hlast.ArithmeticFunction) -> sir.ArithmeticFunction:
    _MAPPING = {
        hlast.ArithmeticFunction.ADD: sir.ArithmeticFunction.ADD,
        hlast.ArithmeticFunction.SUB: sir.ArithmeticFunction.SUB,
        hlast.ArithmeticFunction.MUL: sir.ArithmeticFunction.MUL,
        hlast.ArithmeticFunction.DIV: sir.ArithmeticFunction.DIV,
        hlast.ArithmeticFunction.MOD: sir.ArithmeticFunction.MOD,
        hlast.ArithmeticFunction.BIT_AND: sir.ArithmeticFunction.BIT_AND,
        hlast.ArithmeticFunction.BIT_OR: sir.ArithmeticFunction.BIT_OR,
        hlast.ArithmeticFunction.BIT_XOR: sir.ArithmeticFunction.BIT_XOR,
        hlast.ArithmeticFunction.BIT_SHL: sir.ArithmeticFunction.BIT_SHL,
        hlast.ArithmeticFunction.BIT_SHR: sir.ArithmeticFunction.BIT_SHR,
    }
    return _MAPPING[func]


def as_sir_comparison(func: hlast.ComparisonFunction) -> sir.ComparisonFunction:
    _MAPPING = {
        hlast.ComparisonFunction.EQ: sir.ComparisonFunction.EQ,
        hlast.ComparisonFunction.NEQ: sir.ComparisonFunction.NEQ,
        hlast.ComparisonFunction.LT: sir.ComparisonFunction.LT,
        hlast.ComparisonFunction.GT: sir.ComparisonFunction.GT,
        hlast.ComparisonFunction.LTE: sir.ComparisonFunction.LTE,
        hlast.ComparisonFunction.GTE: sir.ComparisonFunction.GTE,
    }
    return _MAPPING[func]


def get_dim_index(dimensions: list[concepts.Dimension], dim: concepts.Dimension):
    try:
        return dimensions.index(dim)
    except ValueError:
        raise KeyError(f"dimension {dim} is not associated with field")


def elementwise_dims_to_arg(
        dimensions: list[concepts.Dimension],
        arg_types: list[ts.Type]
) -> dict[concepts.Dimension, int]:
    dims_to_arg: dict[concepts.Dimension, int] = {}
    for dim in dimensions:
        for arg_idx, type_ in enumerate(arg_types):
            if not isinstance(type_, ts.FieldType):
                continue
            arg_dims = type_.dimensions
            if not dim in arg_dims:
                continue
            dims_to_arg[dim] = arg_idx
    return dims_to_arg


class ShapeFunctionPass(NodeTransformer):
    @staticmethod
    def shape_var(name: str, dim: concepts.Dimension):
        return f"__shape_{dim.id}_{name}"

    @staticmethod
    def shape_func(name: str):
        return f"__shapes_{name}"

    def visit_Module(self, node: hlast.Module) -> hlast.Module:
        shape_funcs = [self.visit(func) for func in node.functions]

        funcs = [
            self._normalize_lower(),  # Helper functions
            self._normalize_upper(),
            self._slice_size(),
            *shape_funcs,  # Shape functions added
            *node.functions  # Original functions kept without change
        ]
        stencils = node.stencils  # Stencils are unchanged
        return hlast.Module(node.location, node.type_, funcs, stencils)

    def visit_Function(self, node: hlast.Function) -> hlast.Function:
        param_dims = []
        for param in node.parameters:
            ref = hlast.SymbolRef(node.location, param.type_, param.name)
            if isinstance(param.type_, ts.FieldType):
                for dim in param.type_.dimensions:
                    var = self.shape_var(param.name, dim)
                    size = hlast.Shape(node.location, ts.IndexType(), ref, dim)
                    assign = hlast.Assign(node.location, ts.VoidType(), [var], [size])
                    param_dims.append(assign)

        statements = [self.visit(statement) for statement in node.body]
        body = [*param_dims, *statements]

        results: list[ts.Type] = []
        for statement in body:
            if isinstance(statement, hlast.Return):
                results = [expr.type_ for expr in statement.values]

        type_ = ts.FunctionType([p.type_ for p in node.parameters], results)
        name = self.shape_func(node.name)

        return hlast.Function(node.location, type_, name, node.parameters, results, body)

    def visit_Return(self, node: hlast.Return) -> hlast.Return:
        values = [self.visit(value) for value in node.values]
        shapes = [value for value in values if isinstance(value, dict)]
        sorted_shapes = [sorted(shape.items(), key=lambda x: x[0]) for shape in shapes]
        dimless_shapes = [map(lambda x: x[1], shape) for shape in sorted_shapes]
        flattened = list(itertools.chain(*dimless_shapes))
        return hlast.Return(node.location, ts.VoidType(), flattened)

    def visit_Constant(self, node: hlast.Constant) -> hlast.Expr:
        return node

    def visit_SymbolRef(self, node: hlast.SymbolRef) -> hlast.Expr | dict[concepts.Dimension, hlast.Expr]:
        if isinstance(node.type_, ts.FieldType):
            sizes = {
                dim: hlast.SymbolRef(node.location, ts.IndexType(), self.shape_var(node.name, dim))
                for dim in node.type_.dimensions
            }
            return sizes
        return node

    def visit_Assign(self, node: hlast.Assign) -> hlast.Assign:
        raw_values = [self.visit(value) for value in node.values]
        names = []
        values = []
        for raw_name, raw_value in zip(node.names, raw_values):
            if isinstance(raw_value, Mapping):
                for dim, shape in raw_value.items():
                    names.append(self.shape_var(raw_name, dim))
                    values.append(shape)
            else:
                names.append(raw_name)
                values.append(raw_value)

        return hlast.Assign(node.location, node.type_, names, values)

    def visit_Shape(self, node: hlast.Shape) -> hlast.Expr:
        shape = self.visit(node.field)
        return shape[node.dim]

    def visit_Apply(self, node: hlast.Apply) -> dict[concepts.Dimension, hlast.Expr]:
        shape = {dim: self.visit(size) for dim, size in node.shape.items()}
        return shape

    def visit_ArithmeticOperation(self, node: hlast.ArithmeticOperation):
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        return hlast.ArithmeticOperation(node.location, node.type_, lhs, rhs, node.func)

    def visit_ComparisonOperation(self, node: hlast.ComparisonOperation):
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        return hlast.ComparisonOperation(node.location, node.type_, lhs, rhs, node.func)

    def visit_ElementwiseOperation(self, node: hlast.ElementwiseOperation) -> dict[concepts.Dimension, hlast.Expr]:
        assert isinstance(node.type_, ts.FieldType)
        dimensions = node.type_.dimensions
        arg_types = [arg.type_ for arg in node.args]
        arg_shapes = [self.visit(arg) for arg in node.args]
        dims_to_arg = elementwise_dims_to_arg(dimensions, arg_types)
        shape = {
            dim: arg_shapes[arg_idx][dim]
            for dim, arg_idx in dims_to_arg.items()
        }
        return shape

    def visit_If(self, node: hlast.If) -> hlast.If:
        loc = node.location
        type_ = node.type_

        if isinstance(node.type_, ts.FieldType):
            raise CompilationError(loc, "yielding field expressions are not supported from ifs")

        cond = self.visit(node.cond)
        then_body = [self.visit(statement) for statement in node.then_body]
        else_body = [self.visit(statement) for statement in node.else_body]
        return hlast.If(loc, type_, cond, then_body, else_body)

    def visit_Yield(self, node: hlast.Yield):
        loc = node.location
        type_ = node.type_
        values = [self.visit(value) for value in node.values]
        return hlast.Yield(loc, type_, values)

    def visit_ExtractSlice(self, node: hlast.ExtractSlice) -> dict[concepts.Dimension, hlast.Expr]:
        loc = node.location
        source_shape: dict[concepts.Dimension, hlast.Expr] = self.visit(node.source)
        slices = {slc.dimension: self._visit_slice(slc) for slc in node.slices}
        index_t = ts.IndexType()
        def as_index(expr: hlast.Expr):
            return hlast.Cast(loc, index_t, expr, index_t)
        shape = {
            dim: hlast.Call(
                loc,
                size.type_,
                "__slice_size",
                [
                    as_index(slices[dim].lower),
                    as_index(slices[dim].upper) if slices[dim].upper else as_index(size),
                    as_index(slices[dim].step),
                    as_index(size)]
            )
            for dim, size in source_shape.items()
        }
        return shape

    def visit_Cast(self, node: hlast.Cast) -> hlast.Cast:
        value = self.visit(node.value)
        return hlast.Cast(node.location, node.type_, value, node.type_)

    def _visit_slice(self, slc: hlast.Slice):
        return hlast.Slice(
            slc.dimension,
            self.visit(slc.lower),
            self.visit(slc.upper) if slc.upper else None,
            self.visit(slc.step)
        )

    def _normalize_lower(self) -> hlast.Function:
        index_type = ts.IndexType()
        bool_type = ts.IntegerType(1, True)

        loc = concepts.Location.unknown()
        name = "__normalize_lower"
        parameters = [
            hlast.Parameter("lower", index_type),
            hlast.Parameter("size", index_type),
        ]
        results = [index_type]
        type_ = ts.FunctionType([p.type_ for p in parameters], results)

        lower = hlast.SymbolRef(loc, index_type, "lower")
        size = hlast.SymbolRef(loc, index_type, "size")
        c_0 = hlast.Constant(loc, ts.IndexType(), 0)
        c_1 = hlast.Constant(loc, ts.IndexType(), 1)
        is_negative = hlast.ComparisonOperation(loc, bool_type, lower, c_0, hlast.ComparisonFunction.LT)
        correction = hlast.If(
            loc,
            index_type,
            is_negative,
            [hlast.Yield(loc, ts.VoidType(), [c_1])],
            [hlast.Yield(loc, ts.VoidType(), [c_0])],
        )
        count = hlast.ArithmeticOperation(loc, index_type, lower, size, hlast.ArithmeticFunction.DIV)
        corr_count = hlast.ArithmeticOperation(loc, index_type, count, correction, hlast.ArithmeticFunction.SUB)
        prod = hlast.ArithmeticOperation(loc, index_type, corr_count, size, hlast.ArithmeticFunction.MUL)
        rem = hlast.ArithmeticOperation(loc, index_type, lower, prod, hlast.ArithmeticFunction.SUB)
        body = [hlast.Return(loc, ts.VoidType(), [rem])]
        return hlast.Function(loc, type_, name, parameters, results, body)

    def _normalize_upper(self) -> hlast.Function:
        index_type = ts.IndexType()
        bool_type = ts.IntegerType(1, True)

        loc = concepts.Location.unknown()
        name = "__normalize_upper"
        parameters = [
            hlast.Parameter("upper", index_type),
            hlast.Parameter("size", index_type),
        ]
        results = [index_type]
        type_ = ts.FunctionType([p.type_ for p in parameters], results)

        upper = hlast.SymbolRef(loc, index_type, "upper")
        size = hlast.SymbolRef(loc, index_type, "size")
        c_0 = hlast.Constant(loc, index_type, 0)
        rem_expr = hlast.Call(loc, index_type, "__normalize_lower", [upper, size])
        rem_assign = hlast.Assign(loc, ts.VoidType(), ["rem"], [rem_expr])
        rem_ref = hlast.SymbolRef(loc, index_type, "rem")
        is_zero = hlast.ComparisonOperation(loc, bool_type, rem_ref, c_0, hlast.ComparisonFunction.EQ)
        rem_corr = hlast.If(
            loc,
            index_type,
            is_zero,
            [hlast.Yield(loc, ts.VoidType(), [size])],
            [hlast.Yield(loc, ts.VoidType(), [rem_ref])],
        )
        body = [rem_assign, hlast.Return(loc, ts.VoidType(), [rem_corr])]
        return hlast.Function(loc, type_, name, parameters, results, body)

    def _slice_size(self) -> hlast.Function:
        index_type = ts.IndexType()

        loc = concepts.Location.unknown()
        name = "__slice_size"
        parameters = [
            hlast.Parameter("lower", index_type),
            hlast.Parameter("upper", index_type),
            hlast.Parameter("step", index_type),
            hlast.Parameter("size", index_type),
        ]
        results = [index_type]
        type_ = ts.FunctionType([p.type_ for p in parameters], results)

        lower = hlast.SymbolRef(loc, index_type, "lower")
        upper = hlast.SymbolRef(loc, index_type, "upper")
        step = hlast.SymbolRef(loc, index_type, "step")
        size = hlast.SymbolRef(loc, index_type, "size")
        norm_lower = hlast.Call(loc, index_type, "__normalize_lower", [lower, size])
        norm_upper = hlast.Call(loc, index_type, "__normalize_upper", [upper, size])
        diff = hlast.ArithmeticOperation(loc, index_type, norm_upper, norm_lower, hlast.ArithmeticFunction.SUB)
        count = hlast.ArithmeticOperation(loc, index_type, diff, step, hlast.ArithmeticFunction.DIV)
        body = [hlast.Return(loc, ts.VoidType(), [count])]
        return hlast.Function(loc, type_, name, parameters, results, body)


class HlastToSirPass(NodeTransformer):
    immediate_stencils: list[sir.Stencil]

    def __init__(self):
        self.immediate_stencils = []

    @staticmethod
    def _out_param_name(idx: int):
        return f"__out_{idx}"

    def visit_Module(self, node: hlast.Module) -> sir.Module:
        loc = as_sir_loc(node.location)
        funcs = [self.visit(func) for func in node.functions]
        stencils = [self.visit(stencil) for stencil in node.stencils]
        all_stencils = [*self.immediate_stencils, *stencils]
        return sir.Module(funcs, all_stencils, loc)

    def visit_Function(self, node: hlast.Function) -> sir.Function:
        loc = as_sir_loc(node.location)
        name = node.name
        parameters = [sir.Parameter(p.name, as_sir_type(p.type_)) for p in node.parameters]
        out_parameters = [
            sir.Parameter(self._out_param_name(idx), as_sir_type(type_))
            for idx, type_ in enumerate(node.results)
            if isinstance(type_, ts.FieldType)
        ]
        results = [as_sir_type(type_) for type_ in node.results]
        statements = [self.visit(statement) for statement in node.body]
        return sir.Function(name, [*parameters, *out_parameters], results, statements, True, loc)

    def visit_Stencil(self, node: hlast.Stencil) -> sir.Stencil:
        loc = as_sir_loc(node.location)
        name = node.name
        parameters = [sir.Parameter(p.name, as_sir_type(p.type_)) for p in node.parameters]
        out_parameters = [
            sir.Parameter(self._out_param_name(idx), as_sir_type(type_))
            for idx, type_ in enumerate(node.results)
            if isinstance(type_, ts.FieldType)
        ]
        results = [as_sir_type(type_) for type_ in node.results]
        statements = [self.visit(statement) for statement in node.body]
        ndims = len(node.dims)
        return sir.Stencil(name, [*parameters, *out_parameters], results, statements, ndims, True, loc)

    def visit_Return(self, node: hlast.Return) -> sir.Return:
        loc = as_sir_loc(node.location)
        values = []
        for idx, value in enumerate(node.values):
            if isinstance(value.type_, ts.FieldType):
                ndims = len(value.type_.dimensions)

                source = self.visit(value)
                source_assign = sir.Assign([f"__value_{idx}"], [source], loc)
                source_ref = sir.SymbolRef(f"__value_{idx}", loc)

                dest = sir.SymbolRef(self._out_param_name(idx), loc)
                offsets = [sir.Constant.index(0, loc) for _ in range(ndims)]
                sizes = [sir.Dim(source_ref, sir.Constant.index(i, loc), loc) for i in range(ndims)]
                strides = [sir.Constant.index(1, loc) for _ in range(ndims)]
                insert = sir.InsertSlice(source_ref, dest, offsets, sizes, strides, loc)
                block_yield = sir.Yield([insert], loc)

                block = sir.Block([source_assign, block_yield], loc)
                values.append(block)
            else:
                values.append(self.visit(value))

        return sir.Return(values, loc)

    def visit_Constant(self, node: hlast.Constant) -> sir.Constant:
        loc = as_sir_loc(node.location)
        type_ = as_sir_type(node.type_)
        value = node.value
        if isinstance(type_, sir.IndexType):
            return sir.Constant.index(value, loc)
        elif isinstance(type_, sir.IntegerType):
            return sir.Constant.integral(value, type_, loc)
        elif isinstance(type_, sir.FloatType):
            return sir.Constant.floating(value, type_, loc)
        raise NotImplementedError()

    def visit_SymbolRef(self, node: hlast.SymbolRef) -> sir.SymbolRef:
        loc = as_sir_loc(node.location)
        name = node.name
        return sir.SymbolRef(name, loc)

    def visit_Assign(self, node: hlast.Assign) -> sir.SymbolRef:
        loc = as_sir_loc(node.location)
        values = [self.visit(value) for value in node.values]
        return sir.Assign(node.names, values, loc)

    def visit_Shape(self, node: hlast.Shape) -> sir.Dim:
        loc = as_sir_loc(node.location)
        if not isinstance(node.field.type_, ts.FieldType):
            raise CompilationError(node.field.location, f"shape expects a field, got {node.field.type_}")
        try:
            idx_val = get_dim_index(node.field.type_.dimensions, node.dim)
        except KeyError:
            raise MissingDimensionError(node.location, node.field.type_, node.dim)
        field = self.visit(node.field)
        idx = sir.Constant.index(idx_val, loc)
        return sir.Dim(field, idx, loc)

    def visit_Index(self, node: hlast.Index) -> sir.Dim:
        loc = as_sir_loc(node.location)
        return sir.Index(loc)

    def visit_Sample(self, node: hlast.Sample) -> sir.Sample:
        loc = as_sir_loc(node.location)
        assert isinstance(node.field.type_, ts.FieldType)
        assert isinstance(node.index.type_, ts.NDIndexType)
        try:
            projection = [get_dim_index(node.index.type_.dims, dim) for dim in node.field.type_.dimensions]
            field = self.visit(node.field)
            index = self.visit(node.index)
            if sorted(projection) != [range(len(node.field.type_.dimensions))]:
                index = sir.Project(index, projection, loc)
            return sir.Sample(field, index, loc)
        except KeyError:
            raise CompilationError(
                loc,
                f"cannot sample field of type {node.field.type_} with index of type {node.index.type_}"
            )

    def visit_Call(self, node: hlast.Call) -> sir.Call:
        loc = as_sir_loc(node.location)
        name = node.name
        args = [self.visit(arg) for arg in node.args]
        return sir.Call(name, args, loc)

    def visit_Apply(self, node: hlast.Apply) -> sir.Apply:
        loc = as_sir_loc(node.location)
        callee = node.stencil.name
        inputs = [self.visit(arg) for arg in node.args]
        shape = [(dim, self.visit(size)) for dim, size in node.shape.items()]
        shape = sorted(shape, key=lambda v: v[0])
        shape = [size[1] for size in shape]
        shape = [sir.Cast(size, sir.IndexType(), loc) for size in shape]
        assert isinstance(node.type_, ts.FieldType)
        dtype = as_sir_type(node.type_.element_type)
        outputs = [sir.AllocTensor(dtype, shape, loc)]
        return sir.Apply(callee, inputs, outputs, [], [0]*len(shape), loc)

    def visit_ArithmeticOperation(self, node: hlast.ArithmeticOperation):
        loc = as_sir_loc(node.location)
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        func = as_sir_arithmetic(node.func)
        return sir.ArithmeticOperator(lhs, rhs, func, loc)

    def visit_ComparisonOperation(self, node: hlast.ComparisonOperation):
        loc = as_sir_loc(node.location)
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        func = as_sir_comparison(node.func)
        return sir.ComparisonOperator(lhs, rhs, func, loc)

    def visit_ElementwiseOperation(self, node: hlast.ElementwiseOperation):
        loc = as_sir_loc(node.location)
        args = [self.visit(arg) for arg in node.args]
        names = [f"__elemwise_arg{i}" for i in range(len(args))]
        arg_assign = sir.Assign(names, args, loc)
        arg_refs = [sir.SymbolRef(name, loc) for name in names]
        assert isinstance(node.type_, ts.FieldType)

        dims_to_arg = elementwise_dims_to_arg(node.type_.dimensions, [arg.type_ for arg in node.args])
        shape = [
            sir.Dim(
                arg_refs[arg_idx],
                sir.Constant.index(get_dim_index(node.args[arg_idx].type_.dimensions, dim), loc),
                loc
            )
            for dim, arg_idx in dims_to_arg.items()
        ]

        dtype = as_sir_type(node.type_.element_type)
        output = sir.AllocTensor(dtype, shape, loc)
        stencil = self._elementwise_stencil(node)
        self.immediate_stencils.append(self.visit(stencil))
        apply = sir.Apply(stencil.name, args, [output], [], [0]*len(shape), loc)
        return sir.Block([arg_assign, sir.Yield([apply], loc)], loc)

    def visit_If(self, node: hlast.If) -> sir.If:
        loc = as_sir_loc(node.location)
        cond = self.visit(node.cond)
        then_body = [self.visit(statement) for statement in node.then_body]
        else_body = [self.visit(statement) for statement in node.else_body]
        return sir.If(cond, then_body, else_body, loc)

    def visit_Yield(self, node: hlast.Yield) -> sir.Yield:
        loc = as_sir_loc(node.location)
        values = [self.visit(value) for value in node.values]
        return sir.Yield(values, loc)

    def visit_ExtractSlice(self, node: hlast.ExtractSlice) -> sir.ExtractSlice:
        assert isinstance(node.type_, ts.FieldType)
        loc = as_sir_loc(node.location)
        dimensions = node.type_.dimensions
        lower_lookup = {slc.dimension: slc.lower for slc in node.slices}
        upper_lookup = {slc.dimension: slc.upper for slc in node.slices}
        step_lookup = {slc.dimension: slc.step for slc in node.slices}

        source_expr = self.visit(node.source)
        source_assign = sir.Assign(["__extract_slice_src"], [source_expr], loc)
        source_ref = sir.SymbolRef("__extract_slice_src", loc)

        index_type = sir.IndexType()
        lowers = [sir.Cast(self.visit(lower_lookup[dim]), index_type, loc) for dim in dimensions]
        uppers = [sir.Cast(self.visit(upper_lookup[dim]), index_type, loc) for dim in dimensions]
        steps = [sir.Cast(self.visit(step_lookup[dim]), index_type, loc) for dim in dimensions]
        shape = [sir.Dim(source_ref, sir.Constant.index(i, loc), loc) for i in range(len(dimensions))]

        lowers_norm = [sir.Call("__normalize_lower", [lower, size], loc) for lower, size in zip(lowers, shape)]
        sizes = [
            sir.Call("__slice_size", [lower, upper, step, size], loc)
            for lower, upper, step, size in zip(lowers, uppers, steps, shape)
        ]

        extract = sir.ExtractSlice(source_ref, lowers_norm, sizes, steps, loc)

        return sir.Block([source_assign, sir.Yield([extract], loc)], loc)

    def visit_Cast(self, node: hlast.Cast) -> sir.Cast:
        loc = as_sir_loc(node.location)
        value = self.visit(node.value)
        type_ = as_sir_type(node.type_)
        return sir.Cast(value, type_, loc)

    def _elementwise_stencil(self, node: hlast.ElementwiseOperation):
        assert isinstance(node.type_, ts.FieldType)
        loc = node.location
        name = f"__elemwise_sn_{next(utility.unique_id)}"
        type_ = ts.StencilType([arg.type_ for arg in node.args], [node.type_.element_type], node.type_.dimensions)
        parameters = [hlast.Parameter(f"__arg{i}", p_type) for i, p_type in enumerate(type_.parameters)]
        arg_refs = [hlast.SymbolRef(loc, p_type, f"__arg{i}") for i, p_type in enumerate(type_.parameters)]
        ndindex_type = ts.NDIndexType(type_.dims)
        samples = [
            hlast.Sample(loc, arg_ref.type_.element_type, arg_ref, hlast.Index(loc, ndindex_type))
            for arg_ref in arg_refs
        ]
        body = [hlast.Return(loc, type_.results[0], [node.element_expr(samples)])]
        return hlast.Stencil(loc, type_, name, parameters, type_.results, body, type_.dims)

def lower(hast_module: hlast.Module) -> sir.Module:
    shaped_module = ShapeFunctionPass().visit(hast_module)
    sir_module = HlastToSirPass().visit(shaped_module)
    return sir_module