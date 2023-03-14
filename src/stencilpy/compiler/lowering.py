import itertools

import stencilir as sir
from stencilir import ops
from stencilpy.compiler import hlast
from stencilpy import utility
from .node_transformer import NodeTransformer
from stencilpy.compiler import types as ts
from stencilpy.compiler.symbol_table import SymbolTable
from stencilpy.error import *
from stencilpy import concepts
from typing import Optional


def as_sir_loc(loc: hlast.Location) -> ops.Location:
    return ops.Location(loc.file, loc.line, loc.column)


def as_sir_type(type_: ts.Type) -> sir.Type:
    if isinstance(type_, ts.IndexType):
        return sir.IndexType()
    elif isinstance(type_, ts.IntegerType):
        return sir.IntegerType(type_.width, type_.signed)
    elif isinstance(type_, ts.FloatType):
        return sir.FloatType(type_.width)
    elif isinstance(type_, ts.FieldType):
        return sir.FieldType(as_sir_type(type_.element_type), len(type_.dimensions))
    elif isinstance(type_, ts.FunctionType):
        return sir.FunctionType(
            [as_sir_type(p) for p in type_.parameters],
            [as_sir_type(r) for r in type_.results],
        )
    else:
        raise ValueError(f"no SIR type equivalent for {type_.__class__}")


def as_sir_arithmetic(func: hlast.ArithmeticFunction) -> ops.ArithmeticFunction:
    _MAPPING = {
        hlast.ArithmeticFunction.ADD: ops.ArithmeticFunction.ADD,
        hlast.ArithmeticFunction.SUB: ops.ArithmeticFunction.SUB,
        hlast.ArithmeticFunction.MUL: ops.ArithmeticFunction.MUL,
        hlast.ArithmeticFunction.DIV: ops.ArithmeticFunction.DIV,
        hlast.ArithmeticFunction.MOD: ops.ArithmeticFunction.MOD,
        hlast.ArithmeticFunction.BIT_AND: ops.ArithmeticFunction.BIT_AND,
        hlast.ArithmeticFunction.BIT_OR: ops.ArithmeticFunction.BIT_OR,
        hlast.ArithmeticFunction.BIT_XOR: ops.ArithmeticFunction.BIT_XOR,
        hlast.ArithmeticFunction.BIT_SHL: ops.ArithmeticFunction.BIT_SHL,
        hlast.ArithmeticFunction.BIT_SHR: ops.ArithmeticFunction.BIT_SHR,
    }
    return _MAPPING[func]


def as_sir_comparison(func: hlast.ComparisonFunction) -> ops.ComparisonFunction:
    _MAPPING = {
        hlast.ComparisonFunction.EQ: ops.ComparisonFunction.EQ,
        hlast.ComparisonFunction.NEQ: ops.ComparisonFunction.NEQ,
        hlast.ComparisonFunction.LT: ops.ComparisonFunction.LT,
        hlast.ComparisonFunction.GT: ops.ComparisonFunction.GT,
        hlast.ComparisonFunction.LTE: ops.ComparisonFunction.LTE,
        hlast.ComparisonFunction.GTE: ops.ComparisonFunction.GTE,
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


def slice_size_function() -> ops.FuncOp:
    loc = ops.Location("<__slice_size>", 1, 1)

    name = "__slice_size"
    function_type = sir.FunctionType(
        [sir.IndexType(), sir.IndexType(), sir.IndexType()],
        [sir.IndexType()]
    )

    func = ops.FuncOp(name, function_type, False, loc)

    start = func.get_region_arg(0)
    stop = func.get_region_arg(1)
    step = func.get_region_arg(2)

    distance = func.add(ops.ArithmeticOp(stop, start, ops.ArithmeticFunction.SUB, loc)).get_result()
    size = func.add(ops.ArithmeticOp(distance, step, ops.ArithmeticFunction.DIV, loc)).get_result()
    c0 = func.add(ops.ConstantOp(0, sir.IndexType(), loc)).get_result()
    clamped = func.add(ops.MaxOp(size, c0, loc)).get_result()
    func.add(ops.ReturnOp([clamped], loc))

    return func


def is_slice_adjustment_trivial(
        start: hlast.Expr,
        step: hlast.Expr,
) -> bool:
    is_start_trivial = isinstance(start, hlast.Constant) and start.value >= 0
    is_step_trivial = isinstance(step, hlast.Constant) and step.value > 0
    return is_start_trivial and is_step_trivial


def adjust_slice_trivial_function() -> ops.FuncOp:
    """
    Simple method for limited cases to help optimization.
    Use only when:
    - start is a constant expression >= 0
    - stop is an arbitrary expression
    - step is a constant expression > 0
    - length is an arbitrary expression (>=0)
    """
    loc = ops.Location("<__adjust_slice_trivial>", 1, 1)

    name = "__adjust_slice_trivial"
    function_type = sir.FunctionType(
        [sir.IndexType(), sir.IndexType(), sir.IndexType(), sir.IndexType()],
        [sir.IndexType(), sir.IndexType()]
    )

    func = ops.FuncOp(name, function_type, False, loc)

    start = func.get_region_arg(0)
    stop = func.get_region_arg(1)
    length = func.get_region_arg(3)

    c0 = func.add(ops.ConstantOp(0, sir.IndexType(), loc)).get_result()
    is_stop_negative = func.add(ops.ComparisonOp(stop, c0, ops.ComparisonFunction.LT, loc)).get_result()
    stop_incr = func.add(ops.ArithmeticOp(stop, length, ops.ArithmeticFunction.ADD, loc)).get_result()
    stop_zero_clamped = func.add(ops.MaxOp(c0, stop_incr, loc)).get_result()
    stop_length_clamped = func.add(ops.MinOp(stop, length, loc)).get_result()
    stop_adj_op: ops.IfOp = func.add(ops.IfOp(is_stop_negative, 1, loc))
    stop_adj_op.get_then_region().add(ops.YieldOp([stop_zero_clamped], loc))
    stop_adj_op.get_else_region().add(ops.YieldOp([stop_length_clamped], loc))
    func.add(ops.ReturnOp([start, stop_adj_op.get_results()[0]], loc))

    return func


def adjust_slice_function() -> ops.FuncOp:
    loc = ops.Location("<__adjust_slice>", 1, 1)

    name = "__adjust_slice"
    function_type = sir.FunctionType(
        [sir.IndexType(), sir.IndexType(), sir.IndexType(), sir.IndexType()],
        [sir.IndexType(), sir.IndexType()]
    )
    func = ops.FuncOp(name, function_type, True, loc)

    c0 = func.add(ops.ConstantOp(0, sir.IndexType(), loc)).get_result()
    cm1 = func.add(ops.ConstantOp(-1, sir.IndexType(), loc)).get_result()

    start = func.get_region_arg(0)
    stop = func.get_region_arg(1)
    step = func.get_region_arg(2)
    length = func.get_region_arg(3)

    def lt(lhs, rhs):
        return func.add(ops.ComparisonOp(lhs, rhs, ops.ComparisonFunction.LT, loc)).get_result()

    def gte(lhs, rhs):
        return func.add(ops.ComparisonOp(lhs, rhs, ops.ComparisonFunction.GTE, loc)).get_result()

    def add(lhs, rhs):
        return func.add(ops.ArithmeticOp(lhs, rhs, ops.ArithmeticFunction.ADD, loc)).get_result()

    def select(cond, lhs, rhs):
        select_op: ops.IfOp = func.add(ops.IfOp(cond, 1, loc))
        select_op.get_then_region().add(ops.YieldOp([lhs], loc))
        select_op.get_else_region().add(ops.YieldOp([rhs], loc))
        return select_op.get_results()[0]

    start_adj = select(
        lt(start, c0),
        select(
            lt(add(start, length), c0),
            select(lt(step, c0), cm1, c0),
            add(start, length)
        ),
        select(
            gte(start, length),
            select(lt(step, c0), add(length, cm1), length),
            start
        )
    )

    stop_adj = select(
        lt(stop, c0),
        select(
            lt(add(stop, length), c0),
            select(lt(step, c0), cm1, c0),
            add(stop, length)
        ),
        select(
            gte(stop, length),
            select(lt(step, c0), add(length, cm1), length),
            length
        )
    )

    func.add(ops.ReturnOp([start_adj, stop_adj], loc))
    return func


class SirOpTransformer(NodeTransformer):
    region_stack: list[ops.Region] = None
    current_region: ops.Region = None
    symtable: SymbolTable = None

    def push_region(self, region: ops.Region):
        self.region_stack.append(region)
        self.current_region = region

    def pop_region(self):
        self.region_stack.pop(len(self.region_stack) - 1)
        self.current_region = self.region_stack[-1] if self.region_stack else None

    def __init__(self):
        self.symtable = SymbolTable()
        self.region_stack = []


class ShapeFunctionTransformer(SirOpTransformer):
    @staticmethod
    def shape_func(name: str):
        return f"__shapes_{name}"

    def visit_Module(self, node: hlast.Module) -> ops.ModuleOp:
        module = ops.ModuleOp()

        def sc():
            self.push_region(module.get_body())

            self.symtable.assign("__slice_size", self.current_region.add(slice_size_function()))
            self.symtable.assign("__adjust_slice", self.current_region.add(adjust_slice_function()))
            self.symtable.assign("__adjust_slice_trivial", self.current_region.add(adjust_slice_trivial_function()))

            for func in node.functions:
                self.visit(func)

            self.pop_region()
        self.symtable.scope(sc, module)

        return module

    def visit_Function(self, node: hlast.Function) -> list[ops.Value]:
        loc = as_sir_loc(node.location)

        parameters = []
        for param in node.parameters:
            if isinstance(param.type_, ts.FieldType):
                ndims = len(param.type_.dimensions)
                for _ in range(ndims):
                    parameters.append(sir.IndexType())
            else:
                parameters.append(as_sir_type(param.type_))

        results = []
        for result in node.results:
            if isinstance(result, ts.FieldType):
                ndims = len(result.dimensions)
                for _ in range(ndims):
                    results.append(sir.IndexType())

        name = self.shape_func(node.name)
        function_type = sir.FunctionType(parameters, results)

        func: ops.FuncOp = self.current_region.add(ops.FuncOp(name, function_type, True, loc))
        self.symtable.assign(name, func)

        def sc():
            self.push_region(func.get_body())

            arg_idx = 0
            for param in node.parameters:
                if isinstance(param.type_, ts.FieldType):
                    ndims = len(param.type_.dimensions)
                    shape = self.current_region.get_args()[arg_idx:(arg_idx + ndims)]
                    self.symtable.assign(param.name, shape)
                    arg_idx += ndims
                else:
                    self.symtable.assign(param.name, [self.current_region.get_args()[arg_idx]])
                    arg_idx += 1

            for statement in node.body:
                self.visit(statement)

            self.pop_region()

        self.symtable.scope(sc)
        return []

    def visit_Return(self, node: hlast.Return) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        values = [self.visit(value) for value in node.values if isinstance(value.type_, ts.FieldType)]
        flattened = list(itertools.chain(*values))
        converted = [self.current_region.add(ops.CastOp(v, sir.IndexType(), loc)).get_result() for v in flattened]
        self.current_region.add(ops.ReturnOp(converted, loc))
        return []

    def visit_Constant(self, node: hlast.Constant) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        value = node.value
        type_ = as_sir_type(node.type_)
        return self.current_region.add(ops.ConstantOp(value, type_, loc)).get_results()

    def visit_SymbolRef(self, node: hlast.SymbolRef) -> list[ops.Value]:
        return self.symtable.lookup(node.name)

    def visit_Assign(self, node: hlast.Assign) -> list[ops.Value]:
        for name, value in zip(node.names, node.values):
            self.symtable.assign(name, self.visit(value))
        return []

    def visit_Shape(self, node: hlast.Shape) -> list[ops.Value]:
        assert isinstance(node.type_, ts.FieldType)
        shape = self.visit(node.field)
        index = get_dim_index(node.type_.dimensions, node.dim)
        return [shape[index]]

    def visit_Apply(self, node: hlast.Apply) -> list[ops.Value]:
        node_shape = sorted(node.shape.items(), key=lambda x: x[0])
        shape = [self.visit(size) for _, size in node_shape]
        assert all(len(size) == 1 for size in shape)
        shape = [size[0] for size in shape]
        return shape

    def visit_ArithmeticOperation(self, node: hlast.ArithmeticOperation) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        lhs = self.visit(node.lhs)[0]
        rhs = self.visit(node.rhs)[0]
        func = as_sir_arithmetic(node.func)
        return self.current_region.add(ops.ArithmeticOp(lhs, rhs, func, loc)).get_results()

    def visit_ComparisonOperation(self, node: hlast.ComparisonOperation) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        lhs = self.visit(node.lhs)[0]
        rhs = self.visit(node.rhs)[0]
        func = as_sir_comparison(node.func)
        return self.current_region.add(ops.ComparisonOp(lhs, rhs, func, loc)).get_results()

    def visit_ElementwiseOperation(self, node: hlast.ElementwiseOperation) -> list[ops.Value]:
        assert isinstance(node.type_, ts.FieldType)
        dimensions = node.type_.dimensions
        arg_types = [arg.type_ for arg in node.args]
        dims_to_arg = elementwise_dims_to_arg(dimensions, arg_types)

        arg_shapes = [self.visit(arg) for arg in node.args]

        shape = [
            arg_shapes[arg_idx][get_dim_index(arg_types[arg_idx].dimensions, dim)]
            for dim, arg_idx in sorted(dims_to_arg.items(), key=lambda x: x[0])
        ]
        return shape

    def visit_If(self, node: hlast.If) -> list[ops.Value]:
        loc = as_sir_loc(node.location)

        nresults = len(node.type_.dimensions) if isinstance(node.type_, ts.FieldType) else 1

        cond = self.visit(node.cond)[0]
        ifop: ops.IfOp = self.current_region.add(ops.IfOp(cond, nresults, loc))

        self.push_region(ifop.get_then_region())
        for statement in node.then_body:
            self.visit(statement)
        self.pop_region()

        self.push_region(ifop.get_else_region())
        for statement in node.else_body:
            self.visit(statement)
        self.pop_region()

        return ifop.get_results()

    def visit_Yield(self, node: hlast.Yield) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        values = [self.visit(value) for value in node.values if isinstance(value.type_, ts.FieldType)]
        flattened = list(itertools.chain(*values))
        self.current_region.add(ops.YieldOp(flattened, loc))
        return []

    def visit_ExtractSlice(self, node: hlast.ExtractSlice) -> list[ops.Value]:
        loc = as_sir_loc(node.location)

        def as_index(expr: ops.Value) -> ops.Value:
            return self.current_region.add(ops.CastOp(expr, sir.IndexType(), loc)).get_result()

        slices = [self._visit_slice(slc) for slc in sorted(node.slices, key=lambda slc: slc.dimension)]

        starts = [as_index(slc[0]) for slc in slices]
        stops = [as_index(slc[1]) for slc in slices]
        steps = [as_index(slc[2]) for slc in slices]
        lengths: list[ops.Value] = self.visit(node.source)

        adjust_slice_fun = self.symtable.lookup("__adjust_slice")
        adjs = [
            self.current_region.add(ops.CallOp(adjust_slice_fun, [start, stop, step, length], loc)).get_results()
            for start, stop, step, length in zip(starts, stops, steps, lengths)
        ]
        start_adjs = [adj[0] for adj in adjs]
        stop_adjs = [adj[1] for adj in adjs]

        slice_size_fun = self.symtable.lookup("__slice_size")
        shape = [
            self.current_region.add(ops.CallOp(slice_size_fun, [start, stop, step], loc)).get_results()[0]
            for start, stop, step in zip(start_adjs, stop_adjs, steps)
        ]
        return shape

    def visit_Cast(self, node: hlast.Cast) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        value = self.visit(node.value)[0]
        type_ = as_sir_type(node.type_)
        return self.current_region.add(ops.ConstantOp(value, type_, loc)).get_result()

    def _visit_slice(self, slc: hlast.Slice) -> tuple[ops.Value, ops.Value, ops.Value]:
        return self.visit(slc.lower)[0], self.visit(slc.upper)[0], self.visit(slc.step)[0]


class HlastToSirTransformer(SirOpTransformer):
    def visit_Module(self, node: hlast.Module) -> ops.ModuleOp:
        module = ops.ModuleOp()

        def sc():
            self.push_region(module.get_body())

            for stencil in node.stencils:
                self.visit(stencil)
            for func in node.functions:
                self.visit(func)

            self.pop_region()

        self.symtable.scope(sc, module)
        return module

    def visit_Function(self, node: hlast.Function) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        name = node.name
        parameters = [as_sir_type(p.type_) for p in node.parameters]
        results = [as_sir_type(r) for r in node.results]
        outs = [as_sir_type(r) for r in node.results if isinstance(r, ts.FieldType)]
        function_type = sir.FunctionType([*parameters, *outs], results)

        func: ops.FuncOp = self.current_region.add(ops.FuncOp(name, function_type, True, loc))
        self.symtable.assign(name, func)

        def sc():
            for i, p in enumerate(node.parameters):
                self.symtable.assign(p.name, [func.get_region_arg(i)])
            self.push_region(func.get_body())
            for statement in node.body:
                self.visit(statement)
            self.pop_region()
        self.symtable.scope(sc, func)

        return []

    def visit_Stencil(self, node: hlast.Stencil) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        name = node.name
        parameters = [as_sir_type(p.type_) for p in node.parameters]
        results = [as_sir_type(r) for r in node.results]
        function_type = sir.FunctionType(parameters, results)
        ndims = len(node.dims)

        stencil: ops.StencilOp = self.current_region.add(ops.StencilOp(name, function_type, ndims, True, loc))
        self.symtable.assign(name, stencil)

        def sc():
            for i, p in enumerate(node.parameters):
                self.symtable.assign(p.name, [stencil.get_region_arg(i)])
            self.push_region(stencil.get_body())
            for statement in node.body:
                self.visit(statement)
            self.pop_region()
        self.symtable.scope(sc, stencil)

        return []

    def visit_Return(self, node: hlast.Return) -> list[ops.Value]:
        loc = as_sir_loc(node.location)

        def insert(src: ops.Value, dst: ops.Value, ndims: int):
            c0 = self.current_region.add(ops.ConstantOp(0, sir.IndexType(), loc)).get_result()
            c1 = self.current_region.add(ops.ConstantOp(1, sir.IndexType(), loc)).get_result()
            indices = [
                self.current_region.add(ops.ConstantOp(i, sir.IndexType(), loc)).get_result()
                for i in range(ndims)
            ]
            offsets = [c0]*ndims
            sizes = [self.current_region.add(ops.DimOp(src, index, loc)).get_result() for index in indices]
            strides = [c1]*ndims
            return self.current_region.add(ops.InsertSliceOp(src, dst, offsets, sizes, strides, loc)).get_result()

        is_fields = [isinstance(value.type_, ts.FieldType) for value in node.values]
        ndims = [len(value.type_.dimensions) if isinstance(value.type_, ts.FieldType) else 0 for value in node.values]
        values = [self.visit(value)[0] for value in node.values]
        num_outs = sum(1 if is_field else 0 for is_field in is_fields)
        func_region: Optional[ops.Region] = None
        for info in self.symtable.infos():
            if isinstance(info, (ops.FuncOp, ops.StencilOp)):
                func_region = info.get_body()
        assert func_region is not None

        out_idx = len(func_region.get_args()) - num_outs
        for i in range(len(values)):
            if is_fields[i]:
                values[i] = insert(values[i], func_region.get_args()[out_idx], ndims[i])
                out_idx += 1

        self.current_region.add(ops.ReturnOp(values, loc))
        return []

    def visit_Constant(self, node: hlast.Constant) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        type_ = as_sir_type(node.type_)
        value = node.value
        return self.current_region.add(ops.ConstantOp(value, type_, loc)).get_results()

    def visit_SymbolRef(self, node: hlast.SymbolRef) -> list[ops.Value]:
        return self.symtable.lookup(node.name)

    def visit_Assign(self, node: hlast.Assign) -> list[ops.Value]:
        for name, value in zip(node.names, node.values):
            self.symtable.assign(name, self.visit(value))
        return []

    def visit_Shape(self, node: hlast.Shape) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        if not isinstance(node.field.type_, ts.FieldType):
            raise CompilationError(node.field.location, f"shape expects a field, got {node.field.type_}")
        try:
            idx_val = get_dim_index(node.field.type_.dimensions, node.dim)
        except KeyError:
            raise MissingDimensionError(node.location, node.field.type_, node.dim)
        field = self.visit(node.field)[0]
        idx = self.current_region.add(ops.ConstantOp(idx_val, sir.IndexType(), loc)).get_result()
        return self.current_region.add(ops.DimOp(field, idx, loc)).get_results()

    def visit_Index(self, node: hlast.Index) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        return self.current_region.add(ops.IndexOp(loc)).get_results()

    def visit_Sample(self, node: hlast.Sample) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        assert isinstance(node.field.type_, ts.FieldType)
        assert isinstance(node.index.type_, ts.NDIndexType)
        try:
            projection = [get_dim_index(node.index.type_.dims, dim) for dim in node.field.type_.dimensions]
            field = self.visit(node.field)[0]
            index = self.visit(node.index)[0]
            if sorted(projection) != [range(len(node.field.type_.dimensions))]:
                index = self.current_region.add(ops.ProjectOp(index, projection, loc)).get_result()
            return self.current_region.add(ops.SampleOp(field, index, loc)).get_results()
        except KeyError:
            raise CompilationError(
                loc,
                f"cannot sample field of type {node.field.type_} with index of type {node.index.type_}"
            )

    def visit_Call(self, node: hlast.Call) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        func = self.symtable.lookup(node.name)
        assert isinstance(func, ops.FuncOp)
        args = itertools.chain(self.visit(arg) for arg in node.args)
        return self.current_region.add(ops.CallOp(func, args, loc)).get_results()

    def visit_Apply(self, node: hlast.Apply) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        stencil = self.symtable.lookup(node.stencil.name)
        assert isinstance(stencil, ops.StencilOp)
        inputs = [self.visit(arg)[0] for arg in node.args]
        shape = [self.visit(size)[0] for dim, size in sorted(node.shape.items(), key=lambda x: x[0])]
        shape = [self.current_region.add(ops.CastOp(size, sir.IndexType(), loc)).get_result() for size in shape]
        assert isinstance(node.type_, ts.FieldType)
        dtype = as_sir_type(node.type_.element_type)
        outputs = [self.current_region.add(ops.AllocTensorOp(dtype, shape, loc)).get_result()]
        static_offsets = [0]*len(node.type_.dimensions)
        return self.current_region.add(ops.ApplyOp(stencil, inputs, outputs, [], static_offsets, loc)).get_results()

    def visit_ArithmeticOperation(self, node: hlast.ArithmeticOperation) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        lhs = self.visit(node.lhs)[0]
        rhs = self.visit(node.rhs)[0]
        func = as_sir_arithmetic(node.func)
        return self.current_region.add(ops.ArithmeticOp(lhs, rhs, func, loc)).get_results()

    def visit_ComparisonOperation(self, node: hlast.ComparisonOperation) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        lhs = self.visit(node.lhs)[0]
        rhs = self.visit(node.rhs)[0]
        func = as_sir_comparison(node.func)
        return self.current_region.add(ops.ComparisonOp(lhs, rhs, func, loc)).get_results()

    def visit_ElementwiseOperation(self, node: hlast.ElementwiseOperation) -> list[ops.Value]:
        loc = as_sir_loc(node.location)

        stencil = self._elementwise_stencil(node)
        args = [self.visit(arg)[0] for arg in node.args]

        assert isinstance(node.type_, ts.FieldType)
        dims_to_arg = elementwise_dims_to_arg(node.type_.dimensions, [arg.type_ for arg in node.args])
        dims_to_arg_sorted = sorted(dims_to_arg.items(), key=lambda x: x[0])
        shape = [
            self.current_region.add(
                ops.DimOp(
                    args[arg_idx],
                    self.current_region.add(
                        ops.ConstantOp(
                            get_dim_index(node.args[arg_idx].type_.dimensions, dim),
                            sir.IndexType(),
                        loc)
                    ).get_result(),
                    loc
                )
            ).get_result() for dim, arg_idx in dims_to_arg_sorted
        ]

        dtype = as_sir_type(node.type_.element_type)
        output = self.current_region.add(ops.AllocTensorOp(dtype, shape, loc)).get_result()
        return self.current_region.add(ops.ApplyOp(stencil, args, [output], [], [0] * len(shape), loc)).get_results()

    def visit_If(self, node: hlast.If) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        cond = self.visit(node.cond)[0]

        ifop: ops.IfOp = self.current_region.add(ops.IfOp(cond, 1, loc))

        self.push_region(ifop.get_then_region())
        for statement in node.then_body:
            self.visit(statement)
        self.pop_region()

        self.push_region(ifop.get_else_region())
        for statement in node.else_body:
            self.visit(statement)
        self.pop_region()

        return ifop.get_results()

    def visit_Yield(self, node: hlast.Yield) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        values = [self.visit(value)[0] for value in node.values]
        self.current_region.add(ops.YieldOp(values, loc))
        return []

    def visit_ExtractSlice(self, node: hlast.ExtractSlice) -> list[ops.Value]:
        assert isinstance(node.type_, ts.FieldType)
        loc = as_sir_loc(node.location)
        ndims = len(node.type_.dimensions)

        sorted_slices = sorted(node.slices, key=lambda slc: slc.dimension)
        indices = [self.current_region.add(ops.ConstantOp(i, sir.IndexType(), loc)).get_result() for i in range(ndims)]
        source = self.visit(node.source)[0]
        starts = [self.visit(slc.lower)[0] for slc in sorted_slices]
        stops = [self.visit(slc.upper)[0] for slc in sorted_slices]
        steps = [self.visit(slc.step)[0] for slc in sorted_slices]
        lengths = [self.current_region.add(ops.DimOp(source, index, loc)).get_result() for index in indices]

        adjust_slice_fun = self.symtable.lookup("__adjust_slice")
        assert adjust_slice_fun
        adjs = [
            self.current_region.add(ops.CallOp(adjust_slice_fun, [start, stop, step, length], loc)).get_results()
            for start, stop, step, length in zip(starts, stops, steps, lengths)
        ]
        start_adjs = [adj[0] for adj in adjs]
        stop_adjs = [adj[1] for adj in adjs]

        slice_size_fun = self.symtable.lookup("__slice_size")
        assert slice_size_fun
        sizes = [
            self.current_region.add(ops.CallOp(slice_size_fun, [start, stop, step], loc)).get_results()[0]
            for start, stop, step in zip(start_adjs, stop_adjs, steps)
        ]

        return self.current_region.add(ops.ExtractSliceOp(source, start_adjs, sizes, steps, loc)).get_results()

    def visit_Cast(self, node: hlast.Cast) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        value = self.visit(node.value)
        type_ = as_sir_type(node.type_)
        return self.current_region.add(ops.CastOp(value, type_, loc)).get_results()

    def visit_Min(self, node: hlast.Max) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        return self.current_region.add(ops.MinOp(lhs, rhs, loc)).get_results()

    def visit_Max(self, node: hlast.Max) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        return self.current_region.add(ops.MaxOp(lhs, rhs, loc)).get_results()

    def _elementwise_stencil(self, node: hlast.ElementwiseOperation):
        assert isinstance(node.type_, ts.FieldType)
        loc = node.location

        name = f"__elemwise_sn_{utility.unique_id()}"
        type_ = ts.StencilType([arg.type_ for arg in node.args], [node.type_.element_type], node.type_.dimensions)
        parameters = [hlast.Parameter(f"__arg{i}", p_type) for i, p_type in enumerate(type_.parameters)]
        arg_refs = [hlast.SymbolRef(loc, p_type, f"__arg{i}") for i, p_type in enumerate(type_.parameters)]
        ndindex_type = ts.NDIndexType(type_.dims)
        samples = [
            hlast.Sample(loc, arg_ref.type_.element_type, arg_ref, hlast.Index(loc, ndindex_type))
            if isinstance(arg_ref.type_, ts.FieldType)
            else arg_ref
            for arg_ref in arg_refs
        ]
        body = [hlast.Return(loc, type_.results[0], [node.element_expr(samples)])]
        stencil = hlast.Stencil(loc, type_, name, parameters, type_.results, body, type_.dims)

        module: Optional[ops.ModuleOp] = None
        for info in self.symtable.infos():
            if isinstance(info, ops.ModuleOp):
                module = info
        assert module

        def sc():
            self.push_region(module.get_body())
            self.visit(stencil)
            converted = self.symtable.lookup(name)
            self.pop_region()
            return converted
        return self.symtable.scope(sc)

def lower(hlast_module: hlast.Module) -> ops.ModuleOp:
    shape_module: ops.ModuleOp = ShapeFunctionTransformer().visit(hlast_module)
    code_module: ops.ModuleOp = HlastToSirTransformer().visit(hlast_module)
    merged = ops.ModuleOp()
    for op in shape_module.get_body().get_operations():
        merged.get_body().add(op)
    for op in code_module.get_body().get_operations():
        merged.get_body().add(op)
    return merged
