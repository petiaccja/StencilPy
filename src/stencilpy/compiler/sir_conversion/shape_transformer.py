from .basic_transformer import SirOpTransformer
from stencilir import ops
import stencilir as sir
from stencilpy.compiler import hlast
from stencilpy.compiler import types as ts
from .utility import (
    as_sir_loc,
    as_sir_type,
    as_sir_arithmetic,
    as_sir_comparison,
    elementwise_dims_to_arg,
    get_dim_index,
    shape_func_name
)
import itertools


class ShapeTransformer(SirOpTransformer):
    def visit_Module(self, node: hlast.Module) -> ops.ModuleOp:
        module = ops.ModuleOp()

        def sc():
            self.push_region(module.get_body())

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

        name = shape_func_name(node.name)
        function_type = sir.FunctionType(parameters, results)

        func: ops.FuncOp = self.current_region.add(ops.FuncOp(name, function_type, node.is_public, loc))
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

    def visit_Call(self, node: hlast.Call) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        name = shape_func_name(node.name)
        num_outs = len(node.type_.dimensions) if isinstance(node.type_, ts.FieldType) else 0
        args = list(itertools.chain(*[self.visit(arg) for arg in node.args]))
        return self.current_region.add(ops.CallOp(name, num_outs, args, loc)).get_results()

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

        slices = [self.visit_slice(slc) for slc in sorted(node.slices, key=lambda slc: slc.dimension)]

        starts = [as_index(slc[0]) for slc in slices]
        stops = [as_index(slc[1]) for slc in slices]
        steps = [as_index(slc[2]) for slc in slices]
        lengths: list[ops.Value] = self.visit(node.source)

        adjs = [
            self.current_region.add(ops.CallOp("__adjust_slice", 2, [start, stop, step, length], loc)).get_results()
            for start, stop, step, length in zip(starts, stops, steps, lengths)
        ]
        start_adjs = [adj[0] for adj in adjs]
        stop_adjs = [adj[1] for adj in adjs]

        shape = [
            self.current_region.add(ops.CallOp("__slice_size", 1, [start, stop, step], loc)).get_results()[0]
            for start, stop, step in zip(start_adjs, stop_adjs, steps)
        ]
        return shape

    def visit_Cast(self, node: hlast.Cast) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        value = self.visit(node.value)[0]
        type_ = as_sir_type(node.type_)
        return self.current_region.add(ops.ConstantOp(value, type_, loc)).get_result()
