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
    map_elementwise_shape,
    shape_func_name,
    flatten
)


class ShapeTransformer(SirOpTransformer):
    # -----------------------------------
    # Module structure
    # -----------------------------------
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
            if isinstance(param.type_, ts.FieldLikeType):
                ndims = len(param.type_.dimensions)
                parameters = [*parameters, *[sir.IndexType()]*ndims]
            else:
                parameters.append(as_sir_type(param.type_))

        results = []
        if isinstance(node.result, ts.FieldLikeType):
            ndims = len(node.result.dimensions)
            results = [sir.IndexType()]*ndims

        name = shape_func_name(node.name)
        function_type = sir.FunctionType(parameters, results)

        func: ops.FuncOp = self.insert_op(ops.FuncOp(name, function_type, node.is_public, loc))
        self.symtable.assign(name, func)

        def sc():
            self.push_region(func.get_body())

            arg_idx = 0
            for param in node.parameters:
                if isinstance(param.type_, ts.FieldLikeType):
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
        values = self.visit(node.value) if isinstance(node.value.type_, ts.FieldLikeType) else []
        converted = [self.insert_op(ops.CastOp(v, sir.IndexType(), loc)).get_result() for v in values]
        self.insert_op(ops.ReturnOp(converted, loc))
        return []

    def visit_Call(self, node: hlast.Call) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        name = shape_func_name(node.name)
        num_outs = len(node.type_.dimensions) if isinstance(node.type_, ts.FieldLikeType) else 0
        args = flatten(self.visit(arg) for arg in node.args)
        return self.insert_op(ops.CallOp(name, num_outs, args, loc)).get_results()

    def visit_Apply(self, node: hlast.Apply) -> list[ops.Value]:
        node_shape = sorted(node.shape.items(), key=lambda x: x[0])
        shape = [self.visit(size) for _, size in node_shape]
        assert all(len(size) == 1 for size in shape)
        shape = [size[0] for size in shape]
        return shape

    # -----------------------------------
    # Symbols
    # -----------------------------------
    def visit_SymbolRef(self, node: hlast.SymbolRef) -> list[ops.Value]:
        return self.symtable.lookup(node.name)

    def visit_Assign(self, node: hlast.Assign) -> list[ops.Value]:
        for name, value in zip(node.names, node.values):
            self.symtable.assign(name, self.visit(value))
        return []

    # -----------------------------------
    # Arithmetic & logic
    # -----------------------------------
    def visit_Cast(self, node: hlast.Cast) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        value = self.visit(node.value)[0]
        type_ = as_sir_type(node.type_)
        return self.insert_op(ops.ConstantOp(value, type_, loc)).get_result()

    def visit_Constant(self, node: hlast.Constant) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        value = node.value
        type_ = as_sir_type(node.type_)
        return self.insert_op(ops.ConstantOp(value, type_, loc)).get_results()

    def visit_ArithmeticOperation(self, node: hlast.ArithmeticOperation) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        lhs = self.visit(node.lhs)[0]
        rhs = self.visit(node.rhs)[0]
        func = as_sir_arithmetic(node.func)
        return self.insert_op(ops.ArithmeticOp(lhs, rhs, func, loc)).get_results()

    def visit_ComparisonOperation(self, node: hlast.ComparisonOperation) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        lhs = self.visit(node.lhs)[0]
        rhs = self.visit(node.rhs)[0]
        func = as_sir_comparison(node.func)
        return self.insert_op(ops.ComparisonOp(lhs, rhs, func, loc)).get_results()

    def visit_ElementwiseOperation(self, node: hlast.ElementwiseOperation) -> list[ops.Value]:
        assert isinstance(node.type_, ts.FieldType)
        dimensions = node.type_.dimensions
        arg_types = [arg.type_ for arg in node.args]
        shape_mapping = map_elementwise_shape(dimensions, arg_types)

        arg_shapes = [self.visit(arg) for arg in node.args]
        shape = [
            arg_shapes[arg_idx][arg_types[arg_idx].dimensions.index(dim)]
            for dim, arg_idx in sorted(shape_mapping.items(), key=lambda x: x[0])
        ]
        return shape

    # -----------------------------------
    # Control flow
    # -----------------------------------
    def visit_If(self, node: hlast.If) -> list[ops.Value]:
        loc = as_sir_loc(node.location)

        nresults = len(node.type_.dimensions) if isinstance(node.type_, ts.FieldLikeType) else 1

        cond = self.visit(node.cond)[0]
        ifop: ops.IfOp = self.insert_op(ops.IfOp(cond, nresults, loc))

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
        values = self.visit(node.value)
        self.insert_op(ops.YieldOp(values, loc))
        return []

    # -----------------------------------
    # Tensor
    # -----------------------------------
    def visit_Shape(self, node: hlast.Shape) -> list[ops.Value]:
        assert isinstance(node.field.type_, ts.FieldLikeType)
        shape = self.visit(node.field)
        index = node.field.type_.dimensions.index(node.dim)
        return [shape[index]]

    def visit_ExtractSlice(self, node: hlast.ExtractSlice) -> list[ops.Value]:
        loc = as_sir_loc(node.location)

        def as_index(expr: ops.Value) -> ops.Value:
            return self.insert_op(ops.CastOp(expr, sir.IndexType(), loc)).get_result()

        slices = [self._visit_slice(slc) for slc in sorted(node.slices, key=lambda slc: slc.dimension)]
        starts = [as_index(slc[0]) for slc in slices]
        stops = [as_index(slc[1]) for slc in slices]
        steps = [as_index(slc[2]) for slc in slices]
        lengths: list[ops.Value] = self.visit(node.source)

        sizes: list[ops.Value] = []
        for start, stop, step, length in zip(starts, stops, steps, lengths):
            adj = self.insert_op(ops.CallOp("__adjust_slice", 2, [start, stop, step, length], loc)).get_results()
            size = self.insert_op(ops.CallOp("__slice_size", 1, [adj[0], adj[1], step], loc)).get_results()[0]
            sizes.append(size)

        return sizes

    # -----------------------------------
    # Special
    # -----------------------------------
    def visit_Noop(self, _: hlast.Noop) -> list[ops.Value]:
        return []
