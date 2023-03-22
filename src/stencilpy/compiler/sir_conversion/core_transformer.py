import copy

from .basic_transformer import SirOpTransformer
from stencilir import ops
import stencilir as sir
from stencilpy.compiler import hlast
from stencilpy.compiler import types as ts
from stencilpy import utility
from .utility import (
    as_sir_loc,
    as_sir_type,
    as_sir_arithmetic,
    as_sir_comparison,
    is_slice_adjustment_trivial,
    map_elementwise_shape,
    shape_func_name,
    flatten,
)
from typing import Optional


class CoreTransformer(SirOpTransformer):
    # -----------------------------------
    # Module structure
    # -----------------------------------
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
        function_type = self._get_function_type(node)

        func: ops.FuncOp = self.insert_op(ops.FuncOp(name, function_type, node.is_public, loc))
        self.symtable.assign(name, func)

        def sc():
            for param, value in zip(node.parameters, self._get_function_args(node, func.get_body())):
                self.symtable.assign(param.name, value)
            self.push_region(func.get_body())
            for statement in node.body:
                self.visit(statement)
            self.pop_region()
        self.symtable.scope(sc, func)

        return []

    def visit_Stencil(self, node: hlast.Stencil) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        name = node.name
        function_type = self._get_function_type(node)
        ndims = len(node.dims)

        stencil: ops.StencilOp = self.insert_op(ops.StencilOp(name, function_type, ndims, False, loc))
        self.symtable.assign(name, stencil)

        def sc():
            for param, value in zip(node.parameters, self._get_function_args(node, stencil.get_body())):
                self.symtable.assign(param.name, value)
            self.push_region(stencil.get_body())
            for statement in node.body:
                self.visit(statement)
            self.pop_region()
        self.symtable.scope(sc, stencil)

        return []

    def visit_Return(self, node: hlast.Return) -> list[ops.Value]:
        loc = as_sir_loc(node.location)

        def insert(src: ops.Value, dst: ops.Value, src_type: ts.FieldLikeType):
            c0 = self.insert_op(ops.ConstantOp(0, sir.IndexType(), loc)).get_result()
            c1 = self.insert_op(ops.ConstantOp(1, sir.IndexType(), loc)).get_result()
            offsets = [c0] * len(src_type.dimensions)
            sizes = self._get_shape_or_empty(src, src_type, loc)
            strides = [c1] * len(src_type.dimensions)
            return self.insert_op(ops.InsertSliceOp(src, dst, offsets, sizes, strides, loc)).get_result()

        types = [node.value.type_]
        values = self.visit(node.value)
        function_body = self._get_enclosing_function_body()
        assert function_body is not None

        updated_values: list[ops.Value] = []
        for idx, type_, value in zip(range(len(values)), reversed(types), reversed(values)):
            if isinstance(type_, ts.FieldLikeType):
                new_value = insert(value, function_body.get_args()[-idx - 1], type_)
            else:
                new_value = value
            updated_values.append(new_value)

        self.insert_op(ops.ReturnOp(list(reversed(updated_values)), loc))
        return []

    def visit_Call(self, node: hlast.Call) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        name = node.name
        args = flatten(self.visit(arg) for arg in node.args)
        out_args = self._allocate_output_tensors(node, args)
        return self.insert_op(ops.CallOp(name, 1, [*args, *out_args], loc)).get_results()

    def visit_Apply(self, node: hlast.Apply) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        inputs = flatten(self.visit(arg) for arg in node.args)
        shape = [self.visit(size)[0] for dim, size in sorted(node.shape.items(), key=lambda x: x[0])]
        shape = [self.insert_op(ops.CastOp(size, sir.IndexType(), loc)).get_result() for size in shape]
        assert isinstance(node.type_, ts.FieldType)
        dtype = as_sir_type(node.type_.element_type)
        outputs = [self.insert_op(ops.AllocTensorOp(dtype, shape, loc)).get_result()]
        static_offsets = [0]*len(node.type_.dimensions)
        return self.insert_op(ops.ApplyOp(node.stencil, inputs, outputs, [], static_offsets, loc)).get_results()

    # -----------------------------------
    # Symbols
    # -----------------------------------
    def visit_SymbolRef(self, node: hlast.SymbolRef) -> list[ops.Value]:
        return self.symtable.lookup(node.name)

    def visit_Assign(self, node: hlast.Assign) -> list[ops.Value]:
        for name, value in zip(node.names, node.values):
            self.symtable.assign(name, self.visit(value))
        return []

    #-----------------------------------
    # Arithmetic & logic
    # -----------------------------------
    def visit_Cast(self, node: hlast.Cast) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        value = self.visit(node.value)
        type_ = as_sir_type(node.type_)
        return self.insert_op(ops.CastOp(*value, type_, loc)).get_results()

    def visit_Constant(self, node: hlast.Constant) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        type_ = as_sir_type(node.type_)
        value = node.value
        return self.insert_op(ops.ConstantOp(value, type_, loc)).get_results()

    def visit_ArithmeticOperation(self, node: hlast.ArithmeticOperation) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        func = as_sir_arithmetic(node.func)
        return self.insert_op(ops.ArithmeticOp(*lhs, *rhs, func, loc)).get_results()

    def visit_ComparisonOperation(self, node: hlast.ComparisonOperation) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        func = as_sir_comparison(node.func)
        return self.insert_op(ops.ComparisonOp(*lhs, *rhs, func, loc)).get_results()

    def visit_Min(self, node: hlast.Max) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        return self.insert_op(ops.MinOp(*lhs, *rhs, loc)).get_results()

    def visit_Max(self, node: hlast.Max) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        return self.insert_op(ops.MaxOp(*lhs, *rhs, loc)).get_results()

    def visit_ElementwiseOperation(self, node: hlast.ElementwiseOperation) -> list[ops.Value]:
        loc = as_sir_loc(node.location)

        assert isinstance(node.type_, ts.FieldType)
        dimensions = node.type_.dimensions
        arg_types = [arg.type_ for arg in node.args]
        shape_mapping = map_elementwise_shape(dimensions, arg_types)

        stencil = self._elementwise_stencil(node)
        args = [self.visit(arg)[0] for arg in node.args]

        shape_args = [args[shape_mapping[dim]] for dim in dimensions]
        shape_args_types = [node.args[shape_mapping[dim]].type_ for dim in dimensions]
        shape_arg_vs = [type_.dimensions.index(dim) for dim, type_ in zip(dimensions, shape_args_types)]
        shape_arg_cs = [self.insert_op(ops.ConstantOp(v, sir.IndexType(), loc)).get_result() for v in shape_arg_vs]
        shape = [self.insert_op(ops.DimOp(arg, c, loc)).get_result() for arg, c in zip(shape_args, shape_arg_cs)]

        dtype = as_sir_type(node.type_.element_type)
        output = self.insert_op(ops.AllocTensorOp(dtype, shape, loc)).get_result()
        return self.insert_op(ops.ApplyOp(stencil, args, [output], [], [0] * len(shape), loc)).get_results()

    # -----------------------------------
    # Control flow
    # -----------------------------------
    def visit_If(self, node: hlast.If) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        cond = self.visit(node.cond)[0]

        ifop: ops.IfOp = self.insert_op(ops.IfOp(cond, 1, loc))

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
        loc = as_sir_loc(node.location)
        assert isinstance(node.field.type_, ts.FieldLikeType)
        idx_val = node.field.type_.dimensions.index(node.dim)
        field = self.visit(node.field)[0]
        idx = self.insert_op(ops.ConstantOp(idx_val, sir.IndexType(), loc)).get_result()
        return self.insert_op(ops.DimOp(field, idx, loc)).get_results()

    def visit_ExtractSlice(self, node: hlast.ExtractSlice) -> list[ops.Value]:
        assert isinstance(node.type_, ts.FieldType)
        loc = as_sir_loc(node.location)

        def as_index(expr: ops.Value) -> ops.Value:
            return self.insert_op(ops.CastOp(expr, sir.IndexType(), loc)).get_result()

        sorted_slices = sorted(node.slices, key=lambda slc: slc.dimension)
        source = self.visit(node.source)
        slices = [self._visit_slice(slc) for slc in sorted_slices]
        trivials = [is_slice_adjustment_trivial(slc.lower, slc.step) for slc in sorted_slices]
        starts = [as_index(slc[0]) for slc in slices]
        stops = [as_index(slc[1]) for slc in slices]
        steps = [as_index(slc[2]) for slc in slices]
        lengths = self._get_shape_or_empty(source[0], node.source.type_, loc)

        start_adjs: list[ops.Value] = []
        stop_adjs: list[ops.Value] = []
        sizes: list[ops.Value] = []
        for start, stop, step, length, trivial in zip(starts, stops, steps, lengths, trivials):
            adjust_func = "__adjust_slice_trivial" if trivial else "__adjust_slice"
            adj = self.insert_op(ops.CallOp(adjust_func, 2, [start, stop, step, length], loc)).get_results()
            start_adjs.append(adj[0])
            stop_adjs.append(adj[1])
            size = self.insert_op(ops.CallOp("__slice_size", 1, [adj[0], adj[1], step], loc)).get_results()[0]
            sizes.append(size)

        return self.insert_op(ops.ExtractSliceOp(*source, start_adjs, sizes, steps, loc)).get_results()

    # -----------------------------------
    # Stencil instrinsics
    # -----------------------------------
    def visit_Index(self, node: hlast.Index) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        return self.insert_op(ops.IndexOp(loc)).get_results()

    def visit_Exchange(self, node: hlast.Exchange) -> list[ops.Value]:
        assert isinstance(node.index.type_, ts.NDIndexType)
        loc = as_sir_loc(node.location)
        index = self.visit(node.index)
        value = self.visit(node.value)
        old_dim_value = node.index.type_.dims.index(node.old_dim)
        value_cast = self.insert_op(ops.CastOp(*value, sir.IndexType(), loc)).get_result()
        exch = self.insert_op(ops.ExchangeOp(*index, old_dim_value, value_cast, loc)).get_result()
        new_dims = copy.deepcopy(node.index.type_.dims)
        new_dims[old_dim_value] = node.new_dim
        sorted_new_dims = sorted(new_dims)
        positions = [sorted_new_dims.index(dim) for dim in new_dims]
        return self.insert_op(ops.ProjectOp(exch, positions, loc)).get_results()

    def visit_Sample(self, node: hlast.Sample) -> list[ops.Value]:
        loc = as_sir_loc(node.location)
        assert isinstance(node.field.type_, ts.FieldLikeType)
        assert isinstance(node.index.type_, ts.NDIndexType)
        projection = [node.index.type_.dims.index(dim) for dim in node.field.type_.dimensions]
        field = self.visit(node.field)[0]
        index = self.visit(node.index)[0]
        if sorted(projection) != [range(len(node.field.type_.dimensions))]:
            index = self.insert_op(ops.ProjectOp(index, projection, loc)).get_result()
        return self.insert_op(ops.SampleOp(field, index, loc)).get_results()

    # -----------------------------------
    # Utility functions
    # -----------------------------------
    def _get_function_type(self, node: hlast.Function | hlast.Stencil) -> sir.FunctionType:
        inputs = [as_sir_type(p.type_) for p in node.parameters]
        outputs = [as_sir_type(node.result)]
        out_args = []
        if isinstance(node, hlast.Function):
            out_args = [as_sir_type(node.result)] if isinstance(node.result, ts.FieldType) else []
        return sir.FunctionType([*inputs, *out_args], outputs)

    def _get_function_args(self, node: hlast.Function | hlast.Stencil, body: ops.Region) -> list[list[ops.Value]]:
        values: list[list[ops.Value]] = []
        region_args = body.get_args()
        offset = 0
        for _ in node.parameters:
            num_args = 1  # Flatten parameters of tuple type
            values.append(region_args[offset:(offset + num_args)])
            offset += num_args
        return values

    def _get_enclosing_function_body(self) -> Optional[ops.Region]:
        body: Optional[ops.Region] = None
        for info in self.symtable.infos():
            if isinstance(info, (ops.FuncOp, ops.StencilOp)):
                body = info.get_body()
                break
        return body

    def _allocate_output_tensors(self, node: hlast.Call, args: list[ops.Value]) -> list[ops.Value]:
        if not isinstance(node.type_, ts.FieldType):
            return []
        loc = as_sir_loc(node.location)
        name = shape_func_name(node.name)
        num_outs = len(node.type_.dimensions)
        arg_shapes = flatten(self._get_shape_or_empty(arg, narg.type_, loc) for arg, narg in zip(args, node.args))
        out_shape = self.insert_op(ops.CallOp(name, num_outs, arg_shapes, loc)).get_results()
        element_type = as_sir_type(node.type_.element_type)
        return [self.insert_op(ops.AllocTensorOp(element_type, out_shape, loc)).get_result()]

    def _get_shape_or_empty(self, source: ops.Value, source_type: ts.Type, loc: ops.Location):
        if isinstance(source_type, ts.FieldLikeType):
            ndims = len(source_type.dimensions)
            indices = [
                self.insert_op(ops.ConstantOp(i, sir.IndexType(), loc)).get_result()
                for i in range(ndims)
            ]
            shape = [
                self.insert_op(ops.DimOp(source, index, loc)).get_result()
                for index in indices
            ]
            return shape
        return []

    def _elementwise_stencil(self, node: hlast.ElementwiseOperation):
        assert isinstance(node.type_, ts.FieldType)
        loc = node.location

        name = f"__elemwise_sn_{utility.unique_id()}"
        type_ = ts.StencilType([arg.type_ for arg in node.args], node.type_.element_type, node.type_.dimensions)
        parameters = [hlast.Parameter(f"__arg{i}", p_type) for i, p_type in enumerate(type_.parameters)]
        arg_refs = [hlast.SymbolRef(loc, p_type, f"__arg{i}") for i, p_type in enumerate(type_.parameters)]
        ndindex_type = ts.NDIndexType(type_.dims)
        samples = [
            hlast.Sample(loc, arg_ref.type_.element_type, arg_ref, hlast.Index(loc, ndindex_type))
            if isinstance(arg_ref.type_, ts.FieldType)
            else arg_ref
            for arg_ref in arg_refs
        ]
        body = [hlast.Return(loc, type_.result, node.element_expr(samples))]
        stencil = hlast.Stencil(loc, type_, name, parameters, type_.result, body, type_.dims)

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
