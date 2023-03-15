from .basic_transformer import SirOpTransformer
from stencilir import ops
import stencilir as sir
from stencilpy.compiler import hlast
from stencilpy.compiler import types as ts
from stencilpy import utility
from stencilpy.error import *
from .utility import (
    as_sir_loc,
    as_sir_type,
    as_sir_arithmetic,
    as_sir_comparison,
    elementwise_dims_to_arg,
    get_dim_index
)
import itertools
from typing import Optional



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

        stencil: ops.StencilOp = self.current_region.add(ops.StencilOp(name, function_type, ndims, False, loc))
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

        def as_index(expr: ops.Value) -> ops.Value:
            return self.current_region.add(ops.CastOp(expr, sir.IndexType(), loc)).get_result()

        ndims = len(node.type_.dimensions)
        indices = [self.current_region.add(ops.ConstantOp(i, sir.IndexType(), loc)).get_result() for i in range(ndims)]
        source = self.visit(node.source)[0]
        slices = [self.visit_slice(slc) for slc in sorted(node.slices, key=lambda slc: slc.dimension)]
        starts = [as_index(slc[0]) for slc in slices]
        stops = [as_index(slc[1]) for slc in slices]
        steps = [as_index(slc[2]) for slc in slices]
        lengths = [self.current_region.add(ops.DimOp(source, index, loc)).get_result() for index in indices]

        adjs = [
            self.current_region.add(ops.CallOp("__adjust_slice", 2, [start, stop, step, length], loc)).get_results()
            for start, stop, step, length in zip(starts, stops, steps, lengths)
        ]
        start_adjs = [adj[0] for adj in adjs]
        stop_adjs = [adj[1] for adj in adjs]

        sizes = [
            self.current_region.add(ops.CallOp("__slice_size", 1, [start, stop, step], loc)).get_results()[0]
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
