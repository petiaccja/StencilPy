import itertools

import stencilir as sir
from stencilpy.compiler import hlast
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


def get_dim_index(type_: ts.FieldType, dim: concepts.Dimension):
    try:
        return type_.dimensions.index(dim)
    except ValueError:
        raise KeyError(f"dimension {dim} is not associated with field")


class ShapeFunctionPass(NodeTransformer):
    @staticmethod
    def shape_var(name: str, dim: concepts.Dimension):
        return f"__shape_{dim.id}_{name}"

    @staticmethod
    def shape_func(name: str):
        return f"__shapes_{name}"

    def visit_Module(self, node: hlast.Module) -> hlast.Module:
        shape_funcs = [self.visit(func) for func in node.functions]

        funcs = [*shape_funcs, *node.functions]  # Shape function are added to original functions
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


class HlastToSirPass(NodeTransformer):
    @staticmethod
    def _out_param_name(idx: int):
        return f"__out_{idx}"

    def visit_Module(self, node: hlast.Module) -> sir.Module:
        loc = as_sir_loc(node.location)
        funcs = [self.visit(func) for func in node.functions]
        stencils = [self.visit(stencil) for stencil in node.stencils]
        return sir.Module(funcs, stencils, loc)

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
        elif isinstance(type_, sir.IntegerType):
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
            idx_val = get_dim_index(node.field.type_, node.dim)
        except KeyError:
            raise MissingDimensionError(node.location, node.field.type_, node.dim)
        field = self.visit(node.field)
        idx = sir.Constant.index(idx_val, loc)
        return sir.Dim(field, idx, loc)

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


def lower(hast_module: hlast.Module) -> sir.Module:
    shaped_module = ShapeFunctionPass().visit(hast_module)
    sir_module = HlastToSirPass().visit(shaped_module)
    return sir_module