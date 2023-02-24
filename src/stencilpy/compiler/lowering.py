import stencilir as sir
from stencilpy.compiler import hast
from typing import Any
from stencilpy.compiler import types as ts
from stencilpy.error import *
from stencilpy import concepts


def as_sir_loc(loc: hast.Location) -> sir.Location:
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


class HASTtoSIR:
    def visit(self, node: Any):
        class_name = node.__class__.__name__
        handler_name = f"visit_{class_name}"
        if hasattr(self.__class__, handler_name):
            return getattr(self.__class__, handler_name)(self, node)
        if isinstance(node, hast.Node):
            loc = node.location
        else:
            loc = concepts.Location.unknown()
        raise InternalCompilerError(loc, f"no handler implemented for HAST node `{class_name}`")

    def visit_Module(self, node: hast.Module) -> sir.Module:
        loc = as_sir_loc(node.location)
        funcs = [self.visit(func) for func in node.functions]
        stencils = [self.visit(stencil) for stencil in node.stencils]
        return sir.Module(funcs, stencils, loc)

    def visit_Function(self, node: hast.Function) -> sir.Function:
        loc = as_sir_loc(node.location)
        name = node.name
        parameters = [sir.Parameter(p.name, as_sir_type(p.type_)) for p in node.parameters]
        results = [as_sir_type(type_) for type_ in node.results]
        statements = [self.visit(statement) for statement in node.body]
        return sir.Function(name, parameters, results, statements, True, loc)

    def visit_Return(self, node: hast.Return) -> sir.Return:
        loc = as_sir_loc(node.location)
        values = [self.visit(value) for value in node.values]
        return sir.Return(values, loc)

    def visit_Constant(self, node: hast.Constant) -> sir.Constant:
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

    def visit_SymbolRef(self, node: hast.SymbolRef) -> sir.SymbolRef:
        loc = as_sir_loc(node.location)
        name = node.name
        return sir.SymbolRef(name, loc)

    def visit_Shape(self, node: hast.Shape) -> sir.Dim:
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




def lower(hast_module: hast.Module) -> sir.Module:
    sir_module = HASTtoSIR().visit(hast_module)
    return sir_module