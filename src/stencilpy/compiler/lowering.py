import stencilir as sir
from stencilpy.compiler import hast
from typing import Any
from stencilpy.compiler import types as ts


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


class HASTtoSIR:
    def visit(self, node: Any):
        class_name = node.__class__.__name__
        handler_name = f"visit_{class_name}"
        if hasattr(self.__class__, handler_name):
            return getattr(self.__class__, handler_name)(self, node)
        raise ValueError(f"no handler for node `{class_name}`")

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



def lower(hast_module: hast.Module) -> sir.Module:
    sir_module = HASTtoSIR().visit(hast_module)
    return sir_module