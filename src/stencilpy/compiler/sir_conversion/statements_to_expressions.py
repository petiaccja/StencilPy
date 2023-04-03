import copy
import dataclasses

from stencilpy.compiler import hlast
from stencilpy.compiler.node_transformer import NodeTransformer
from stencilpy import concepts
from stencilpy.error import *
from stencilpy.compiler import types as ts
from typing import Any, Optional
from collections.abc import Sequence, Mapping
from stencilpy import utility


class PassthroughTransformer(NodeTransformer):
    def generic_visit(self, node: Any, **kwargs):
        if isinstance(node, concepts.Dimension):
            return node
        if dataclasses.is_dataclass(node):
            fields = dataclasses.fields(node)
            fields = {field.name: self.visit(getattr(node, field.name), **kwargs) for field in fields if field.init}
            return node.__class__(**fields)
        if isinstance(node, str):
            return node
        if isinstance(node, Sequence):
            return node.__class__(self.visit(item, **kwargs) for item in node)
        if isinstance(node, Mapping):
            return node.__class__({name: self.visit(item, **kwargs) for name, item in node.items()})
        return copy.deepcopy(node)


class CollectSymbolsPass(PassthroughTransformer):
    unknown_symbols: list[hlast.SymbolRef]
    known_symbols: set[str]

    def __init__(self):
        self.unknown_symbols = []
        self.known_symbols = set()

    def visit_SymbolRef(self, node: hlast.SymbolRef):
        if node.name not in self.known_symbols:
            self.unknown_symbols.append(node)
        self.known_symbols.add(node.name)

    def visit_Assign(self, node: hlast.Assign):
        self.visit(node.value)
        for name in node.names:
            self.known_symbols.add(name)


class StatementsToExpressionsPass(PassthroughTransformer):
    continuation_funcs: list[hlast.Function]

    def __init__(self):
        self.continuation_funcs = []

    def visit_Module(self, node: hlast.Module) -> hlast.Module:
        stencils = [self.visit(stencil) for stencil in node.stencils]
        functions = [self.visit(function) for function in node.functions]
        return hlast.Module(node.location, node.type_, [*self.continuation_funcs, *functions], stencils)

    def visit_Function(self, node: hlast.Function) -> hlast.Function:
        body = self._expressionify_statements(node.body)
        return hlast.Function(node.location, node.type_, node.name, node.parameters, node.result, body, node.is_public)

    def visit_Stencil(self, node: hlast.Stencil) -> hlast.Stencil:
        body = self._expressionify_statements(node.body)
        return hlast.Stencil(node.location, node.type_, node.name, node.parameters, node.result, body, node.dims)

    def visit_IfStatement(
            self, node: hlast.IfStatement, continuation: list[hlast.Statement]
    ) -> Optional[list[hlast.Statement]]:
        cont_func, cont_call = self._create_continuation_function(continuation, node.location)

        if cont_func:
            self.continuation_funcs.append(cont_func)

        then_body = self._create_yield(node.then_body, cont_call)
        then_body = self._expressionify_statements(then_body)
        else_body = self._create_yield(node.else_body, cont_call)
        else_body = self._expressionify_statements(else_body)

        then_terminator = then_body[-1]
        assert isinstance(then_terminator, hlast.Yield)

        expr = hlast.If(node.location, then_terminator.value.type_, node.cond, then_body, else_body)

        terminator_cls = type(continuation[-1]) if continuation else hlast.Return
        if terminator_cls == hlast.Yield:
            terminator_stmt = [hlast.Yield(node.location, ts.void_t, expr)]
        elif terminator_cls == hlast.Return:
            terminator_stmt = [hlast.Return(node.location, ts.void_t, expr)]
        else:
            terminator_stmt = []
        return [*terminator_stmt]

    def visit_ForStatement(
            self, node: hlast.ForStatement, continuation: list[hlast.Statement]
    ) -> Optional[list[hlast.Statement]]:
        syms_for_cont = self._collect_unknown_symbols(continuation)
        syms_for_self = self._collect_unknown_symbols(node.body)
        unique_syms = {sym.name: sym for sym in [*syms_for_self, *syms_for_cont]}
        del unique_syms[node.loop_index.name]
        syms_all = list(unique_syms.values())
        sym_types = [sym.type_ for sym in syms_all]

        yield_value = hlast.TupleCreate(node.location, ts.TupleType(sym_types), syms_all)
        type_ = yield_value.type_
        yield_stmt = hlast.Yield(node.location, type_, yield_value)
        loop_carried = hlast.Parameter(node.location, type_, "__loop_carried")
        loop_carried_ref = hlast.SymbolRef(node.location, type_, "__loop_carried")
        unpack_carried = hlast.Assign(node.location, ts.void_t, [sym.name for sym in syms_all], loop_carried_ref)

        body = [unpack_carried, *node.body, yield_stmt]
        body = self._expressionify_statements(body)

        expr = hlast.For(
            node.location,
            type_,
            node.start,
            node.stop,
            node.step,
            yield_value,
            node.loop_index,
            loop_carried,
            body
        )
        unpack_for = hlast.Assign(node.location, ts.void_t, [sym.name for sym in syms_all], expr)
        return [unpack_for, *continuation]

    def visit_Statement(self, _1: hlast.Statement, **_kw) -> Optional[list[hlast.Statement]]:
        return None

    def _expressionify_statements(self, statements: list[hlast.Statement]):
        for i in range(len(statements) - 1, -1, -1):
            statement = statements[i]
            continuation = statements[i + 1:]
            new_statements = self.visit(statement, continuation=continuation)
            if new_statements:
                statements = [*statements[0:i], *new_statements]
        return statements

    def _create_yield(self, body: list[hlast.Statement], fallback_value: Optional[hlast.Expr]):
        terminator = body[-1] if body else None
        if terminator and isinstance(terminator, (hlast.Return, hlast.Yield)):
            new_terminator = hlast.Yield(terminator.location, ts.void_t, terminator.value)
            return [*body[0:-1], new_terminator]
        elif fallback_value:
            new_terminator = hlast.Yield(fallback_value.location, ts.void_t, fallback_value)
            return [*body, new_terminator]
        else:
            loc = terminator.location if terminator else concepts.Location.unknown()
            raise CompilationError(
                loc,
                "statement block must be terminated by Yield or Return, or a fallback value must be provided"
            )

    def _create_continuation_function(self, statements: list[hlast.Statement], location: concepts.Location):
        if not statements:
            return None, None
        body = statements[0:-1]
        terminator = statements[-1]

        if isinstance(terminator, hlast.Return):
            return_statement = terminator
        elif isinstance(terminator, hlast.Yield):
            return_statement = hlast.Return(terminator.location, terminator.type_, terminator.value)
        else:
            raise CompilationError(terminator.location, "block terminator must be a Yield or a Return")

        unknown_symbols = self._collect_unknown_symbols(statements)
        parameters = [hlast.Parameter(sym.location, sym.type_, sym.name) for sym in unknown_symbols]
        param_types = [p.type_ for p in parameters]
        result_type = return_statement.value.type_ if return_statement.value else ts.void_t
        function_type = ts.FunctionType(param_types, result_type)
        name = f"__tail_{utility.unique_id()}"
        func = hlast.Function(location, function_type, name, parameters, result_type, [*body, return_statement], False)
        call = hlast.Call(location, result_type, name, unknown_symbols)
        return func, call

    def _collect_unknown_symbols(self, node: Any) -> list[hlast.SymbolRef]:
        collect_symbols_pass = CollectSymbolsPass()
        collect_symbols_pass.visit(node)
        return collect_symbols_pass.unknown_symbols