import copy
import inspect
import ast
from typing import Optional

import stencilir as sir


def _get_location(file: str, node: ast.AST) -> sir.Location:
    return sir.Location(file, node.lineno, node.col_offset)


def _translate_arith_op(op):
    if isinstance(op, ast.Add):
        return sir.ArithmeticFunction.ADD
    elif isinstance(op, ast.Sub):
        return sir.ArithmeticFunction.SUB
    elif isinstance(op, ast.Mult):
        return sir.ArithmeticFunction.MUL
    elif isinstance(op, ast.Div):
        return sir.ArithmeticFunction.DIV
    else:
        raise NotImplementedError("Unknown binary function.")


def _translate_compare_op(op):
    if isinstance(op, ast.Lt):
        return sir.ComparisonFunction.LT
    elif isinstance(op, ast.LtE):
        return sir.ComparisonFunction.LTE
    elif isinstance(op, ast.Gt):
        return sir.ComparisonFunction.GT
    elif isinstance(op, ast.GtE):
        return sir.ComparisonFunction.GTE
    elif isinstance(op, ast.Eq):
        return sir.ComparisonFunction.EQ
    elif isinstance(op, ast.NotEq):
        return sir.ComparisonFunction.NEQ
    else:
        raise NotImplementedError("Unknown comparison function.")


class UsedDefinedVars(ast.NodeVisitor):
    defined_vars: dict[ast.AST, set[str]] = {}
    used_vars: dict[ast.AST, set[str]] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for arg in node.args.args:
            self.visit(arg)
        for statement in node.body:
            self.visit(statement)

    def visit_Name(self, node: ast.Name, **kwargs):
        if isinstance(node.ctx, ast.Load):
            self.used_vars[node] = {node.id}
        elif isinstance(node.ctx, ast.Store):
            self.defined_vars[node] = {node.id}

    def visit_Load(self, node: ast.Load):
        pass

    def visit_Store(self, node: ast.Load):
        pass

    def visit_arg(self, node: ast.arg):
        self.defined_vars[node] = {node.arg}


class DefinedBeforeVars(ast.NodeVisitor):
    defined_vars: dict[ast.AST, set[str]]
    defined_before_vars: dict[ast.AST, set[str]] = {}
    _cumulative: set[str] = set()

    def __init__(self, def_annotations: dict[ast.AST, set[str]]):
        self.defined_vars = def_annotations

    def generic_visit(self, node: ast.AST):
        self.defined_before_vars[node] = self._cumulative
        if node in self.defined_vars:
            self._cumulative = self._cumulative | self.defined_vars[node]
        ast.NodeVisitor.generic_visit(self, node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for arg in node.args.args:
            self.visit(arg)
        for statement in node.body:
            self.visit(statement)

    def visit_If(self, node: ast.If):
        self.defined_before_vars[node] = self._cumulative

        cumulative_original = copy.deepcopy(self._cumulative)
        for statement in node.body:
            self.visit(statement)
        cumulative_then = copy.deepcopy(self._cumulative)
        self._cumulative = copy.deepcopy(cumulative_original)
        for statement in node.orelse:
            self.visit(statement)
        cumulative_else = copy.deepcopy(self._cumulative)

        self._cumulative = cumulative_then | cumulative_else

    def visit_For(self, node: ast.For):
        raise NotImplementedError()

    def visit_Load(self, node: ast.Load):
        pass

    def visit_Store(self, node: ast.Load):
        pass


class UsedAfterVars(ast.NodeVisitor):
    defined_vars: dict[ast.AST, set[str]]
    used_vars: dict[ast.AST, set[str]]
    used_after_vars: dict[ast.AST, set[str]] = {}
    refreshed_vars: dict[ast.AST, set[str]] = {}
    _cumulative: set[str] = set()

    def __init__(self, def_annotations: dict[ast.AST, set[str]], use_annotations: dict[ast.AST, set[str]]):
        self.defined_vars = def_annotations
        self.used_vars = use_annotations

    def generic_visit(self, node: ast.AST):
        self.used_after_vars[node] = self._cumulative
        ast.NodeVisitor.generic_visit(self, node)
        if node in self.used_vars:
            self._cumulative = self._cumulative | self.used_vars[node]
        if node in self.defined_vars:
            self._cumulative = self._cumulative - self.defined_vars[node]
        self.refreshed_vars[node] = self.used_after_vars[node] - self._cumulative

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for statement in reversed(node.body):
            self.visit(statement)

    def visit_If(self, node: ast.If):
        self.used_after_vars[node] = self._cumulative

        cumulative_original = copy.deepcopy(self._cumulative)
        for statement in reversed(node.body):
            self.visit(statement)
        cumulative_then = copy.deepcopy(self._cumulative)
        self._cumulative = copy.deepcopy(cumulative_original)
        for statement in reversed(node.orelse):
            self.visit(statement)
        cumulative_else = copy.deepcopy(self._cumulative)

        self._cumulative = cumulative_then & cumulative_else
        self.refreshed_vars[node] = self.used_after_vars[node] - self._cumulative

    def visit_For(self, node: ast.For):
        raise NotImplementedError()

    def visit_Load(self, node: ast.Load):
        pass

    def visit_Store(self, node: ast.Load):
        pass


class ScopeLeakAnnotator(ast.NodeVisitor):
    use_after_annotations: dict[ast.AST, set[str]] = {}
    refreshes_annotations: dict[ast.AST, set[str]] = {}
    scope_leaks: dict[ast.AST, set[str]] = {}

    def __init__(
            self,
            use_after_annotations: dict[ast.AST, set[str]],
            refreshes_annotations: dict[ast.AST, set[str]]
    ):
        self.use_after_annotations = use_after_annotations
        self.refreshes_annotations = refreshes_annotations

    def visit_If(self, node: ast.If):
        self.scope_leaks[node] = self.refreshes_annotations[node]

        self.generic_visit(node)


class PythonToStencilAST(ast.NodeTransformer):
    file: str
    input_types: list[sir.Type]
    output_types: list[sir.FieldType]
    refreshed_vars: dict[ast.AST, set[str]]
    defined_before_vars: dict[ast.AST, set[str]]

    def __init__(
            self,
            file: str,
            input_types: list[sir.Type],
            output_types: list[sir.Type],
            refreshed_vars: dict[ast.AST, set[str]],
            defined_before_vars: dict[ast.AST, set[str]]
    ):
        self.file = file
        self.input_types = input_types
        self.output_types = output_types
        self.refreshed_vars = refreshed_vars
        self.defined_before_vars = defined_before_vars

    def visit_Module(self, node: ast.Module) -> sir.Module:
        functions: list[sir.Function] = []
        stencils: list[sir.Stencil] = []
        for statement_node in node.body:
            statement = self.visit(statement_node)
            if isinstance(statement, tuple) and isinstance(statement[0], sir.Function):
                functions.append(statement[0])
                stencils.append(statement[1])
            else:
                raise ValueError("Illegal statement in module body.")
        return sir.Module(functions, stencils, None)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> tuple[sir.Function, sir.Stencil]:
        loc = _get_location(self.file, node)
        rank = self.output_types[0].num_dimensions
        stencil_body = [self.visit(statement) for statement in node.body]
        names = [arg.arg for arg in node.args.args]
        parameters = [sir.Parameter(name, type_) for name, type_ in zip(names, self.input_types)]
        stencil = sir.Stencil(
            node.name,
            parameters,
            [output_type.element_type for output_type in self.output_types],
            stencil_body,
            rank,
            loc
        )
        function_body = [
            sir.Apply(
                node.name,
                [sir.SymbolRef(param.name, loc) for param in parameters],
                [sir.SymbolRef(f"__out_{i}", loc) for i in range(len(self.output_types))],
                [],
                [0] * rank,
                loc
            ),
            sir.Return([], loc)
        ]
        function = sir.Function(
            "main",
            [*parameters, *[sir.Parameter(f"__out_{i}", t) for i, t in enumerate(self.output_types)]],
            [],
            function_body,
            loc
        )
        return function, stencil

    def visit_Assign(self, node: ast.Assign) -> sir.Assign:
        names = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.append(str(target.id))
        exprs = self.visit(node.value)
        return sir.Assign(names, [exprs], _get_location(self.file, node))

    def visit_Return(self, node: ast.Return) -> sir.Return:
        exprs = self.visit(node.value)
        return sir.Return([exprs], _get_location(self.file, node))

    def visit_If(self, node: ast.If) -> sir.Assign | sir.If:
        loc = _get_location(self.file, node)
        refreshed_vars = self.refreshed_vars[node]
        yield_stmt = sir.Yield([sir.SymbolRef(var, loc) for var in refreshed_vars], loc)
        if_stmt = sir.If(
            self.visit(node.test),
            [*[self.visit(stmt) for stmt in node.body], yield_stmt],
            [*[self.visit(stmt) for stmt in (node.orelse or [])], yield_stmt],
            loc
        )
        if refreshed_vars:
            assign_stmt = sir.Assign(list(refreshed_vars), [if_stmt], loc)
            return assign_stmt
        return if_stmt

    def visit_For(self, node: ast.For) -> sir.Assign | sir.For:
        raise NotImplementedError()

    def visit_Name(self, node: ast.Name) -> sir.SymbolRef:
        if not isinstance(node.ctx, ast.Load):
            raise ValueError("Names for stores have to be handled separately.")
        return sir.SymbolRef(node.id, _get_location(self.file, node))

    def visit_BinOp(self, node: ast.BinOp) -> sir.ArithmeticOperator:
        op = _translate_arith_op(node.op)
        left = self.visit(node.left)
        right = self.visit(node.right)
        return sir.ArithmeticOperator(left, right, op, _get_location(self.file, node))

    def visit_Compare(self, node: ast.Compare) -> sir.Expression:
        loc = _get_location(self.file, node)
        values = [self.visit(node.left), *[self.visit(v) for v in node.comparators]]
        operators = [_translate_compare_op(op) for op in node.ops]

        def get_compare_chain(
                condition,
                left,
                values,
                operators,
        ) -> sir.Expression:
            if not operators:
                return sir.Constant.boolean(True, loc)
            i = len(values)
            op = operators[0]
            right = values[0]
            return sir.If(
                condition,
                [
                    sir.Assign([f"__value_{i}"], [right]),
                    sir.Assign(
                        ["__condition"],
                        [sir.ComparisonOperator(left, sir.SymbolRef(f"__value_{i}", loc), op, loc)],
                        loc
                    ),
                    sir.Yield([get_compare_chain(
                        sir.SymbolRef("__condition", loc),
                        sir.SymbolRef(f"__value_{i}", loc),
                        values[1:],
                        operators[1:]
                    )], loc)
                ],
                [
                    sir.Yield([sir.Constant.boolean(False, loc)], loc)
                ],
                loc
            )

        return get_compare_chain(
            sir.Constant.boolean(True, loc),
            values[0],
            values[1:],
            operators
        )

    def visit_Constant(self, node: ast.Constant) -> sir.Constant:
        raise NotImplementedError()


def parse_function(fun: callable, input_types: list[sir.Type], output_types: list[sir.Type]):
    source = inspect.getsource(fun)
    source_file = inspect.getsourcefile(fun)
    lineno = inspect.getsourcelines(fun)[1]
    python_ast = ast.parse(source)

    use_defs = UsedDefinedVars()
    use_defs.visit(python_ast)

    print("\n")
    defs_before = DefinedBeforeVars(use_defs.defined_vars)
    defs_before.visit(python_ast)
    print({k: v for k, v in defs_before.defined_before_vars.items() if not isinstance(k, (ast.Name, ast.Module))})

    uses_after = UsedAfterVars(use_defs.defined_vars, use_defs.used_vars)
    uses_after.visit(python_ast)
    print({k: v for k, v in uses_after.used_after_vars.items() if not isinstance(k, (ast.Name, ast.Module))})

    ast.increment_lineno(python_ast, lineno - 1)
    return PythonToStencilAST(
        source_file,
        input_types,
        output_types,
        defs_before.defined_before_vars,
        uses_after.refreshed_vars
    ).visit(python_ast)
