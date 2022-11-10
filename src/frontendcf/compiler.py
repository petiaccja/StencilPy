import copy
import inspect
import ast
from dataclasses import dataclass

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
    def __init__(self):
        self.defined_vars: dict[ast.AST, set[str]] = {}
        self.used_vars: dict[ast.AST, set[str]] = {}

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
    def __init__(self, def_annotations: dict[ast.AST, set[str]]):
        self.defined_vars = def_annotations
        self.defined_before_vars: dict[ast.AST, set[str]] = {}
        self._cumulative: set[str] = set()

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

        self.visit(node.test)

    def visit_For(self, node: ast.For):
        self.defined_before_vars[node] = self._cumulative

        for statement in node.body:
            self.visit(statement)
        if node.orelse:
            raise NotImplementedError("For loop else clause not supported.")
        self.visit(node.target)

    def visit_Load(self, node: ast.Load):
        pass

    def visit_Store(self, node: ast.Load):
        pass


class UsedAfterVars(ast.NodeVisitor):
    def __init__(self, defined_vars: dict[ast.AST, set[str]], used_vars: dict[ast.AST, set[str]]):
        self.defined_vars: dict[ast.AST, set[str]] = defined_vars
        self.used_vars: dict[ast.AST, set[str]] = used_vars
        self.used_after_vars: dict[ast.AST, set[str]] = {}
        self.refreshed_vars: dict[ast.AST, set[str]] = {}
        self._cumulative: set[str] = set()

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

    def visit_Assign(self, node: ast.Assign):
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)

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
        self.visit(node.test)

    def visit_For(self, node: ast.For):
        self.used_after_vars[node] = self._cumulative

        defs = UsedDefinedVars()
        for statement in reversed(node.body):
            self.visit(statement)
            defs.visit(statement)
        if node.orelse:
            raise NotImplementedError("For loop else clause not supported.")
        self.visit(node.target)

        defined_vars: set[str] = set()
        for defined_vars_by_node in defs.defined_vars.values():
            for var in defined_vars_by_node:
                defined_vars.add(var)

        self.refreshed_vars[node] = (self.used_after_vars[node] - self._cumulative) | (defined_vars & self._cumulative)

    def visit_Load(self, node: ast.Load):
        pass

    def visit_Store(self, node: ast.Load):
        pass


@dataclass
class PythonToStencilAST(ast.NodeTransformer):
    file: str
    input_types: list[sir.Type]
    output_types: list[sir.FieldType]
    refreshed_vars: dict[ast.AST, set[str]]
    defined_before_vars: dict[ast.AST, set[str]]

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
        loc = _get_location(self.file, node)
        refreshed_vars = self.refreshed_vars[node]
        if not isinstance(node.target, ast.Name):
            raise ValueError("For loops can have only a single index variable.")
        index_var_name = node.target.id

        yield_stmt = sir.Yield([sir.SymbolRef(var, loc) for var in refreshed_vars], loc)
        start, end, step = self._visit_range(node.iter)
        for_stmt = sir.For(
            start,
            end,
            step,
            index_var_name,
            [*[self.visit(stmt) for stmt in node.body], yield_stmt],
            [sir.SymbolRef(var, loc) for var in refreshed_vars],
            list(refreshed_vars),
            loc
        )
        if refreshed_vars:
            assign_stmt = sir.Assign(list(refreshed_vars), [for_stmt], loc)
            return assign_stmt
        return for_stmt

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
                return condition
            i = len(values)
            op = operators[0]
            right = values[0]
            return sir.If(
                condition,
                [
                    sir.Assign([f"__value_{i}"], [right], loc),
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

    def visit_Call(self, node: ast.Call) -> sir.Expression:
        loc = _get_location(self.file, node)
        if isinstance(node.func, ast.Name) and node.func.id == "index":
            return sir.Index(loc)
        raise NotImplementedError("Function calls are not supported.")

    def visit_Subscript(self, node: ast.Subscript) -> sir.Sample:
        loc = _get_location(self.file, node)
        field = self.visit(node.value)
        index = self.visit(node.slice)
        return sir.Sample(field, index, loc)

    def visit_Constant(self, node: ast.Constant) -> sir.Constant:
        loc = _get_location(self.file, node)
        if isinstance(node.value, float):
            return sir.Constant.floating(node.value, sir.ScalarType.FLOAT64, loc)
        elif isinstance(node.value, bool):
            return sir.Constant.boolean(node.value, loc)
        elif isinstance(node.value, int):
            return sir.Constant.integral(node.value, sir.ScalarType.SINT64, loc)
        else:
            raise NotImplementedError(f"Constant of type {type(node.value)} is not supported.")

    def _visit_range(self, node: ast.AST):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "range":
            raise ValueError("range(...) call expected")
        loc = _get_location(self.file, node)
        args = node.args
        c0 = sir.Constant.index(0, loc)
        c1 = sir.Constant.index(1, loc)

        def as_index(arg):
            return sir.Cast(arg, sir.ScalarType.INDEX, loc)

        if len(args) == 1:
            return c0, as_index(self.visit(args[0])), c1
        if len(args) == 2:
            return as_index(self.visit(args[0])), as_index(self.visit(args[1])), c1
        return as_index(self.visit(args[0])), as_index(self.visit(args[1])), as_index(self.visit(args[2]))



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
    print("def before: ", {k: v for k, v in defs_before.defined_before_vars.items() if not isinstance(k, (ast.Name, ast.Module))})

    uses_after = UsedAfterVars(use_defs.defined_vars, use_defs.used_vars)
    uses_after.visit(python_ast)
    print("used after: ", {k: v for k, v in uses_after.used_after_vars.items() if not isinstance(k, (ast.Name, ast.Module))})
    print("refreshed:  ", {k: v for k, v in uses_after.refreshed_vars.items() if not isinstance(k, (ast.Name, ast.Module))})

    ast.increment_lineno(python_ast, lineno - 1)
    return PythonToStencilAST(
        source_file,
        input_types,
        output_types,
        uses_after.refreshed_vars,
        defs_before.defined_before_vars
    ).visit(python_ast)
