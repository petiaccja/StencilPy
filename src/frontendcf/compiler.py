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


class PythonToStencilAST(ast.NodeTransformer):
    file: str
    input_types: list[sir.Type]
    output_types: list[sir.FieldType]

    def __init__(self, file: str, input_types: list[sir.Type], output_types: list[sir.Type]):
        self.file = file
        self.input_types = input_types
        self.output_types = output_types

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
                [0]*rank,
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
        return sir.Assign(names, exprs, _get_location(self.file, node))

    def visit_Name(self, node: ast.Name) -> sir.SymbolRef:
        if not isinstance(node.ctx, ast.Load):
            raise ValueError("Names for stores have to be handled separately.")
        return sir.SymbolRef(node.id, _get_location(self.file, node))

    def visit_Return(self, node: ast.Return) -> sir.Return:
        exprs = self.visit(node.value)
        return sir.Return([exprs], _get_location(self.file, node))

    def visit_If(self, node: ast.If) -> sir.Assign | sir.If:
        raise NotImplementedError()

    def visit_For(self, node: ast.For) -> sir.Assign | sir.For:
        raise NotImplementedError()

    def visit_BinOp(self, node: ast.BinOp) -> sir.ArithmeticOperator:
        op = _translate_arith_op(node.op)
        left = self.visit(node.left)
        right = self.visit(node.right)
        return sir.ArithmeticOperator(left, right, op, _get_location(self.file, node))

    def visit_Compare(self, node: ast.Compare) -> sir.Expression:
        loc = _get_location(self.file, node)
        values = [self.visit(node.left), *[self.visit(v) for v in node.comparators]]
        operators = [_translate_compare_op(op) for op in node.ops]
        lefts = values[0:-1]
        rights = values[1:]
        results = [
            sir.ComparisonOperator(left, right, op, loc)
            for left, right, op in zip(lefts, rights, operators)
        ]
        combined = results[0]
        for result in results[1:]:
            combined = sir.ArithmeticOperator(combined, result, sir.ArithmeticFunction.BIT_AND, loc)
        return combined

    def visit_Constant(self, node: ast.Constant) -> sir.Constant:
        raise NotImplementedError()


def parse_function(fun: callable, input_types: list[sir.Type], output_types: list[sir.Type]):
    source = inspect.getsource(fun)
    source_file = inspect.getsourcefile(fun)
    lineno = inspect.getsourcelines(fun)[1]
    python_ast = ast.parse(source)
    ast.increment_lineno(python_ast, lineno)
    return PythonToStencilAST(source_file, input_types, output_types).visit(python_ast)
