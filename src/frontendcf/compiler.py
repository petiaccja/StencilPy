import inspect
import ast

import stencilir as sir


def _get_location(file: str, node: ast.AST) -> sir.Location:
    return sir.Location(file, node.lineno, node.col_offset)


class PythonToStencilAST(ast.NodeTransformer):
    file: str
    input_types: list[type]
    output_types: list[type]

    def __init__(self, file: str, input_types: list[type], output_types: list[type]):
        self.file = file
        self.input_types = input_types
        self.output_types = output_types

    def visit_FunctionDef(self, node: FunctionDef) -> sir.Stencil:
        raise NotImplementedError()

    def visit_Assign(self, node: Assign) -> sir.Assign:
        raise NotImplementedError()

    def visit_If(self, node: If) -> sir.Assign | sir.If:
        raise NotImplementedError()

    def visit_For(self, node: For) -> sir.Assign | sir.For:
        raise NotImplementedError()

    def visit_BinOp(self, node: BinOp) -> sir.ArithmeticOperator:
        if isinstance(node.op, ast.Add):
            op = sir.ArithmeticFunction.ADD
        elif isinstance(node.op, ast.Sub):
            op = sir.ArithmeticFunction.SUB
        elif isinstance(node.op, ast.Mult):
            op = sir.ArithmeticFunction.MUL
        elif isinstance(node.op, ast.Div):
            op = sir.ArithmeticFunction.DIV
        else:
            raise NotImplementedError("Unknown binary function.")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return sir.ArithmeticOperator(left, right, op, _get_location(self.file, node))

    def visit_Compare(self, node: Compare) -> sir.ComparisonOperator:
        raise NotImplementedError()

    def visit_Constant(self, node: Constant) -> sir.Constant:
        raise NotImplementedError()


def parse_function(fun: callable, input_types: list[type], output_types: list[type]):
    source = inspect.getsource(fun)
    source_file = inspect.getsourcefile(fun)
    lineno = inspect.getsourcelines(fun)[1]
    python_ast = ast.parse(source)
    ast.increment_lineno(python_ast, lineno)
    return PythonToStencilAST(source_file, input_types, output_types).visit()
