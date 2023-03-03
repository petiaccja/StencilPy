import dataclasses
import ast

from typing import Any
from stencilpy import concepts


def format_error(location: concepts.Location, message: str):
    return f"{location}: error: {message}"


@dataclasses.dataclass
class CompilationError(Exception):
    def __init__(self, location: concepts.Location, message: str):
        self.location = location
        self.message = message

    def __str__(self):
        return format_error(self.location, self.message)


class InternalCompilerError(CompilationError):
    def __init__(self, location: concepts.Location, message: str):
        super().__init__(location, f"internal compiler error: {message}")


class UndefinedSymbolError(CompilationError):
    def __init__(self, location: concepts.Location, symbol: str):
        super().__init__(location, f"undefined symbol `{symbol}`")
        self.symbol = symbol


class UnsupportedLanguageError(CompilationError):
    def __init__(self, location: concepts.Location, node: ast.AST):
        super().__init__(location, f"unsupported python language feature `{type(node).__name__}`")


class MissingDimensionError(CompilationError):
    def __init__(self, location: concepts.Location, field_type: Any, dim: concepts.Dimension):
        super().__init__(location, f"field of type {field_type} has no dimension {dim}")


class ArgumentCountError(CompilationError):
    def __init__(self, location: concepts.Location, num_expected: int, num_provided: int):
        super().__init__(location, f"function expects {num_expected} arguments but {num_provided} were provided")


class ArgumentTypeError(CompilationError):
    def __init__(self, location: concepts.Location, expected_types: list, provided_types: list):
        expected = [', '.join(str(t) for t in expected_types)]
        provided = [', '.join(str(t) for t in provided_types)]
        super().__init__(
            location,
            f"function expects argument types {expected} but types {provided} were provided"
        )

class ArgumentCompatibilityError(CompilationError):
    def __init__(self, location: concepts.Location, operation: str, arg_types: list):
        args = ', '.join(str(t) for t in arg_types)
        super().__init__(location, f"{operation} does not accept arguments of incompatible type {args}")