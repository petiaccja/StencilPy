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
        super().__init__(location, f"bug in the compiler: {message}")


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
