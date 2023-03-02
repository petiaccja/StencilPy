import dataclasses
from typing import Any, Callable, Optional


@dataclasses.dataclass
class SymbolScope:
    info: Optional[Any]
    symbols: dict[str, Any]


class SymbolTable:
    scopes: list[SymbolScope]

    def __init__(self):
        self.scopes = [SymbolScope(None, {})]

    def assign(self, name: str, value: Any):
        self.scopes[-1].symbols[name] = value

    def lookup(self, name: str) -> Any:
        for scope in reversed(self.scopes):
            if name in scope.symbols:
                return scope.symbols[name]
        return None

    def infos(self):
        return (scope.info for scope in reversed(self.scopes))

    def scope(self, callback: Callable, scope_info: Optional[Any] = None) -> Any:
        self.scopes.append(SymbolScope(scope_info, {}))
        result = callback()
        self.scopes.pop(-1)
        return result
