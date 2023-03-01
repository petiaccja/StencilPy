from typing import Any, Callable


class SymbolTable:
    tables: list[dict[str, Any]]

    def __init__(self):
        self.tables = [{}]

    def assign(self, name: str, value: Any):
        self.tables[-1][name] = value

    def lookup(self, name: str) -> Any:
        for table in reversed(self.tables):
            if name in table:
                return table[name]
        return None

    def scope(self, callback: Callable) -> Any:
        self.tables.append({})
        result = callback()
        self.tables.pop(-1)
        return result
