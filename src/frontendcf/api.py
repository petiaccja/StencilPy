from typing import Optional


_index: Optional[tuple[int, ...]] = None


def set_index(index_: tuple[int, ...]):
    global _index
    _index = index_


def index():
    global _index
    return _index
