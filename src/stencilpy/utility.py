from typing import Iterable, Any
import itertools


def _unique_id_generator():
    i = 0
    while True:
        i = i + 1
        yield i


_unique_id = _unique_id_generator()


def unique_id():
    global _unique_id
    return next(_unique_id)


def flatten(values: Iterable[Iterable[Any]]) -> list[Any]:
    return list(itertools.chain(*values))


def flatten_recursive(value: Any) -> list[Any]:
    if not isinstance(value, Iterable):
        return [value]
    return list(itertools.chain(*(flatten_recursive(v) for v in value)))
