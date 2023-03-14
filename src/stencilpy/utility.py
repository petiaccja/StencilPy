def _unique_id_generator():
    i = 0
    while True:
        i = i + 1
        yield i


_unique_id = _unique_id_generator()


def unique_id():
    global _unique_id
    return next(_unique_id)
