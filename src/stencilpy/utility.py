def _unique_id_generator():
    i = 0
    while True:
        i = i + 1
        yield i

unique_id = _unique_id_generator()
