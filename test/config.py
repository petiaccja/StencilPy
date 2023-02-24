import pytest


@pytest.fixture(params=[False, True])
def use_jit(request):
    yield request.param