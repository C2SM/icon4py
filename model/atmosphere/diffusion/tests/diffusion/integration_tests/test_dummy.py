# python
import pytest

@pytest.fixture(scope="session", params=["a", "b"])
def resource(request):
    # setup runs once per param for the whole session
    val = f"setup-{request.param}"
    yield val
    # teardown runs at session end for this param
    print(f"teardown-{request.param}")
    del val

def test_one(resource):
    assert resource.startswith("setup-")

def test_two(resource):
    assert len(resource) > 0
