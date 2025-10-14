import dataclasses

import boltons.funcutils
import pytest

from icon4py.model.atmosphere.diffusion import config_reader
import omegaconf as oc


@dataclasses.dataclass
class Foo:
    a: int
    b: str
    c: list[int]

def test_config_reader_raises_missing_value():
    reader = config_reader.ConfigReader(Foo)
    with pytest.raises(oc.MissingMandatoryValue):
        reader.get_config()






def test_config_reader():
    reader = config_reader.ConfigReader(Foo(1, "b", [1,2,3]))
    foo = reader.get_config()
    assert foo.a == 1
    assert foo.b == "b"
    assert foo.c == [1,2,3]


def test_config_reader_update():
    reader = config_reader.ConfigReader(Foo(1, "b", [1, 2]))
    original_config = reader.get_config()
    assert original_config.a == 1
    assert original_config.b == "b"
    assert original_config.c == [1,2]
    reader.update(Foo(2, "b*", [8,7]))
    update = reader.get_config()
    assert update.a == 2
    assert update.b == "b*"
    # TODO should Sequences replace or append?, same question for general dicts
    assert update.c == [8,7]

def test_config_reader_type_validates():
    reader = config_reader.ConfigReader(Foo)
    wrong_foo = Foo("a", "b", [1, 2, 3])
    with pytest.raises(oc.ValidationError):
        reader.update(wrong_foo)


def test_config_reader_default_config_equals_config():
    reader = config_reader.ConfigReader(Foo(1, "b", [1,2,3]))
    foo = reader.get_config()
    default = reader.default
    assert default == foo
