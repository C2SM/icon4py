import pytest
import numpy as np
from functional.ffront.fbuiltins import Field


class Utils:
    @staticmethod
    def assert_equality(out: Field, ref: np.array):
        out = np.asarray(out)
        assert np.allclose(out, ref)


@pytest.fixture
def utils():
    return Utils
