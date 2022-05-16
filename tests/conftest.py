import pytest
import numpy as np
from functional.ffront.fbuiltins import Field


class Utils:
    @staticmethod
    def assert_equality(out: Field, ref: np.array):
        out = np.asarray(out)
        truth_arr = np.isclose(out, ref).flatten()
        assert all(truth_arr)


@pytest.fixture
def utils():
    return Utils
