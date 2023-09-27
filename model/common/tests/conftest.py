"""
Initialize pytest.

Workaround for pytest not discovering those configuration function, when they are added to the
diffusion_test/conftest.py folder
"""
from icon4py.model.common.test_utils.helpers import backend, mesh  # noqa: F401 # fixtures
from icon4py.model.common.test_utils.pytest_config import (  # noqa: F401 # pytest config
    pytest_addoption,
    pytest_configure,
    pytest_runtest_setup,
)
