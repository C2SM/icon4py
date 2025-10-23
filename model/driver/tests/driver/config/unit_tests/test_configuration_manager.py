from icon4py.model.driver.config import configuration_manager
import pathlib
import pytest
def test_configuration_manager_happy_path():
    path = pathlib.Path(__file__).parent.joinpath("model.yaml")
    manager = configuration_manager.ConfigurationManager(path)
    manager.read_config()
    assert len(manager.get_configured_modules()) == 3
    assert "diffusion" in manager.get_configured_modules()
    assert manager.get_config("diffusion").temperature_boundary_diffusion_denom == 45.0



