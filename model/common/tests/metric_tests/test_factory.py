from icon4py.model.common.metrics import factory
from icon4py.model.common.metrics.factory import RetrievalType


def test_field_provider(icon_grid, metrics_savepoint):
    z_ifc = factory.SimpleFieldProvider(icon_grid, metrics_savepoint.z_ifc(), factory._attrs["height_on_interface_levels"])
    z_mc = factory.FieldProvider(grid=icon_grid, deps=(z_ifc,), attrs=factory._attrs["height"])
    data_array = z_mc(RetrievalType.FIELD)
    
    #assert dallclose(metrics_savepoint.z_mc(), data_array.ndarray)
    
    
    #provider = factory.FieldProviderImpl(icon_grid, (z_ifc, z_mc), attrs=factory.attrs["functional_determinant_of_the_metrics_on_half_levels"])
    #provider()
    
    