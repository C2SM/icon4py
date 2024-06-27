import pytest

from icon4py.model.common.metrics import factory
from icon4py.model.common.test_utils.helpers import dallclose


@pytest.mark.datatest
def test_field_provider(icon_grid, metrics_savepoint, backend):
    fields_factory = factory.MetricsFieldsFactory(icon_grid, metrics_savepoint.z_ifc(), backend)
       
    data = fields_factory.get("functional_determinant_of_the_metrics_on_half_levels", type_=factory.RetrievalType.FIELD)
    ref = metrics_savepoint.ddqz_z_half().ndarray   
    assert dallclose(data.ndarray, ref)
    
    