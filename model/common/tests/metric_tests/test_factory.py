import pytest

import icon4py.model.common.test_utils.helpers as helpers
from icon4py.model.common import dimension as dims
from icon4py.model.common.metrics import factory, metric_fields as mf


@pytest.mark.datatest
def test_field_provider(icon_grid, metrics_savepoint, backend):
    fields_factory = factory.MetricsFieldsFactory(icon_grid, metrics_savepoint.z_ifc(), backend)
    height_provider = factory.ProgramFieldProvider(func=mf.compute_z_mc,
                                                   domain={dims.CellDim: (0, icon_grid.num_cells),
                                                           dims.KDim: (0, icon_grid.num_levels)},
                                                   fields=["height"],
                                                   deps=["height_on_interface_levels"], 
                                                   outer=fields_factory)
    fields_factory.register_provider(height_provider)
    functional_determinant_provider = factory.ProgramFieldProvider(func=mf.compute_ddqz_z_half,
                                                                   domain={dims.CellDim: (0,icon_grid.num_cells),
                                                                       dims.KHalfDim: (
                                                                           0,
                                                                           icon_grid.num_levels + 1)},
                                                                   fields=[
                                                                       "functional_determinant_of_the_metrics_on_half_levels"],
                                                                   deps=[
                                                                       "height_on_interface_levels",
                                                                       "height",
                                                                       "model_level_number"],
                                                                   params=[
                                                                       "num_lev"], outer=fields_factory)
    fields_factory.register_provider(functional_determinant_provider)
    
    data = fields_factory.get("functional_determinant_of_the_metrics_on_half_levels",
                              type_=factory.RetrievalType.FIELD)
    ref = metrics_savepoint.ddqz_z_half().ndarray
    assert helpers.dallclose(data.ndarray, ref)
