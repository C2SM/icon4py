import gt4py.next as gtx
import pytest

import icon4py.model.common.test_utils.helpers as helpers
from icon4py.model.common import dimension as dims
from icon4py.model.common.io import cf_utils
from icon4py.model.common.metrics import factory, metric_fields as mf
from icon4py.model.common.settings import xp


def test_check_dependencies_on_register(icon_grid, backend):
    fields_factory = factory.FieldsFactory(icon_grid, backend)
    provider = factory.ProgramFieldProvider(func=mf.compute_z_mc,
                                                           domain={dims.CellDim: (0, icon_grid.num_cells),
                                                                   dims.KDim: (0, icon_grid.num_levels)},
                                                           fields=["height"],
                                                           deps=["height_on_interface_levels"],
                                                           )
    with pytest.raises(ValueError) as e:
        fields_factory.register_provider(provider)
        assert e.value.match("'height_on_interface_levels' not found")
    

@pytest.mark.datatest
def test_field_provider(icon_grid, metrics_savepoint, backend):
    fields_factory = factory.FieldsFactory(icon_grid, backend)
    k_index = gtx.as_field((dims.KDim,), xp.arange(icon_grid.num_levels + 1, dtype=gtx.int32))
    z_ifc = metrics_savepoint.z_ifc()

    pre_computed_fields = factory.PrecomputedFieldsProvider(
        {"height_on_interface_levels": z_ifc, cf_utils.INTERFACE_LEVEL_STANDARD_NAME: k_index})
    
    fields_factory.register_provider(pre_computed_fields)
   
    height_provider = factory.ProgramFieldProvider(func=mf.compute_z_mc,
                                                   domain={dims.CellDim: (0, icon_grid.num_cells),
                                                           dims.KDim: (0, icon_grid.num_levels)},
                                                   fields=["height"],
                                                   deps=["height_on_interface_levels"], 
                                                   )
    fields_factory.register_provider(height_provider)
    functional_determinant_provider = factory.ProgramFieldProvider(func=mf.compute_ddqz_z_half,
                                                                   domain={dims.CellDim: (0,icon_grid.num_cells),
                                                                       dims.KHalfDim: (
                                                                           0,
                                                                           icon_grid.num_levels + 1)},
                                                                   fields=[
                                                                       "functional_determinant_of_metrics_on_interface_levels"],
                                                                   deps=[
                                                                       "height_on_interface_levels",
                                                                       "height",
                                                                       cf_utils.INTERFACE_LEVEL_STANDARD_NAME],
                                                                   params={
                                                                       "num_lev": icon_grid.num_levels})
    fields_factory.register_provider(functional_determinant_provider)
    
    data = fields_factory.get("functional_determinant_of_metrics_on_interface_levels",
                              type_=factory.RetrievalType.FIELD)
    ref = metrics_savepoint.ddqz_z_half().ndarray
    assert helpers.dallclose(data.ndarray, ref)
