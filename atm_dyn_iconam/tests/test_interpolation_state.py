import numpy as np
import pytest


@pytest.mark.datatest
def test_cecdim(interpolation_savepoint, icon_grid):
    interpolation_fields = interpolation_savepoint.construct_interpolation_state()
    geofac_n2s = np.asarray(interpolation_fields.geofac_n2s)
    geofac_n2s_nbh = np.asarray(interpolation_fields.geofac_n2s_nbh)
    assert np.count_nonzero(geofac_n2s_nbh) > 0
    c2cec= icon_grid.get_c2cec_connectivity().table
    ported = geofac_n2s_nbh[c2cec]
    assert ported.shape == geofac_n2s[:, 1:].shape
    assert np.allclose(ported, geofac_n2s[:, 1:])
