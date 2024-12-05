import icon4py.model.common.test_utils.datatest_utils as dt_utils
from icon4py.model.common.grid import geometry_attributes as attrs


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, rank, grid_file",
    [
        (dt_utils.REGIONAL_EXPERIMENT, 0, dt_utils.REGIONAL_EXPERIMENT),
        #(dt_utils.REGIONAL_EXPERIMENT, 0, dt_utils.REGIONAL_EXPERIMENT),
    ],
)
def test_compute_rbf_vec_coeff(grid_savepoint, metrics_savepoint, backend, data_provider, icon_grid, experiment, rank, grid_file):
    rbf_vec_coeff_c1_ref = data_provider.from_interpolation_savepoint().rbf_vec_coeff_c1()
    lower_bound = metrics_savepoint.i_startidx_c()
    upper_bound = metrics_savepoint.i_endidx_c()
    #z_diag_c = metrics_savepoint.z_diag_c()
    z_nx1_c = metrics_savepoint.z_nx1_c().ndarray
    lon = grid_savepoint.cell_center_lon().ndarray # these are edges
    lat = grid_savepoint.cell_center_lat().ndarray
    lon_e = grid_savepoint.edges_center_lon()
    lat_e = grid_savepoint.edges_center_lat()

    c2e = icon_grid.connectivities[dims.C2EDim]
    c2e2c = icon_grid.connectivities[dims.C2E2CDim]
    z_rbfmat_c = metrics_savepoint.z_rbfmat_c().ndarray
    owner_mask = grid_savepoint.c_owner_mask()
    mean_cell_area = grid_savepoint.mean_cell_area()
    num_cells = icon_grid.num_cells
    rbf_vec_dim_c = 9
    rbf_vec_kern_c = 1
    import math
    mean_characteristic_length = math.sqrt(mean_cell_area)
    cartesian_center_e = (metrics_savepoint.cartesian_center_e_c_x()[:],
                          metrics_savepoint.cartesian_center_e_c_y()[:],
                          metrics_savepoint.cartesian_center_e_c_z()[:],)
    primal_cart_normal = (metrics_savepoint.primal_cart_normal_c_x()[:],
                          metrics_savepoint.primal_cart_normal_c_y()[:],
                          metrics_savepoint.primal_cart_normal_c_z()[:],)
    z_rbfmat, istencil = _compute_z_rbfmat_istencil(
        c2e2c,
        c2e,
        cartesian_center_e,
        primal_cart_normal,
        mean_characteristic_length,
        True,
        rbf_vec_dim_c,
        rbf_vec_kern_c,
        lower_bound=lower_bound - 1,  # ptr_patch%cells%start_blk(i_rcstartlev,1)
        upper_bound=num_cells,
        num_cells=num_cells
    )
    maxdim = rbf_vec_dim_c
    grid_geometry = get_grid_geometry(backend, grid_file)
    primal_cart_normal_x = grid_geometry.get(attrs.EDGE_NORMAL_X)
    primal_cart_normal_y = grid_geometry.get(attrs.EDGE_NORMAL_Y)
    primal_cart_normal_z = grid_geometry.get(attrs.EDGE_NORMAL_Z)
    primal_cart_normal = (primal_cart_normal_x, primal_cart_normal_y, primal_cart_normal_z)

    cartesian_center_e = geographical_to_cartesian_on_edges(lat_e, lon_e)
    cartesian_center_c = geographical_to_cartesian_on_cells(lat, lon)
    lower_bound = icon_grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    mean_characteristic_length = math.sqrt(mean_cell_area)

    lon = metrics_savepoint.lon_c().ndarray
    lat = metrics_savepoint.lat_c().ndarray

    z_nx1, z_nx2 = _compute_z_xn1_z_xn2(
        lon,
        lat,
        None,
        True,
        0,  # ptr_patch%cells%start_blk(i_rcstartlev,1)
        lon.shape[0],
        lon.shape[0]
    )

    rbf_vec_coeff = compute_rbf_vec_coeff(
        c2e2c,
        c2e,
        lon.asnumpy(),
        lat.asnumpy(),
        cartesian_center_e,
        cartesian_center_c,
        mean_cell_area,
        primal_cart_normal,
        owner_mask.asnumpy(),
        rbf_vec_dim_c,
        rbf_vec_kern_c,
        maxdim,
        lower_bound, # ptr_patch%cells%start_blk(i_rcstartlev,1)
        num_cells
    )

    assert test_helpers.dallclose(np.transpose(rbf_vec_coeff), rbf_vec_coeff_c1_ref.asnumpy())