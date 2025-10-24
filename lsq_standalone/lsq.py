import numpy
from mesh_generator import mesh_generator

def lsq_stencil_create(lsq_dim_c, e2c, c2e2c, c2e2c2e2c):

    if lsq_dim_c == 3:
        #
        # 3-point stencil
        #
        (idx_c, dim_stencil) = create_stencil_c3(e2c, c2e2c)
    elif lsq_dim_c == 9:
        #
        # 9-point stencil
        #
        (idx_c, dim_stencil) = create_stencil_c9(e2c, c2e2c2e2c)
    elif lsq_dim_c == 12:
        #
        # 12-point stencil
        #
        (idx_c, dim_stencil) = create_stencil_c12()
    else:
        print("Could not create lsq stencil; invalid stencil size lsq_dim_c=", lsq_dim_c)
        sys.exit('routine lsq_stencil_create')
    return (idx_c, dim_stencil)

def create_stencil_c3(e2c, c2e2c):
    """
    For each cell create a 3-point stencil which consists of the 3 cells
    surrounding the control volume, i.e. the direct neighbors.
    """

    print("create_stencil_c3: create 3-point stencil")

    #! sanity check
    #IF (ANY((/SIZE(idx_c,3),SIZE(blk_c,3)/) /= 3)) THEN
    #  CALL finish(routine, "Invalid size of output fields")
    #ENDIF

    # the cell and block indices are copied from p_patch%cells%neighbor_idx
    # and p_patch%cells%neighbor_blk

    idx_c = c2e2c
    dim_stencil = e2c.shape[0]
    return (idx_c, dim_stencil)

def create_stencil_c9(e2c, c2e2c2e2c):
    """
    For each cell create a 9-point stencil which consists of the
    3 cells surrounding the control volume, i.e. the direct neighbors,
    and the neighbors of the neighbors.
    """

    print("create_stencil_c9: create 9-point stencil")

    #! sanity check
    #IF (ANY((/SIZE(idx_c,3),SIZE(blk_c,3)/) /= 9)) THEN
    #  CALL finish(routine, "Invalid size of output fields")
    #ENDIF

    idx_c = c2e2c2e2c
    dim_stencil = e2c.shape[0]
    return (idx_c, dim_stencil)

def lsq_compute_coeff_cell(
    lsq_dim_c,
    lsq_dim_unk,
    lsq_wgt_exp
):
    """
    This routine bifurcates into lsq_compute_coeff_cell based on geometry type

    Anurag Dipankar, MPIM (2013-04)
    """
    if geometry_type == planar_torus_geometry:
        lsq_compute_coeff_cell_torus( lsq_dim_c, lsq_dim_unk, lsq_wgt_exp )
    elif geometry_type == sphere_geometry:
        lsq_compute_coeff_cell_sphere( lsq_dim_c, lsq_dim_unk, lsq_wgt_exp )
    else:
        sys.exit('lsq_compute_coeff_cell: Undefined geometry type')
    return

def plane_torus_closest_coordinates(
    cc_cv,
    cc_cell,
    domain_length,
    domain_height
):
    length_of_torus = domain_length
    height_of_torus = domain_height
    for i in range(len(cc_cv[:, 0])):
        if abs(cc_cell[i, 0] - cc_cv[i, 0]) > length_of_torus * 0.5:
            if cc_cv[i, 0] > cc_cell[i, 0]:
                cc_cell[i, 0] = cc_cell[i, 0] + length_of_torus
            else:
                cc_cell[i, 0] = cc_cell[i, 0] - length_of_torus
        if abs(cc_cell[i, 1] - cc_cv[i, 1]) > height_of_torus * 0.5:
            if cc_cv[i, 1] > cc_cell[i, 1]:
                cc_cell[i, 1] = cc_cell[i, 1] + height_of_torus
            else:
                cc_cell[i, 1] = cc_cell[i, 1] - height_of_torus
    return cc_cell

def lsq_compute_coeff_cell_torus(
    lsq_dim_c,
    lsq_dim_unk,
    lsq_wgt_exp
):
    """
    This is same routine as lsq_compute_coeff_cell_sphere just modified for
    flat geometry
    """
    print("mo_interpolation:lsq_compute_coeff_cell_torus")
    (nodes,
     c2v_table,
     e2v_table,
     v2c_table,
     v2e_table,
     e2c_table,
     e2c2v_table,
     c2e_table,
     e2c2e0_table,
     e2c2e_table,
     c2e2cO_table,
     c2e2c_table,
     c2e2c2e_table,
     c2e2c2e2c_table,
     cartesian_vertex_coordinates,
     cartesian_cell_centers,
     cartesian_edge_centers,
     primal_edge_length,
     edge_orientation,
     area) = mesh_generator()
    cc_cv = cartesian_cell_centers
    (idx_c, dim_stencil) = lsq_stencil_create(lsq_dim_c = 3, e2c = e2c_table, c2e2c = c2e2c_table, c2e2c2e2c = c2e2c2e2c_table)
    cc_cell = cc_cv[idx_c[:], :]
    domain_length = numpy.max(cartesian_cell_centers[:, 0]) - numpy.min(cartesian_cell_centers[:, 0])
    domain_height = numpy.max(cartesian_cell_centers[:, 1]) - numpy.min(cartesian_cell_centers[:, 1])
    z_dist_g = numpy.empty(cc_cell.shape, dtype=float)
    z_norm = numpy.empty(cc_cell[:, :, 0].shape, dtype=float)
    lsq_weights_c = numpy.empty(cc_cell[:, :, 0].shape, dtype=float)
    for js in range(lsq_dim_c):
        #Get the actual location of the cell w.r.t the cc_cv
        cc_cell[:, js] = plane_torus_closest_coordinates(cc_cv, cc_cell[:, js], domain_length, domain_height)

        #the distance vector: z coord is 0
        #last index equals x or y
        z_dist_g[:, js, :] = cc_cell[:, js, :] - cc_cv[:, :]

        z_norm[:, js] = numpy.sqrt(numpy.square(z_dist_g[:, js, 0]) + numpy.square(z_dist_g[:, js, 1]))

        #
        # weights for weighted least squares system
        #
        lsq_weights_c[:, js] = 1.0 / numpy.power(z_norm[:, js], lsq_wgt_exp)

    #
    # Normalization
    #
    lsq_weights_c[:, js] = lsq_weights_c[:, js] / numpy.max(lsq_weights_c[:, :], axis=1)

    #
    # 4. for each cell, calculate LSQ design matrix A
    #
    # Take care that z_lsq_mat_c isn't singular
    z_lsq_mat_c_dim = min(lsq_dim_c, lsq_dim_unk)
    z_lsq_mat_c = numpy.zeros((len(cc_cell[:, 0, 0]), lsq_dim_c, lsq_dim_unk), dtype=float)
    for js in range(z_lsq_mat_c_dim):
        z_lsq_mat_c[:, js, js] = 1.0

    # line and block indices of cells in the stencil
    # ilc_s(1:ptr_ncells(jc,jb)) = ptr_int_lsq%lsq_idx_c(jc,jb,1:ptr_ncells(jc,jb))
    # ibc_s(1:ptr_ncells(jc,jb)) = ptr_int_lsq%lsq_blk_c(jc,jb,1:ptr_ncells(jc,jb))


    # Calculate full moments lsq_moments_hat(ilc_s(js),ibc_s(js),ju)
    #
    # Storage docu for x^ny^m:
    # lsq_moments_hat(:,:,:,1) : \hat{x^1y^0}
    # lsq_moments_hat(:,:,:,2) : \hat{x^0y^1}
    # lsq_moments_hat(:,:,:,3) : \hat{x^2y^0}
    # lsq_moments_hat(:,:,:,4) : \hat{x^0y^2}
    # lsq_moments_hat(:,:,:,5) : \hat{x^1y^1}
    # lsq_moments_hat(:,:,:,6) : \hat{x^3y^0}
    # lsq_moments_hat(:,:,:,7) : \hat{x^0y^3}
    # lsq_moments_hat(:,:,:,8) : \hat{x^2y^1}
    # lsq_moments_hat(:,:,:,9) : \hat{x^1y^2}
    #

    lsq_moments = numpy.zeros((len(z_dist_g[:, 0, 0]), len(z_dist_g[0, :, 0]), 9), dtype=float)
    lsq_moments_hat = numpy.empty((len(z_dist_g[:, 0, 0]), len(z_dist_g[0, :, 0]), lsq_dim_unk), dtype=float)
    for js in range(len(z_dist_g[0, :, 0])):
        lsq_moments_hat[:, js, 0:2] = lsq_moments[c2e2c_table[:, js], js, 0:2] + z_dist_g[:, js, 0:2]
    if len(lsq_moments_hat[0, 0, :]) > 2:
        for js in range(len(z_dist_g[0, :, 0])):
            lsq_moments_hat[:, js, 2:4] = lsq_moments[c2e2c_table[:, js], js, 2:4] + 2 * lsq_moments[c2e2c_table[:, js], js, 0:2] * z_dist_g[:, js, 0:2] + numpy.square(z_dist_g[:, js, 0:2])
            lsq_moments_hat[:, js, 4] = lsq_moments[c2e2c_table[:, js], js, 4] + lsq_moments[c2e2c_table[:, js], js, 0] * z_dist_g[:, js, 1] + lsq_moments[c2e2c_table[:, js], js, 1] * z_dist_g[:, js, 0] + z_dist_g[:, js, 0] * z_dist_g[:, js, 1]
    if len(lsq_moments_hat[0, 0, :]) > 5:
        for js in range(len(z_dist_g[0, :, 0])):
            lsq_moments_hat[:, js, 5:7] = lsq_moments[c2e2c_table[:, js], js, 5:7] + 3 * lsq_moments[c2e2c_table[:, js], js, 2:4] * z_dist_g[:, js, 0:2] + 3 * lsq_moments[c2e2c_table[:, js], js, 0:2] * numpy.square(z_dist_g[:, js, 0:2]) + numpy.power(z_dist_g[:, js, 0:2], 3)
            lsq_moments_hat[:, js, 7] = lsq_moments[c2e2c_table[:, js], js, 7] + lsq_moments[c2e2c_table[:, js], js, 2] * z_dist_g[:, js, 1] + 2 * lsq_moments[c2e2c_table[:, js], js, 4] * z_dist_g[:, js, 0] + 2 * lsq_moments[c2e2c_table[:, js], js, 0] * z_dist_g[:, js, 0] * z_dist_g[:, js, 1] + lsq_moments[c2e2c_table[:, js], js, 1] * numpy.square(z_dist_g[:, js, 0]) + numpy.square(z_dist_g[:, js, 0]) * z_dist_g[:, js, 1]
            lsq_moments_hat[:, js, 8] = lsq_moments[c2e2c_table[:, js], js, 8] + lsq_moments[c2e2c_table[:, js], js, 3] * z_dist_g[:, js, 0] + 2 * lsq_moments[c2e2c_table[:, js], js, 4] * z_dist_g[:, js, 1] + 2 * lsq_moments[c2e2c_table[:, js], js, 1] * z_dist_g[:, js, 1] * z_dist_g[:, js, 0] + lsq_moments[c2e2c_table[:, js], js, 0] * numpy.square(z_dist_g[:, js, 1]) + numpy.square(z_dist_g[:, js, 1]) * z_dist_g[:, js, 0]
    # loop over rows of lsq design matrix (all cells in the stencil)
    for js in range(lsq_dim_c):
        # loop over columns of lsq design matrix (number of unknowns)
        for ju in range(lsq_dim_unk):
            z_lsq_mat_c[:, js, ju] = lsq_weights_c[:, js] * (lsq_moments_hat[:, js, ju] - lsq_moments[:, js, ju])
    #
    # compute QR decomposition and Singular Value Decomposition (SVD)
    # of least squares design matrix A. For the time being both methods are
    # retained.

    #
    # 5a. QR-factorization of design matrix A
    #
#    if not llsq_svd:
    if not False:
        (z_qmat, z_rmat) = numpy.linalg.qr(z_lsq_mat_c)

        # 7. Save transposed Q-Matrix
        lsq_qtmat_c = numpy.transpose(z_qmat[:, 0:lsq_dim_c, 0:lsq_dim_unk], axes=(0, 2, 1))

        # 8. Save R-Matrix
        #
        # a. Save reciprocal values of the diagonal elements
        #
        lsq_rmat_rdiag_c = numpy.empty((len(z_rmat[:, 0]), lsq_dim_unk), dtype=float)
        for ju in range(lsq_dim_unk):
            lsq_rmat_rdiag_c[:, ju] = 1.0 / z_rmat[:, ju, ju]

        #
        # b. Save upper triangular elements without diagonal elements in a 1D-array
        #    (starting from the bottom right)
        #
        cnt = 1
        for jrow in range(lsq_dim_unk - 1, 0, -1):
            nel = lsq_dim_unk - jrow
            cnt = cnt + nel
        lsq_rmat_utri_c = numpy.empty((len(z_rmat[:, 0]), cnt - 1 + nel - 1), dtype=float)
        cnt = 1
        for jrow in range(lsq_dim_unk - 1, 0, -1):
            # number of elements to store
            nel = lsq_dim_unk - jrow
            lsq_rmat_utri_c[:, cnt - 1:cnt + nel - 1] = z_rmat[:, jrow - 1, jrow:lsq_dim_unk]
            cnt = cnt + nel

        # Multiply ith column of the transposed Q-matrix (corresponds to the
        # different members of the stencil) with the ith weight. This avoids
        # multiplication of the RHS of the LSQ-System with this weight during
        # runtime.
        for ju in range(lsq_dim_unk):
            for js in range(lsq_dim_c):
                lsq_qtmat_c[:, ju, js] = lsq_qtmat_c[:, ju, js] * lsq_weights_c[:, js]
    return

lsq_compute_coeff_cell_torus(
    lsq_dim_c = 3,
    lsq_dim_unk = 2,
    lsq_wgt_exp = 0.4
)
