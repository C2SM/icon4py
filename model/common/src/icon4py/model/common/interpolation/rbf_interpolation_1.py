import numpy as np

from icon4py.model.common import math


def _compute_rbf_vec_idx_c(c2e2c, c2e, num_cells):
    rbf_vec_idx_c = np.zeros((9, num_cells)) # TODO: do not hard code 9
    rbf_vec_idx_c[0, :] = c2e[c2e2c[:, 0], 0]
    rbf_vec_idx_c[1, :] = c2e[c2e2c[:, 0], 1]
    rbf_vec_idx_c[2, :] = c2e[c2e2c[:, 0], 2]
    rbf_vec_idx_c[3, :] = c2e[c2e2c[:, 1], 0]
    rbf_vec_idx_c[4, :] = c2e[c2e2c[:, 1], 1]
    rbf_vec_idx_c[5, :] = c2e[c2e2c[:, 1], 2]
    rbf_vec_idx_c[6, :] = c2e[c2e2c[:, 2], 0]
    rbf_vec_idx_c[7, :] = c2e[c2e2c[:, 2], 1]
    rbf_vec_idx_c[8, :] = c2e[c2e2c[:, 2], 2]
    return rbf_vec_idx_c

def _gvec2cvec(p_gu, p_gv, p_long, p_lat):
    z_sln = math.sin(p_long)
    z_cln = math.cos(p_long)
    z_slt = math.sin(p_lat)
    z_clt = math.cos(p_lat)

    p_cu = z_sln * p_gu + z_slt * z_cln * p_gv
    p_cu = -1.0 * p_cu
    p_cv = z_cln * p_gu - z_slt * z_sln * p_gv
    p_cw = z_clt * p_gv
    return p_cu, p_cv, p_cw


def _compute_z_xn1_z_xn2(
    lon,
    lat,
    cartesian_center,
    owner_mask,
    lower_bound, # ptr_patch%cells%start_blk(i_rcstartlev,1)
    upper_bound,
    num_cells
):
    z_nx1 = np.zeros((num_cells, 3))
    z_nx2 = np.zeros((num_cells, 3))

    for jc in range(lower_bound, upper_bound):
        #if not owner_mask[jc]: continue
        z_nx1[jc, 0], z_nx1[jc, 1], z_nx1[jc, 2] = _gvec2cvec(1., 0., lon[jc], lat[jc])
        z_norm = math.sqrt(np.dot(z_nx1[jc, :], z_nx1[jc, :]))
        z_nx1[jc, :] = 1. / z_norm * z_nx1[jc, :]
        z_nx2[jc, 0], z_nx2[jc, 1], z_nx2[jc, 2] = _gvec2cvec(0., 1., lon[jc], lat[jc])
        z_norm = math.sqrt(np.dot(z_nx2[jc, :], z_nx2[jc, :]))
        z_nx2[jc, :] = 1. / z_norm * z_nx2[jc, :]

    return z_nx1, z_nx2

def _compute_rbf_vec_scale_c(mean_characteristic_length):
    resol = mean_characteristic_length/1000.0
    rbf_vec_scale_c = 0.5 / (1. + 1.8 * math.log(2.5/resol) ** 3.75) if resol < 2.5 else 0.5
    rbf_vec_scale_c = rbf_vec_scale_c*(resol/0.125)**0.9 if resol <= 0.125 else rbf_vec_scale_c
    return rbf_vec_scale_c

def _compute_arc_length_v(p_x, p_y):
    z_lx = math.sqrt(np.dot(p_x, p_x))
    z_ly = math.sqrt(np.dot(p_y, p_y))

    z_cc = np.dot(p_x, p_y)/(z_lx*z_ly)

    if z_cc > 1.0: z_cc = 1.0
    if z_cc < -1.0: z_cc = -1.0

    p_arc = np.arccos(z_cc)

    return p_arc

def _compute_rhs1_rhs2(
    c2e,
    c2e2c,
    cartesian_center_c,
    cartesian_center_e,
    mean_characteristic_length,
    z_nx1,
    z_nx2,
    istencil,
    owner_mask,
    primal_cart_normal_x,
    jg,
    rbf_vec_kern_c,  # rbf_vec_kern_c from config
    lower_bound,  # ptr_patch%cells%start_blk(i_rcstartlev,1)
    num_cells
):
    rbf_vec_idx_c = _compute_rbf_vec_idx_c(c2e2c, c2e, num_cells)
    rbf_vec_dim_c = 9 # rbf_vec_dim_c from config
    z_rhs1 = np.zeros((num_cells, rbf_vec_dim_c))
    z_rhs2 = np.zeros((num_cells, rbf_vec_dim_c))
    z_rbfval = np.zeros((num_cells, rbf_vec_dim_c))
    z_nx3 = np.zeros((num_cells, 3))
    rbf_vec_scale_c = _compute_rbf_vec_scale_c(mean_characteristic_length)
    for je2 in range(rbf_vec_dim_c):
        for jc in range(lower_bound, num_cells):
            if not owner_mask[jc]: continue
            if je2 > istencil[jc]: continue
            cc_c = (cartesian_center_c[0][jc].ndarray,
                    cartesian_center_c[1][jc].ndarray,
                    cartesian_center_c[2][jc].ndarray)
            ile2 = rbf_vec_idx_c[je2, jc]
            cc_e2 = (cartesian_center_e[0][int(ile2)].ndarray,
             cartesian_center_e[1][int(ile2)].ndarray,
             cartesian_center_e[2][int(ile2)].ndarray)
            z_nx3[jc, 0] = primal_cart_normal_x[0][int(ile2)].ndarray
            z_nx3[jc, 1] = primal_cart_normal_x[1][int(ile2)].ndarray
            z_nx3[jc, 2] = primal_cart_normal_x[2][int(ile2)].ndarray
            z_dist = _compute_arc_length_v(cc_c, cc_e2)
            if rbf_vec_kern_c == 1:  # rbf_vec_kern_c from config
                z_rbfval[jc, je2] = _gaussi(z_dist, rbf_vec_scale_c)
            elif rbf_vec_kern_c == 3:  # rbf_vec_kern_c from config
                z_rbfval[jc, je2] = _inv_multiq(z_dist, rbf_vec_scale_c)

            z_rhs1[jc, je2] = z_rbfval[jc, je2] * np.dot(z_nx1[jc, :], z_nx3[jc, :])
            z_rhs2[jc, je2] = z_rbfval[jc, je2] * np.dot(z_nx2[jc, :], z_nx3[jc, :])

    return z_rhs1, z_rhs2

import cmath


def _compute_z_diag(
    k_dim,
    rbf_vec_dim_c,
    p_a,
    lower_bound, # ptr_patch%cells%start_blk(i_rcstartlev,1)
    maxdim,
    num_cells
):
    # # non-vectorized version
    # p_diag = np.zeros((num_cells, rbf_vec_dim_c))
    # for jc in range(lower_bound, num_cells):
    #     for ji in range(int(k_dim[jc])):
    #         jj = ji
    #         z_sum = p_a[jc, ji, jj]
    #         for jk in reversed(range(ji-1)):
    #             z_sum = z_sum - p_a[jc, ji, jk] * p_a[jc, jj, jk]
    #             if z_sum < 0.:
    #                 a = 1
    #         p_diag[jc, ji] = math.sqrt(z_sum)
    #
    #         for jj in range(ji + 1, int(k_dim[jc])):
    #             z_sum = p_a[jc, ji, jj]
    #             for jk in reversed(range(ji-1)):
    #                 z_sum = z_sum - p_a[jc, ji, jk] * p_a[jc, jj, jk]
    #
    #             p_a[jc, jj, ji] = z_sum / p_diag[jc, ji]

    # vectorized version
    z_sum = np.zeros((num_cells))
    p_diag = np.zeros((num_cells, maxdim))
    for ji in range(maxdim):
        for jj in range(ji, maxdim):
            for jc in range(lower_bound, num_cells):
                if (ji > k_dim[jc] or jj > k_dim[jc]): continue
                z_sum[jc] = p_a[jc, ji, jj]
            for jk in reversed(range(ji-1)):
                for jc in range(lower_bound, num_cells):
                    if (ji > k_dim[jc] or jj > k_dim[jc]): continue
                    z_sum[jc] = z_sum[jc] - p_a[jc, ji, jk] * p_a[jc, jj, jk]
            if ji == jj:
                for jc in range(lower_bound, num_cells):
                    if (ji > k_dim[jc]): continue
                    p_diag[jc, ji] = cmath.sqrt(z_sum[jc])
            else:
                for jc in range(lower_bound, num_cells):
                    if (ji > k_dim[jc] or jj > k_dim[jc]): continue
                    p_a[jc, jj, ji] = z_sum[jc] / p_diag[jc, ji]
    return p_diag

def _gaussi(p_x, p_scale):
    p_rbf_val = p_x / p_scale
    p_rbf_val = -1. * p_rbf_val * p_rbf_val
    p_rbf_val = np.exp(p_rbf_val)
    return p_rbf_val

def _inv_multiq(p_x, p_scale):
    p_rbf_val = p_x / p_scale
    p_rbf_val = p_rbf_val * p_rbf_val
    p_rbf_val = np.sqrt(1. + p_rbf_val)
    p_rbf_val = 1. / p_rbf_val
    return p_rbf_val

def _compute_z_rbfmat_istencil(
    c2e2c,
    c2e,
    cartesian_center_e,
    primal_cart_normal_x,
    mean_characteristic_length,
    owner_mask,
    rbf_vec_dim_c,
    rbf_vec_kern_c,
    lower_bound, # ptr_patch%cells%start_blk(i_rcstartlev,1)
    upper_bound,
    num_cells
):
    jg = 1
    rbf_vec_idx_c = _compute_rbf_vec_idx_c(c2e2c, c2e, num_cells)
    rbf_vec_scale_c = _compute_rbf_vec_scale_c(mean_characteristic_length)
    rbf_vec_stencil_c = np.zeros((num_cells,))
    z_rbfmat = np.zeros((num_cells, rbf_vec_dim_c, rbf_vec_dim_c))
    z_nx1 = np.zeros((num_cells, 3))
    z_nx2 = np.zeros((num_cells, 3))
    istencil = np.zeros(num_cells)
    for je1 in list(range(rbf_vec_dim_c)):
        for je2 in range(je1+1):
            for jc in range(lower_bound, upper_bound):
                if jc == 403:
                    a = 1
                # if not owner_mask[jc]:
                #     istencil[jc] = 0
                #     continue
                rbf_vec_stencil_c[jc] = len(np.argwhere(rbf_vec_idx_c[:, jc] != 0))
                istencil[jc] = rbf_vec_stencil_c[jc]
                ile1 = rbf_vec_idx_c[je1, jc]
                ile2 = rbf_vec_idx_c[je2, jc]
                if (je1 > istencil[jc] or je2 > istencil[jc]):
                    continue
                cc_e1 = (cartesian_center_e[0][int(ile1)].ndarray,
                         cartesian_center_e[1][int(ile1)].ndarray,
                         cartesian_center_e[2][int(ile1)].ndarray)
                cc_e2 = (cartesian_center_e[0][int(ile2)].ndarray,
                         cartesian_center_e[1][int(ile2)].ndarray,
                         cartesian_center_e[2][int(ile2)].ndarray)
                z_nx1[jc, 0] = primal_cart_normal_x[0][int(ile1)].ndarray
                z_nx1[jc, 1] = primal_cart_normal_x[1][int(ile1)].ndarray
                z_nx1[jc, 2] = primal_cart_normal_x[2][int(ile1)].ndarray
                z_nx2[jc, 0] = primal_cart_normal_x[0][int(ile2)].ndarray
                z_nx2[jc, 1] = primal_cart_normal_x[1][int(ile2)].ndarray
                z_nx2[jc, 2] = primal_cart_normal_x[2][int(ile2)].ndarray
                z_nxprod = np.dot(z_nx1[jc, :], z_nx2[jc, :])
                z_dist = _compute_arc_length_v(cc_e1, cc_e2)
                if rbf_vec_kern_c == 1:
                    z_rbfmat[jc, je1, je2] = z_nxprod * _gaussi(z_dist, rbf_vec_scale_c) #[max(jg, 0)])
                elif rbf_vec_kern_c == 3:
                    z_rbfmat[jc, je1, je2] = z_nxprod * _inv_multiq(z_dist, rbf_vec_scale_c) #[max(jg, 0)])

                if je1 > je2:
                    z_rbfmat[jc, je2, je1] = z_rbfmat[jc, je1, je2]

    return z_rbfmat, istencil

def _compute_rbf_vec_coeff(
    k_dim,
    p_a,
    p_diag,
    p_b,
    maxdim,
    lower_bound, # ptr_patch%cells%start_blk(i_rcstartlev,1)
    num_cells
):
    # non-vectorized version
    # p_x = np.zeros((maxdim, num_cells))
    # #z_sum = np.zeros(num_cells)
    # for jc in range(lower_bound, num_cells):
    #     for ji in range(int(k_dim[jc])):
    #         z_sum = p_b[jc, ji]
    #         # z_sum[lower_bound:] = p_b[lower_bound:, ji]
    #         for jj in reversed(range(ji-1)):
    #             # z_sum[lower_bound:] = z_sum[lower_bound:] - p_a[lower_bound:, ji, jj] * p_x[jj, lower_bound:]
    #             z_sum = z_sum - p_a[jc, ji, jj] * p_x[jj, jc]
    #
    #         if p_diag[jc, ji] == 0:
    #             a = 1
    #
    #         p_x[ji, jc] = z_sum / p_diag[jc, ji]
    #
    #     for ji in reversed(range(int(k_dim[jc]))):
    #         z_sum = p_x[ji, jc]
    #         # z_sum[lower_bound:] = p_x[ji, lower_bound:]
    #         # z_sum[lower_bound:] = z_sum[lower_bound:] - p_a[lower_bound:, ji + 1: maxdim, ji] * p_x[ji + 1: maxdim, lower_bound:]
    #         for jj in range(ji+1, int(k_dim[jc])):
    #             z_sum = z_sum - p_a[jc, jj, ji] * p_x[jj, jc]
    #
    #         if p_diag[jc, ji] == 0:
    #             a = 1
    #
    #         p_x[ji, jc] = z_sum / p_diag[jc, ji]

    # vectorized version
    p_x = np.zeros((maxdim, num_cells))
    z_sum = np.zeros(num_cells)
    for ji in range(maxdim):
        for jc in range(lower_bound, num_cells):
            if ji > k_dim[jc]: continue
            z_sum[jc] = p_b[jc, ji]
        for jj in reversed(range(ji-1)):
            for jc in range(lower_bound, num_cells):
                if ji > k_dim[jc]: continue
                z_sum[jc] = z_sum[jc] - p_a[jc, ji, jj] * p_x[jj, jc]

        for jc in range(lower_bound, num_cells):
            if ji > k_dim[jc]: continue
            p_x[ji, jc] = z_sum[jc] / p_diag[jc, ji]

    for ji in reversed(range(maxdim)):
        for jc in range(lower_bound, num_cells):
            if ji > k_dim[jc]: continue
            z_sum[jc] = p_x[ji, jc]
        for jj in range(ji+1, maxdim):
            for jc in range(lower_bound, num_cells):
                if (ji > k_dim[jc] or jj > k_dim[jc]): continue
                z_sum[jc] = z_sum[jc] - p_a[jc, jj, ji] * p_x[jj, jc]

        for jc in range(lower_bound, num_cells):
            if ji > k_dim[jc]: continue
            p_x[ji, jc] = z_sum[jc] / p_diag[jc, ji]

    return p_x

def compute_rbf_vec_coeff(
    c2e2c,
    c2e,
    lon,
    lat,
    cartesian_center_e,
    cartesian_center_c,
    mean_cell_area,
    primal_cart_normal_x,
    owner_mask,
    rbf_vec_dim_c,
    rbf_vec_kern_c,
    maxdim,
    lower_bound, # ptr_patch%cells%start_blk(i_rcstartlev,1)
    num_cells
):
    rbf_vec_coeff_c = np.zeros((maxdim, 3, num_cells))
    mean_characteristic_length = math.sqrt(mean_cell_area)
    jg = 0

    z_rbfmat, istencil = _compute_z_rbfmat_istencil(
        c2e2c,
        c2e,
        cartesian_center_e,
        primal_cart_normal_x,
        mean_characteristic_length,
        owner_mask,
        rbf_vec_dim_c,
        rbf_vec_kern_c,
        lower_bound,  # ptr_patch%cells%start_blk(i_rcstartlev,1)
        num_cells
    )

    z_nx1, z_nx2 = _compute_z_xn1_z_xn2(
        lon,
        lat,
        cartesian_center_c,
        owner_mask,
        lower_bound,  # ptr_patch%cells%start_blk(i_rcstartlev,1)
        num_cells
    )

    z_rhs1, z_rhs2 = _compute_rhs1_rhs2(
        c2e,
        c2e2c,
        cartesian_center_c,
        cartesian_center_e,
        mean_characteristic_length,
        z_nx1,
        z_nx2,
        istencil,
        owner_mask,
        primal_cart_normal_x,
        jg,
        rbf_vec_kern_c,  # rbf_vec_kern_c from config
        lower_bound,  # ptr_patch%cells%start_blk(i_rcstartlev,1)
        num_cells
    )

    z_diag = _compute_z_diag(
        istencil,
        rbf_vec_dim_c,
        z_rbfmat,
        lower_bound, # ptr_patch%cells%start_blk(i_rcstartlev,1)
        maxdim,
        num_cells
    )

    rbf_vec_coeff_c[:, 0, :] = _compute_rbf_vec_coeff(
        istencil,
        z_rbfmat,
        z_diag,
        z_rhs1,
        rbf_vec_dim_c,
        lower_bound,  # ptr_patch%cells%start_blk(i_rcstartlev,1)
        num_cells
    )

    return rbf_vec_coeff_c[:, 0, :]