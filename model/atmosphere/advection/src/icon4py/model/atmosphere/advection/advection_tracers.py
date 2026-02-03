
import numpy as np
from icon4py.model.common.grid import geometry_attributes as geometry_attrs, horizontal as h_grid
from icon4py.model.common import dimension as dims, constants
import scipy
import math

# lsq_dim_unk = 2
# lsq_dim_c = 3
# lsq_wgt_exp = 2

# xloc = geometry_attrs.CELL_LAT
# yloc = geometry_attrs.CELL_LON
cell_domain = h_grid.domain(dims.CellDim)
min_rlcell_int = cell_domain(h_grid.Zone.LOCAL)


# grid_sphere_radius = constants.EARTH_RADIUS,
# cell_owner_mask = np.zeros((cell_domain))# change to grid_savepoint.c_owner_mask().ndarray
# domain_length = 42.0 # change to grid.global_properties.domain_length
# domain_height = 110.5 # change to grid.global_properties.domain_height

def compute_lsq_pseudoinv(cell_owner_mask, lsq_pseudoinv, z_lsq_mat_c, lsq_weights_c, start_idx, min_rlcell_int, lsq_dim_unk: int, lsq_dim_c: int):
    for jja in range(lsq_dim_unk):
        for jjb in range(lsq_dim_c):
            for jjk in range(lsq_dim_unk):
                for jc in range(start_idx, min_rlcell_int):
                    A = z_lsq_mat_c[jc, :, :]
                    u, s, v_t, icheck = scipy.linalg.lapack.dgesdd(A)
                    lsq_pseudoinv[jc, jja, jjb] = np.where(cell_owner_mask[jc],
                                                                 (lsq_pseudoinv[jc, jja, jjb] +
                                                                  v_t[jjk, jja] / s[jjk] * u[jjb, jjk]
                                                                  * lsq_weights_c[jc, jjb]),
                                                                 lsq_pseudoinv[jc, jja, jjb])
    return lsq_pseudoinv

def compute_lsq_weights_c(z_dist_g, lsq_weights_c, jc, lsq_dim_stencil, lsq_wgt_exp):
    for js in range(lsq_dim_stencil):
        z_norm = np.sqrt(np.dot(z_dist_g[jc, js, :], z_dist_g[jc, js, :]))
        lsq_weights_c[jc, js] = 1. / (z_norm ** lsq_wgt_exp)
    return lsq_weights_c[jc,:]


def plane_torus_closest_coordinates(cc_cv_x, cc_cell, domain_length, domain_height):
    x0 = cc_cv_x[0]
    x1 = cc_cv_x[1]
    y0 = cc_cell[0]
    y1 = cc_cell[1]

    x1 = np.where(abs(x1 - x0) <= 0.5 * domain_length, x1, np.where(x0 > x1, x1 + domain_length, x1 - domain_length))
    y1 = np.where(abs(y1 - y0) <= 0.5 * domain_height, y1, np.where(y0 > y1, y1 + domain_height, y1 - domain_height))
    return x1, y1

def gnomonic_proj(lon_c, lat_c, lon, lat):
    cosc = math.sin(lat_c)*math.sin(lat) + math.cos(lat_c)*math.cos(lat)*math.cos(lon - lon_c)
    zk = 1./cosc

    x = zk*math.cos(lat)*math.sin(lon - lon_c)
    y = zk*(math.cos(lat_c)*math.sin(lat) - math.sin(lat_c)*math.cos(lat)*math.cos(lon - lon_c))
    return x,y

def compute_lsq_moments(lsq_moments, cc_cv, cc_vert, domain_length, domain_height, lsq_dim_unk, nverts, min_rlcell_int):
#for jc in range(min_rlcell_int):
    # # for sphere
    # xytemp_v = np.zeros((nverts, 2))
    # for jec in range(nverts):
    #     xytemp_v[jec, 0] = vertex_lon[jlv[jc, jec]].ndarray
    #     xytemp_v[jec, 1] = vertex_lat[jlv[jc, jec]].ndarray
    fx = np.zeros((nverts,))
    fy = np.zeros((nverts,))
    delx = np.zeros((nverts,))
    dely = np.zeros((nverts,))
    distxy_v = np.zeros((nverts,lsq_dim_unk))
    # # for sphere
    # for jec in range(nverts):
    #     distxy_v[jec, :] = gnomonic_proj(xloc.ndarray[jc], yloc.ndarray[jc], xytemp_v[jec,0], xytemp_v[jec,1])# ,distxy_v(jec, 1), distxy_v(jec, 2) )
    # distxy_v[:, :] = constants.EARTH_RADIUS * distxy_v[:, :]

    # for torus
    for jec in range(nverts):
        cc_vert[jec] = plane_torus_closest_coordinates(cc_cv, cc_vert[jec], domain_length, domain_height)
        distxy_v[jec, 0] = cc_vert[jec][0] - cc_cv[0]
        distxy_v[jec, 1] = cc_vert[jec][1] - cc_cv[1]

    for jec in range(nverts):
        jecp = 0 if jec == nverts-1 else jec+1
        delx[jec] = distxy_v[jecp,0] - distxy_v[jec,0]
        dely[jec] = distxy_v[jecp,1] - distxy_v[jec,1]
        fx[jec] = distxy_v[jecp,0] + distxy_v[jec,0]

    z_rcarea = 2. / np.dot(fx, dely)

    for jec in range(nverts):
        jecp = 0 if jec == nverts-1 else jec+1
        fx[jec] = distxy_v[jecp,0] ** 2 + distxy_v[jecp,0] * distxy_v[jec,0] + distxy_v[jec,0] ** 2
        fy[jec] = distxy_v[jecp, 1] ** 2 + distxy_v[jecp, 1] * distxy_v[jec, 1] + distxy_v[jec, 1] ** 2

    lsq_moments[0] = z_rcarea / 6. * np.dot(fx, dely)
    lsq_moments[1] = -z_rcarea / 6. * np.dot(fy, delx)

    return lsq_moments

#########################
# Sphere

def lsq_compute_coeff_cell_sphere(cell_lat, cell_lon, vertex_lat, vertex_lon, c2e2c, c2v, cell_owner_mask, grid_sphere_radius, lsq_dim_unk, lsq_dim_c, lsq_wgt_exp, start_idx, min_rlcell_int):
    lsq_dim_stencil = 3  # change to c2e[1], which should be 3 #ccells % num_edges(jc_range)
    nverts = 3  # change to c2e[1], which should be 3 #cells % num_edges(jc_range)

    z_dist_g = np.zeros((min_rlcell_int, lsq_dim_c, 2))
    lsq_weights_c = np.zeros((min_rlcell_int, lsq_dim_stencil))
    lsq_pseudoinv = np.zeros((min_rlcell_int, lsq_dim_unk, lsq_dim_c))
    z_lsq_mat_c = np.zeros((min_rlcell_int, lsq_dim_c, lsq_dim_c))
    lsq_moments = np.zeros((min_rlcell_int, 2))
    lsq_moments_hat = np.zeros((min_rlcell_int, lsq_dim_stencil, lsq_dim_unk))

    jlv = c2v
    #lsq_moments = compute_lsq_moments(lsq_moments, cell_lon, cell_lat, vertex_lon, vertex_lat, jlv, lsq_dim_unk, nverts, min_rlcell_int)
    for jc in range(start_idx, min_rlcell_int):
        xytemp_c = np.zeros((lsq_dim_stencil, 2))
        ilc_s = c2e2c[jc, :lsq_dim_stencil]
        for js in range(lsq_dim_stencil):
            xytemp_c[js, 0] = cell_lon[ilc_s[js]].ndarray
            xytemp_c[js, 1] = cell_lat[ilc_s[js]].ndarray

        xytemp_v = np.zeros((nverts, 2))
        jlv = c2v[jc, :nverts]
        for jec in range(nverts):
            xytemp_v[jec, 0] = vertex_lon[jlv[jec]].ndarray
            xytemp_v[jec, 1] = vertex_lat[jlv[jec]].ndarray

        for js in range(lsq_dim_stencil):
            z_dist_g[jc, js, :] = gnomonic_proj(cell_lon.ndarray[jc], cell_lat.ndarray[jc], xytemp_c[js, 0], xytemp_c[js, 1])
        z_dist_g[jc, :lsq_dim_stencil, :] = grid_sphere_radius * z_dist_g[jc, :lsq_dim_stencil,:]

        lsq_weights_c[jc, :] = compute_lsq_weights_c(z_dist_g, lsq_weights_c, jc, lsq_dim_stencil, lsq_wgt_exp)
        lsq_weights_c[jc, :] = lsq_weights_c[jc, :] / np.max(lsq_weights_c[jc, :])

        for js in range(min(lsq_dim_unk, lsq_dim_c)):
            z_lsq_mat_c[jc, js, js] = np.where(cell_owner_mask[jc], 1.0, 0.0)

        for js in range(lsq_dim_stencil):
            lsq_moments_hat[jc, js, 0] = lsq_moments[ilc_s[js], 0] + z_dist_g[jc, js, 0]
            lsq_moments_hat[jc, js, 1] = lsq_moments[ilc_s[js], 1] + z_dist_g[jc, js, 1]
        for js in range(lsq_dim_c):
            for ju in range(lsq_dim_unk):
                z_lsq_mat_c[jc, js, ju] = lsq_weights_c[jc, js] *(lsq_moments_hat[jc, js, ju] - lsq_moments[jc, ju])
            if lsq_dim_stencil < lsq_dim_c:
                z_lsq_mat_c[jc, lsq_dim_c, :] = 0.

    lsq_pseudoinv = compute_lsq_pseudoinv(cell_owner_mask, lsq_pseudoinv, z_lsq_mat_c, lsq_weights_c, start_idx, min_rlcell_int, lsq_dim_unk, lsq_dim_c)

    return lsq_pseudoinv


#######################
# Torus grid

def lsq_compute_coeff_cell_torus(vertex_x, vertex_y, cell_center_x, cell_center_y, c2e2c, c2v, cell_owner_mask, domain_length, domain_height, lsq_dim_unk, lsq_dim_c, lsq_wgt_exp, start_idx, min_rlcell_int):
    lsq_dim_stencil = 3  # change to c2e[1], which should be 3 #ccells % num_edges(jc_range)
    nverts = 3  # change to c2e[1], which should be 3 #cells % num_edges(jc_range)

    z_dist_g = np.zeros((min_rlcell_int, lsq_dim_c, 2))
    lsq_weights_c = np.zeros((min_rlcell_int, lsq_dim_stencil))
    lsq_pseudoinv = np.zeros((min_rlcell_int, lsq_dim_unk, lsq_dim_c))
    z_lsq_mat_c = np.zeros((min_rlcell_int, lsq_dim_c, lsq_dim_c))
    lsq_moments = np.zeros((min_rlcell_int, 2))
    lsq_moments_hat = np.zeros((min_rlcell_int, lsq_dim_stencil, lsq_dim_unk))

    jlv = c2v
    cc_cv = (cell_center_x.ndarray, cell_center_y.ndarray)
    # lsq_moments = compute_lsq_moments(lsq_moments, cc_cv, cc_vert, domain_length, domain_height, jlv, lsq_dim_unk, nverts, min_rlcell_int)
    for jc in range(start_idx, min_rlcell_int):
        ilc_s = c2e2c[jc, :lsq_dim_stencil]
        cc_cell = np.zeros((lsq_dim_stencil, 2))
        cc_cv = (cell_center_x.ndarray[jc], cell_center_y.ndarray[jc])
        jlv = c2v[jc]
        for js in range(lsq_dim_stencil):
            cc_cell[js, 0] = cell_center_x.ndarray[ilc_s[js]]
            cc_cell[js, 1] = cell_center_y.ndarray[ilc_s[js]]

        cc_vert = np.zeros((nverts, 2))
        for jec in range(nverts):
            cc_vert[jec, 0] = vertex_x[jlv[jec]]
            cc_vert[jec, 1] = vertex_y[jlv[jec]]

        # lsq_moments = compute_lsq_moments(lsq_moments, cc_cv, cc_vert, domain_length, domain_height, lsq_dim_unk,
        #                                   nverts, min_rlcell_int)

        for js in range(lsq_dim_stencil):
            cc_cell[js, :] = plane_torus_closest_coordinates(cc_cv, cc_cell[js], domain_length, domain_height)
            z_dist_g[jc, js, 0] = cc_cell[js, 0] - cc_cv[0] # CELL_CENTER_X
            z_dist_g[jc, js, 1] = cc_cell[js, 1] - cc_cv[1] # CELL_CENTER_Y


        lsq_weights_c[jc, :] = compute_lsq_weights_c(z_dist_g, lsq_weights_c, jc, lsq_dim_stencil, lsq_wgt_exp)
        lsq_weights_c[jc, :] = lsq_weights_c[jc, :] / np.max(lsq_weights_c[jc, :])

        for js in range(min(lsq_dim_unk, lsq_dim_c)):
            z_lsq_mat_c[jc, js, js] = np.where(cell_owner_mask[jc], 1.0, 0.0)

        for js in range(lsq_dim_stencil):
            lsq_moments_hat[jc, js, 0] = lsq_moments[ilc_s[js], 0] + z_dist_g[jc, js, 0]
            lsq_moments_hat[jc, js, 1] = lsq_moments[ilc_s[js], 1] + z_dist_g[jc, js, 1]
        for js in range(lsq_dim_c):
            for ju in range(lsq_dim_unk):
                z_lsq_mat_c[jc, js, ju] = lsq_weights_c[jc, js] * (lsq_moments_hat[jc, js, ju] - lsq_moments[jc, ju])
            if lsq_dim_stencil < lsq_dim_c:
                z_lsq_mat_c[jc, js, ju] = 0.

    lsq_pseudoinv = compute_lsq_pseudoinv(cell_owner_mask, lsq_pseudoinv, z_lsq_mat_c, lsq_weights_c, start_idx, min_rlcell_int, lsq_dim_unk, lsq_dim_c)

    return lsq_pseudoinv



