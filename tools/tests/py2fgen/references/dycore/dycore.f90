module dycore
   use, intrinsic :: iso_c_binding
   implicit none

   public :: solve_nh_run

   public :: solve_nh_init

   public :: grid_init

   interface

      function solve_nh_run_wrapper(rho_now, &
                                    rho_now_size_0, &
                                    rho_now_size_1, &
                                    rho_new, &
                                    rho_new_size_0, &
                                    rho_new_size_1, &
                                    exner_now, &
                                    exner_now_size_0, &
                                    exner_now_size_1, &
                                    exner_new, &
                                    exner_new_size_0, &
                                    exner_new_size_1, &
                                    w_now, &
                                    w_now_size_0, &
                                    w_now_size_1, &
                                    w_new, &
                                    w_new_size_0, &
                                    w_new_size_1, &
                                    theta_v_now, &
                                    theta_v_now_size_0, &
                                    theta_v_now_size_1, &
                                    theta_v_new, &
                                    theta_v_new_size_0, &
                                    theta_v_new_size_1, &
                                    vn_now, &
                                    vn_now_size_0, &
                                    vn_now_size_1, &
                                    vn_new, &
                                    vn_new_size_0, &
                                    vn_new_size_1, &
                                    w_concorr_c, &
                                    w_concorr_c_size_0, &
                                    w_concorr_c_size_1, &
                                    ddt_vn_apc_ntl1, &
                                    ddt_vn_apc_ntl1_size_0, &
                                    ddt_vn_apc_ntl1_size_1, &
                                    ddt_vn_apc_ntl2, &
                                    ddt_vn_apc_ntl2_size_0, &
                                    ddt_vn_apc_ntl2_size_1, &
                                    ddt_w_adv_ntl1, &
                                    ddt_w_adv_ntl1_size_0, &
                                    ddt_w_adv_ntl1_size_1, &
                                    ddt_w_adv_ntl2, &
                                    ddt_w_adv_ntl2_size_0, &
                                    ddt_w_adv_ntl2_size_1, &
                                    theta_v_ic, &
                                    theta_v_ic_size_0, &
                                    theta_v_ic_size_1, &
                                    rho_ic, &
                                    rho_ic_size_0, &
                                    rho_ic_size_1, &
                                    exner_pr, &
                                    exner_pr_size_0, &
                                    exner_pr_size_1, &
                                    exner_dyn_incr, &
                                    exner_dyn_incr_size_0, &
                                    exner_dyn_incr_size_1, &
                                    ddt_exner_phy, &
                                    ddt_exner_phy_size_0, &
                                    ddt_exner_phy_size_1, &
                                    grf_tend_rho, &
                                    grf_tend_rho_size_0, &
                                    grf_tend_rho_size_1, &
                                    grf_tend_thv, &
                                    grf_tend_thv_size_0, &
                                    grf_tend_thv_size_1, &
                                    grf_tend_w, &
                                    grf_tend_w_size_0, &
                                    grf_tend_w_size_1, &
                                    mass_fl_e, &
                                    mass_fl_e_size_0, &
                                    mass_fl_e_size_1, &
                                    ddt_vn_phy, &
                                    ddt_vn_phy_size_0, &
                                    ddt_vn_phy_size_1, &
                                    grf_tend_vn, &
                                    grf_tend_vn_size_0, &
                                    grf_tend_vn_size_1, &
                                    vn_ie, &
                                    vn_ie_size_0, &
                                    vn_ie_size_1, &
                                    vt, &
                                    vt_size_0, &
                                    vt_size_1, &
                                    mass_flx_me, &
                                    mass_flx_me_size_0, &
                                    mass_flx_me_size_1, &
                                    mass_flx_ic, &
                                    mass_flx_ic_size_0, &
                                    mass_flx_ic_size_1, &
                                    vn_traj, &
                                    vn_traj_size_0, &
                                    vn_traj_size_1, &
                                    dtime, &
                                    lprep_adv, &
                                    at_initial_timestep, &
                                    divdamp_fac_o2, &
                                    ndyn_substeps, &
                                    idyn_timestep) bind(c, name="solve_nh_run_wrapper") result(rc)
         import :: c_int, c_double, c_bool, c_ptr
         integer(c_int) :: rc  ! Stores the return code

         type(c_ptr), value, target :: rho_now

         integer(c_int), value :: rho_now_size_0

         integer(c_int), value :: rho_now_size_1

         type(c_ptr), value, target :: rho_new

         integer(c_int), value :: rho_new_size_0

         integer(c_int), value :: rho_new_size_1

         type(c_ptr), value, target :: exner_now

         integer(c_int), value :: exner_now_size_0

         integer(c_int), value :: exner_now_size_1

         type(c_ptr), value, target :: exner_new

         integer(c_int), value :: exner_new_size_0

         integer(c_int), value :: exner_new_size_1

         type(c_ptr), value, target :: w_now

         integer(c_int), value :: w_now_size_0

         integer(c_int), value :: w_now_size_1

         type(c_ptr), value, target :: w_new

         integer(c_int), value :: w_new_size_0

         integer(c_int), value :: w_new_size_1

         type(c_ptr), value, target :: theta_v_now

         integer(c_int), value :: theta_v_now_size_0

         integer(c_int), value :: theta_v_now_size_1

         type(c_ptr), value, target :: theta_v_new

         integer(c_int), value :: theta_v_new_size_0

         integer(c_int), value :: theta_v_new_size_1

         type(c_ptr), value, target :: vn_now

         integer(c_int), value :: vn_now_size_0

         integer(c_int), value :: vn_now_size_1

         type(c_ptr), value, target :: vn_new

         integer(c_int), value :: vn_new_size_0

         integer(c_int), value :: vn_new_size_1

         type(c_ptr), value, target :: w_concorr_c

         integer(c_int), value :: w_concorr_c_size_0

         integer(c_int), value :: w_concorr_c_size_1

         type(c_ptr), value, target :: ddt_vn_apc_ntl1

         integer(c_int), value :: ddt_vn_apc_ntl1_size_0

         integer(c_int), value :: ddt_vn_apc_ntl1_size_1

         type(c_ptr), value, target :: ddt_vn_apc_ntl2

         integer(c_int), value :: ddt_vn_apc_ntl2_size_0

         integer(c_int), value :: ddt_vn_apc_ntl2_size_1

         type(c_ptr), value, target :: ddt_w_adv_ntl1

         integer(c_int), value :: ddt_w_adv_ntl1_size_0

         integer(c_int), value :: ddt_w_adv_ntl1_size_1

         type(c_ptr), value, target :: ddt_w_adv_ntl2

         integer(c_int), value :: ddt_w_adv_ntl2_size_0

         integer(c_int), value :: ddt_w_adv_ntl2_size_1

         type(c_ptr), value, target :: theta_v_ic

         integer(c_int), value :: theta_v_ic_size_0

         integer(c_int), value :: theta_v_ic_size_1

         type(c_ptr), value, target :: rho_ic

         integer(c_int), value :: rho_ic_size_0

         integer(c_int), value :: rho_ic_size_1

         type(c_ptr), value, target :: exner_pr

         integer(c_int), value :: exner_pr_size_0

         integer(c_int), value :: exner_pr_size_1

         type(c_ptr), value, target :: exner_dyn_incr

         integer(c_int), value :: exner_dyn_incr_size_0

         integer(c_int), value :: exner_dyn_incr_size_1

         type(c_ptr), value, target :: ddt_exner_phy

         integer(c_int), value :: ddt_exner_phy_size_0

         integer(c_int), value :: ddt_exner_phy_size_1

         type(c_ptr), value, target :: grf_tend_rho

         integer(c_int), value :: grf_tend_rho_size_0

         integer(c_int), value :: grf_tend_rho_size_1

         type(c_ptr), value, target :: grf_tend_thv

         integer(c_int), value :: grf_tend_thv_size_0

         integer(c_int), value :: grf_tend_thv_size_1

         type(c_ptr), value, target :: grf_tend_w

         integer(c_int), value :: grf_tend_w_size_0

         integer(c_int), value :: grf_tend_w_size_1

         type(c_ptr), value, target :: mass_fl_e

         integer(c_int), value :: mass_fl_e_size_0

         integer(c_int), value :: mass_fl_e_size_1

         type(c_ptr), value, target :: ddt_vn_phy

         integer(c_int), value :: ddt_vn_phy_size_0

         integer(c_int), value :: ddt_vn_phy_size_1

         type(c_ptr), value, target :: grf_tend_vn

         integer(c_int), value :: grf_tend_vn_size_0

         integer(c_int), value :: grf_tend_vn_size_1

         type(c_ptr), value, target :: vn_ie

         integer(c_int), value :: vn_ie_size_0

         integer(c_int), value :: vn_ie_size_1

         type(c_ptr), value, target :: vt

         integer(c_int), value :: vt_size_0

         integer(c_int), value :: vt_size_1

         type(c_ptr), value, target :: mass_flx_me

         integer(c_int), value :: mass_flx_me_size_0

         integer(c_int), value :: mass_flx_me_size_1

         type(c_ptr), value, target :: mass_flx_ic

         integer(c_int), value :: mass_flx_ic_size_0

         integer(c_int), value :: mass_flx_ic_size_1

         type(c_ptr), value, target :: vn_traj

         integer(c_int), value :: vn_traj_size_0

         integer(c_int), value :: vn_traj_size_1

         real(c_double), value, target :: dtime

         logical(c_int), value, target :: lprep_adv

         logical(c_int), value, target :: at_initial_timestep

         real(c_double), value, target :: divdamp_fac_o2

         real(c_double), value, target :: ndyn_substeps

         integer(c_int), value, target :: idyn_timestep

      end function solve_nh_run_wrapper

      function solve_nh_init_wrapper(vct_a, &
                                     vct_a_size_0, &
                                     vct_b, &
                                     vct_b_size_0, &
                                     cell_areas, &
                                     cell_areas_size_0, &
                                     primal_normal_cell_x, &
                                     primal_normal_cell_x_size_0, &
                                     primal_normal_cell_x_size_1, &
                                     primal_normal_cell_y, &
                                     primal_normal_cell_y_size_0, &
                                     primal_normal_cell_y_size_1, &
                                     dual_normal_cell_x, &
                                     dual_normal_cell_x_size_0, &
                                     dual_normal_cell_x_size_1, &
                                     dual_normal_cell_y, &
                                     dual_normal_cell_y_size_0, &
                                     dual_normal_cell_y_size_1, &
                                     edge_areas, &
                                     edge_areas_size_0, &
                                     tangent_orientation, &
                                     tangent_orientation_size_0, &
                                     inverse_primal_edge_lengths, &
                                     inverse_primal_edge_lengths_size_0, &
                                     inverse_dual_edge_lengths, &
                                     inverse_dual_edge_lengths_size_0, &
                                     inverse_vertex_vertex_lengths, &
                                     inverse_vertex_vertex_lengths_size_0, &
                                     primal_normal_vert_x, &
                                     primal_normal_vert_x_size_0, &
                                     primal_normal_vert_x_size_1, &
                                     primal_normal_vert_y, &
                                     primal_normal_vert_y_size_0, &
                                     primal_normal_vert_y_size_1, &
                                     dual_normal_vert_x, &
                                     dual_normal_vert_x_size_0, &
                                     dual_normal_vert_x_size_1, &
                                     dual_normal_vert_y, &
                                     dual_normal_vert_y_size_0, &
                                     dual_normal_vert_y_size_1, &
                                     f_e, &
                                     f_e_size_0, &
                                     c_lin_e, &
                                     c_lin_e_size_0, &
                                     c_lin_e_size_1, &
                                     c_intp, &
                                     c_intp_size_0, &
                                     c_intp_size_1, &
                                     e_flx_avg, &
                                     e_flx_avg_size_0, &
                                     e_flx_avg_size_1, &
                                     geofac_grdiv, &
                                     geofac_grdiv_size_0, &
                                     geofac_grdiv_size_1, &
                                     geofac_rot, &
                                     geofac_rot_size_0, &
                                     geofac_rot_size_1, &
                                     pos_on_tplane_e_1, &
                                     pos_on_tplane_e_1_size_0, &
                                     pos_on_tplane_e_1_size_1, &
                                     pos_on_tplane_e_2, &
                                     pos_on_tplane_e_2_size_0, &
                                     pos_on_tplane_e_2_size_1, &
                                     rbf_vec_coeff_e, &
                                     rbf_vec_coeff_e_size_0, &
                                     rbf_vec_coeff_e_size_1, &
                                     e_bln_c_s, &
                                     e_bln_c_s_size_0, &
                                     e_bln_c_s_size_1, &
                                     rbf_coeff_1, &
                                     rbf_coeff_1_size_0, &
                                     rbf_coeff_1_size_1, &
                                     rbf_coeff_2, &
                                     rbf_coeff_2_size_0, &
                                     rbf_coeff_2_size_1, &
                                     geofac_div, &
                                     geofac_div_size_0, &
                                     geofac_div_size_1, &
                                     geofac_n2s, &
                                     geofac_n2s_size_0, &
                                     geofac_n2s_size_1, &
                                     geofac_grg_x, &
                                     geofac_grg_x_size_0, &
                                     geofac_grg_x_size_1, &
                                     geofac_grg_y, &
                                     geofac_grg_y_size_0, &
                                     geofac_grg_y_size_1, &
                                     nudgecoeff_e, &
                                     nudgecoeff_e_size_0, &
                                     bdy_halo_c, &
                                     bdy_halo_c_size_0, &
                                     mask_prog_halo_c, &
                                     mask_prog_halo_c_size_0, &
                                     rayleigh_w, &
                                     rayleigh_w_size_0, &
                                     exner_exfac, &
                                     exner_exfac_size_0, &
                                     exner_exfac_size_1, &
                                     exner_ref_mc, &
                                     exner_ref_mc_size_0, &
                                     exner_ref_mc_size_1, &
                                     wgtfac_c, &
                                     wgtfac_c_size_0, &
                                     wgtfac_c_size_1, &
                                     wgtfacq_c, &
                                     wgtfacq_c_size_0, &
                                     wgtfacq_c_size_1, &
                                     inv_ddqz_z_full, &
                                     inv_ddqz_z_full_size_0, &
                                     inv_ddqz_z_full_size_1, &
                                     rho_ref_mc, &
                                     rho_ref_mc_size_0, &
                                     rho_ref_mc_size_1, &
                                     theta_ref_mc, &
                                     theta_ref_mc_size_0, &
                                     theta_ref_mc_size_1, &
                                     vwind_expl_wgt, &
                                     vwind_expl_wgt_size_0, &
                                     d_exner_dz_ref_ic, &
                                     d_exner_dz_ref_ic_size_0, &
                                     d_exner_dz_ref_ic_size_1, &
                                     ddqz_z_half, &
                                     ddqz_z_half_size_0, &
                                     ddqz_z_half_size_1, &
                                     theta_ref_ic, &
                                     theta_ref_ic_size_0, &
                                     theta_ref_ic_size_1, &
                                     d2dexdz2_fac1_mc, &
                                     d2dexdz2_fac1_mc_size_0, &
                                     d2dexdz2_fac1_mc_size_1, &
                                     d2dexdz2_fac2_mc, &
                                     d2dexdz2_fac2_mc_size_0, &
                                     d2dexdz2_fac2_mc_size_1, &
                                     rho_ref_me, &
                                     rho_ref_me_size_0, &
                                     rho_ref_me_size_1, &
                                     theta_ref_me, &
                                     theta_ref_me_size_0, &
                                     theta_ref_me_size_1, &
                                     ddxn_z_full, &
                                     ddxn_z_full_size_0, &
                                     ddxn_z_full_size_1, &
                                     zdiff_gradp, &
                                     zdiff_gradp_size_0, &
                                     zdiff_gradp_size_1, &
                                     zdiff_gradp_size_2, &
                                     vertoffset_gradp, &
                                     vertoffset_gradp_size_0, &
                                     vertoffset_gradp_size_1, &
                                     vertoffset_gradp_size_2, &
                                     ipeidx_dsl, &
                                     ipeidx_dsl_size_0, &
                                     ipeidx_dsl_size_1, &
                                     pg_exdist, &
                                     pg_exdist_size_0, &
                                     pg_exdist_size_1, &
                                     ddqz_z_full_e, &
                                     ddqz_z_full_e_size_0, &
                                     ddqz_z_full_e_size_1, &
                                     ddxt_z_full, &
                                     ddxt_z_full_size_0, &
                                     ddxt_z_full_size_1, &
                                     wgtfac_e, &
                                     wgtfac_e_size_0, &
                                     wgtfac_e_size_1, &
                                     wgtfacq_e, &
                                     wgtfacq_e_size_0, &
                                     wgtfacq_e_size_1, &
                                     vwind_impl_wgt, &
                                     vwind_impl_wgt_size_0, &
                                     hmask_dd3d, &
                                     hmask_dd3d_size_0, &
                                     scalfac_dd3d, &
                                     scalfac_dd3d_size_0, &
                                     coeff1_dwdz, &
                                     coeff1_dwdz_size_0, &
                                     coeff1_dwdz_size_1, &
                                     coeff2_dwdz, &
                                     coeff2_dwdz_size_0, &
                                     coeff2_dwdz_size_1, &
                                     coeff_gradekin, &
                                     coeff_gradekin_size_0, &
                                     coeff_gradekin_size_1, &
                                     c_owner_mask, &
                                     c_owner_mask_size_0, &
                                     cell_center_lat, &
                                     cell_center_lat_size_0, &
                                     cell_center_lon, &
                                     cell_center_lon_size_0, &
                                     edge_center_lat, &
                                     edge_center_lat_size_0, &
                                     edge_center_lon, &
                                     edge_center_lon_size_0, &
                                     primal_normal_x, &
                                     primal_normal_x_size_0, &
                                     primal_normal_y, &
                                     primal_normal_y_size_0, &
                                     rayleigh_damping_height, &
                                     itime_scheme, &
                                     iadv_rhotheta, &
                                     igradp_method, &
                                     ndyn_substeps, &
                                     rayleigh_type, &
                                     rayleigh_coeff, &
                                     divdamp_order, &
                                     is_iau_active, &
                                     iau_wgt_dyn, &
                                     divdamp_type, &
                                     divdamp_trans_start, &
                                     divdamp_trans_end, &
                                     l_vert_nested, &
                                     rhotheta_offctr, &
                                     veladv_offctr, &
                                     max_nudging_coeff, &
                                     divdamp_fac, &
                                     divdamp_fac2, &
                                     divdamp_fac3, &
                                     divdamp_fac4, &
                                     divdamp_z, &
                                     divdamp_z2, &
                                     divdamp_z3, &
                                     divdamp_z4, &
                                     lowest_layer_thickness, &
                                     model_top_height, &
                                     stretch_factor, &
                                     nflat_gradp, &
                                     num_levels) bind(c, name="solve_nh_init_wrapper") result(rc)
         import :: c_int, c_double, c_bool, c_ptr
         integer(c_int) :: rc  ! Stores the return code

         type(c_ptr), value, target :: vct_a

         integer(c_int), value :: vct_a_size_0

         type(c_ptr), value, target :: vct_b

         integer(c_int), value :: vct_b_size_0

         type(c_ptr), value, target :: cell_areas

         integer(c_int), value :: cell_areas_size_0

         type(c_ptr), value, target :: primal_normal_cell_x

         integer(c_int), value :: primal_normal_cell_x_size_0

         integer(c_int), value :: primal_normal_cell_x_size_1

         type(c_ptr), value, target :: primal_normal_cell_y

         integer(c_int), value :: primal_normal_cell_y_size_0

         integer(c_int), value :: primal_normal_cell_y_size_1

         type(c_ptr), value, target :: dual_normal_cell_x

         integer(c_int), value :: dual_normal_cell_x_size_0

         integer(c_int), value :: dual_normal_cell_x_size_1

         type(c_ptr), value, target :: dual_normal_cell_y

         integer(c_int), value :: dual_normal_cell_y_size_0

         integer(c_int), value :: dual_normal_cell_y_size_1

         type(c_ptr), value, target :: edge_areas

         integer(c_int), value :: edge_areas_size_0

         type(c_ptr), value, target :: tangent_orientation

         integer(c_int), value :: tangent_orientation_size_0

         type(c_ptr), value, target :: inverse_primal_edge_lengths

         integer(c_int), value :: inverse_primal_edge_lengths_size_0

         type(c_ptr), value, target :: inverse_dual_edge_lengths

         integer(c_int), value :: inverse_dual_edge_lengths_size_0

         type(c_ptr), value, target :: inverse_vertex_vertex_lengths

         integer(c_int), value :: inverse_vertex_vertex_lengths_size_0

         type(c_ptr), value, target :: primal_normal_vert_x

         integer(c_int), value :: primal_normal_vert_x_size_0

         integer(c_int), value :: primal_normal_vert_x_size_1

         type(c_ptr), value, target :: primal_normal_vert_y

         integer(c_int), value :: primal_normal_vert_y_size_0

         integer(c_int), value :: primal_normal_vert_y_size_1

         type(c_ptr), value, target :: dual_normal_vert_x

         integer(c_int), value :: dual_normal_vert_x_size_0

         integer(c_int), value :: dual_normal_vert_x_size_1

         type(c_ptr), value, target :: dual_normal_vert_y

         integer(c_int), value :: dual_normal_vert_y_size_0

         integer(c_int), value :: dual_normal_vert_y_size_1

         type(c_ptr), value, target :: f_e

         integer(c_int), value :: f_e_size_0

         type(c_ptr), value, target :: c_lin_e

         integer(c_int), value :: c_lin_e_size_0

         integer(c_int), value :: c_lin_e_size_1

         type(c_ptr), value, target :: c_intp

         integer(c_int), value :: c_intp_size_0

         integer(c_int), value :: c_intp_size_1

         type(c_ptr), value, target :: e_flx_avg

         integer(c_int), value :: e_flx_avg_size_0

         integer(c_int), value :: e_flx_avg_size_1

         type(c_ptr), value, target :: geofac_grdiv

         integer(c_int), value :: geofac_grdiv_size_0

         integer(c_int), value :: geofac_grdiv_size_1

         type(c_ptr), value, target :: geofac_rot

         integer(c_int), value :: geofac_rot_size_0

         integer(c_int), value :: geofac_rot_size_1

         type(c_ptr), value, target :: pos_on_tplane_e_1

         integer(c_int), value :: pos_on_tplane_e_1_size_0

         integer(c_int), value :: pos_on_tplane_e_1_size_1

         type(c_ptr), value, target :: pos_on_tplane_e_2

         integer(c_int), value :: pos_on_tplane_e_2_size_0

         integer(c_int), value :: pos_on_tplane_e_2_size_1

         type(c_ptr), value, target :: rbf_vec_coeff_e

         integer(c_int), value :: rbf_vec_coeff_e_size_0

         integer(c_int), value :: rbf_vec_coeff_e_size_1

         type(c_ptr), value, target :: e_bln_c_s

         integer(c_int), value :: e_bln_c_s_size_0

         integer(c_int), value :: e_bln_c_s_size_1

         type(c_ptr), value, target :: rbf_coeff_1

         integer(c_int), value :: rbf_coeff_1_size_0

         integer(c_int), value :: rbf_coeff_1_size_1

         type(c_ptr), value, target :: rbf_coeff_2

         integer(c_int), value :: rbf_coeff_2_size_0

         integer(c_int), value :: rbf_coeff_2_size_1

         type(c_ptr), value, target :: geofac_div

         integer(c_int), value :: geofac_div_size_0

         integer(c_int), value :: geofac_div_size_1

         type(c_ptr), value, target :: geofac_n2s

         integer(c_int), value :: geofac_n2s_size_0

         integer(c_int), value :: geofac_n2s_size_1

         type(c_ptr), value, target :: geofac_grg_x

         integer(c_int), value :: geofac_grg_x_size_0

         integer(c_int), value :: geofac_grg_x_size_1

         type(c_ptr), value, target :: geofac_grg_y

         integer(c_int), value :: geofac_grg_y_size_0

         integer(c_int), value :: geofac_grg_y_size_1

         type(c_ptr), value, target :: nudgecoeff_e

         integer(c_int), value :: nudgecoeff_e_size_0

         type(c_ptr), value, target :: bdy_halo_c

         integer(c_int), value :: bdy_halo_c_size_0

         type(c_ptr), value, target :: mask_prog_halo_c

         integer(c_int), value :: mask_prog_halo_c_size_0

         type(c_ptr), value, target :: rayleigh_w

         integer(c_int), value :: rayleigh_w_size_0

         type(c_ptr), value, target :: exner_exfac

         integer(c_int), value :: exner_exfac_size_0

         integer(c_int), value :: exner_exfac_size_1

         type(c_ptr), value, target :: exner_ref_mc

         integer(c_int), value :: exner_ref_mc_size_0

         integer(c_int), value :: exner_ref_mc_size_1

         type(c_ptr), value, target :: wgtfac_c

         integer(c_int), value :: wgtfac_c_size_0

         integer(c_int), value :: wgtfac_c_size_1

         type(c_ptr), value, target :: wgtfacq_c

         integer(c_int), value :: wgtfacq_c_size_0

         integer(c_int), value :: wgtfacq_c_size_1

         type(c_ptr), value, target :: inv_ddqz_z_full

         integer(c_int), value :: inv_ddqz_z_full_size_0

         integer(c_int), value :: inv_ddqz_z_full_size_1

         type(c_ptr), value, target :: rho_ref_mc

         integer(c_int), value :: rho_ref_mc_size_0

         integer(c_int), value :: rho_ref_mc_size_1

         type(c_ptr), value, target :: theta_ref_mc

         integer(c_int), value :: theta_ref_mc_size_0

         integer(c_int), value :: theta_ref_mc_size_1

         type(c_ptr), value, target :: vwind_expl_wgt

         integer(c_int), value :: vwind_expl_wgt_size_0

         type(c_ptr), value, target :: d_exner_dz_ref_ic

         integer(c_int), value :: d_exner_dz_ref_ic_size_0

         integer(c_int), value :: d_exner_dz_ref_ic_size_1

         type(c_ptr), value, target :: ddqz_z_half

         integer(c_int), value :: ddqz_z_half_size_0

         integer(c_int), value :: ddqz_z_half_size_1

         type(c_ptr), value, target :: theta_ref_ic

         integer(c_int), value :: theta_ref_ic_size_0

         integer(c_int), value :: theta_ref_ic_size_1

         type(c_ptr), value, target :: d2dexdz2_fac1_mc

         integer(c_int), value :: d2dexdz2_fac1_mc_size_0

         integer(c_int), value :: d2dexdz2_fac1_mc_size_1

         type(c_ptr), value, target :: d2dexdz2_fac2_mc

         integer(c_int), value :: d2dexdz2_fac2_mc_size_0

         integer(c_int), value :: d2dexdz2_fac2_mc_size_1

         type(c_ptr), value, target :: rho_ref_me

         integer(c_int), value :: rho_ref_me_size_0

         integer(c_int), value :: rho_ref_me_size_1

         type(c_ptr), value, target :: theta_ref_me

         integer(c_int), value :: theta_ref_me_size_0

         integer(c_int), value :: theta_ref_me_size_1

         type(c_ptr), value, target :: ddxn_z_full

         integer(c_int), value :: ddxn_z_full_size_0

         integer(c_int), value :: ddxn_z_full_size_1

         type(c_ptr), value, target :: zdiff_gradp

         integer(c_int), value :: zdiff_gradp_size_0

         integer(c_int), value :: zdiff_gradp_size_1

         integer(c_int), value :: zdiff_gradp_size_2

         type(c_ptr), value, target :: vertoffset_gradp

         integer(c_int), value :: vertoffset_gradp_size_0

         integer(c_int), value :: vertoffset_gradp_size_1

         integer(c_int), value :: vertoffset_gradp_size_2

         type(c_ptr), value, target :: ipeidx_dsl

         integer(c_int), value :: ipeidx_dsl_size_0

         integer(c_int), value :: ipeidx_dsl_size_1

         type(c_ptr), value, target :: pg_exdist

         integer(c_int), value :: pg_exdist_size_0

         integer(c_int), value :: pg_exdist_size_1

         type(c_ptr), value, target :: ddqz_z_full_e

         integer(c_int), value :: ddqz_z_full_e_size_0

         integer(c_int), value :: ddqz_z_full_e_size_1

         type(c_ptr), value, target :: ddxt_z_full

         integer(c_int), value :: ddxt_z_full_size_0

         integer(c_int), value :: ddxt_z_full_size_1

         type(c_ptr), value, target :: wgtfac_e

         integer(c_int), value :: wgtfac_e_size_0

         integer(c_int), value :: wgtfac_e_size_1

         type(c_ptr), value, target :: wgtfacq_e

         integer(c_int), value :: wgtfacq_e_size_0

         integer(c_int), value :: wgtfacq_e_size_1

         type(c_ptr), value, target :: vwind_impl_wgt

         integer(c_int), value :: vwind_impl_wgt_size_0

         type(c_ptr), value, target :: hmask_dd3d

         integer(c_int), value :: hmask_dd3d_size_0

         type(c_ptr), value, target :: scalfac_dd3d

         integer(c_int), value :: scalfac_dd3d_size_0

         type(c_ptr), value, target :: coeff1_dwdz

         integer(c_int), value :: coeff1_dwdz_size_0

         integer(c_int), value :: coeff1_dwdz_size_1

         type(c_ptr), value, target :: coeff2_dwdz

         integer(c_int), value :: coeff2_dwdz_size_0

         integer(c_int), value :: coeff2_dwdz_size_1

         type(c_ptr), value, target :: coeff_gradekin

         integer(c_int), value :: coeff_gradekin_size_0

         integer(c_int), value :: coeff_gradekin_size_1

         type(c_ptr), value, target :: c_owner_mask

         integer(c_int), value :: c_owner_mask_size_0

         type(c_ptr), value, target :: cell_center_lat

         integer(c_int), value :: cell_center_lat_size_0

         type(c_ptr), value, target :: cell_center_lon

         integer(c_int), value :: cell_center_lon_size_0

         type(c_ptr), value, target :: edge_center_lat

         integer(c_int), value :: edge_center_lat_size_0

         type(c_ptr), value, target :: edge_center_lon

         integer(c_int), value :: edge_center_lon_size_0

         type(c_ptr), value, target :: primal_normal_x

         integer(c_int), value :: primal_normal_x_size_0

         type(c_ptr), value, target :: primal_normal_y

         integer(c_int), value :: primal_normal_y_size_0

         real(c_double), value, target :: rayleigh_damping_height

         integer(c_int), value, target :: itime_scheme

         integer(c_int), value, target :: iadv_rhotheta

         integer(c_int), value, target :: igradp_method

         real(c_double), value, target :: ndyn_substeps

         integer(c_int), value, target :: rayleigh_type

         real(c_double), value, target :: rayleigh_coeff

         integer(c_int), value, target :: divdamp_order

         logical(c_int), value, target :: is_iau_active

         real(c_double), value, target :: iau_wgt_dyn

         integer(c_int), value, target :: divdamp_type

         real(c_double), value, target :: divdamp_trans_start

         real(c_double), value, target :: divdamp_trans_end

         logical(c_int), value, target :: l_vert_nested

         real(c_double), value, target :: rhotheta_offctr

         real(c_double), value, target :: veladv_offctr

         real(c_double), value, target :: max_nudging_coeff

         real(c_double), value, target :: divdamp_fac

         real(c_double), value, target :: divdamp_fac2

         real(c_double), value, target :: divdamp_fac3

         real(c_double), value, target :: divdamp_fac4

         real(c_double), value, target :: divdamp_z

         real(c_double), value, target :: divdamp_z2

         real(c_double), value, target :: divdamp_z3

         real(c_double), value, target :: divdamp_z4

         real(c_double), value, target :: lowest_layer_thickness

         real(c_double), value, target :: model_top_height

         real(c_double), value, target :: stretch_factor

         integer(c_int), value, target :: nflat_gradp

         integer(c_int), value, target :: num_levels

      end function solve_nh_init_wrapper

      function grid_init_wrapper(cell_starts, &
                                 cell_starts_size_0, &
                                 cell_ends, &
                                 cell_ends_size_0, &
                                 vertex_starts, &
                                 vertex_starts_size_0, &
                                 vertex_ends, &
                                 vertex_ends_size_0, &
                                 edge_starts, &
                                 edge_starts_size_0, &
                                 edge_ends, &
                                 edge_ends_size_0, &
                                 c2e, &
                                 c2e_size_0, &
                                 c2e_size_1, &
                                 e2c, &
                                 e2c_size_0, &
                                 e2c_size_1, &
                                 c2e2c, &
                                 c2e2c_size_0, &
                                 c2e2c_size_1, &
                                 e2c2e, &
                                 e2c2e_size_0, &
                                 e2c2e_size_1, &
                                 e2v, &
                                 e2v_size_0, &
                                 e2v_size_1, &
                                 v2e, &
                                 v2e_size_0, &
                                 v2e_size_1, &
                                 v2c, &
                                 v2c_size_0, &
                                 v2c_size_1, &
                                 e2c2v, &
                                 e2c2v_size_0, &
                                 e2c2v_size_1, &
                                 c2v, &
                                 c2v_size_0, &
                                 c2v_size_1, &
                                 global_root, &
                                 global_level, &
                                 num_vertices, &
                                 num_cells, &
                                 num_edges, &
                                 vertical_size, &
                                 limited_area) bind(c, name="grid_init_wrapper") result(rc)
         import :: c_int, c_double, c_bool, c_ptr
         integer(c_int) :: rc  ! Stores the return code

         type(c_ptr), value, target :: cell_starts

         integer(c_int), value :: cell_starts_size_0

         type(c_ptr), value, target :: cell_ends

         integer(c_int), value :: cell_ends_size_0

         type(c_ptr), value, target :: vertex_starts

         integer(c_int), value :: vertex_starts_size_0

         type(c_ptr), value, target :: vertex_ends

         integer(c_int), value :: vertex_ends_size_0

         type(c_ptr), value, target :: edge_starts

         integer(c_int), value :: edge_starts_size_0

         type(c_ptr), value, target :: edge_ends

         integer(c_int), value :: edge_ends_size_0

         type(c_ptr), value, target :: c2e

         integer(c_int), value :: c2e_size_0

         integer(c_int), value :: c2e_size_1

         type(c_ptr), value, target :: e2c

         integer(c_int), value :: e2c_size_0

         integer(c_int), value :: e2c_size_1

         type(c_ptr), value, target :: c2e2c

         integer(c_int), value :: c2e2c_size_0

         integer(c_int), value :: c2e2c_size_1

         type(c_ptr), value, target :: e2c2e

         integer(c_int), value :: e2c2e_size_0

         integer(c_int), value :: e2c2e_size_1

         type(c_ptr), value, target :: e2v

         integer(c_int), value :: e2v_size_0

         integer(c_int), value :: e2v_size_1

         type(c_ptr), value, target :: v2e

         integer(c_int), value :: v2e_size_0

         integer(c_int), value :: v2e_size_1

         type(c_ptr), value, target :: v2c

         integer(c_int), value :: v2c_size_0

         integer(c_int), value :: v2c_size_1

         type(c_ptr), value, target :: e2c2v

         integer(c_int), value :: e2c2v_size_0

         integer(c_int), value :: e2c2v_size_1

         type(c_ptr), value, target :: c2v

         integer(c_int), value :: c2v_size_0

         integer(c_int), value :: c2v_size_1

         integer(c_int), value, target :: global_root

         integer(c_int), value, target :: global_level

         integer(c_int), value, target :: num_vertices

         integer(c_int), value, target :: num_cells

         integer(c_int), value, target :: num_edges

         integer(c_int), value, target :: vertical_size

         logical(c_int), value, target :: limited_area

      end function grid_init_wrapper

   end interface

contains

   subroutine solve_nh_run(rho_now, &
                           rho_new, &
                           exner_now, &
                           exner_new, &
                           w_now, &
                           w_new, &
                           theta_v_now, &
                           theta_v_new, &
                           vn_now, &
                           vn_new, &
                           w_concorr_c, &
                           ddt_vn_apc_ntl1, &
                           ddt_vn_apc_ntl2, &
                           ddt_w_adv_ntl1, &
                           ddt_w_adv_ntl2, &
                           theta_v_ic, &
                           rho_ic, &
                           exner_pr, &
                           exner_dyn_incr, &
                           ddt_exner_phy, &
                           grf_tend_rho, &
                           grf_tend_thv, &
                           grf_tend_w, &
                           mass_fl_e, &
                           ddt_vn_phy, &
                           grf_tend_vn, &
                           vn_ie, &
                           vt, &
                           mass_flx_me, &
                           mass_flx_ic, &
                           vn_traj, &
                           dtime, &
                           lprep_adv, &
                           at_initial_timestep, &
                           divdamp_fac_o2, &
                           ndyn_substeps, &
                           idyn_timestep, &
                           rc)
      use, intrinsic :: iso_c_binding

      real(c_double), dimension(:, :), target :: rho_now

      real(c_double), dimension(:, :), target :: rho_new

      real(c_double), dimension(:, :), target :: exner_now

      real(c_double), dimension(:, :), target :: exner_new

      real(c_double), dimension(:, :), target :: w_now

      real(c_double), dimension(:, :), target :: w_new

      real(c_double), dimension(:, :), target :: theta_v_now

      real(c_double), dimension(:, :), target :: theta_v_new

      real(c_double), dimension(:, :), target :: vn_now

      real(c_double), dimension(:, :), target :: vn_new

      real(c_double), dimension(:, :), target :: w_concorr_c

      real(c_double), dimension(:, :), target :: ddt_vn_apc_ntl1

      real(c_double), dimension(:, :), target :: ddt_vn_apc_ntl2

      real(c_double), dimension(:, :), target :: ddt_w_adv_ntl1

      real(c_double), dimension(:, :), target :: ddt_w_adv_ntl2

      real(c_double), dimension(:, :), target :: theta_v_ic

      real(c_double), dimension(:, :), target :: rho_ic

      real(c_double), dimension(:, :), target :: exner_pr

      real(c_double), dimension(:, :), target :: exner_dyn_incr

      real(c_double), dimension(:, :), target :: ddt_exner_phy

      real(c_double), dimension(:, :), target :: grf_tend_rho

      real(c_double), dimension(:, :), target :: grf_tend_thv

      real(c_double), dimension(:, :), target :: grf_tend_w

      real(c_double), dimension(:, :), target :: mass_fl_e

      real(c_double), dimension(:, :), target :: ddt_vn_phy

      real(c_double), dimension(:, :), target :: grf_tend_vn

      real(c_double), dimension(:, :), target :: vn_ie

      real(c_double), dimension(:, :), target :: vt

      real(c_double), dimension(:, :), target :: mass_flx_me

      real(c_double), dimension(:, :), target :: mass_flx_ic

      real(c_double), dimension(:, :), target :: vn_traj

      real(c_double), value, target :: dtime

      logical(c_int), value, target :: lprep_adv

      logical(c_int), value, target :: at_initial_timestep

      real(c_double), value, target :: divdamp_fac_o2

      real(c_double), value, target :: ndyn_substeps

      integer(c_int), value, target :: idyn_timestep

      integer(c_int) :: rho_now_size_0

      integer(c_int) :: rho_now_size_1

      integer(c_int) :: rho_new_size_0

      integer(c_int) :: rho_new_size_1

      integer(c_int) :: exner_now_size_0

      integer(c_int) :: exner_now_size_1

      integer(c_int) :: exner_new_size_0

      integer(c_int) :: exner_new_size_1

      integer(c_int) :: w_now_size_0

      integer(c_int) :: w_now_size_1

      integer(c_int) :: w_new_size_0

      integer(c_int) :: w_new_size_1

      integer(c_int) :: theta_v_now_size_0

      integer(c_int) :: theta_v_now_size_1

      integer(c_int) :: theta_v_new_size_0

      integer(c_int) :: theta_v_new_size_1

      integer(c_int) :: vn_now_size_0

      integer(c_int) :: vn_now_size_1

      integer(c_int) :: vn_new_size_0

      integer(c_int) :: vn_new_size_1

      integer(c_int) :: w_concorr_c_size_0

      integer(c_int) :: w_concorr_c_size_1

      integer(c_int) :: ddt_vn_apc_ntl1_size_0

      integer(c_int) :: ddt_vn_apc_ntl1_size_1

      integer(c_int) :: ddt_vn_apc_ntl2_size_0

      integer(c_int) :: ddt_vn_apc_ntl2_size_1

      integer(c_int) :: ddt_w_adv_ntl1_size_0

      integer(c_int) :: ddt_w_adv_ntl1_size_1

      integer(c_int) :: ddt_w_adv_ntl2_size_0

      integer(c_int) :: ddt_w_adv_ntl2_size_1

      integer(c_int) :: theta_v_ic_size_0

      integer(c_int) :: theta_v_ic_size_1

      integer(c_int) :: rho_ic_size_0

      integer(c_int) :: rho_ic_size_1

      integer(c_int) :: exner_pr_size_0

      integer(c_int) :: exner_pr_size_1

      integer(c_int) :: exner_dyn_incr_size_0

      integer(c_int) :: exner_dyn_incr_size_1

      integer(c_int) :: ddt_exner_phy_size_0

      integer(c_int) :: ddt_exner_phy_size_1

      integer(c_int) :: grf_tend_rho_size_0

      integer(c_int) :: grf_tend_rho_size_1

      integer(c_int) :: grf_tend_thv_size_0

      integer(c_int) :: grf_tend_thv_size_1

      integer(c_int) :: grf_tend_w_size_0

      integer(c_int) :: grf_tend_w_size_1

      integer(c_int) :: mass_fl_e_size_0

      integer(c_int) :: mass_fl_e_size_1

      integer(c_int) :: ddt_vn_phy_size_0

      integer(c_int) :: ddt_vn_phy_size_1

      integer(c_int) :: grf_tend_vn_size_0

      integer(c_int) :: grf_tend_vn_size_1

      integer(c_int) :: vn_ie_size_0

      integer(c_int) :: vn_ie_size_1

      integer(c_int) :: vt_size_0

      integer(c_int) :: vt_size_1

      integer(c_int) :: mass_flx_me_size_0

      integer(c_int) :: mass_flx_me_size_1

      integer(c_int) :: mass_flx_ic_size_0

      integer(c_int) :: mass_flx_ic_size_1

      integer(c_int) :: vn_traj_size_0

      integer(c_int) :: vn_traj_size_1

      integer(c_int) :: rc  ! Stores the return code
      ! ptrs

      !$acc host_data use_device(rho_now)
      !$acc host_data use_device(rho_new)
      !$acc host_data use_device(exner_now)
      !$acc host_data use_device(exner_new)
      !$acc host_data use_device(w_now)
      !$acc host_data use_device(w_new)
      !$acc host_data use_device(theta_v_now)
      !$acc host_data use_device(theta_v_new)
      !$acc host_data use_device(vn_now)
      !$acc host_data use_device(vn_new)
      !$acc host_data use_device(w_concorr_c)
      !$acc host_data use_device(ddt_vn_apc_ntl1)
      !$acc host_data use_device(ddt_vn_apc_ntl2)
      !$acc host_data use_device(ddt_w_adv_ntl1)
      !$acc host_data use_device(ddt_w_adv_ntl2)
      !$acc host_data use_device(theta_v_ic)
      !$acc host_data use_device(rho_ic)
      !$acc host_data use_device(exner_pr)
      !$acc host_data use_device(exner_dyn_incr)
      !$acc host_data use_device(ddt_exner_phy)
      !$acc host_data use_device(grf_tend_rho)
      !$acc host_data use_device(grf_tend_thv)
      !$acc host_data use_device(grf_tend_w)
      !$acc host_data use_device(mass_fl_e)
      !$acc host_data use_device(ddt_vn_phy)
      !$acc host_data use_device(grf_tend_vn)
      !$acc host_data use_device(vn_ie)
      !$acc host_data use_device(vt)
      !$acc host_data use_device(mass_flx_me)
      !$acc host_data use_device(mass_flx_ic)
      !$acc host_data use_device(vn_traj)

      rho_now_size_0 = SIZE(rho_now, 1)
      rho_now_size_1 = SIZE(rho_now, 2)

      rho_new_size_0 = SIZE(rho_new, 1)
      rho_new_size_1 = SIZE(rho_new, 2)

      exner_now_size_0 = SIZE(exner_now, 1)
      exner_now_size_1 = SIZE(exner_now, 2)

      exner_new_size_0 = SIZE(exner_new, 1)
      exner_new_size_1 = SIZE(exner_new, 2)

      w_now_size_0 = SIZE(w_now, 1)
      w_now_size_1 = SIZE(w_now, 2)

      w_new_size_0 = SIZE(w_new, 1)
      w_new_size_1 = SIZE(w_new, 2)

      theta_v_now_size_0 = SIZE(theta_v_now, 1)
      theta_v_now_size_1 = SIZE(theta_v_now, 2)

      theta_v_new_size_0 = SIZE(theta_v_new, 1)
      theta_v_new_size_1 = SIZE(theta_v_new, 2)

      vn_now_size_0 = SIZE(vn_now, 1)
      vn_now_size_1 = SIZE(vn_now, 2)

      vn_new_size_0 = SIZE(vn_new, 1)
      vn_new_size_1 = SIZE(vn_new, 2)

      w_concorr_c_size_0 = SIZE(w_concorr_c, 1)
      w_concorr_c_size_1 = SIZE(w_concorr_c, 2)

      ddt_vn_apc_ntl1_size_0 = SIZE(ddt_vn_apc_ntl1, 1)
      ddt_vn_apc_ntl1_size_1 = SIZE(ddt_vn_apc_ntl1, 2)

      ddt_vn_apc_ntl2_size_0 = SIZE(ddt_vn_apc_ntl2, 1)
      ddt_vn_apc_ntl2_size_1 = SIZE(ddt_vn_apc_ntl2, 2)

      ddt_w_adv_ntl1_size_0 = SIZE(ddt_w_adv_ntl1, 1)
      ddt_w_adv_ntl1_size_1 = SIZE(ddt_w_adv_ntl1, 2)

      ddt_w_adv_ntl2_size_0 = SIZE(ddt_w_adv_ntl2, 1)
      ddt_w_adv_ntl2_size_1 = SIZE(ddt_w_adv_ntl2, 2)

      theta_v_ic_size_0 = SIZE(theta_v_ic, 1)
      theta_v_ic_size_1 = SIZE(theta_v_ic, 2)

      rho_ic_size_0 = SIZE(rho_ic, 1)
      rho_ic_size_1 = SIZE(rho_ic, 2)

      exner_pr_size_0 = SIZE(exner_pr, 1)
      exner_pr_size_1 = SIZE(exner_pr, 2)

      exner_dyn_incr_size_0 = SIZE(exner_dyn_incr, 1)
      exner_dyn_incr_size_1 = SIZE(exner_dyn_incr, 2)

      ddt_exner_phy_size_0 = SIZE(ddt_exner_phy, 1)
      ddt_exner_phy_size_1 = SIZE(ddt_exner_phy, 2)

      grf_tend_rho_size_0 = SIZE(grf_tend_rho, 1)
      grf_tend_rho_size_1 = SIZE(grf_tend_rho, 2)

      grf_tend_thv_size_0 = SIZE(grf_tend_thv, 1)
      grf_tend_thv_size_1 = SIZE(grf_tend_thv, 2)

      grf_tend_w_size_0 = SIZE(grf_tend_w, 1)
      grf_tend_w_size_1 = SIZE(grf_tend_w, 2)

      mass_fl_e_size_0 = SIZE(mass_fl_e, 1)
      mass_fl_e_size_1 = SIZE(mass_fl_e, 2)

      ddt_vn_phy_size_0 = SIZE(ddt_vn_phy, 1)
      ddt_vn_phy_size_1 = SIZE(ddt_vn_phy, 2)

      grf_tend_vn_size_0 = SIZE(grf_tend_vn, 1)
      grf_tend_vn_size_1 = SIZE(grf_tend_vn, 2)

      vn_ie_size_0 = SIZE(vn_ie, 1)
      vn_ie_size_1 = SIZE(vn_ie, 2)

      vt_size_0 = SIZE(vt, 1)
      vt_size_1 = SIZE(vt, 2)

      mass_flx_me_size_0 = SIZE(mass_flx_me, 1)
      mass_flx_me_size_1 = SIZE(mass_flx_me, 2)

      mass_flx_ic_size_0 = SIZE(mass_flx_ic, 1)
      mass_flx_ic_size_1 = SIZE(mass_flx_ic, 2)

      vn_traj_size_0 = SIZE(vn_traj, 1)
      vn_traj_size_1 = SIZE(vn_traj, 2)

      rc = solve_nh_run_wrapper(rho_now=c_loc(rho_now), &
                                rho_now_size_0=rho_now_size_0, &
                                rho_now_size_1=rho_now_size_1, &
                                rho_new=c_loc(rho_new), &
                                rho_new_size_0=rho_new_size_0, &
                                rho_new_size_1=rho_new_size_1, &
                                exner_now=c_loc(exner_now), &
                                exner_now_size_0=exner_now_size_0, &
                                exner_now_size_1=exner_now_size_1, &
                                exner_new=c_loc(exner_new), &
                                exner_new_size_0=exner_new_size_0, &
                                exner_new_size_1=exner_new_size_1, &
                                w_now=c_loc(w_now), &
                                w_now_size_0=w_now_size_0, &
                                w_now_size_1=w_now_size_1, &
                                w_new=c_loc(w_new), &
                                w_new_size_0=w_new_size_0, &
                                w_new_size_1=w_new_size_1, &
                                theta_v_now=c_loc(theta_v_now), &
                                theta_v_now_size_0=theta_v_now_size_0, &
                                theta_v_now_size_1=theta_v_now_size_1, &
                                theta_v_new=c_loc(theta_v_new), &
                                theta_v_new_size_0=theta_v_new_size_0, &
                                theta_v_new_size_1=theta_v_new_size_1, &
                                vn_now=c_loc(vn_now), &
                                vn_now_size_0=vn_now_size_0, &
                                vn_now_size_1=vn_now_size_1, &
                                vn_new=c_loc(vn_new), &
                                vn_new_size_0=vn_new_size_0, &
                                vn_new_size_1=vn_new_size_1, &
                                w_concorr_c=c_loc(w_concorr_c), &
                                w_concorr_c_size_0=w_concorr_c_size_0, &
                                w_concorr_c_size_1=w_concorr_c_size_1, &
                                ddt_vn_apc_ntl1=c_loc(ddt_vn_apc_ntl1), &
                                ddt_vn_apc_ntl1_size_0=ddt_vn_apc_ntl1_size_0, &
                                ddt_vn_apc_ntl1_size_1=ddt_vn_apc_ntl1_size_1, &
                                ddt_vn_apc_ntl2=c_loc(ddt_vn_apc_ntl2), &
                                ddt_vn_apc_ntl2_size_0=ddt_vn_apc_ntl2_size_0, &
                                ddt_vn_apc_ntl2_size_1=ddt_vn_apc_ntl2_size_1, &
                                ddt_w_adv_ntl1=c_loc(ddt_w_adv_ntl1), &
                                ddt_w_adv_ntl1_size_0=ddt_w_adv_ntl1_size_0, &
                                ddt_w_adv_ntl1_size_1=ddt_w_adv_ntl1_size_1, &
                                ddt_w_adv_ntl2=c_loc(ddt_w_adv_ntl2), &
                                ddt_w_adv_ntl2_size_0=ddt_w_adv_ntl2_size_0, &
                                ddt_w_adv_ntl2_size_1=ddt_w_adv_ntl2_size_1, &
                                theta_v_ic=c_loc(theta_v_ic), &
                                theta_v_ic_size_0=theta_v_ic_size_0, &
                                theta_v_ic_size_1=theta_v_ic_size_1, &
                                rho_ic=c_loc(rho_ic), &
                                rho_ic_size_0=rho_ic_size_0, &
                                rho_ic_size_1=rho_ic_size_1, &
                                exner_pr=c_loc(exner_pr), &
                                exner_pr_size_0=exner_pr_size_0, &
                                exner_pr_size_1=exner_pr_size_1, &
                                exner_dyn_incr=c_loc(exner_dyn_incr), &
                                exner_dyn_incr_size_0=exner_dyn_incr_size_0, &
                                exner_dyn_incr_size_1=exner_dyn_incr_size_1, &
                                ddt_exner_phy=c_loc(ddt_exner_phy), &
                                ddt_exner_phy_size_0=ddt_exner_phy_size_0, &
                                ddt_exner_phy_size_1=ddt_exner_phy_size_1, &
                                grf_tend_rho=c_loc(grf_tend_rho), &
                                grf_tend_rho_size_0=grf_tend_rho_size_0, &
                                grf_tend_rho_size_1=grf_tend_rho_size_1, &
                                grf_tend_thv=c_loc(grf_tend_thv), &
                                grf_tend_thv_size_0=grf_tend_thv_size_0, &
                                grf_tend_thv_size_1=grf_tend_thv_size_1, &
                                grf_tend_w=c_loc(grf_tend_w), &
                                grf_tend_w_size_0=grf_tend_w_size_0, &
                                grf_tend_w_size_1=grf_tend_w_size_1, &
                                mass_fl_e=c_loc(mass_fl_e), &
                                mass_fl_e_size_0=mass_fl_e_size_0, &
                                mass_fl_e_size_1=mass_fl_e_size_1, &
                                ddt_vn_phy=c_loc(ddt_vn_phy), &
                                ddt_vn_phy_size_0=ddt_vn_phy_size_0, &
                                ddt_vn_phy_size_1=ddt_vn_phy_size_1, &
                                grf_tend_vn=c_loc(grf_tend_vn), &
                                grf_tend_vn_size_0=grf_tend_vn_size_0, &
                                grf_tend_vn_size_1=grf_tend_vn_size_1, &
                                vn_ie=c_loc(vn_ie), &
                                vn_ie_size_0=vn_ie_size_0, &
                                vn_ie_size_1=vn_ie_size_1, &
                                vt=c_loc(vt), &
                                vt_size_0=vt_size_0, &
                                vt_size_1=vt_size_1, &
                                mass_flx_me=c_loc(mass_flx_me), &
                                mass_flx_me_size_0=mass_flx_me_size_0, &
                                mass_flx_me_size_1=mass_flx_me_size_1, &
                                mass_flx_ic=c_loc(mass_flx_ic), &
                                mass_flx_ic_size_0=mass_flx_ic_size_0, &
                                mass_flx_ic_size_1=mass_flx_ic_size_1, &
                                vn_traj=c_loc(vn_traj), &
                                vn_traj_size_0=vn_traj_size_0, &
                                vn_traj_size_1=vn_traj_size_1, &
                                dtime=dtime, &
                                lprep_adv=lprep_adv, &
                                at_initial_timestep=at_initial_timestep, &
                                divdamp_fac_o2=divdamp_fac_o2, &
                                ndyn_substeps=ndyn_substeps, &
                                idyn_timestep=idyn_timestep)
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
   end subroutine solve_nh_run

   subroutine solve_nh_init(vct_a, &
                            vct_b, &
                            cell_areas, &
                            primal_normal_cell_x, &
                            primal_normal_cell_y, &
                            dual_normal_cell_x, &
                            dual_normal_cell_y, &
                            edge_areas, &
                            tangent_orientation, &
                            inverse_primal_edge_lengths, &
                            inverse_dual_edge_lengths, &
                            inverse_vertex_vertex_lengths, &
                            primal_normal_vert_x, &
                            primal_normal_vert_y, &
                            dual_normal_vert_x, &
                            dual_normal_vert_y, &
                            f_e, &
                            c_lin_e, &
                            c_intp, &
                            e_flx_avg, &
                            geofac_grdiv, &
                            geofac_rot, &
                            pos_on_tplane_e_1, &
                            pos_on_tplane_e_2, &
                            rbf_vec_coeff_e, &
                            e_bln_c_s, &
                            rbf_coeff_1, &
                            rbf_coeff_2, &
                            geofac_div, &
                            geofac_n2s, &
                            geofac_grg_x, &
                            geofac_grg_y, &
                            nudgecoeff_e, &
                            bdy_halo_c, &
                            mask_prog_halo_c, &
                            rayleigh_w, &
                            exner_exfac, &
                            exner_ref_mc, &
                            wgtfac_c, &
                            wgtfacq_c, &
                            inv_ddqz_z_full, &
                            rho_ref_mc, &
                            theta_ref_mc, &
                            vwind_expl_wgt, &
                            d_exner_dz_ref_ic, &
                            ddqz_z_half, &
                            theta_ref_ic, &
                            d2dexdz2_fac1_mc, &
                            d2dexdz2_fac2_mc, &
                            rho_ref_me, &
                            theta_ref_me, &
                            ddxn_z_full, &
                            zdiff_gradp, &
                            vertoffset_gradp, &
                            ipeidx_dsl, &
                            pg_exdist, &
                            ddqz_z_full_e, &
                            ddxt_z_full, &
                            wgtfac_e, &
                            wgtfacq_e, &
                            vwind_impl_wgt, &
                            hmask_dd3d, &
                            scalfac_dd3d, &
                            coeff1_dwdz, &
                            coeff2_dwdz, &
                            coeff_gradekin, &
                            c_owner_mask, &
                            cell_center_lat, &
                            cell_center_lon, &
                            edge_center_lat, &
                            edge_center_lon, &
                            primal_normal_x, &
                            primal_normal_y, &
                            rayleigh_damping_height, &
                            itime_scheme, &
                            iadv_rhotheta, &
                            igradp_method, &
                            ndyn_substeps, &
                            rayleigh_type, &
                            rayleigh_coeff, &
                            divdamp_order, &
                            is_iau_active, &
                            iau_wgt_dyn, &
                            divdamp_type, &
                            divdamp_trans_start, &
                            divdamp_trans_end, &
                            l_vert_nested, &
                            rhotheta_offctr, &
                            veladv_offctr, &
                            max_nudging_coeff, &
                            divdamp_fac, &
                            divdamp_fac2, &
                            divdamp_fac3, &
                            divdamp_fac4, &
                            divdamp_z, &
                            divdamp_z2, &
                            divdamp_z3, &
                            divdamp_z4, &
                            lowest_layer_thickness, &
                            model_top_height, &
                            stretch_factor, &
                            nflat_gradp, &
                            num_levels, &
                            rc)
      use, intrinsic :: iso_c_binding

      real(c_double), dimension(:), target :: vct_a

      real(c_double), dimension(:), target :: vct_b

      real(c_double), dimension(:), target :: cell_areas

      real(c_double), dimension(:, :), target :: primal_normal_cell_x

      real(c_double), dimension(:, :), target :: primal_normal_cell_y

      real(c_double), dimension(:, :), target :: dual_normal_cell_x

      real(c_double), dimension(:, :), target :: dual_normal_cell_y

      real(c_double), dimension(:), target :: edge_areas

      real(c_double), dimension(:), target :: tangent_orientation

      real(c_double), dimension(:), target :: inverse_primal_edge_lengths

      real(c_double), dimension(:), target :: inverse_dual_edge_lengths

      real(c_double), dimension(:), target :: inverse_vertex_vertex_lengths

      real(c_double), dimension(:, :), target :: primal_normal_vert_x

      real(c_double), dimension(:, :), target :: primal_normal_vert_y

      real(c_double), dimension(:, :), target :: dual_normal_vert_x

      real(c_double), dimension(:, :), target :: dual_normal_vert_y

      real(c_double), dimension(:), target :: f_e

      real(c_double), dimension(:, :), target :: c_lin_e

      real(c_double), dimension(:, :), target :: c_intp

      real(c_double), dimension(:, :), target :: e_flx_avg

      real(c_double), dimension(:, :), target :: geofac_grdiv

      real(c_double), dimension(:, :), target :: geofac_rot

      real(c_double), dimension(:, :), target :: pos_on_tplane_e_1

      real(c_double), dimension(:, :), target :: pos_on_tplane_e_2

      real(c_double), dimension(:, :), target :: rbf_vec_coeff_e

      real(c_double), dimension(:, :), target :: e_bln_c_s

      real(c_double), dimension(:, :), target :: rbf_coeff_1

      real(c_double), dimension(:, :), target :: rbf_coeff_2

      real(c_double), dimension(:, :), target :: geofac_div

      real(c_double), dimension(:, :), target :: geofac_n2s

      real(c_double), dimension(:, :), target :: geofac_grg_x

      real(c_double), dimension(:, :), target :: geofac_grg_y

      real(c_double), dimension(:), target :: nudgecoeff_e

      logical(c_int), dimension(:), target :: bdy_halo_c

      logical(c_int), dimension(:), target :: mask_prog_halo_c

      real(c_double), dimension(:), target :: rayleigh_w

      real(c_double), dimension(:, :), target :: exner_exfac

      real(c_double), dimension(:, :), target :: exner_ref_mc

      real(c_double), dimension(:, :), target :: wgtfac_c

      real(c_double), dimension(:, :), target :: wgtfacq_c

      real(c_double), dimension(:, :), target :: inv_ddqz_z_full

      real(c_double), dimension(:, :), target :: rho_ref_mc

      real(c_double), dimension(:, :), target :: theta_ref_mc

      real(c_double), dimension(:), target :: vwind_expl_wgt

      real(c_double), dimension(:, :), target :: d_exner_dz_ref_ic

      real(c_double), dimension(:, :), target :: ddqz_z_half

      real(c_double), dimension(:, :), target :: theta_ref_ic

      real(c_double), dimension(:, :), target :: d2dexdz2_fac1_mc

      real(c_double), dimension(:, :), target :: d2dexdz2_fac2_mc

      real(c_double), dimension(:, :), target :: rho_ref_me

      real(c_double), dimension(:, :), target :: theta_ref_me

      real(c_double), dimension(:, :), target :: ddxn_z_full

      real(c_double), dimension(:, :, :), target :: zdiff_gradp

      integer(c_int), dimension(:, :, :), target :: vertoffset_gradp

      logical(c_int), dimension(:, :), target :: ipeidx_dsl

      real(c_double), dimension(:, :), target :: pg_exdist

      real(c_double), dimension(:, :), target :: ddqz_z_full_e

      real(c_double), dimension(:, :), target :: ddxt_z_full

      real(c_double), dimension(:, :), target :: wgtfac_e

      real(c_double), dimension(:, :), target :: wgtfacq_e

      real(c_double), dimension(:), target :: vwind_impl_wgt

      real(c_double), dimension(:), target :: hmask_dd3d

      real(c_double), dimension(:), target :: scalfac_dd3d

      real(c_double), dimension(:, :), target :: coeff1_dwdz

      real(c_double), dimension(:, :), target :: coeff2_dwdz

      real(c_double), dimension(:, :), target :: coeff_gradekin

      logical(c_int), dimension(:), target :: c_owner_mask

      real(c_double), dimension(:), target :: cell_center_lat

      real(c_double), dimension(:), target :: cell_center_lon

      real(c_double), dimension(:), target :: edge_center_lat

      real(c_double), dimension(:), target :: edge_center_lon

      real(c_double), dimension(:), target :: primal_normal_x

      real(c_double), dimension(:), target :: primal_normal_y

      real(c_double), value, target :: rayleigh_damping_height

      integer(c_int), value, target :: itime_scheme

      integer(c_int), value, target :: iadv_rhotheta

      integer(c_int), value, target :: igradp_method

      real(c_double), value, target :: ndyn_substeps

      integer(c_int), value, target :: rayleigh_type

      real(c_double), value, target :: rayleigh_coeff

      integer(c_int), value, target :: divdamp_order

      logical(c_int), value, target :: is_iau_active

      real(c_double), value, target :: iau_wgt_dyn

      integer(c_int), value, target :: divdamp_type

      real(c_double), value, target :: divdamp_trans_start

      real(c_double), value, target :: divdamp_trans_end

      logical(c_int), value, target :: l_vert_nested

      real(c_double), value, target :: rhotheta_offctr

      real(c_double), value, target :: veladv_offctr

      real(c_double), value, target :: max_nudging_coeff

      real(c_double), value, target :: divdamp_fac

      real(c_double), value, target :: divdamp_fac2

      real(c_double), value, target :: divdamp_fac3

      real(c_double), value, target :: divdamp_fac4

      real(c_double), value, target :: divdamp_z

      real(c_double), value, target :: divdamp_z2

      real(c_double), value, target :: divdamp_z3

      real(c_double), value, target :: divdamp_z4

      real(c_double), value, target :: lowest_layer_thickness

      real(c_double), value, target :: model_top_height

      real(c_double), value, target :: stretch_factor

      integer(c_int), value, target :: nflat_gradp

      integer(c_int), value, target :: num_levels

      integer(c_int) :: vct_a_size_0

      integer(c_int) :: vct_b_size_0

      integer(c_int) :: cell_areas_size_0

      integer(c_int) :: primal_normal_cell_x_size_0

      integer(c_int) :: primal_normal_cell_x_size_1

      integer(c_int) :: primal_normal_cell_y_size_0

      integer(c_int) :: primal_normal_cell_y_size_1

      integer(c_int) :: dual_normal_cell_x_size_0

      integer(c_int) :: dual_normal_cell_x_size_1

      integer(c_int) :: dual_normal_cell_y_size_0

      integer(c_int) :: dual_normal_cell_y_size_1

      integer(c_int) :: edge_areas_size_0

      integer(c_int) :: tangent_orientation_size_0

      integer(c_int) :: inverse_primal_edge_lengths_size_0

      integer(c_int) :: inverse_dual_edge_lengths_size_0

      integer(c_int) :: inverse_vertex_vertex_lengths_size_0

      integer(c_int) :: primal_normal_vert_x_size_0

      integer(c_int) :: primal_normal_vert_x_size_1

      integer(c_int) :: primal_normal_vert_y_size_0

      integer(c_int) :: primal_normal_vert_y_size_1

      integer(c_int) :: dual_normal_vert_x_size_0

      integer(c_int) :: dual_normal_vert_x_size_1

      integer(c_int) :: dual_normal_vert_y_size_0

      integer(c_int) :: dual_normal_vert_y_size_1

      integer(c_int) :: f_e_size_0

      integer(c_int) :: c_lin_e_size_0

      integer(c_int) :: c_lin_e_size_1

      integer(c_int) :: c_intp_size_0

      integer(c_int) :: c_intp_size_1

      integer(c_int) :: e_flx_avg_size_0

      integer(c_int) :: e_flx_avg_size_1

      integer(c_int) :: geofac_grdiv_size_0

      integer(c_int) :: geofac_grdiv_size_1

      integer(c_int) :: geofac_rot_size_0

      integer(c_int) :: geofac_rot_size_1

      integer(c_int) :: pos_on_tplane_e_1_size_0

      integer(c_int) :: pos_on_tplane_e_1_size_1

      integer(c_int) :: pos_on_tplane_e_2_size_0

      integer(c_int) :: pos_on_tplane_e_2_size_1

      integer(c_int) :: rbf_vec_coeff_e_size_0

      integer(c_int) :: rbf_vec_coeff_e_size_1

      integer(c_int) :: e_bln_c_s_size_0

      integer(c_int) :: e_bln_c_s_size_1

      integer(c_int) :: rbf_coeff_1_size_0

      integer(c_int) :: rbf_coeff_1_size_1

      integer(c_int) :: rbf_coeff_2_size_0

      integer(c_int) :: rbf_coeff_2_size_1

      integer(c_int) :: geofac_div_size_0

      integer(c_int) :: geofac_div_size_1

      integer(c_int) :: geofac_n2s_size_0

      integer(c_int) :: geofac_n2s_size_1

      integer(c_int) :: geofac_grg_x_size_0

      integer(c_int) :: geofac_grg_x_size_1

      integer(c_int) :: geofac_grg_y_size_0

      integer(c_int) :: geofac_grg_y_size_1

      integer(c_int) :: nudgecoeff_e_size_0

      integer(c_int) :: bdy_halo_c_size_0

      integer(c_int) :: mask_prog_halo_c_size_0

      integer(c_int) :: rayleigh_w_size_0

      integer(c_int) :: exner_exfac_size_0

      integer(c_int) :: exner_exfac_size_1

      integer(c_int) :: exner_ref_mc_size_0

      integer(c_int) :: exner_ref_mc_size_1

      integer(c_int) :: wgtfac_c_size_0

      integer(c_int) :: wgtfac_c_size_1

      integer(c_int) :: wgtfacq_c_size_0

      integer(c_int) :: wgtfacq_c_size_1

      integer(c_int) :: inv_ddqz_z_full_size_0

      integer(c_int) :: inv_ddqz_z_full_size_1

      integer(c_int) :: rho_ref_mc_size_0

      integer(c_int) :: rho_ref_mc_size_1

      integer(c_int) :: theta_ref_mc_size_0

      integer(c_int) :: theta_ref_mc_size_1

      integer(c_int) :: vwind_expl_wgt_size_0

      integer(c_int) :: d_exner_dz_ref_ic_size_0

      integer(c_int) :: d_exner_dz_ref_ic_size_1

      integer(c_int) :: ddqz_z_half_size_0

      integer(c_int) :: ddqz_z_half_size_1

      integer(c_int) :: theta_ref_ic_size_0

      integer(c_int) :: theta_ref_ic_size_1

      integer(c_int) :: d2dexdz2_fac1_mc_size_0

      integer(c_int) :: d2dexdz2_fac1_mc_size_1

      integer(c_int) :: d2dexdz2_fac2_mc_size_0

      integer(c_int) :: d2dexdz2_fac2_mc_size_1

      integer(c_int) :: rho_ref_me_size_0

      integer(c_int) :: rho_ref_me_size_1

      integer(c_int) :: theta_ref_me_size_0

      integer(c_int) :: theta_ref_me_size_1

      integer(c_int) :: ddxn_z_full_size_0

      integer(c_int) :: ddxn_z_full_size_1

      integer(c_int) :: zdiff_gradp_size_0

      integer(c_int) :: zdiff_gradp_size_1

      integer(c_int) :: zdiff_gradp_size_2

      integer(c_int) :: vertoffset_gradp_size_0

      integer(c_int) :: vertoffset_gradp_size_1

      integer(c_int) :: vertoffset_gradp_size_2

      integer(c_int) :: ipeidx_dsl_size_0

      integer(c_int) :: ipeidx_dsl_size_1

      integer(c_int) :: pg_exdist_size_0

      integer(c_int) :: pg_exdist_size_1

      integer(c_int) :: ddqz_z_full_e_size_0

      integer(c_int) :: ddqz_z_full_e_size_1

      integer(c_int) :: ddxt_z_full_size_0

      integer(c_int) :: ddxt_z_full_size_1

      integer(c_int) :: wgtfac_e_size_0

      integer(c_int) :: wgtfac_e_size_1

      integer(c_int) :: wgtfacq_e_size_0

      integer(c_int) :: wgtfacq_e_size_1

      integer(c_int) :: vwind_impl_wgt_size_0

      integer(c_int) :: hmask_dd3d_size_0

      integer(c_int) :: scalfac_dd3d_size_0

      integer(c_int) :: coeff1_dwdz_size_0

      integer(c_int) :: coeff1_dwdz_size_1

      integer(c_int) :: coeff2_dwdz_size_0

      integer(c_int) :: coeff2_dwdz_size_1

      integer(c_int) :: coeff_gradekin_size_0

      integer(c_int) :: coeff_gradekin_size_1

      integer(c_int) :: c_owner_mask_size_0

      integer(c_int) :: cell_center_lat_size_0

      integer(c_int) :: cell_center_lon_size_0

      integer(c_int) :: edge_center_lat_size_0

      integer(c_int) :: edge_center_lon_size_0

      integer(c_int) :: primal_normal_x_size_0

      integer(c_int) :: primal_normal_y_size_0

      integer(c_int) :: rc  ! Stores the return code
      ! ptrs

      !$acc host_data use_device(vct_a)
      !$acc host_data use_device(vct_b)
      !$acc host_data use_device(cell_areas)
      !$acc host_data use_device(primal_normal_cell_x)
      !$acc host_data use_device(primal_normal_cell_y)
      !$acc host_data use_device(dual_normal_cell_x)
      !$acc host_data use_device(dual_normal_cell_y)
      !$acc host_data use_device(edge_areas)
      !$acc host_data use_device(tangent_orientation)
      !$acc host_data use_device(inverse_primal_edge_lengths)
      !$acc host_data use_device(inverse_dual_edge_lengths)
      !$acc host_data use_device(inverse_vertex_vertex_lengths)
      !$acc host_data use_device(primal_normal_vert_x)
      !$acc host_data use_device(primal_normal_vert_y)
      !$acc host_data use_device(dual_normal_vert_x)
      !$acc host_data use_device(dual_normal_vert_y)
      !$acc host_data use_device(f_e)
      !$acc host_data use_device(c_lin_e)
      !$acc host_data use_device(c_intp)
      !$acc host_data use_device(e_flx_avg)
      !$acc host_data use_device(geofac_grdiv)
      !$acc host_data use_device(geofac_rot)
      !$acc host_data use_device(pos_on_tplane_e_1)
      !$acc host_data use_device(pos_on_tplane_e_2)
      !$acc host_data use_device(rbf_vec_coeff_e)
      !$acc host_data use_device(e_bln_c_s)
      !$acc host_data use_device(rbf_coeff_1)
      !$acc host_data use_device(rbf_coeff_2)
      !$acc host_data use_device(geofac_div)
      !$acc host_data use_device(geofac_n2s)
      !$acc host_data use_device(geofac_grg_x)
      !$acc host_data use_device(geofac_grg_y)
      !$acc host_data use_device(nudgecoeff_e)
      !$acc host_data use_device(bdy_halo_c)
      !$acc host_data use_device(mask_prog_halo_c)
      !$acc host_data use_device(rayleigh_w)
      !$acc host_data use_device(exner_exfac)
      !$acc host_data use_device(exner_ref_mc)
      !$acc host_data use_device(wgtfac_c)
      !$acc host_data use_device(wgtfacq_c)
      !$acc host_data use_device(inv_ddqz_z_full)
      !$acc host_data use_device(rho_ref_mc)
      !$acc host_data use_device(theta_ref_mc)
      !$acc host_data use_device(vwind_expl_wgt)
      !$acc host_data use_device(d_exner_dz_ref_ic)
      !$acc host_data use_device(ddqz_z_half)
      !$acc host_data use_device(theta_ref_ic)
      !$acc host_data use_device(d2dexdz2_fac1_mc)
      !$acc host_data use_device(d2dexdz2_fac2_mc)
      !$acc host_data use_device(rho_ref_me)
      !$acc host_data use_device(theta_ref_me)
      !$acc host_data use_device(ddxn_z_full)
      !$acc host_data use_device(zdiff_gradp)
      !$acc host_data use_device(vertoffset_gradp)
      !$acc host_data use_device(ipeidx_dsl)
      !$acc host_data use_device(pg_exdist)
      !$acc host_data use_device(ddqz_z_full_e)
      !$acc host_data use_device(ddxt_z_full)
      !$acc host_data use_device(wgtfac_e)
      !$acc host_data use_device(wgtfacq_e)
      !$acc host_data use_device(vwind_impl_wgt)
      !$acc host_data use_device(hmask_dd3d)
      !$acc host_data use_device(scalfac_dd3d)
      !$acc host_data use_device(coeff1_dwdz)
      !$acc host_data use_device(coeff2_dwdz)
      !$acc host_data use_device(coeff_gradekin)
      !$acc host_data use_device(c_owner_mask)
      !$acc host_data use_device(cell_center_lat)
      !$acc host_data use_device(cell_center_lon)
      !$acc host_data use_device(edge_center_lat)
      !$acc host_data use_device(edge_center_lon)
      !$acc host_data use_device(primal_normal_x)
      !$acc host_data use_device(primal_normal_y)

      vct_a_size_0 = SIZE(vct_a, 1)

      vct_b_size_0 = SIZE(vct_b, 1)

      cell_areas_size_0 = SIZE(cell_areas, 1)

      primal_normal_cell_x_size_0 = SIZE(primal_normal_cell_x, 1)
      primal_normal_cell_x_size_1 = SIZE(primal_normal_cell_x, 2)

      primal_normal_cell_y_size_0 = SIZE(primal_normal_cell_y, 1)
      primal_normal_cell_y_size_1 = SIZE(primal_normal_cell_y, 2)

      dual_normal_cell_x_size_0 = SIZE(dual_normal_cell_x, 1)
      dual_normal_cell_x_size_1 = SIZE(dual_normal_cell_x, 2)

      dual_normal_cell_y_size_0 = SIZE(dual_normal_cell_y, 1)
      dual_normal_cell_y_size_1 = SIZE(dual_normal_cell_y, 2)

      edge_areas_size_0 = SIZE(edge_areas, 1)

      tangent_orientation_size_0 = SIZE(tangent_orientation, 1)

      inverse_primal_edge_lengths_size_0 = SIZE(inverse_primal_edge_lengths, 1)

      inverse_dual_edge_lengths_size_0 = SIZE(inverse_dual_edge_lengths, 1)

      inverse_vertex_vertex_lengths_size_0 = SIZE(inverse_vertex_vertex_lengths, 1)

      primal_normal_vert_x_size_0 = SIZE(primal_normal_vert_x, 1)
      primal_normal_vert_x_size_1 = SIZE(primal_normal_vert_x, 2)

      primal_normal_vert_y_size_0 = SIZE(primal_normal_vert_y, 1)
      primal_normal_vert_y_size_1 = SIZE(primal_normal_vert_y, 2)

      dual_normal_vert_x_size_0 = SIZE(dual_normal_vert_x, 1)
      dual_normal_vert_x_size_1 = SIZE(dual_normal_vert_x, 2)

      dual_normal_vert_y_size_0 = SIZE(dual_normal_vert_y, 1)
      dual_normal_vert_y_size_1 = SIZE(dual_normal_vert_y, 2)

      f_e_size_0 = SIZE(f_e, 1)

      c_lin_e_size_0 = SIZE(c_lin_e, 1)
      c_lin_e_size_1 = SIZE(c_lin_e, 2)

      c_intp_size_0 = SIZE(c_intp, 1)
      c_intp_size_1 = SIZE(c_intp, 2)

      e_flx_avg_size_0 = SIZE(e_flx_avg, 1)
      e_flx_avg_size_1 = SIZE(e_flx_avg, 2)

      geofac_grdiv_size_0 = SIZE(geofac_grdiv, 1)
      geofac_grdiv_size_1 = SIZE(geofac_grdiv, 2)

      geofac_rot_size_0 = SIZE(geofac_rot, 1)
      geofac_rot_size_1 = SIZE(geofac_rot, 2)

      pos_on_tplane_e_1_size_0 = SIZE(pos_on_tplane_e_1, 1)
      pos_on_tplane_e_1_size_1 = SIZE(pos_on_tplane_e_1, 2)

      pos_on_tplane_e_2_size_0 = SIZE(pos_on_tplane_e_2, 1)
      pos_on_tplane_e_2_size_1 = SIZE(pos_on_tplane_e_2, 2)

      rbf_vec_coeff_e_size_0 = SIZE(rbf_vec_coeff_e, 1)
      rbf_vec_coeff_e_size_1 = SIZE(rbf_vec_coeff_e, 2)

      e_bln_c_s_size_0 = SIZE(e_bln_c_s, 1)
      e_bln_c_s_size_1 = SIZE(e_bln_c_s, 2)

      rbf_coeff_1_size_0 = SIZE(rbf_coeff_1, 1)
      rbf_coeff_1_size_1 = SIZE(rbf_coeff_1, 2)

      rbf_coeff_2_size_0 = SIZE(rbf_coeff_2, 1)
      rbf_coeff_2_size_1 = SIZE(rbf_coeff_2, 2)

      geofac_div_size_0 = SIZE(geofac_div, 1)
      geofac_div_size_1 = SIZE(geofac_div, 2)

      geofac_n2s_size_0 = SIZE(geofac_n2s, 1)
      geofac_n2s_size_1 = SIZE(geofac_n2s, 2)

      geofac_grg_x_size_0 = SIZE(geofac_grg_x, 1)
      geofac_grg_x_size_1 = SIZE(geofac_grg_x, 2)

      geofac_grg_y_size_0 = SIZE(geofac_grg_y, 1)
      geofac_grg_y_size_1 = SIZE(geofac_grg_y, 2)

      nudgecoeff_e_size_0 = SIZE(nudgecoeff_e, 1)

      bdy_halo_c_size_0 = SIZE(bdy_halo_c, 1)

      mask_prog_halo_c_size_0 = SIZE(mask_prog_halo_c, 1)

      rayleigh_w_size_0 = SIZE(rayleigh_w, 1)

      exner_exfac_size_0 = SIZE(exner_exfac, 1)
      exner_exfac_size_1 = SIZE(exner_exfac, 2)

      exner_ref_mc_size_0 = SIZE(exner_ref_mc, 1)
      exner_ref_mc_size_1 = SIZE(exner_ref_mc, 2)

      wgtfac_c_size_0 = SIZE(wgtfac_c, 1)
      wgtfac_c_size_1 = SIZE(wgtfac_c, 2)

      wgtfacq_c_size_0 = SIZE(wgtfacq_c, 1)
      wgtfacq_c_size_1 = SIZE(wgtfacq_c, 2)

      inv_ddqz_z_full_size_0 = SIZE(inv_ddqz_z_full, 1)
      inv_ddqz_z_full_size_1 = SIZE(inv_ddqz_z_full, 2)

      rho_ref_mc_size_0 = SIZE(rho_ref_mc, 1)
      rho_ref_mc_size_1 = SIZE(rho_ref_mc, 2)

      theta_ref_mc_size_0 = SIZE(theta_ref_mc, 1)
      theta_ref_mc_size_1 = SIZE(theta_ref_mc, 2)

      vwind_expl_wgt_size_0 = SIZE(vwind_expl_wgt, 1)

      d_exner_dz_ref_ic_size_0 = SIZE(d_exner_dz_ref_ic, 1)
      d_exner_dz_ref_ic_size_1 = SIZE(d_exner_dz_ref_ic, 2)

      ddqz_z_half_size_0 = SIZE(ddqz_z_half, 1)
      ddqz_z_half_size_1 = SIZE(ddqz_z_half, 2)

      theta_ref_ic_size_0 = SIZE(theta_ref_ic, 1)
      theta_ref_ic_size_1 = SIZE(theta_ref_ic, 2)

      d2dexdz2_fac1_mc_size_0 = SIZE(d2dexdz2_fac1_mc, 1)
      d2dexdz2_fac1_mc_size_1 = SIZE(d2dexdz2_fac1_mc, 2)

      d2dexdz2_fac2_mc_size_0 = SIZE(d2dexdz2_fac2_mc, 1)
      d2dexdz2_fac2_mc_size_1 = SIZE(d2dexdz2_fac2_mc, 2)

      rho_ref_me_size_0 = SIZE(rho_ref_me, 1)
      rho_ref_me_size_1 = SIZE(rho_ref_me, 2)

      theta_ref_me_size_0 = SIZE(theta_ref_me, 1)
      theta_ref_me_size_1 = SIZE(theta_ref_me, 2)

      ddxn_z_full_size_0 = SIZE(ddxn_z_full, 1)
      ddxn_z_full_size_1 = SIZE(ddxn_z_full, 2)

      zdiff_gradp_size_0 = SIZE(zdiff_gradp, 1)
      zdiff_gradp_size_1 = SIZE(zdiff_gradp, 2)
      zdiff_gradp_size_2 = SIZE(zdiff_gradp, 3)

      vertoffset_gradp_size_0 = SIZE(vertoffset_gradp, 1)
      vertoffset_gradp_size_1 = SIZE(vertoffset_gradp, 2)
      vertoffset_gradp_size_2 = SIZE(vertoffset_gradp, 3)

      ipeidx_dsl_size_0 = SIZE(ipeidx_dsl, 1)
      ipeidx_dsl_size_1 = SIZE(ipeidx_dsl, 2)

      pg_exdist_size_0 = SIZE(pg_exdist, 1)
      pg_exdist_size_1 = SIZE(pg_exdist, 2)

      ddqz_z_full_e_size_0 = SIZE(ddqz_z_full_e, 1)
      ddqz_z_full_e_size_1 = SIZE(ddqz_z_full_e, 2)

      ddxt_z_full_size_0 = SIZE(ddxt_z_full, 1)
      ddxt_z_full_size_1 = SIZE(ddxt_z_full, 2)

      wgtfac_e_size_0 = SIZE(wgtfac_e, 1)
      wgtfac_e_size_1 = SIZE(wgtfac_e, 2)

      wgtfacq_e_size_0 = SIZE(wgtfacq_e, 1)
      wgtfacq_e_size_1 = SIZE(wgtfacq_e, 2)

      vwind_impl_wgt_size_0 = SIZE(vwind_impl_wgt, 1)

      hmask_dd3d_size_0 = SIZE(hmask_dd3d, 1)

      scalfac_dd3d_size_0 = SIZE(scalfac_dd3d, 1)

      coeff1_dwdz_size_0 = SIZE(coeff1_dwdz, 1)
      coeff1_dwdz_size_1 = SIZE(coeff1_dwdz, 2)

      coeff2_dwdz_size_0 = SIZE(coeff2_dwdz, 1)
      coeff2_dwdz_size_1 = SIZE(coeff2_dwdz, 2)

      coeff_gradekin_size_0 = SIZE(coeff_gradekin, 1)
      coeff_gradekin_size_1 = SIZE(coeff_gradekin, 2)

      c_owner_mask_size_0 = SIZE(c_owner_mask, 1)

      cell_center_lat_size_0 = SIZE(cell_center_lat, 1)

      cell_center_lon_size_0 = SIZE(cell_center_lon, 1)

      edge_center_lat_size_0 = SIZE(edge_center_lat, 1)

      edge_center_lon_size_0 = SIZE(edge_center_lon, 1)

      primal_normal_x_size_0 = SIZE(primal_normal_x, 1)

      primal_normal_y_size_0 = SIZE(primal_normal_y, 1)

      rc = solve_nh_init_wrapper(vct_a=c_loc(vct_a), &
                                 vct_a_size_0=vct_a_size_0, &
                                 vct_b=c_loc(vct_b), &
                                 vct_b_size_0=vct_b_size_0, &
                                 cell_areas=c_loc(cell_areas), &
                                 cell_areas_size_0=cell_areas_size_0, &
                                 primal_normal_cell_x=c_loc(primal_normal_cell_x), &
                                 primal_normal_cell_x_size_0=primal_normal_cell_x_size_0, &
                                 primal_normal_cell_x_size_1=primal_normal_cell_x_size_1, &
                                 primal_normal_cell_y=c_loc(primal_normal_cell_y), &
                                 primal_normal_cell_y_size_0=primal_normal_cell_y_size_0, &
                                 primal_normal_cell_y_size_1=primal_normal_cell_y_size_1, &
                                 dual_normal_cell_x=c_loc(dual_normal_cell_x), &
                                 dual_normal_cell_x_size_0=dual_normal_cell_x_size_0, &
                                 dual_normal_cell_x_size_1=dual_normal_cell_x_size_1, &
                                 dual_normal_cell_y=c_loc(dual_normal_cell_y), &
                                 dual_normal_cell_y_size_0=dual_normal_cell_y_size_0, &
                                 dual_normal_cell_y_size_1=dual_normal_cell_y_size_1, &
                                 edge_areas=c_loc(edge_areas), &
                                 edge_areas_size_0=edge_areas_size_0, &
                                 tangent_orientation=c_loc(tangent_orientation), &
                                 tangent_orientation_size_0=tangent_orientation_size_0, &
                                 inverse_primal_edge_lengths=c_loc(inverse_primal_edge_lengths), &
                                 inverse_primal_edge_lengths_size_0=inverse_primal_edge_lengths_size_0, &
                                 inverse_dual_edge_lengths=c_loc(inverse_dual_edge_lengths), &
                                 inverse_dual_edge_lengths_size_0=inverse_dual_edge_lengths_size_0, &
                                 inverse_vertex_vertex_lengths=c_loc(inverse_vertex_vertex_lengths), &
                                 inverse_vertex_vertex_lengths_size_0=inverse_vertex_vertex_lengths_size_0, &
                                 primal_normal_vert_x=c_loc(primal_normal_vert_x), &
                                 primal_normal_vert_x_size_0=primal_normal_vert_x_size_0, &
                                 primal_normal_vert_x_size_1=primal_normal_vert_x_size_1, &
                                 primal_normal_vert_y=c_loc(primal_normal_vert_y), &
                                 primal_normal_vert_y_size_0=primal_normal_vert_y_size_0, &
                                 primal_normal_vert_y_size_1=primal_normal_vert_y_size_1, &
                                 dual_normal_vert_x=c_loc(dual_normal_vert_x), &
                                 dual_normal_vert_x_size_0=dual_normal_vert_x_size_0, &
                                 dual_normal_vert_x_size_1=dual_normal_vert_x_size_1, &
                                 dual_normal_vert_y=c_loc(dual_normal_vert_y), &
                                 dual_normal_vert_y_size_0=dual_normal_vert_y_size_0, &
                                 dual_normal_vert_y_size_1=dual_normal_vert_y_size_1, &
                                 f_e=c_loc(f_e), &
                                 f_e_size_0=f_e_size_0, &
                                 c_lin_e=c_loc(c_lin_e), &
                                 c_lin_e_size_0=c_lin_e_size_0, &
                                 c_lin_e_size_1=c_lin_e_size_1, &
                                 c_intp=c_loc(c_intp), &
                                 c_intp_size_0=c_intp_size_0, &
                                 c_intp_size_1=c_intp_size_1, &
                                 e_flx_avg=c_loc(e_flx_avg), &
                                 e_flx_avg_size_0=e_flx_avg_size_0, &
                                 e_flx_avg_size_1=e_flx_avg_size_1, &
                                 geofac_grdiv=c_loc(geofac_grdiv), &
                                 geofac_grdiv_size_0=geofac_grdiv_size_0, &
                                 geofac_grdiv_size_1=geofac_grdiv_size_1, &
                                 geofac_rot=c_loc(geofac_rot), &
                                 geofac_rot_size_0=geofac_rot_size_0, &
                                 geofac_rot_size_1=geofac_rot_size_1, &
                                 pos_on_tplane_e_1=c_loc(pos_on_tplane_e_1), &
                                 pos_on_tplane_e_1_size_0=pos_on_tplane_e_1_size_0, &
                                 pos_on_tplane_e_1_size_1=pos_on_tplane_e_1_size_1, &
                                 pos_on_tplane_e_2=c_loc(pos_on_tplane_e_2), &
                                 pos_on_tplane_e_2_size_0=pos_on_tplane_e_2_size_0, &
                                 pos_on_tplane_e_2_size_1=pos_on_tplane_e_2_size_1, &
                                 rbf_vec_coeff_e=c_loc(rbf_vec_coeff_e), &
                                 rbf_vec_coeff_e_size_0=rbf_vec_coeff_e_size_0, &
                                 rbf_vec_coeff_e_size_1=rbf_vec_coeff_e_size_1, &
                                 e_bln_c_s=c_loc(e_bln_c_s), &
                                 e_bln_c_s_size_0=e_bln_c_s_size_0, &
                                 e_bln_c_s_size_1=e_bln_c_s_size_1, &
                                 rbf_coeff_1=c_loc(rbf_coeff_1), &
                                 rbf_coeff_1_size_0=rbf_coeff_1_size_0, &
                                 rbf_coeff_1_size_1=rbf_coeff_1_size_1, &
                                 rbf_coeff_2=c_loc(rbf_coeff_2), &
                                 rbf_coeff_2_size_0=rbf_coeff_2_size_0, &
                                 rbf_coeff_2_size_1=rbf_coeff_2_size_1, &
                                 geofac_div=c_loc(geofac_div), &
                                 geofac_div_size_0=geofac_div_size_0, &
                                 geofac_div_size_1=geofac_div_size_1, &
                                 geofac_n2s=c_loc(geofac_n2s), &
                                 geofac_n2s_size_0=geofac_n2s_size_0, &
                                 geofac_n2s_size_1=geofac_n2s_size_1, &
                                 geofac_grg_x=c_loc(geofac_grg_x), &
                                 geofac_grg_x_size_0=geofac_grg_x_size_0, &
                                 geofac_grg_x_size_1=geofac_grg_x_size_1, &
                                 geofac_grg_y=c_loc(geofac_grg_y), &
                                 geofac_grg_y_size_0=geofac_grg_y_size_0, &
                                 geofac_grg_y_size_1=geofac_grg_y_size_1, &
                                 nudgecoeff_e=c_loc(nudgecoeff_e), &
                                 nudgecoeff_e_size_0=nudgecoeff_e_size_0, &
                                 bdy_halo_c=c_loc(bdy_halo_c), &
                                 bdy_halo_c_size_0=bdy_halo_c_size_0, &
                                 mask_prog_halo_c=c_loc(mask_prog_halo_c), &
                                 mask_prog_halo_c_size_0=mask_prog_halo_c_size_0, &
                                 rayleigh_w=c_loc(rayleigh_w), &
                                 rayleigh_w_size_0=rayleigh_w_size_0, &
                                 exner_exfac=c_loc(exner_exfac), &
                                 exner_exfac_size_0=exner_exfac_size_0, &
                                 exner_exfac_size_1=exner_exfac_size_1, &
                                 exner_ref_mc=c_loc(exner_ref_mc), &
                                 exner_ref_mc_size_0=exner_ref_mc_size_0, &
                                 exner_ref_mc_size_1=exner_ref_mc_size_1, &
                                 wgtfac_c=c_loc(wgtfac_c), &
                                 wgtfac_c_size_0=wgtfac_c_size_0, &
                                 wgtfac_c_size_1=wgtfac_c_size_1, &
                                 wgtfacq_c=c_loc(wgtfacq_c), &
                                 wgtfacq_c_size_0=wgtfacq_c_size_0, &
                                 wgtfacq_c_size_1=wgtfacq_c_size_1, &
                                 inv_ddqz_z_full=c_loc(inv_ddqz_z_full), &
                                 inv_ddqz_z_full_size_0=inv_ddqz_z_full_size_0, &
                                 inv_ddqz_z_full_size_1=inv_ddqz_z_full_size_1, &
                                 rho_ref_mc=c_loc(rho_ref_mc), &
                                 rho_ref_mc_size_0=rho_ref_mc_size_0, &
                                 rho_ref_mc_size_1=rho_ref_mc_size_1, &
                                 theta_ref_mc=c_loc(theta_ref_mc), &
                                 theta_ref_mc_size_0=theta_ref_mc_size_0, &
                                 theta_ref_mc_size_1=theta_ref_mc_size_1, &
                                 vwind_expl_wgt=c_loc(vwind_expl_wgt), &
                                 vwind_expl_wgt_size_0=vwind_expl_wgt_size_0, &
                                 d_exner_dz_ref_ic=c_loc(d_exner_dz_ref_ic), &
                                 d_exner_dz_ref_ic_size_0=d_exner_dz_ref_ic_size_0, &
                                 d_exner_dz_ref_ic_size_1=d_exner_dz_ref_ic_size_1, &
                                 ddqz_z_half=c_loc(ddqz_z_half), &
                                 ddqz_z_half_size_0=ddqz_z_half_size_0, &
                                 ddqz_z_half_size_1=ddqz_z_half_size_1, &
                                 theta_ref_ic=c_loc(theta_ref_ic), &
                                 theta_ref_ic_size_0=theta_ref_ic_size_0, &
                                 theta_ref_ic_size_1=theta_ref_ic_size_1, &
                                 d2dexdz2_fac1_mc=c_loc(d2dexdz2_fac1_mc), &
                                 d2dexdz2_fac1_mc_size_0=d2dexdz2_fac1_mc_size_0, &
                                 d2dexdz2_fac1_mc_size_1=d2dexdz2_fac1_mc_size_1, &
                                 d2dexdz2_fac2_mc=c_loc(d2dexdz2_fac2_mc), &
                                 d2dexdz2_fac2_mc_size_0=d2dexdz2_fac2_mc_size_0, &
                                 d2dexdz2_fac2_mc_size_1=d2dexdz2_fac2_mc_size_1, &
                                 rho_ref_me=c_loc(rho_ref_me), &
                                 rho_ref_me_size_0=rho_ref_me_size_0, &
                                 rho_ref_me_size_1=rho_ref_me_size_1, &
                                 theta_ref_me=c_loc(theta_ref_me), &
                                 theta_ref_me_size_0=theta_ref_me_size_0, &
                                 theta_ref_me_size_1=theta_ref_me_size_1, &
                                 ddxn_z_full=c_loc(ddxn_z_full), &
                                 ddxn_z_full_size_0=ddxn_z_full_size_0, &
                                 ddxn_z_full_size_1=ddxn_z_full_size_1, &
                                 zdiff_gradp=c_loc(zdiff_gradp), &
                                 zdiff_gradp_size_0=zdiff_gradp_size_0, &
                                 zdiff_gradp_size_1=zdiff_gradp_size_1, &
                                 zdiff_gradp_size_2=zdiff_gradp_size_2, &
                                 vertoffset_gradp=c_loc(vertoffset_gradp), &
                                 vertoffset_gradp_size_0=vertoffset_gradp_size_0, &
                                 vertoffset_gradp_size_1=vertoffset_gradp_size_1, &
                                 vertoffset_gradp_size_2=vertoffset_gradp_size_2, &
                                 ipeidx_dsl=c_loc(ipeidx_dsl), &
                                 ipeidx_dsl_size_0=ipeidx_dsl_size_0, &
                                 ipeidx_dsl_size_1=ipeidx_dsl_size_1, &
                                 pg_exdist=c_loc(pg_exdist), &
                                 pg_exdist_size_0=pg_exdist_size_0, &
                                 pg_exdist_size_1=pg_exdist_size_1, &
                                 ddqz_z_full_e=c_loc(ddqz_z_full_e), &
                                 ddqz_z_full_e_size_0=ddqz_z_full_e_size_0, &
                                 ddqz_z_full_e_size_1=ddqz_z_full_e_size_1, &
                                 ddxt_z_full=c_loc(ddxt_z_full), &
                                 ddxt_z_full_size_0=ddxt_z_full_size_0, &
                                 ddxt_z_full_size_1=ddxt_z_full_size_1, &
                                 wgtfac_e=c_loc(wgtfac_e), &
                                 wgtfac_e_size_0=wgtfac_e_size_0, &
                                 wgtfac_e_size_1=wgtfac_e_size_1, &
                                 wgtfacq_e=c_loc(wgtfacq_e), &
                                 wgtfacq_e_size_0=wgtfacq_e_size_0, &
                                 wgtfacq_e_size_1=wgtfacq_e_size_1, &
                                 vwind_impl_wgt=c_loc(vwind_impl_wgt), &
                                 vwind_impl_wgt_size_0=vwind_impl_wgt_size_0, &
                                 hmask_dd3d=c_loc(hmask_dd3d), &
                                 hmask_dd3d_size_0=hmask_dd3d_size_0, &
                                 scalfac_dd3d=c_loc(scalfac_dd3d), &
                                 scalfac_dd3d_size_0=scalfac_dd3d_size_0, &
                                 coeff1_dwdz=c_loc(coeff1_dwdz), &
                                 coeff1_dwdz_size_0=coeff1_dwdz_size_0, &
                                 coeff1_dwdz_size_1=coeff1_dwdz_size_1, &
                                 coeff2_dwdz=c_loc(coeff2_dwdz), &
                                 coeff2_dwdz_size_0=coeff2_dwdz_size_0, &
                                 coeff2_dwdz_size_1=coeff2_dwdz_size_1, &
                                 coeff_gradekin=c_loc(coeff_gradekin), &
                                 coeff_gradekin_size_0=coeff_gradekin_size_0, &
                                 coeff_gradekin_size_1=coeff_gradekin_size_1, &
                                 c_owner_mask=c_loc(c_owner_mask), &
                                 c_owner_mask_size_0=c_owner_mask_size_0, &
                                 cell_center_lat=c_loc(cell_center_lat), &
                                 cell_center_lat_size_0=cell_center_lat_size_0, &
                                 cell_center_lon=c_loc(cell_center_lon), &
                                 cell_center_lon_size_0=cell_center_lon_size_0, &
                                 edge_center_lat=c_loc(edge_center_lat), &
                                 edge_center_lat_size_0=edge_center_lat_size_0, &
                                 edge_center_lon=c_loc(edge_center_lon), &
                                 edge_center_lon_size_0=edge_center_lon_size_0, &
                                 primal_normal_x=c_loc(primal_normal_x), &
                                 primal_normal_x_size_0=primal_normal_x_size_0, &
                                 primal_normal_y=c_loc(primal_normal_y), &
                                 primal_normal_y_size_0=primal_normal_y_size_0, &
                                 rayleigh_damping_height=rayleigh_damping_height, &
                                 itime_scheme=itime_scheme, &
                                 iadv_rhotheta=iadv_rhotheta, &
                                 igradp_method=igradp_method, &
                                 ndyn_substeps=ndyn_substeps, &
                                 rayleigh_type=rayleigh_type, &
                                 rayleigh_coeff=rayleigh_coeff, &
                                 divdamp_order=divdamp_order, &
                                 is_iau_active=is_iau_active, &
                                 iau_wgt_dyn=iau_wgt_dyn, &
                                 divdamp_type=divdamp_type, &
                                 divdamp_trans_start=divdamp_trans_start, &
                                 divdamp_trans_end=divdamp_trans_end, &
                                 l_vert_nested=l_vert_nested, &
                                 rhotheta_offctr=rhotheta_offctr, &
                                 veladv_offctr=veladv_offctr, &
                                 max_nudging_coeff=max_nudging_coeff, &
                                 divdamp_fac=divdamp_fac, &
                                 divdamp_fac2=divdamp_fac2, &
                                 divdamp_fac3=divdamp_fac3, &
                                 divdamp_fac4=divdamp_fac4, &
                                 divdamp_z=divdamp_z, &
                                 divdamp_z2=divdamp_z2, &
                                 divdamp_z3=divdamp_z3, &
                                 divdamp_z4=divdamp_z4, &
                                 lowest_layer_thickness=lowest_layer_thickness, &
                                 model_top_height=model_top_height, &
                                 stretch_factor=stretch_factor, &
                                 nflat_gradp=nflat_gradp, &
                                 num_levels=num_levels)
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
   end subroutine solve_nh_init

   subroutine grid_init(cell_starts, &
                        cell_ends, &
                        vertex_starts, &
                        vertex_ends, &
                        edge_starts, &
                        edge_ends, &
                        c2e, &
                        e2c, &
                        c2e2c, &
                        e2c2e, &
                        e2v, &
                        v2e, &
                        v2c, &
                        e2c2v, &
                        c2v, &
                        global_root, &
                        global_level, &
                        num_vertices, &
                        num_cells, &
                        num_edges, &
                        vertical_size, &
                        limited_area, &
                        rc)
      use, intrinsic :: iso_c_binding

      integer(c_int), dimension(:), target :: cell_starts

      integer(c_int), dimension(:), target :: cell_ends

      integer(c_int), dimension(:), target :: vertex_starts

      integer(c_int), dimension(:), target :: vertex_ends

      integer(c_int), dimension(:), target :: edge_starts

      integer(c_int), dimension(:), target :: edge_ends

      integer(c_int), dimension(:, :), target :: c2e

      integer(c_int), dimension(:, :), target :: e2c

      integer(c_int), dimension(:, :), target :: c2e2c

      integer(c_int), dimension(:, :), target :: e2c2e

      integer(c_int), dimension(:, :), target :: e2v

      integer(c_int), dimension(:, :), target :: v2e

      integer(c_int), dimension(:, :), target :: v2c

      integer(c_int), dimension(:, :), target :: e2c2v

      integer(c_int), dimension(:, :), target :: c2v

      integer(c_int), value, target :: global_root

      integer(c_int), value, target :: global_level

      integer(c_int), value, target :: num_vertices

      integer(c_int), value, target :: num_cells

      integer(c_int), value, target :: num_edges

      integer(c_int), value, target :: vertical_size

      logical(c_int), value, target :: limited_area

      integer(c_int) :: cell_starts_size_0

      integer(c_int) :: cell_ends_size_0

      integer(c_int) :: vertex_starts_size_0

      integer(c_int) :: vertex_ends_size_0

      integer(c_int) :: edge_starts_size_0

      integer(c_int) :: edge_ends_size_0

      integer(c_int) :: c2e_size_0

      integer(c_int) :: c2e_size_1

      integer(c_int) :: e2c_size_0

      integer(c_int) :: e2c_size_1

      integer(c_int) :: c2e2c_size_0

      integer(c_int) :: c2e2c_size_1

      integer(c_int) :: e2c2e_size_0

      integer(c_int) :: e2c2e_size_1

      integer(c_int) :: e2v_size_0

      integer(c_int) :: e2v_size_1

      integer(c_int) :: v2e_size_0

      integer(c_int) :: v2e_size_1

      integer(c_int) :: v2c_size_0

      integer(c_int) :: v2c_size_1

      integer(c_int) :: e2c2v_size_0

      integer(c_int) :: e2c2v_size_1

      integer(c_int) :: c2v_size_0

      integer(c_int) :: c2v_size_1

      integer(c_int) :: rc  ! Stores the return code
      ! ptrs

      !$acc host_data use_device(cell_starts)
      !$acc host_data use_device(cell_ends)
      !$acc host_data use_device(vertex_starts)
      !$acc host_data use_device(vertex_ends)
      !$acc host_data use_device(edge_starts)
      !$acc host_data use_device(edge_ends)
      !$acc host_data use_device(c2e)
      !$acc host_data use_device(e2c)
      !$acc host_data use_device(c2e2c)
      !$acc host_data use_device(e2c2e)
      !$acc host_data use_device(e2v)
      !$acc host_data use_device(v2e)
      !$acc host_data use_device(v2c)
      !$acc host_data use_device(e2c2v)
      !$acc host_data use_device(c2v)

      cell_starts_size_0 = SIZE(cell_starts, 1)

      cell_ends_size_0 = SIZE(cell_ends, 1)

      vertex_starts_size_0 = SIZE(vertex_starts, 1)

      vertex_ends_size_0 = SIZE(vertex_ends, 1)

      edge_starts_size_0 = SIZE(edge_starts, 1)

      edge_ends_size_0 = SIZE(edge_ends, 1)

      c2e_size_0 = SIZE(c2e, 1)
      c2e_size_1 = SIZE(c2e, 2)

      e2c_size_0 = SIZE(e2c, 1)
      e2c_size_1 = SIZE(e2c, 2)

      c2e2c_size_0 = SIZE(c2e2c, 1)
      c2e2c_size_1 = SIZE(c2e2c, 2)

      e2c2e_size_0 = SIZE(e2c2e, 1)
      e2c2e_size_1 = SIZE(e2c2e, 2)

      e2v_size_0 = SIZE(e2v, 1)
      e2v_size_1 = SIZE(e2v, 2)

      v2e_size_0 = SIZE(v2e, 1)
      v2e_size_1 = SIZE(v2e, 2)

      v2c_size_0 = SIZE(v2c, 1)
      v2c_size_1 = SIZE(v2c, 2)

      e2c2v_size_0 = SIZE(e2c2v, 1)
      e2c2v_size_1 = SIZE(e2c2v, 2)

      c2v_size_0 = SIZE(c2v, 1)
      c2v_size_1 = SIZE(c2v, 2)

      rc = grid_init_wrapper(cell_starts=c_loc(cell_starts), &
                             cell_starts_size_0=cell_starts_size_0, &
                             cell_ends=c_loc(cell_ends), &
                             cell_ends_size_0=cell_ends_size_0, &
                             vertex_starts=c_loc(vertex_starts), &
                             vertex_starts_size_0=vertex_starts_size_0, &
                             vertex_ends=c_loc(vertex_ends), &
                             vertex_ends_size_0=vertex_ends_size_0, &
                             edge_starts=c_loc(edge_starts), &
                             edge_starts_size_0=edge_starts_size_0, &
                             edge_ends=c_loc(edge_ends), &
                             edge_ends_size_0=edge_ends_size_0, &
                             c2e=c_loc(c2e), &
                             c2e_size_0=c2e_size_0, &
                             c2e_size_1=c2e_size_1, &
                             e2c=c_loc(e2c), &
                             e2c_size_0=e2c_size_0, &
                             e2c_size_1=e2c_size_1, &
                             c2e2c=c_loc(c2e2c), &
                             c2e2c_size_0=c2e2c_size_0, &
                             c2e2c_size_1=c2e2c_size_1, &
                             e2c2e=c_loc(e2c2e), &
                             e2c2e_size_0=e2c2e_size_0, &
                             e2c2e_size_1=e2c2e_size_1, &
                             e2v=c_loc(e2v), &
                             e2v_size_0=e2v_size_0, &
                             e2v_size_1=e2v_size_1, &
                             v2e=c_loc(v2e), &
                             v2e_size_0=v2e_size_0, &
                             v2e_size_1=v2e_size_1, &
                             v2c=c_loc(v2c), &
                             v2c_size_0=v2c_size_0, &
                             v2c_size_1=v2c_size_1, &
                             e2c2v=c_loc(e2c2v), &
                             e2c2v_size_0=e2c2v_size_0, &
                             e2c2v_size_1=e2c2v_size_1, &
                             c2v=c_loc(c2v), &
                             c2v_size_0=c2v_size_0, &
                             c2v_size_1=c2v_size_1, &
                             global_root=global_root, &
                             global_level=global_level, &
                             num_vertices=num_vertices, &
                             num_cells=num_cells, &
                             num_edges=num_edges, &
                             vertical_size=vertical_size, &
                             limited_area=limited_area)
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
   end subroutine grid_init

end module