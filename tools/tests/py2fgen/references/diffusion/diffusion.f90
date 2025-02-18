module diffusion
   use, intrinsic :: iso_c_binding
   implicit none

   public :: diffusion_run

   public :: diffusion_init

   public :: grid_init_diffusion

   interface

      function diffusion_run_wrapper(w, &
                                     w_size_0, &
                                     w_size_1, &
                                     vn, &
                                     vn_size_0, &
                                     vn_size_1, &
                                     exner, &
                                     exner_size_0, &
                                     exner_size_1, &
                                     theta_v, &
                                     theta_v_size_0, &
                                     theta_v_size_1, &
                                     rho, &
                                     rho_size_0, &
                                     rho_size_1, &
                                     hdef_ic, &
                                     hdef_ic_size_0, &
                                     hdef_ic_size_1, &
                                     div_ic, &
                                     div_ic_size_0, &
                                     div_ic_size_1, &
                                     dwdx, &
                                     dwdx_size_0, &
                                     dwdx_size_1, &
                                     dwdy, &
                                     dwdy_size_0, &
                                     dwdy_size_1, &
                                     dtime, &
                                     linit) bind(c, name="diffusion_run_wrapper") result(rc)
         import :: c_int, c_double, c_bool, c_ptr
         integer(c_int) :: rc  ! Stores the return code

         type(c_ptr), value, target :: w

         integer(c_int), value :: w_size_0

         integer(c_int), value :: w_size_1

         type(c_ptr), value, target :: vn

         integer(c_int), value :: vn_size_0

         integer(c_int), value :: vn_size_1

         type(c_ptr), value, target :: exner

         integer(c_int), value :: exner_size_0

         integer(c_int), value :: exner_size_1

         type(c_ptr), value, target :: theta_v

         integer(c_int), value :: theta_v_size_0

         integer(c_int), value :: theta_v_size_1

         type(c_ptr), value, target :: rho

         integer(c_int), value :: rho_size_0

         integer(c_int), value :: rho_size_1

         type(c_ptr), value, target :: hdef_ic

         integer(c_int), value :: hdef_ic_size_0

         integer(c_int), value :: hdef_ic_size_1

         type(c_ptr), value, target :: div_ic

         integer(c_int), value :: div_ic_size_0

         integer(c_int), value :: div_ic_size_1

         type(c_ptr), value, target :: dwdx

         integer(c_int), value :: dwdx_size_0

         integer(c_int), value :: dwdx_size_1

         type(c_ptr), value, target :: dwdy

         integer(c_int), value :: dwdy_size_0

         integer(c_int), value :: dwdy_size_1

         real(c_double), value, target :: dtime

         logical(c_int), value, target :: linit

      end function diffusion_run_wrapper

      function diffusion_init_wrapper(vct_a, &
                                      vct_a_size_0, &
                                      vct_b, &
                                      vct_b_size_0, &
                                      theta_ref_mc, &
                                      theta_ref_mc_size_0, &
                                      theta_ref_mc_size_1, &
                                      wgtfac_c, &
                                      wgtfac_c_size_0, &
                                      wgtfac_c_size_1, &
                                      e_bln_c_s, &
                                      e_bln_c_s_size_0, &
                                      e_bln_c_s_size_1, &
                                      geofac_div, &
                                      geofac_div_size_0, &
                                      geofac_div_size_1, &
                                      geofac_grg_x, &
                                      geofac_grg_x_size_0, &
                                      geofac_grg_x_size_1, &
                                      geofac_grg_y, &
                                      geofac_grg_y_size_0, &
                                      geofac_grg_y_size_1, &
                                      geofac_n2s, &
                                      geofac_n2s_size_0, &
                                      geofac_n2s_size_1, &
                                      nudgecoeff_e, &
                                      nudgecoeff_e_size_0, &
                                      rbf_coeff_1, &
                                      rbf_coeff_1_size_0, &
                                      rbf_coeff_1_size_1, &
                                      rbf_coeff_2, &
                                      rbf_coeff_2_size_0, &
                                      rbf_coeff_2_size_1, &
                                      mask_hdiff, &
                                      mask_hdiff_size_0, &
                                      mask_hdiff_size_1, &
                                      zd_diffcoef, &
                                      zd_diffcoef_size_0, &
                                      zd_diffcoef_size_1, &
                                      zd_vertoffset, &
                                      zd_vertoffset_size_0, &
                                      zd_vertoffset_size_1, &
                                      zd_vertoffset_size_2, &
                                      zd_intcoef, &
                                      zd_intcoef_size_0, &
                                      zd_intcoef_size_1, &
                                      zd_intcoef_size_2, &
                                      ndyn_substeps, &
                                      rayleigh_damping_height, &
                                      nflat_gradp, &
                                      diffusion_type, &
                                      hdiff_w, &
                                      hdiff_vn, &
                                      zdiffu_t, &
                                      type_t_diffu, &
                                      type_vn_diffu, &
                                      hdiff_efdt_ratio, &
                                      smagorinski_scaling_factor, &
                                      hdiff_temp, &
                                      thslp_zdiffu, &
                                      thhgtd_zdiffu, &
                                      denom_diffu_v, &
                                      nudge_max_coeff, &
                                      itype_sher, &
                                      ltkeshs, &
                                      tangent_orientation, &
                                      tangent_orientation_size_0, &
                                      inverse_primal_edge_lengths, &
                                      inverse_primal_edge_lengths_size_0, &
                                      inv_dual_edge_length, &
                                      inv_dual_edge_length_size_0, &
                                      inv_vert_vert_length, &
                                      inv_vert_vert_length_size_0, &
                                      edge_areas, &
                                      edge_areas_size_0, &
                                      f_e, &
                                      f_e_size_0, &
                                      cell_center_lat, &
                                      cell_center_lat_size_0, &
                                      cell_center_lon, &
                                      cell_center_lon_size_0, &
                                      cell_areas, &
                                      cell_areas_size_0, &
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
                                      edge_center_lat, &
                                      edge_center_lat_size_0, &
                                      edge_center_lon, &
                                      edge_center_lon_size_0, &
                                      primal_normal_x, &
                                      primal_normal_x_size_0, &
                                      primal_normal_y, &
                                      primal_normal_y_size_0, &
                                      global_root, &
                                      global_level, &
                                      lowest_layer_thickness, &
                                      model_top_height, &
                                      stretch_factor) bind(c, name="diffusion_init_wrapper") result(rc)
         import :: c_int, c_double, c_bool, c_ptr
         integer(c_int) :: rc  ! Stores the return code

         type(c_ptr), value, target :: vct_a

         integer(c_int), value :: vct_a_size_0

         type(c_ptr), value, target :: vct_b

         integer(c_int), value :: vct_b_size_0

         type(c_ptr), value, target :: theta_ref_mc

         integer(c_int), value :: theta_ref_mc_size_0

         integer(c_int), value :: theta_ref_mc_size_1

         type(c_ptr), value, target :: wgtfac_c

         integer(c_int), value :: wgtfac_c_size_0

         integer(c_int), value :: wgtfac_c_size_1

         type(c_ptr), value, target :: e_bln_c_s

         integer(c_int), value :: e_bln_c_s_size_0

         integer(c_int), value :: e_bln_c_s_size_1

         type(c_ptr), value, target :: geofac_div

         integer(c_int), value :: geofac_div_size_0

         integer(c_int), value :: geofac_div_size_1

         type(c_ptr), value, target :: geofac_grg_x

         integer(c_int), value :: geofac_grg_x_size_0

         integer(c_int), value :: geofac_grg_x_size_1

         type(c_ptr), value, target :: geofac_grg_y

         integer(c_int), value :: geofac_grg_y_size_0

         integer(c_int), value :: geofac_grg_y_size_1

         type(c_ptr), value, target :: geofac_n2s

         integer(c_int), value :: geofac_n2s_size_0

         integer(c_int), value :: geofac_n2s_size_1

         type(c_ptr), value, target :: nudgecoeff_e

         integer(c_int), value :: nudgecoeff_e_size_0

         type(c_ptr), value, target :: rbf_coeff_1

         integer(c_int), value :: rbf_coeff_1_size_0

         integer(c_int), value :: rbf_coeff_1_size_1

         type(c_ptr), value, target :: rbf_coeff_2

         integer(c_int), value :: rbf_coeff_2_size_0

         integer(c_int), value :: rbf_coeff_2_size_1

         type(c_ptr), value, target :: mask_hdiff

         integer(c_int), value :: mask_hdiff_size_0

         integer(c_int), value :: mask_hdiff_size_1

         type(c_ptr), value, target :: zd_diffcoef

         integer(c_int), value :: zd_diffcoef_size_0

         integer(c_int), value :: zd_diffcoef_size_1

         type(c_ptr), value, target :: zd_vertoffset

         integer(c_int), value :: zd_vertoffset_size_0

         integer(c_int), value :: zd_vertoffset_size_1

         integer(c_int), value :: zd_vertoffset_size_2

         type(c_ptr), value, target :: zd_intcoef

         integer(c_int), value :: zd_intcoef_size_0

         integer(c_int), value :: zd_intcoef_size_1

         integer(c_int), value :: zd_intcoef_size_2

         integer(c_int), value, target :: ndyn_substeps

         real(c_double), value, target :: rayleigh_damping_height

         integer(c_int), value, target :: nflat_gradp

         integer(c_int), value, target :: diffusion_type

         logical(c_int), value, target :: hdiff_w

         logical(c_int), value, target :: hdiff_vn

         logical(c_int), value, target :: zdiffu_t

         integer(c_int), value, target :: type_t_diffu

         integer(c_int), value, target :: type_vn_diffu

         real(c_double), value, target :: hdiff_efdt_ratio

         real(c_double), value, target :: smagorinski_scaling_factor

         logical(c_int), value, target :: hdiff_temp

         real(c_double), value, target :: thslp_zdiffu

         real(c_double), value, target :: thhgtd_zdiffu

         real(c_double), value, target :: denom_diffu_v

         real(c_double), value, target :: nudge_max_coeff

         integer(c_int), value, target :: itype_sher

         logical(c_int), value, target :: ltkeshs

         type(c_ptr), value, target :: tangent_orientation

         integer(c_int), value :: tangent_orientation_size_0

         type(c_ptr), value, target :: inverse_primal_edge_lengths

         integer(c_int), value :: inverse_primal_edge_lengths_size_0

         type(c_ptr), value, target :: inv_dual_edge_length

         integer(c_int), value :: inv_dual_edge_length_size_0

         type(c_ptr), value, target :: inv_vert_vert_length

         integer(c_int), value :: inv_vert_vert_length_size_0

         type(c_ptr), value, target :: edge_areas

         integer(c_int), value :: edge_areas_size_0

         type(c_ptr), value, target :: f_e

         integer(c_int), value :: f_e_size_0

         type(c_ptr), value, target :: cell_center_lat

         integer(c_int), value :: cell_center_lat_size_0

         type(c_ptr), value, target :: cell_center_lon

         integer(c_int), value :: cell_center_lon_size_0

         type(c_ptr), value, target :: cell_areas

         integer(c_int), value :: cell_areas_size_0

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

         type(c_ptr), value, target :: edge_center_lat

         integer(c_int), value :: edge_center_lat_size_0

         type(c_ptr), value, target :: edge_center_lon

         integer(c_int), value :: edge_center_lon_size_0

         type(c_ptr), value, target :: primal_normal_x

         integer(c_int), value :: primal_normal_x_size_0

         type(c_ptr), value, target :: primal_normal_y

         integer(c_int), value :: primal_normal_y_size_0

         integer(c_int), value, target :: global_root

         integer(c_int), value, target :: global_level

         real(c_double), value, target :: lowest_layer_thickness

         real(c_double), value, target :: model_top_height

         real(c_double), value, target :: stretch_factor

      end function diffusion_init_wrapper

      function grid_init_diffusion_wrapper(cell_starts, &
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
                                           c_owner_mask, &
                                           c_owner_mask_size_0, &
                                           e_owner_mask, &
                                           e_owner_mask_size_0, &
                                           v_owner_mask, &
                                           v_owner_mask_size_0, &
                                           c_glb_index, &
                                           c_glb_index_size_0, &
                                           e_glb_index, &
                                           e_glb_index_size_0, &
                                           v_glb_index, &
                                           v_glb_index_size_0, &
                                           comm_id, &
                                           global_root, &
                                           global_level, &
                                           num_vertices, &
                                           num_cells, &
                                           num_edges, &
                                           vertical_size, &
                                           limited_area) bind(c, name="grid_init_diffusion_wrapper") result(rc)
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

         type(c_ptr), value, target :: c_owner_mask

         integer(c_int), value :: c_owner_mask_size_0

         type(c_ptr), value, target :: e_owner_mask

         integer(c_int), value :: e_owner_mask_size_0

         type(c_ptr), value, target :: v_owner_mask

         integer(c_int), value :: v_owner_mask_size_0

         type(c_ptr), value, target :: c_glb_index

         integer(c_int), value :: c_glb_index_size_0

         type(c_ptr), value, target :: e_glb_index

         integer(c_int), value :: e_glb_index_size_0

         type(c_ptr), value, target :: v_glb_index

         integer(c_int), value :: v_glb_index_size_0

         integer(c_int), value, target :: comm_id

         integer(c_int), value, target :: global_root

         integer(c_int), value, target :: global_level

         integer(c_int), value, target :: num_vertices

         integer(c_int), value, target :: num_cells

         integer(c_int), value, target :: num_edges

         integer(c_int), value, target :: vertical_size

         logical(c_int), value, target :: limited_area

      end function grid_init_diffusion_wrapper

   end interface

contains

   subroutine diffusion_run(w, &
                            vn, &
                            exner, &
                            theta_v, &
                            rho, &
                            hdef_ic, &
                            div_ic, &
                            dwdx, &
                            dwdy, &
                            dtime, &
                            linit, &
                            rc)
      use, intrinsic :: iso_c_binding

      real(c_double), dimension(:, :), target :: w

      real(c_double), dimension(:, :), target :: vn

      real(c_double), dimension(:, :), target :: exner

      real(c_double), dimension(:, :), target :: theta_v

      real(c_double), dimension(:, :), target :: rho

      real(c_double), dimension(:, :), pointer :: hdef_ic

      real(c_double), dimension(:, :), pointer :: div_ic

      real(c_double), dimension(:, :), pointer :: dwdx

      real(c_double), dimension(:, :), pointer :: dwdy

      real(c_double), value, target :: dtime

      logical(c_int), value, target :: linit

      integer(c_int) :: w_size_0

      integer(c_int) :: w_size_1

      integer(c_int) :: vn_size_0

      integer(c_int) :: vn_size_1

      integer(c_int) :: exner_size_0

      integer(c_int) :: exner_size_1

      integer(c_int) :: theta_v_size_0

      integer(c_int) :: theta_v_size_1

      integer(c_int) :: rho_size_0

      integer(c_int) :: rho_size_1

      integer(c_int) :: hdef_ic_size_0

      integer(c_int) :: hdef_ic_size_1

      integer(c_int) :: div_ic_size_0

      integer(c_int) :: div_ic_size_1

      integer(c_int) :: dwdx_size_0

      integer(c_int) :: dwdx_size_1

      integer(c_int) :: dwdy_size_0

      integer(c_int) :: dwdy_size_1

      integer(c_int) :: rc  ! Stores the return code
      ! ptrs

      type(c_ptr) :: hdef_ic_ptr

      type(c_ptr) :: div_ic_ptr

      type(c_ptr) :: dwdx_ptr

      type(c_ptr) :: dwdy_ptr

      hdef_ic_ptr = c_null_ptr

      div_ic_ptr = c_null_ptr

      dwdx_ptr = c_null_ptr

      dwdy_ptr = c_null_ptr

      !$acc host_data use_device(w)
      !$acc host_data use_device(vn)
      !$acc host_data use_device(exner)
      !$acc host_data use_device(theta_v)
      !$acc host_data use_device(rho)
      !$acc host_data use_device(hdef_ic) if(associated(hdef_ic))
      !$acc host_data use_device(div_ic) if(associated(div_ic))
      !$acc host_data use_device(dwdx) if(associated(dwdx))
      !$acc host_data use_device(dwdy) if(associated(dwdy))

      w_size_0 = SIZE(w, 1)
      w_size_1 = SIZE(w, 2)

      vn_size_0 = SIZE(vn, 1)
      vn_size_1 = SIZE(vn, 2)

      exner_size_0 = SIZE(exner, 1)
      exner_size_1 = SIZE(exner, 2)

      theta_v_size_0 = SIZE(theta_v, 1)
      theta_v_size_1 = SIZE(theta_v, 2)

      rho_size_0 = SIZE(rho, 1)
      rho_size_1 = SIZE(rho, 2)

      if (associated(hdef_ic)) then
         hdef_ic_ptr = c_loc(hdef_ic)
         hdef_ic_size_0 = SIZE(hdef_ic, 1)
         hdef_ic_size_1 = SIZE(hdef_ic, 2)
      end if

      if (associated(div_ic)) then
         div_ic_ptr = c_loc(div_ic)
         div_ic_size_0 = SIZE(div_ic, 1)
         div_ic_size_1 = SIZE(div_ic, 2)
      end if

      if (associated(dwdx)) then
         dwdx_ptr = c_loc(dwdx)
         dwdx_size_0 = SIZE(dwdx, 1)
         dwdx_size_1 = SIZE(dwdx, 2)
      end if

      if (associated(dwdy)) then
         dwdy_ptr = c_loc(dwdy)
         dwdy_size_0 = SIZE(dwdy, 1)
         dwdy_size_1 = SIZE(dwdy, 2)
      end if

      rc = diffusion_run_wrapper(w=c_loc(w), &
                                 w_size_0=w_size_0, &
                                 w_size_1=w_size_1, &
                                 vn=c_loc(vn), &
                                 vn_size_0=vn_size_0, &
                                 vn_size_1=vn_size_1, &
                                 exner=c_loc(exner), &
                                 exner_size_0=exner_size_0, &
                                 exner_size_1=exner_size_1, &
                                 theta_v=c_loc(theta_v), &
                                 theta_v_size_0=theta_v_size_0, &
                                 theta_v_size_1=theta_v_size_1, &
                                 rho=c_loc(rho), &
                                 rho_size_0=rho_size_0, &
                                 rho_size_1=rho_size_1, &
                                 hdef_ic=hdef_ic_ptr, &
                                 hdef_ic_size_0=hdef_ic_size_0, &
                                 hdef_ic_size_1=hdef_ic_size_1, &
                                 div_ic=div_ic_ptr, &
                                 div_ic_size_0=div_ic_size_0, &
                                 div_ic_size_1=div_ic_size_1, &
                                 dwdx=dwdx_ptr, &
                                 dwdx_size_0=dwdx_size_0, &
                                 dwdx_size_1=dwdx_size_1, &
                                 dwdy=dwdy_ptr, &
                                 dwdy_size_0=dwdy_size_0, &
                                 dwdy_size_1=dwdy_size_1, &
                                 dtime=dtime, &
                                 linit=linit)
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
   end subroutine diffusion_run

   subroutine diffusion_init(vct_a, &
                             vct_b, &
                             theta_ref_mc, &
                             wgtfac_c, &
                             e_bln_c_s, &
                             geofac_div, &
                             geofac_grg_x, &
                             geofac_grg_y, &
                             geofac_n2s, &
                             nudgecoeff_e, &
                             rbf_coeff_1, &
                             rbf_coeff_2, &
                             mask_hdiff, &
                             zd_diffcoef, &
                             zd_vertoffset, &
                             zd_intcoef, &
                             ndyn_substeps, &
                             rayleigh_damping_height, &
                             nflat_gradp, &
                             diffusion_type, &
                             hdiff_w, &
                             hdiff_vn, &
                             zdiffu_t, &
                             type_t_diffu, &
                             type_vn_diffu, &
                             hdiff_efdt_ratio, &
                             smagorinski_scaling_factor, &
                             hdiff_temp, &
                             thslp_zdiffu, &
                             thhgtd_zdiffu, &
                             denom_diffu_v, &
                             nudge_max_coeff, &
                             itype_sher, &
                             ltkeshs, &
                             tangent_orientation, &
                             inverse_primal_edge_lengths, &
                             inv_dual_edge_length, &
                             inv_vert_vert_length, &
                             edge_areas, &
                             f_e, &
                             cell_center_lat, &
                             cell_center_lon, &
                             cell_areas, &
                             primal_normal_vert_x, &
                             primal_normal_vert_y, &
                             dual_normal_vert_x, &
                             dual_normal_vert_y, &
                             primal_normal_cell_x, &
                             primal_normal_cell_y, &
                             dual_normal_cell_x, &
                             dual_normal_cell_y, &
                             edge_center_lat, &
                             edge_center_lon, &
                             primal_normal_x, &
                             primal_normal_y, &
                             global_root, &
                             global_level, &
                             lowest_layer_thickness, &
                             model_top_height, &
                             stretch_factor, &
                             rc)
      use, intrinsic :: iso_c_binding

      real(c_double), dimension(:), target :: vct_a

      real(c_double), dimension(:), target :: vct_b

      real(c_double), dimension(:, :), target :: theta_ref_mc

      real(c_double), dimension(:, :), target :: wgtfac_c

      real(c_double), dimension(:, :), target :: e_bln_c_s

      real(c_double), dimension(:, :), target :: geofac_div

      real(c_double), dimension(:, :), target :: geofac_grg_x

      real(c_double), dimension(:, :), target :: geofac_grg_y

      real(c_double), dimension(:, :), target :: geofac_n2s

      real(c_double), dimension(:), target :: nudgecoeff_e

      real(c_double), dimension(:, :), target :: rbf_coeff_1

      real(c_double), dimension(:, :), target :: rbf_coeff_2

      logical(c_int), dimension(:, :), pointer :: mask_hdiff

      real(c_double), dimension(:, :), pointer :: zd_diffcoef

      integer(c_int), dimension(:, :, :), pointer :: zd_vertoffset

      real(c_double), dimension(:, :, :), pointer :: zd_intcoef

      integer(c_int), value, target :: ndyn_substeps

      real(c_double), value, target :: rayleigh_damping_height

      integer(c_int), value, target :: nflat_gradp

      integer(c_int), value, target :: diffusion_type

      logical(c_int), value, target :: hdiff_w

      logical(c_int), value, target :: hdiff_vn

      logical(c_int), value, target :: zdiffu_t

      integer(c_int), value, target :: type_t_diffu

      integer(c_int), value, target :: type_vn_diffu

      real(c_double), value, target :: hdiff_efdt_ratio

      real(c_double), value, target :: smagorinski_scaling_factor

      logical(c_int), value, target :: hdiff_temp

      real(c_double), value, target :: thslp_zdiffu

      real(c_double), value, target :: thhgtd_zdiffu

      real(c_double), value, target :: denom_diffu_v

      real(c_double), value, target :: nudge_max_coeff

      integer(c_int), value, target :: itype_sher

      logical(c_int), value, target :: ltkeshs

      real(c_double), dimension(:), target :: tangent_orientation

      real(c_double), dimension(:), target :: inverse_primal_edge_lengths

      real(c_double), dimension(:), target :: inv_dual_edge_length

      real(c_double), dimension(:), target :: inv_vert_vert_length

      real(c_double), dimension(:), target :: edge_areas

      real(c_double), dimension(:), target :: f_e

      real(c_double), dimension(:), target :: cell_center_lat

      real(c_double), dimension(:), target :: cell_center_lon

      real(c_double), dimension(:), target :: cell_areas

      real(c_double), dimension(:, :), target :: primal_normal_vert_x

      real(c_double), dimension(:, :), target :: primal_normal_vert_y

      real(c_double), dimension(:, :), target :: dual_normal_vert_x

      real(c_double), dimension(:, :), target :: dual_normal_vert_y

      real(c_double), dimension(:, :), target :: primal_normal_cell_x

      real(c_double), dimension(:, :), target :: primal_normal_cell_y

      real(c_double), dimension(:, :), target :: dual_normal_cell_x

      real(c_double), dimension(:, :), target :: dual_normal_cell_y

      real(c_double), dimension(:), target :: edge_center_lat

      real(c_double), dimension(:), target :: edge_center_lon

      real(c_double), dimension(:), target :: primal_normal_x

      real(c_double), dimension(:), target :: primal_normal_y

      integer(c_int), value, target :: global_root

      integer(c_int), value, target :: global_level

      real(c_double), value, target :: lowest_layer_thickness

      real(c_double), value, target :: model_top_height

      real(c_double), value, target :: stretch_factor

      integer(c_int) :: vct_a_size_0

      integer(c_int) :: vct_b_size_0

      integer(c_int) :: theta_ref_mc_size_0

      integer(c_int) :: theta_ref_mc_size_1

      integer(c_int) :: wgtfac_c_size_0

      integer(c_int) :: wgtfac_c_size_1

      integer(c_int) :: e_bln_c_s_size_0

      integer(c_int) :: e_bln_c_s_size_1

      integer(c_int) :: geofac_div_size_0

      integer(c_int) :: geofac_div_size_1

      integer(c_int) :: geofac_grg_x_size_0

      integer(c_int) :: geofac_grg_x_size_1

      integer(c_int) :: geofac_grg_y_size_0

      integer(c_int) :: geofac_grg_y_size_1

      integer(c_int) :: geofac_n2s_size_0

      integer(c_int) :: geofac_n2s_size_1

      integer(c_int) :: nudgecoeff_e_size_0

      integer(c_int) :: rbf_coeff_1_size_0

      integer(c_int) :: rbf_coeff_1_size_1

      integer(c_int) :: rbf_coeff_2_size_0

      integer(c_int) :: rbf_coeff_2_size_1

      integer(c_int) :: mask_hdiff_size_0

      integer(c_int) :: mask_hdiff_size_1

      integer(c_int) :: zd_diffcoef_size_0

      integer(c_int) :: zd_diffcoef_size_1

      integer(c_int) :: zd_vertoffset_size_0

      integer(c_int) :: zd_vertoffset_size_1

      integer(c_int) :: zd_vertoffset_size_2

      integer(c_int) :: zd_intcoef_size_0

      integer(c_int) :: zd_intcoef_size_1

      integer(c_int) :: zd_intcoef_size_2

      integer(c_int) :: tangent_orientation_size_0

      integer(c_int) :: inverse_primal_edge_lengths_size_0

      integer(c_int) :: inv_dual_edge_length_size_0

      integer(c_int) :: inv_vert_vert_length_size_0

      integer(c_int) :: edge_areas_size_0

      integer(c_int) :: f_e_size_0

      integer(c_int) :: cell_center_lat_size_0

      integer(c_int) :: cell_center_lon_size_0

      integer(c_int) :: cell_areas_size_0

      integer(c_int) :: primal_normal_vert_x_size_0

      integer(c_int) :: primal_normal_vert_x_size_1

      integer(c_int) :: primal_normal_vert_y_size_0

      integer(c_int) :: primal_normal_vert_y_size_1

      integer(c_int) :: dual_normal_vert_x_size_0

      integer(c_int) :: dual_normal_vert_x_size_1

      integer(c_int) :: dual_normal_vert_y_size_0

      integer(c_int) :: dual_normal_vert_y_size_1

      integer(c_int) :: primal_normal_cell_x_size_0

      integer(c_int) :: primal_normal_cell_x_size_1

      integer(c_int) :: primal_normal_cell_y_size_0

      integer(c_int) :: primal_normal_cell_y_size_1

      integer(c_int) :: dual_normal_cell_x_size_0

      integer(c_int) :: dual_normal_cell_x_size_1

      integer(c_int) :: dual_normal_cell_y_size_0

      integer(c_int) :: dual_normal_cell_y_size_1

      integer(c_int) :: edge_center_lat_size_0

      integer(c_int) :: edge_center_lon_size_0

      integer(c_int) :: primal_normal_x_size_0

      integer(c_int) :: primal_normal_y_size_0

      integer(c_int) :: rc  ! Stores the return code
      ! ptrs

      type(c_ptr) :: mask_hdiff_ptr

      type(c_ptr) :: zd_diffcoef_ptr

      type(c_ptr) :: zd_vertoffset_ptr

      type(c_ptr) :: zd_intcoef_ptr

      mask_hdiff_ptr = c_null_ptr

      zd_diffcoef_ptr = c_null_ptr

      zd_vertoffset_ptr = c_null_ptr

      zd_intcoef_ptr = c_null_ptr

      !$acc host_data use_device(vct_a)
      !$acc host_data use_device(vct_b)
      !$acc host_data use_device(theta_ref_mc)
      !$acc host_data use_device(wgtfac_c)
      !$acc host_data use_device(e_bln_c_s)
      !$acc host_data use_device(geofac_div)
      !$acc host_data use_device(geofac_grg_x)
      !$acc host_data use_device(geofac_grg_y)
      !$acc host_data use_device(geofac_n2s)
      !$acc host_data use_device(nudgecoeff_e)
      !$acc host_data use_device(rbf_coeff_1)
      !$acc host_data use_device(rbf_coeff_2)
      !$acc host_data use_device(tangent_orientation)
      !$acc host_data use_device(inverse_primal_edge_lengths)
      !$acc host_data use_device(inv_dual_edge_length)
      !$acc host_data use_device(inv_vert_vert_length)
      !$acc host_data use_device(edge_areas)
      !$acc host_data use_device(f_e)
      !$acc host_data use_device(cell_center_lat)
      !$acc host_data use_device(cell_center_lon)
      !$acc host_data use_device(cell_areas)
      !$acc host_data use_device(primal_normal_vert_x)
      !$acc host_data use_device(primal_normal_vert_y)
      !$acc host_data use_device(dual_normal_vert_x)
      !$acc host_data use_device(dual_normal_vert_y)
      !$acc host_data use_device(primal_normal_cell_x)
      !$acc host_data use_device(primal_normal_cell_y)
      !$acc host_data use_device(dual_normal_cell_x)
      !$acc host_data use_device(dual_normal_cell_y)
      !$acc host_data use_device(edge_center_lat)
      !$acc host_data use_device(edge_center_lon)
      !$acc host_data use_device(primal_normal_x)
      !$acc host_data use_device(primal_normal_y)
      !$acc host_data use_device(mask_hdiff) if(associated(mask_hdiff))
      !$acc host_data use_device(zd_diffcoef) if(associated(zd_diffcoef))
      !$acc host_data use_device(zd_vertoffset) if(associated(zd_vertoffset))
      !$acc host_data use_device(zd_intcoef) if(associated(zd_intcoef))

      vct_a_size_0 = SIZE(vct_a, 1)

      vct_b_size_0 = SIZE(vct_b, 1)

      theta_ref_mc_size_0 = SIZE(theta_ref_mc, 1)
      theta_ref_mc_size_1 = SIZE(theta_ref_mc, 2)

      wgtfac_c_size_0 = SIZE(wgtfac_c, 1)
      wgtfac_c_size_1 = SIZE(wgtfac_c, 2)

      e_bln_c_s_size_0 = SIZE(e_bln_c_s, 1)
      e_bln_c_s_size_1 = SIZE(e_bln_c_s, 2)

      geofac_div_size_0 = SIZE(geofac_div, 1)
      geofac_div_size_1 = SIZE(geofac_div, 2)

      geofac_grg_x_size_0 = SIZE(geofac_grg_x, 1)
      geofac_grg_x_size_1 = SIZE(geofac_grg_x, 2)

      geofac_grg_y_size_0 = SIZE(geofac_grg_y, 1)
      geofac_grg_y_size_1 = SIZE(geofac_grg_y, 2)

      geofac_n2s_size_0 = SIZE(geofac_n2s, 1)
      geofac_n2s_size_1 = SIZE(geofac_n2s, 2)

      nudgecoeff_e_size_0 = SIZE(nudgecoeff_e, 1)

      rbf_coeff_1_size_0 = SIZE(rbf_coeff_1, 1)
      rbf_coeff_1_size_1 = SIZE(rbf_coeff_1, 2)

      rbf_coeff_2_size_0 = SIZE(rbf_coeff_2, 1)
      rbf_coeff_2_size_1 = SIZE(rbf_coeff_2, 2)

      tangent_orientation_size_0 = SIZE(tangent_orientation, 1)

      inverse_primal_edge_lengths_size_0 = SIZE(inverse_primal_edge_lengths, 1)

      inv_dual_edge_length_size_0 = SIZE(inv_dual_edge_length, 1)

      inv_vert_vert_length_size_0 = SIZE(inv_vert_vert_length, 1)

      edge_areas_size_0 = SIZE(edge_areas, 1)

      f_e_size_0 = SIZE(f_e, 1)

      cell_center_lat_size_0 = SIZE(cell_center_lat, 1)

      cell_center_lon_size_0 = SIZE(cell_center_lon, 1)

      cell_areas_size_0 = SIZE(cell_areas, 1)

      primal_normal_vert_x_size_0 = SIZE(primal_normal_vert_x, 1)
      primal_normal_vert_x_size_1 = SIZE(primal_normal_vert_x, 2)

      primal_normal_vert_y_size_0 = SIZE(primal_normal_vert_y, 1)
      primal_normal_vert_y_size_1 = SIZE(primal_normal_vert_y, 2)

      dual_normal_vert_x_size_0 = SIZE(dual_normal_vert_x, 1)
      dual_normal_vert_x_size_1 = SIZE(dual_normal_vert_x, 2)

      dual_normal_vert_y_size_0 = SIZE(dual_normal_vert_y, 1)
      dual_normal_vert_y_size_1 = SIZE(dual_normal_vert_y, 2)

      primal_normal_cell_x_size_0 = SIZE(primal_normal_cell_x, 1)
      primal_normal_cell_x_size_1 = SIZE(primal_normal_cell_x, 2)

      primal_normal_cell_y_size_0 = SIZE(primal_normal_cell_y, 1)
      primal_normal_cell_y_size_1 = SIZE(primal_normal_cell_y, 2)

      dual_normal_cell_x_size_0 = SIZE(dual_normal_cell_x, 1)
      dual_normal_cell_x_size_1 = SIZE(dual_normal_cell_x, 2)

      dual_normal_cell_y_size_0 = SIZE(dual_normal_cell_y, 1)
      dual_normal_cell_y_size_1 = SIZE(dual_normal_cell_y, 2)

      edge_center_lat_size_0 = SIZE(edge_center_lat, 1)

      edge_center_lon_size_0 = SIZE(edge_center_lon, 1)

      primal_normal_x_size_0 = SIZE(primal_normal_x, 1)

      primal_normal_y_size_0 = SIZE(primal_normal_y, 1)

      if (associated(mask_hdiff)) then
         mask_hdiff_ptr = c_loc(mask_hdiff)
         mask_hdiff_size_0 = SIZE(mask_hdiff, 1)
         mask_hdiff_size_1 = SIZE(mask_hdiff, 2)
      end if

      if (associated(zd_diffcoef)) then
         zd_diffcoef_ptr = c_loc(zd_diffcoef)
         zd_diffcoef_size_0 = SIZE(zd_diffcoef, 1)
         zd_diffcoef_size_1 = SIZE(zd_diffcoef, 2)
      end if

      if (associated(zd_vertoffset)) then
         zd_vertoffset_ptr = c_loc(zd_vertoffset)
         zd_vertoffset_size_0 = SIZE(zd_vertoffset, 1)
         zd_vertoffset_size_1 = SIZE(zd_vertoffset, 2)
         zd_vertoffset_size_2 = SIZE(zd_vertoffset, 3)
      end if

      if (associated(zd_intcoef)) then
         zd_intcoef_ptr = c_loc(zd_intcoef)
         zd_intcoef_size_0 = SIZE(zd_intcoef, 1)
         zd_intcoef_size_1 = SIZE(zd_intcoef, 2)
         zd_intcoef_size_2 = SIZE(zd_intcoef, 3)
      end if

      rc = diffusion_init_wrapper(vct_a=c_loc(vct_a), &
                                  vct_a_size_0=vct_a_size_0, &
                                  vct_b=c_loc(vct_b), &
                                  vct_b_size_0=vct_b_size_0, &
                                  theta_ref_mc=c_loc(theta_ref_mc), &
                                  theta_ref_mc_size_0=theta_ref_mc_size_0, &
                                  theta_ref_mc_size_1=theta_ref_mc_size_1, &
                                  wgtfac_c=c_loc(wgtfac_c), &
                                  wgtfac_c_size_0=wgtfac_c_size_0, &
                                  wgtfac_c_size_1=wgtfac_c_size_1, &
                                  e_bln_c_s=c_loc(e_bln_c_s), &
                                  e_bln_c_s_size_0=e_bln_c_s_size_0, &
                                  e_bln_c_s_size_1=e_bln_c_s_size_1, &
                                  geofac_div=c_loc(geofac_div), &
                                  geofac_div_size_0=geofac_div_size_0, &
                                  geofac_div_size_1=geofac_div_size_1, &
                                  geofac_grg_x=c_loc(geofac_grg_x), &
                                  geofac_grg_x_size_0=geofac_grg_x_size_0, &
                                  geofac_grg_x_size_1=geofac_grg_x_size_1, &
                                  geofac_grg_y=c_loc(geofac_grg_y), &
                                  geofac_grg_y_size_0=geofac_grg_y_size_0, &
                                  geofac_grg_y_size_1=geofac_grg_y_size_1, &
                                  geofac_n2s=c_loc(geofac_n2s), &
                                  geofac_n2s_size_0=geofac_n2s_size_0, &
                                  geofac_n2s_size_1=geofac_n2s_size_1, &
                                  nudgecoeff_e=c_loc(nudgecoeff_e), &
                                  nudgecoeff_e_size_0=nudgecoeff_e_size_0, &
                                  rbf_coeff_1=c_loc(rbf_coeff_1), &
                                  rbf_coeff_1_size_0=rbf_coeff_1_size_0, &
                                  rbf_coeff_1_size_1=rbf_coeff_1_size_1, &
                                  rbf_coeff_2=c_loc(rbf_coeff_2), &
                                  rbf_coeff_2_size_0=rbf_coeff_2_size_0, &
                                  rbf_coeff_2_size_1=rbf_coeff_2_size_1, &
                                  mask_hdiff=mask_hdiff_ptr, &
                                  mask_hdiff_size_0=mask_hdiff_size_0, &
                                  mask_hdiff_size_1=mask_hdiff_size_1, &
                                  zd_diffcoef=zd_diffcoef_ptr, &
                                  zd_diffcoef_size_0=zd_diffcoef_size_0, &
                                  zd_diffcoef_size_1=zd_diffcoef_size_1, &
                                  zd_vertoffset=zd_vertoffset_ptr, &
                                  zd_vertoffset_size_0=zd_vertoffset_size_0, &
                                  zd_vertoffset_size_1=zd_vertoffset_size_1, &
                                  zd_vertoffset_size_2=zd_vertoffset_size_2, &
                                  zd_intcoef=zd_intcoef_ptr, &
                                  zd_intcoef_size_0=zd_intcoef_size_0, &
                                  zd_intcoef_size_1=zd_intcoef_size_1, &
                                  zd_intcoef_size_2=zd_intcoef_size_2, &
                                  ndyn_substeps=ndyn_substeps, &
                                  rayleigh_damping_height=rayleigh_damping_height, &
                                  nflat_gradp=nflat_gradp, &
                                  diffusion_type=diffusion_type, &
                                  hdiff_w=hdiff_w, &
                                  hdiff_vn=hdiff_vn, &
                                  zdiffu_t=zdiffu_t, &
                                  type_t_diffu=type_t_diffu, &
                                  type_vn_diffu=type_vn_diffu, &
                                  hdiff_efdt_ratio=hdiff_efdt_ratio, &
                                  smagorinski_scaling_factor=smagorinski_scaling_factor, &
                                  hdiff_temp=hdiff_temp, &
                                  thslp_zdiffu=thslp_zdiffu, &
                                  thhgtd_zdiffu=thhgtd_zdiffu, &
                                  denom_diffu_v=denom_diffu_v, &
                                  nudge_max_coeff=nudge_max_coeff, &
                                  itype_sher=itype_sher, &
                                  ltkeshs=ltkeshs, &
                                  tangent_orientation=c_loc(tangent_orientation), &
                                  tangent_orientation_size_0=tangent_orientation_size_0, &
                                  inverse_primal_edge_lengths=c_loc(inverse_primal_edge_lengths), &
                                  inverse_primal_edge_lengths_size_0=inverse_primal_edge_lengths_size_0, &
                                  inv_dual_edge_length=c_loc(inv_dual_edge_length), &
                                  inv_dual_edge_length_size_0=inv_dual_edge_length_size_0, &
                                  inv_vert_vert_length=c_loc(inv_vert_vert_length), &
                                  inv_vert_vert_length_size_0=inv_vert_vert_length_size_0, &
                                  edge_areas=c_loc(edge_areas), &
                                  edge_areas_size_0=edge_areas_size_0, &
                                  f_e=c_loc(f_e), &
                                  f_e_size_0=f_e_size_0, &
                                  cell_center_lat=c_loc(cell_center_lat), &
                                  cell_center_lat_size_0=cell_center_lat_size_0, &
                                  cell_center_lon=c_loc(cell_center_lon), &
                                  cell_center_lon_size_0=cell_center_lon_size_0, &
                                  cell_areas=c_loc(cell_areas), &
                                  cell_areas_size_0=cell_areas_size_0, &
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
                                  edge_center_lat=c_loc(edge_center_lat), &
                                  edge_center_lat_size_0=edge_center_lat_size_0, &
                                  edge_center_lon=c_loc(edge_center_lon), &
                                  edge_center_lon_size_0=edge_center_lon_size_0, &
                                  primal_normal_x=c_loc(primal_normal_x), &
                                  primal_normal_x_size_0=primal_normal_x_size_0, &
                                  primal_normal_y=c_loc(primal_normal_y), &
                                  primal_normal_y_size_0=primal_normal_y_size_0, &
                                  global_root=global_root, &
                                  global_level=global_level, &
                                  lowest_layer_thickness=lowest_layer_thickness, &
                                  model_top_height=model_top_height, &
                                  stretch_factor=stretch_factor)
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
   end subroutine diffusion_init

   subroutine grid_init_diffusion(cell_starts, &
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
                                  c_owner_mask, &
                                  e_owner_mask, &
                                  v_owner_mask, &
                                  c_glb_index, &
                                  e_glb_index, &
                                  v_glb_index, &
                                  comm_id, &
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

      logical(c_int), dimension(:), target :: c_owner_mask

      logical(c_int), dimension(:), target :: e_owner_mask

      logical(c_int), dimension(:), target :: v_owner_mask

      integer(c_int), dimension(:), target :: c_glb_index

      integer(c_int), dimension(:), target :: e_glb_index

      integer(c_int), dimension(:), target :: v_glb_index

      integer(c_int), value, target :: comm_id

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

      integer(c_int) :: c_owner_mask_size_0

      integer(c_int) :: e_owner_mask_size_0

      integer(c_int) :: v_owner_mask_size_0

      integer(c_int) :: c_glb_index_size_0

      integer(c_int) :: e_glb_index_size_0

      integer(c_int) :: v_glb_index_size_0

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
      !$acc host_data use_device(c_owner_mask)
      !$acc host_data use_device(e_owner_mask)
      !$acc host_data use_device(v_owner_mask)
      !$acc host_data use_device(c_glb_index)
      !$acc host_data use_device(e_glb_index)
      !$acc host_data use_device(v_glb_index)

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

      c_owner_mask_size_0 = SIZE(c_owner_mask, 1)

      e_owner_mask_size_0 = SIZE(e_owner_mask, 1)

      v_owner_mask_size_0 = SIZE(v_owner_mask, 1)

      c_glb_index_size_0 = SIZE(c_glb_index, 1)

      e_glb_index_size_0 = SIZE(e_glb_index, 1)

      v_glb_index_size_0 = SIZE(v_glb_index, 1)

      rc = grid_init_diffusion_wrapper(cell_starts=c_loc(cell_starts), &
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
                                       c_owner_mask=c_loc(c_owner_mask), &
                                       c_owner_mask_size_0=c_owner_mask_size_0, &
                                       e_owner_mask=c_loc(e_owner_mask), &
                                       e_owner_mask_size_0=e_owner_mask_size_0, &
                                       v_owner_mask=c_loc(v_owner_mask), &
                                       v_owner_mask_size_0=v_owner_mask_size_0, &
                                       c_glb_index=c_loc(c_glb_index), &
                                       c_glb_index_size_0=c_glb_index_size_0, &
                                       e_glb_index=c_loc(e_glb_index), &
                                       e_glb_index_size_0=e_glb_index_size_0, &
                                       v_glb_index=c_loc(v_glb_index), &
                                       v_glb_index_size_0=v_glb_index_size_0, &
                                       comm_id=comm_id, &
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
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
   end subroutine grid_init_diffusion

end module