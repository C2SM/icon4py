module diffusion_plugin
   use, intrinsic :: iso_c_binding
   implicit none

   public :: diffusion_init

   public :: diffusion_run

   public :: grid_init_diffusion

   interface

      function diffusion_init_wrapper(vct_a, &
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
                                      n_C2E, &
                                      n_C2E2CO, &
                                      n_Cell, &
                                      n_E2C, &
                                      n_E2C2V, &
                                      n_Edge, &
                                      n_K, &
                                      n_KHalf, &
                                      n_V2E, &
                                      n_Vertex) bind(c, name="diffusion_init_wrapper") result(rc)
         import :: c_int, c_double, c_bool, c_ptr

         integer(c_int), value :: n_C2E

         integer(c_int), value :: n_C2E2CO

         integer(c_int), value :: n_Cell

         integer(c_int), value :: n_E2C

         integer(c_int), value :: n_E2C2V

         integer(c_int), value :: n_Edge

         integer(c_int), value :: n_K

         integer(c_int), value :: n_KHalf

         integer(c_int), value :: n_V2E

         integer(c_int), value :: n_Vertex

         integer(c_int) :: rc  ! Stores the return code

         real(c_double), dimension(*), target :: vct_a

         real(c_double), dimension(*), target :: vct_b

         real(c_double), dimension(*), target :: theta_ref_mc

         real(c_double), dimension(*), target :: wgtfac_c

         real(c_double), dimension(*), target :: e_bln_c_s

         real(c_double), dimension(*), target :: geofac_div

         real(c_double), dimension(*), target :: geofac_grg_x

         real(c_double), dimension(*), target :: geofac_grg_y

         real(c_double), dimension(*), target :: geofac_n2s

         real(c_double), dimension(*), target :: nudgecoeff_e

         real(c_double), dimension(*), target :: rbf_coeff_1

         real(c_double), dimension(*), target :: rbf_coeff_2

         type(c_ptr), value :: mask_hdiff

         type(c_ptr), value :: zd_diffcoef

         type(c_ptr), value :: zd_vertoffset

         type(c_ptr), value :: zd_intcoef

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

         real(c_double), dimension(*), target :: tangent_orientation

         real(c_double), dimension(*), target :: inverse_primal_edge_lengths

         real(c_double), dimension(*), target :: inv_dual_edge_length

         real(c_double), dimension(*), target :: inv_vert_vert_length

         real(c_double), dimension(*), target :: edge_areas

         real(c_double), dimension(*), target :: f_e

         real(c_double), dimension(*), target :: cell_center_lat

         real(c_double), dimension(*), target :: cell_center_lon

         real(c_double), dimension(*), target :: cell_areas

         real(c_double), dimension(*), target :: primal_normal_vert_x

         real(c_double), dimension(*), target :: primal_normal_vert_y

         real(c_double), dimension(*), target :: dual_normal_vert_x

         real(c_double), dimension(*), target :: dual_normal_vert_y

         real(c_double), dimension(*), target :: primal_normal_cell_x

         real(c_double), dimension(*), target :: primal_normal_cell_y

         real(c_double), dimension(*), target :: dual_normal_cell_x

         real(c_double), dimension(*), target :: dual_normal_cell_y

         real(c_double), dimension(*), target :: edge_center_lat

         real(c_double), dimension(*), target :: edge_center_lon

         real(c_double), dimension(*), target :: primal_normal_x

         real(c_double), dimension(*), target :: primal_normal_y

         integer(c_int), value, target :: global_root

         integer(c_int), value, target :: global_level

         real(c_double), value, target :: lowest_layer_thickness

         real(c_double), value, target :: model_top_height

         real(c_double), value, target :: stretch_factor

      end function diffusion_init_wrapper

      function diffusion_run_wrapper(w, &
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
                                     n_Cell, &
                                     n_Edge, &
                                     n_K, &
                                     n_KHalf) bind(c, name="diffusion_run_wrapper") result(rc)
         import :: c_int, c_double, c_bool, c_ptr

         integer(c_int), value :: n_Cell

         integer(c_int), value :: n_Edge

         integer(c_int), value :: n_K

         integer(c_int), value :: n_KHalf

         integer(c_int) :: rc  ! Stores the return code

         real(c_double), dimension(*), target :: w

         real(c_double), dimension(*), target :: vn

         real(c_double), dimension(*), target :: exner

         real(c_double), dimension(*), target :: theta_v

         real(c_double), dimension(*), target :: rho

         type(c_ptr), value :: hdef_ic

         type(c_ptr), value :: div_ic

         type(c_ptr), value :: dwdx

         type(c_ptr), value :: dwdy

         real(c_double), value, target :: dtime

         logical(c_int), value, target :: linit

      end function diffusion_run_wrapper

      function grid_init_diffusion_wrapper(cell_starts, &
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
                                           n_C2E, &
                                           n_C2E2C, &
                                           n_C2V, &
                                           n_Cell, &
                                           n_CellGlobalIndex, &
                                           n_CellIndex, &
                                           n_E2C, &
                                           n_E2C2E, &
                                           n_E2C2V, &
                                           n_E2V, &
                                           n_Edge, &
                                           n_EdgeGlobalIndex, &
                                           n_EdgeIndex, &
                                           n_V2C, &
                                           n_V2E, &
                                           n_Vertex, &
                                           n_VertexGlobalIndex, &
                                           n_VertexIndex) bind(c, name="grid_init_diffusion_wrapper") result(rc)
         import :: c_int, c_double, c_bool, c_ptr

         integer(c_int), value :: n_C2E

         integer(c_int), value :: n_C2E2C

         integer(c_int), value :: n_C2V

         integer(c_int), value :: n_Cell

         integer(c_int), value :: n_CellGlobalIndex

         integer(c_int), value :: n_CellIndex

         integer(c_int), value :: n_E2C

         integer(c_int), value :: n_E2C2E

         integer(c_int), value :: n_E2C2V

         integer(c_int), value :: n_E2V

         integer(c_int), value :: n_Edge

         integer(c_int), value :: n_EdgeGlobalIndex

         integer(c_int), value :: n_EdgeIndex

         integer(c_int), value :: n_V2C

         integer(c_int), value :: n_V2E

         integer(c_int), value :: n_Vertex

         integer(c_int), value :: n_VertexGlobalIndex

         integer(c_int), value :: n_VertexIndex

         integer(c_int) :: rc  ! Stores the return code

         integer(c_int), dimension(*), target :: cell_starts

         integer(c_int), dimension(*), target :: cell_ends

         integer(c_int), dimension(*), target :: vertex_starts

         integer(c_int), dimension(*), target :: vertex_ends

         integer(c_int), dimension(*), target :: edge_starts

         integer(c_int), dimension(*), target :: edge_ends

         integer(c_int), dimension(*), target :: c2e

         integer(c_int), dimension(*), target :: e2c

         integer(c_int), dimension(*), target :: c2e2c

         integer(c_int), dimension(*), target :: e2c2e

         integer(c_int), dimension(*), target :: e2v

         integer(c_int), dimension(*), target :: v2e

         integer(c_int), dimension(*), target :: v2c

         integer(c_int), dimension(*), target :: e2c2v

         integer(c_int), dimension(*), target :: c2v

         logical(c_int), dimension(*), target :: c_owner_mask

         logical(c_int), dimension(*), target :: e_owner_mask

         logical(c_int), dimension(*), target :: v_owner_mask

         integer(c_int), dimension(*), target :: c_glb_index

         integer(c_int), dimension(*), target :: e_glb_index

         integer(c_int), dimension(*), target :: v_glb_index

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
                             rc, &
                             mask_hdiff, &
                             zd_diffcoef, &
                             zd_vertoffset, &
                             zd_intcoef)
      use, intrinsic :: iso_c_binding

      integer(c_int) :: n_C2E

      integer(c_int) :: n_C2E2CO

      integer(c_int) :: n_Cell

      integer(c_int) :: n_E2C

      integer(c_int) :: n_E2C2V

      integer(c_int) :: n_Edge

      integer(c_int) :: n_K

      integer(c_int) :: n_KHalf

      integer(c_int) :: n_V2E

      integer(c_int) :: n_Vertex

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

      integer(c_int) :: rc  ! Stores the return code

      logical(c_int), dimension(:, :), target :: mask_hdiff

      real(c_double), dimension(:, :), target :: zd_diffcoef

      integer(c_int), dimension(:, :, :), target :: zd_vertoffset

      real(c_double), dimension(:, :, :), target :: zd_intcoef

      ! ptrs

      type(c_ptr) :: mask_hdiff_ptr

      type(c_ptr) :: zd_diffcoef_ptr

      type(c_ptr) :: zd_vertoffset_ptr

      type(c_ptr) :: zd_intcoef_ptr

      logical(c_int), dimension(:, :), pointer  :: mask_hdiff_ftn_ptr

      real(c_double), dimension(:, :), pointer  :: zd_diffcoef_ftn_ptr

      integer(c_int), dimension(:, :, :), pointer  :: zd_vertoffset_ftn_ptr

      real(c_double), dimension(:, :, :), pointer  :: zd_intcoef_ftn_ptr

      mask_hdiff_ftn_ptr => mask_hdiff

      zd_diffcoef_ftn_ptr => zd_diffcoef

      zd_vertoffset_ftn_ptr => zd_vertoffset

      zd_intcoef_ftn_ptr => zd_intcoef

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
      !$acc host_data use_device(mask_hdiff_ftn_ptr) if(associated(mask_hdiff_ftn_ptr))
      !$acc host_data use_device(zd_diffcoef_ftn_ptr) if(associated(zd_diffcoef_ftn_ptr))
      !$acc host_data use_device(zd_vertoffset_ftn_ptr) if(associated(zd_vertoffset_ftn_ptr))
      !$acc host_data use_device(zd_intcoef_ftn_ptr) if(associated(zd_intcoef_ftn_ptr))

      n_KHalf = SIZE(vct_a, 1)

      n_Cell = SIZE(theta_ref_mc, 1)

      n_K = SIZE(theta_ref_mc, 2)

      n_C2E = SIZE(e_bln_c_s, 2)

      n_C2E2CO = SIZE(geofac_grg_x, 2)

      n_Edge = SIZE(nudgecoeff_e, 1)

      n_Vertex = SIZE(rbf_coeff_1, 1)

      n_V2E = SIZE(rbf_coeff_1, 2)

      n_E2C2V = SIZE(primal_normal_vert_x, 2)

      n_E2C = SIZE(primal_normal_cell_x, 2)

      if (associated(mask_hdiff_ftn_ptr)) then
         mask_hdiff_ptr = c_loc(mask_hdiff_ftn_ptr)
      end if

      if (associated(zd_diffcoef_ftn_ptr)) then
         zd_diffcoef_ptr = c_loc(zd_diffcoef_ftn_ptr)
      end if

      if (associated(zd_vertoffset_ftn_ptr)) then
         zd_vertoffset_ptr = c_loc(zd_vertoffset_ftn_ptr)
      end if

      if (associated(zd_intcoef_ftn_ptr)) then
         zd_intcoef_ptr = c_loc(zd_intcoef_ftn_ptr)
      end if

      rc = diffusion_init_wrapper(vct_a=vct_a, &
                                  vct_b=vct_b, &
                                  theta_ref_mc=theta_ref_mc, &
                                  wgtfac_c=wgtfac_c, &
                                  e_bln_c_s=e_bln_c_s, &
                                  geofac_div=geofac_div, &
                                  geofac_grg_x=geofac_grg_x, &
                                  geofac_grg_y=geofac_grg_y, &
                                  geofac_n2s=geofac_n2s, &
                                  nudgecoeff_e=nudgecoeff_e, &
                                  rbf_coeff_1=rbf_coeff_1, &
                                  rbf_coeff_2=rbf_coeff_2, &
                                  mask_hdiff=mask_hdiff_ptr, &
                                  zd_diffcoef=zd_diffcoef_ptr, &
                                  zd_vertoffset=zd_vertoffset_ptr, &
                                  zd_intcoef=zd_intcoef_ptr, &
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
                                  tangent_orientation=tangent_orientation, &
                                  inverse_primal_edge_lengths=inverse_primal_edge_lengths, &
                                  inv_dual_edge_length=inv_dual_edge_length, &
                                  inv_vert_vert_length=inv_vert_vert_length, &
                                  edge_areas=edge_areas, &
                                  f_e=f_e, &
                                  cell_center_lat=cell_center_lat, &
                                  cell_center_lon=cell_center_lon, &
                                  cell_areas=cell_areas, &
                                  primal_normal_vert_x=primal_normal_vert_x, &
                                  primal_normal_vert_y=primal_normal_vert_y, &
                                  dual_normal_vert_x=dual_normal_vert_x, &
                                  dual_normal_vert_y=dual_normal_vert_y, &
                                  primal_normal_cell_x=primal_normal_cell_x, &
                                  primal_normal_cell_y=primal_normal_cell_y, &
                                  dual_normal_cell_x=dual_normal_cell_x, &
                                  dual_normal_cell_y=dual_normal_cell_y, &
                                  edge_center_lat=edge_center_lat, &
                                  edge_center_lon=edge_center_lon, &
                                  primal_normal_x=primal_normal_x, &
                                  primal_normal_y=primal_normal_y, &
                                  global_root=global_root, &
                                  global_level=global_level, &
                                  lowest_layer_thickness=lowest_layer_thickness, &
                                  model_top_height=model_top_height, &
                                  stretch_factor=stretch_factor, &
                                  n_C2E=n_C2E, &
                                  n_C2E2CO=n_C2E2CO, &
                                  n_Cell=n_Cell, &
                                  n_E2C=n_E2C, &
                                  n_E2C2V=n_E2C2V, &
                                  n_Edge=n_Edge, &
                                  n_K=n_K, &
                                  n_KHalf=n_KHalf, &
                                  n_V2E=n_V2E, &
                                  n_Vertex=n_Vertex)
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

   subroutine diffusion_run(w, &
                            vn, &
                            exner, &
                            theta_v, &
                            rho, &
                            dtime, &
                            linit, &
                            rc, &
                            hdef_ic, &
                            div_ic, &
                            dwdx, &
                            dwdy)
      use, intrinsic :: iso_c_binding

      integer(c_int) :: n_Cell

      integer(c_int) :: n_Edge

      integer(c_int) :: n_K

      integer(c_int) :: n_KHalf

      real(c_double), dimension(:, :), target :: w

      real(c_double), dimension(:, :), target :: vn

      real(c_double), dimension(:, :), target :: exner

      real(c_double), dimension(:, :), target :: theta_v

      real(c_double), dimension(:, :), target :: rho

      real(c_double), value, target :: dtime

      logical(c_int), value, target :: linit

      integer(c_int) :: rc  ! Stores the return code

      real(c_double), dimension(:, :), target :: hdef_ic

      real(c_double), dimension(:, :), target :: div_ic

      real(c_double), dimension(:, :), target :: dwdx

      real(c_double), dimension(:, :), target :: dwdy

      ! ptrs

      type(c_ptr) :: hdef_ic_ptr

      type(c_ptr) :: div_ic_ptr

      type(c_ptr) :: dwdx_ptr

      type(c_ptr) :: dwdy_ptr

      real(c_double), dimension(:, :), pointer  :: hdef_ic_ftn_ptr

      real(c_double), dimension(:, :), pointer  :: div_ic_ftn_ptr

      real(c_double), dimension(:, :), pointer  :: dwdx_ftn_ptr

      real(c_double), dimension(:, :), pointer  :: dwdy_ftn_ptr

      hdef_ic_ftn_ptr => hdef_ic

      div_ic_ftn_ptr => div_ic

      dwdx_ftn_ptr => dwdx

      dwdy_ftn_ptr => dwdy

      hdef_ic_ptr = c_null_ptr

      div_ic_ptr = c_null_ptr

      dwdx_ptr = c_null_ptr

      dwdy_ptr = c_null_ptr

      !$acc host_data use_device(w)
      !$acc host_data use_device(vn)
      !$acc host_data use_device(exner)
      !$acc host_data use_device(theta_v)
      !$acc host_data use_device(rho)
      !$acc host_data use_device(hdef_ic_ftn_ptr) if(associated(hdef_ic_ftn_ptr))
      !$acc host_data use_device(div_ic_ftn_ptr) if(associated(div_ic_ftn_ptr))
      !$acc host_data use_device(dwdx_ftn_ptr) if(associated(dwdx_ftn_ptr))
      !$acc host_data use_device(dwdy_ftn_ptr) if(associated(dwdy_ftn_ptr))

      n_Cell = SIZE(w, 1)

      n_KHalf = SIZE(w, 2)

      n_Edge = SIZE(vn, 1)

      n_K = SIZE(vn, 2)

      if (associated(hdef_ic_ftn_ptr)) then
         hdef_ic_ptr = c_loc(hdef_ic_ftn_ptr)
      end if

      if (associated(div_ic_ftn_ptr)) then
         div_ic_ptr = c_loc(div_ic_ftn_ptr)
      end if

      if (associated(dwdx_ftn_ptr)) then
         dwdx_ptr = c_loc(dwdx_ftn_ptr)
      end if

      if (associated(dwdy_ftn_ptr)) then
         dwdy_ptr = c_loc(dwdy_ftn_ptr)
      end if

      rc = diffusion_run_wrapper(w=w, &
                                 vn=vn, &
                                 exner=exner, &
                                 theta_v=theta_v, &
                                 rho=rho, &
                                 hdef_ic=hdef_ic_ptr, &
                                 div_ic=div_ic_ptr, &
                                 dwdx=dwdx_ptr, &
                                 dwdy=dwdy_ptr, &
                                 dtime=dtime, &
                                 linit=linit, &
                                 n_Cell=n_Cell, &
                                 n_Edge=n_Edge, &
                                 n_K=n_K, &
                                 n_KHalf=n_KHalf)
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

      integer(c_int) :: n_C2E

      integer(c_int) :: n_C2E2C

      integer(c_int) :: n_C2V

      integer(c_int) :: n_Cell

      integer(c_int) :: n_CellGlobalIndex

      integer(c_int) :: n_CellIndex

      integer(c_int) :: n_E2C

      integer(c_int) :: n_E2C2E

      integer(c_int) :: n_E2C2V

      integer(c_int) :: n_E2V

      integer(c_int) :: n_Edge

      integer(c_int) :: n_EdgeGlobalIndex

      integer(c_int) :: n_EdgeIndex

      integer(c_int) :: n_V2C

      integer(c_int) :: n_V2E

      integer(c_int) :: n_Vertex

      integer(c_int) :: n_VertexGlobalIndex

      integer(c_int) :: n_VertexIndex

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

      n_CellIndex = SIZE(cell_starts, 1)

      n_VertexIndex = SIZE(vertex_starts, 1)

      n_EdgeIndex = SIZE(edge_starts, 1)

      n_Cell = SIZE(c2e, 1)

      n_C2E = SIZE(c2e, 2)

      n_Edge = SIZE(e2c, 1)

      n_E2C = SIZE(e2c, 2)

      n_C2E2C = SIZE(c2e2c, 2)

      n_E2C2E = SIZE(e2c2e, 2)

      n_E2V = SIZE(e2v, 2)

      n_Vertex = SIZE(v2e, 1)

      n_V2E = SIZE(v2e, 2)

      n_V2C = SIZE(v2c, 2)

      n_E2C2V = SIZE(e2c2v, 2)

      n_C2V = SIZE(c2v, 2)

      n_CellGlobalIndex = SIZE(c_glb_index, 1)

      n_EdgeGlobalIndex = SIZE(e_glb_index, 1)

      n_VertexGlobalIndex = SIZE(v_glb_index, 1)

      rc = grid_init_diffusion_wrapper(cell_starts=cell_starts, &
                                       cell_ends=cell_ends, &
                                       vertex_starts=vertex_starts, &
                                       vertex_ends=vertex_ends, &
                                       edge_starts=edge_starts, &
                                       edge_ends=edge_ends, &
                                       c2e=c2e, &
                                       e2c=e2c, &
                                       c2e2c=c2e2c, &
                                       e2c2e=e2c2e, &
                                       e2v=e2v, &
                                       v2e=v2e, &
                                       v2c=v2c, &
                                       e2c2v=e2c2v, &
                                       c2v=c2v, &
                                       c_owner_mask=c_owner_mask, &
                                       e_owner_mask=e_owner_mask, &
                                       v_owner_mask=v_owner_mask, &
                                       c_glb_index=c_glb_index, &
                                       e_glb_index=e_glb_index, &
                                       v_glb_index=v_glb_index, &
                                       comm_id=comm_id, &
                                       global_root=global_root, &
                                       global_level=global_level, &
                                       num_vertices=num_vertices, &
                                       num_cells=num_cells, &
                                       num_edges=num_edges, &
                                       vertical_size=vertical_size, &
                                       limited_area=limited_area, &
                                       n_C2E=n_C2E, &
                                       n_C2E2C=n_C2E2C, &
                                       n_C2V=n_C2V, &
                                       n_Cell=n_Cell, &
                                       n_CellGlobalIndex=n_CellGlobalIndex, &
                                       n_CellIndex=n_CellIndex, &
                                       n_E2C=n_E2C, &
                                       n_E2C2E=n_E2C2E, &
                                       n_E2C2V=n_E2C2V, &
                                       n_E2V=n_E2V, &
                                       n_Edge=n_Edge, &
                                       n_EdgeGlobalIndex=n_EdgeGlobalIndex, &
                                       n_EdgeIndex=n_EdgeIndex, &
                                       n_V2C=n_V2C, &
                                       n_V2E=n_V2E, &
                                       n_Vertex=n_Vertex, &
                                       n_VertexGlobalIndex=n_VertexGlobalIndex, &
                                       n_VertexIndex=n_VertexIndex)
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