module grid
   use, intrinsic :: iso_c_binding
   implicit none

   public :: grid_init

   interface

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
                                 mean_cell_area, &
                                 comm_id, &
                                 num_vertices, &
                                 num_cells, &
                                 num_edges, &
                                 vertical_size, &
                                 limited_area, &
                                 backend, &
                                 on_gpu) bind(c, name="grid_init_wrapper") result(rc)
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

         real(c_double), value, target :: mean_cell_area

         integer(c_int), value, target :: comm_id

         integer(c_int), value, target :: num_vertices

         integer(c_int), value, target :: num_cells

         integer(c_int), value, target :: num_edges

         integer(c_int), value, target :: vertical_size

         logical(c_int), value, target :: limited_area

         integer(c_int), value, target :: backend

         logical(c_int), value :: on_gpu

      end function grid_init_wrapper

   end interface

contains

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
                        c_owner_mask, &
                        e_owner_mask, &
                        v_owner_mask, &
                        c_glb_index, &
                        e_glb_index, &
                        v_glb_index, &
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
                        mean_cell_area, &
                        comm_id, &
                        num_vertices, &
                        num_cells, &
                        num_edges, &
                        vertical_size, &
                        limited_area, &
                        backend, &
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

      real(c_double), value, target :: mean_cell_area

      integer(c_int), value, target :: comm_id

      integer(c_int), value, target :: num_vertices

      integer(c_int), value, target :: num_cells

      integer(c_int), value, target :: num_edges

      integer(c_int), value, target :: vertical_size

      logical(c_int), value, target :: limited_area

      integer(c_int), value, target :: backend

      logical(c_int) :: on_gpu

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

      !$acc host_data use_device(c2e)
      !$acc host_data use_device(e2c)
      !$acc host_data use_device(c2e2c)
      !$acc host_data use_device(e2c2e)
      !$acc host_data use_device(e2v)
      !$acc host_data use_device(v2e)
      !$acc host_data use_device(v2c)
      !$acc host_data use_device(e2c2v)
      !$acc host_data use_device(c2v)
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

#ifdef _OPENACC
      on_gpu = .True.
#else
      on_gpu = .False.
#endif

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
                             mean_cell_area=mean_cell_area, &
                             comm_id=comm_id, &
                             num_vertices=num_vertices, &
                             num_cells=num_cells, &
                             num_edges=num_edges, &
                             vertical_size=vertical_size, &
                             limited_area=limited_area, &
                             backend=backend, &
                             on_gpu=on_gpu)
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
   end subroutine grid_init

end module
