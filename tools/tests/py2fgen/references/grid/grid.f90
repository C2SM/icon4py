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
                                 comm_id, &
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
   end subroutine grid_init

end module