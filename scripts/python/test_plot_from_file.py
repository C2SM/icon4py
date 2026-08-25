import matplotlib

matplotlib.use("Agg")

import sys

sys.path.insert(0, "/capstor/scratch/cscs/cong/icon4py/scripts/python")

from plot_utils import plot_torus_plane_from_file, read_connectivity_and_positions_from_grid_file, plot_cloud_evolution_in_wk82exp_from_file, plot_cross_section_in_wk82exp_from_file, plot_kinetic_energy_in_wk82exp_from_file, plot_cloud_snapshots_in_wk82exp_from_file


GRID = "/capstor/scratch/cscs/cong/icon4py/warm_bubble_runs/data/dz200m/output/Torus_Triangles_100km_x_100km_res500m_ugrid.nc"
DATA = "/capstor/scratch/cscs/cong/icon4py/warm_bubble_runs/data/dz200m/output/icon4py_output_0001.nc"
OUT = "/capstor/scratch/cscs/cong/icon4py/warm_bubble_runs/data/dz200m/plots"
OUT2 = "/capstor/scratch/cscs/cong/icon4py/warm_bubble_runs/data/dz200m/plots_for_ec"
# GRID = "/capstor/scratch/cscs/cong/icon4py/tests/output/Torus_Triangles_100km_x_100km_res500m_ugrid.nc"
# DATA = "/capstor/scratch/cscs/cong/icon4py/tests/output/icon4py_output_0001.nc"
# OUT = "/capstor/scratch/cscs/cong/icon4py/tests/plots"


tri, c2v, valid_cells, vertex, edge, cell, mean_edge_length = read_connectivity_and_positions_from_grid_file(grid_file=GRID, length_max=None)

plot_cloud_evolution_in_wk82exp_from_file(
    tri=tri,
    mean_edge_length=mean_edge_length,
    c2v_connectivity=c2v,
    valid_cells=valid_cells,
    cell=cell,
    vertex=vertex,
    data_file=DATA,
    level=80,
    time_step=(14, 29, 79, 119),
    out_file=f"{OUT2}/cloud",
)

# plot_cloud_snapshots_in_wk82exp_from_file(
#     tri=tri,
#     mean_edge_length=mean_edge_length,
#     c2v_connectivity=c2v,
#     valid_cells=valid_cells,
#     cell=cell,
#     vertex=vertex,
#     data_file=DATA,
#     level=99,
#     time_step=(29, 49, 89),
#     out_file=f"{OUT2}/cloud",
# )

# plot_cross_section_in_wk82exp_from_file(
#     data_file=DATA,
#     mean_edge_length=mean_edge_length,
#     cell=cell,
#     time_step=(14, 29, 79, 119),
#     out_file=f"{OUT2}/section",
# )

# plot_kinetic_energy_in_wk82exp_from_file(
#     mean_edge_length=mean_edge_length,
#     data_file=DATA,
#     out_file=f"{OUT2}/time_series",
# )



# cell-located field on a full level
# levels = (49, 40, 30, 20); time_steps = (0, 20, 40, 60, 80, 100, 119)
# for variable, out_file in zip(
#     (
#         "virtual_potential_temperature",
#         "qv",
#         "qc",
#         "qr",
#         "qi",
#         "qs",
#         "qg",
#         "eastward_wind",
#         "northward_wind",
#         "upward_air_velocity",
#     ),
#     (
#         f"{OUT}/theta_v",
#         f"{OUT}/qv",
#         f"{OUT}/qc",
#         f"{OUT}/qr",
#         f"{OUT}/qi",
#         f"{OUT}/qs",
#         f"{OUT}/qg",
#         f"{OUT}/u",
#         f"{OUT}/v",
#         f"{OUT}/w"),
# ):
#     plot_torus_plane_from_file(
#         tri=tri,
#         valid_cells=valid_cells,
#         edge=edge,
#         data_file=DATA,
#         variable=variable,
#         level=levels,
#         time_step=time_steps,
#         out_file=out_file,
#     )






# cell-located field on a half level
# plot_torus_plane_from_file(
#     data_file=DATA,
#     grid_file=GRID,
#     variable="upward_air_velocity",
#     level=49,
#     time_step=5,
#     out_file=f"{OUT}/w_hlevel49_t5.png",
# )
# print("half-level field OK")

# # edge-located field
# plot_torus_plane_from_file(
#     data_file=DATA,
#     grid_file=GRID,
#     variable="normal_velocity",
#     level=45,
#     time_step=0,
#     out_file=f"{OUT}/vn_level45_t0.png",
# )
# print("edge field OK")
