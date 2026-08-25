# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors
from scipy.interpolate import griddata


def create_mpl_triangulation(
    *,
    c2v_connectivity: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    length_max: float,
) -> mpl.tri.Triangulation:
    """
    Create a matplotlib triangulation object for torus grids.

    Args:
        c2v_connectivity: array that contains the cell-to-vertex connectivity
        node_x: array that contains the vertex x-coordinates
        node_y: array that contains the vertex y-coordinates
        length_max: maximum edge length to plot

    """
    # create a matplotlib triangulation from the grid connectivity
    tri = mpl.tri.Triangulation(node_x, node_y, triangles=c2v_connectivity)

    # remove triangles with edge length smaller greater than some max length
    # note: this is necessary to avoid plotting artifacts due to the periodicity of torus grids
    triangles = tri.triangles
    node_x_diff = node_x[triangles] - np.roll(node_x[triangles], 1, axis=1)
    node_y_diff = node_y[triangles] - np.roll(node_y[triangles], 1, axis=1)
    node_dist_max = np.max(np.sqrt(node_x_diff**2 + node_y_diff**2), axis=1)
    tri.set_mask(node_dist_max > length_max)

    return tri


def finalize_plot(
    *,
    fig: mpl.figure.Figure,
    out_file: str = "",
) -> None:
    """
    Save or show the current figure and close it afterwards.

    Args:
        fig: matplotlib figure
        out_file: passed to savefig if present, else plot is shown instead

    """
    if out_file != "":
        fig.savefig(out_file, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_mpl_triangulation(
    *,
    tri: mpl.tri.Triangulation,
    values: np.ndarray,
    cmap: str = "viridis",
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Plot values on a matplotlib triangulation.

    Args:
        tri: matplotlib triangulation
        values: array that contains the values on the triangulation

    """
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)

    tpc = ax.tripcolor(tri, values, edgecolor="none", shading="flat", cmap=cmap)
    # tpc = ax.tricontourf(tri, values, cmap="viridis")
    cbar = fig.colorbar(tpc, ax=ax)
    # cbar.formatter.set_powerlimits((0, 0))  # type: ignore[attr-defined]
    # cbar.formatter.set_useMathText(True)  # type: ignore[attr-defined]
    # ax.triplot(triang, color="k", lw=0.3, alpha=0.4)

    ax.grid("both")  # type: ignore[arg-type]
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    return fig, ax


def plot_mpl_scatter(
    *,
    node_x: np.ndarray,
    node_y: np.ndarray,
    values: np.ndarray,
    mesh_tri: mpl.tri.Triangulation | None = None,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Scatter plot of values at given point locations, e.g. edge midpoints on a torus grid.

    Args:
        node_x: array that contains the point x-coordinates
        node_y: array that contains the point y-coordinates
        values: array that contains the values at the point locations
        mesh_tri: optional matplotlib triangulation whose edges are overlaid on the scatter plot

    """
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)

    if mesh_tri is not None:
        ax.triplot(mesh_tri, color="black", linewidth=0.5, zorder=1.9)

    sc = ax.scatter(node_x, node_y, c=values, cmap="viridis", zorder=2.0)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.formatter.set_powerlimits((0, 0))  # type: ignore[attr-defined]
    cbar.formatter.set_useMathText(True)  # type: ignore[attr-defined]

    ax.grid("both")  # type: ignore[arg-type]
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    return fig, ax


def plot_torus_scatter(
    *,
    node_x: np.ndarray,
    node_y: np.ndarray,
    values: np.ndarray,
    c2v_connectivity: np.ndarray | None = None,
    vertex_x: np.ndarray | None = None,
    vertex_y: np.ndarray | None = None,
    length_max: float | None = None,
    out_file: str = "",
) -> None:
    """
    Scatter plot of values at given point locations for torus grids, e.g. edge-located fields.

    Args:
        node_x: array that contains the point x-coordinates
        node_y: array that contains the point y-coordinates
        values: array that contains the values at the point locations
        c2v_connectivity: optional cell-to-vertex connectivity used to overlay the mesh triangle
            edges; if given, 'vertex_x', 'vertex_y' and 'length_max' must also be given
        vertex_x: array that contains the vertex x-coordinates, required if 'c2v_connectivity' is given
        vertex_y: array that contains the vertex y-coordinates, required if 'c2v_connectivity' is given
        length_max: maximum edge length to plot, required if 'c2v_connectivity' is given
        out_file: passed to savefig if present, else plot is shown instead

    """
    mesh_tri = None
    if c2v_connectivity is not None:
        assert vertex_x is not None and vertex_y is not None and length_max is not None, (
            "vertex_x, vertex_y and length_max must be given if c2v_connectivity is given"
        )
        mesh_tri = create_mpl_triangulation(
            c2v_connectivity=c2v_connectivity,
            node_x=vertex_x,
            node_y=vertex_y,
            length_max=length_max,
        )
    fig, _ = plot_mpl_scatter(
        node_x=node_x,
        node_y=node_y,
        values=values,
        mesh_tri=mesh_tri,
    )
    finalize_plot(fig=fig, out_file=out_file)


def plot_torus_plane(
    *,
    c2v_connectivity: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    values: np.ndarray,
    length_max: float,
    out_file: str = "",
) -> None:
    """
    Plot a single horizontal plane for torus grids.

    Args:
        c2v_connectivity: array that contains the cell-to-vertex connectivity
        node_x: array that contains the vertex x-coordinates
        node_y: array that contains the vertex y-coordinates
        values: array that contains the horizontal values on a single level to plot
        length_max: maximum edge length to plot
        out_file: passed to savefig if present, else plot is shown instead

    """
    tri = create_mpl_triangulation(
        c2v_connectivity=c2v_connectivity,
        node_x=node_x,
        node_y=node_y,
        length_max=length_max,
    )
    fig, _ = plot_mpl_triangulation(
        tri=tri,
        values=values,
    )
    finalize_plot(fig=fig, out_file=out_file)


def _read_c2v_connectivity(grid: xr.Dataset) -> np.ndarray:
    """
    Read the cell-to-vertex connectivity from an ICON grid file.

    Handles both raw ICON grid files (transposed storage, 1-based indices) and
    UGRID-patched grid files ((cell, nv) storage, 0-based indices).
    """
    c2v = np.asarray(grid["vertex_of_cell"].values, dtype=np.int64)
    if c2v.shape[0] == 3 and c2v.shape[1] != 3:
        # raw ICON grid files store the connectivity transposed
        c2v = c2v.T
    if c2v.min() == 1:
        # raw ICON grid files use 1-based indices
        c2v = c2v - 1
    return c2v


def _read_horizontal_positions(grid: xr.Dataset, location: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the horizontal positions of cells, edges or vertices from an ICON grid file.

    Prefers the planar cartesian coordinates in meters, falls back to longitude/latitude
    in radians if they are not present.
    """
    cartesian_names = {
        "cell": ("cell_circumcenter_cartesian_x", "cell_circumcenter_cartesian_y"),
        "edge": ("edge_middle_cartesian_x", "edge_middle_cartesian_y"),
        "vertex": ("cartesian_x_vertices", "cartesian_y_vertices"),
    }
    lonlat_names = {
        "cell": ("clon", "clat"),
        "edge": ("elon", "elat"),
        "vertex": ("vlon", "vlat"),
    }
    names = cartesian_names[location]
    if names[0] not in grid:
        names = lonlat_names[location]
    return grid[names[0]].values, grid[names[1]].values


def _infer_length_max(grid: xr.Dataset, vertex_x: np.ndarray, vertex_y: np.ndarray) -> float:
    """
    Infer the maximum edge length to plot from the grid.

    Uses 1.5 times the longest grid edge if the edge lengths are available, otherwise
    a tenth of the smaller domain extent.
    """
    if "mean_edge_length" in grid.attrs:
        return 1.5 * float(grid.attrs["mean_edge_length"])
    elif "edge_length" in grid:
        return 1.5 * float(grid["edge_length"].max())
    # extrace global attribute mean_edge_length if present
    return 0.1 * float(min(vertex_x.max() - vertex_x.min(), vertex_y.max() - vertex_y.min()))


def read_connectivity_and_positions_from_grid_file(
    *,
    grid_file: str,
    length_max: float | None = None,
) -> tuple[mpl.tri.Triangulation, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]], float]:
    """
    Read the cell-to-vertex connectivity and horizontal positions from an ICON grid file.
    """
    with xr.open_dataset(grid_file) as grid:
        c2v_connectivity = _read_c2v_connectivity(grid)
        # drop cells with invalid (missing) vertices
        valid_cells = np.all(c2v_connectivity >= 0, axis=1)
        c2v_connectivity = c2v_connectivity[valid_cells]
        cell_x, cell_y = _read_horizontal_positions(grid, "cell")
        vertex_x, vertex_y = _read_horizontal_positions(grid, "vertex")
        edge_x, edge_y = _read_horizontal_positions(grid, "edge")
        if length_max is None:
            length_max = _infer_length_max(grid, vertex_x, vertex_y)
    
    tri = create_mpl_triangulation(
        c2v_connectivity=c2v_connectivity,
        node_x=vertex_x,
        node_y=vertex_y,
        length_max=length_max,
    )

    if "mean_edge_length" in grid.attrs:
        mean_edge_length = float(grid.attrs["mean_edge_length"])
    else:
        raise ValueError("Grid file does not contain mean_edge_length attribute.")

    return tri, c2v_connectivity, valid_cells, (vertex_x, vertex_y), (edge_x, edge_y), (cell_x, cell_y), mean_edge_length

def plot_torus_plane_from_file(
    *,
    tri: mpl.tri.Triangulation,
    valid_cells: np.ndarray,
    edge: tuple[np.ndarray, np.ndarray],
    data_file: str,
    variable: str,
    level: int | list[int] = 0,
    time_step: int | list[int] = 0,
    out_file: str = "",
) -> None:
    """
    Plot a single horizontal plane of a variable from an ICON4Py netcdf output file.

    The field at the given time step and vertical level is read from 'data_file' and the
    mesh information is read from 'grid_file'. Cell- and vertex-located fields are drawn
    as colored triangles, edge-located fields as a scatter at the edge midpoints with the
    mesh edges overlaid.

    Args:
        data_file: path to the netcdf output file containing the data
        grid_file: path to the netcdf grid file
        variable: name of the variable to plot
        level: vertical level index to plot, ignored for variables without a vertical dimension
        time_step: time index to plot, ignored for variables without a time dimension
        length_max: maximum edge length to plot, inferred from the grid if not given
        out_file: passed to savefig if present, else plot is shown instead

    """
    time_step = (time_step,) if isinstance(time_step, int) else time_step
    level = (level,) if isinstance(level, int) else level

    with xr.open_dataset(data_file) as data:
        if variable not in data:
            raise ValueError(f"Variable '{variable}' not found in data file '{data_file}'.")
        field = data[variable]
        title = variable
        for time in time_step:
            for lev in level:
                if "time" in field.dims:
                    local_field = field.isel(time=time)
                for vertical_dim in ("level", "half_level"):
                    if vertical_dim in field.dims:
                        local_field = local_field.isel({vertical_dim: lev})
                values = local_field.values

                location = next(
                    (dim for dim in ("cell", "edge", "vertex") if dim in field.dims), None
                )
                if location is None:
                    raise ValueError(f"Variable '{variable}' is not located on the horizontal mesh.")

                title = f"{variable}, time step {time}, level {lev}"
                if location == "edge":
                    fig, ax = plot_mpl_scatter(
                        node_x=edge[0],
                        node_y=edge[1],
                        values=values,
                        mesh_tri=tri,
                    )
                else:
                    if location == "cell":
                        values = values[valid_cells]
                    fig, ax = plot_mpl_triangulation(
                        tri=tri,
                        values=values,
                    )
                ax.set_title(title)
                finalize_plot(fig=fig, out_file=f"{out_file}_t{time}_l{lev}.png")


def plot_cloud_evolution_in_wk82exp_from_file(
    *,
    tri: mpl.tri.Triangulation,
    mean_edge_length: float,
    c2v_connectivity: np.ndarray,
    valid_cells: np.ndarray,
    cell: tuple[np.ndarray, np.ndarray],
    vertex: tuple[np.ndarray, np.ndarray],
    data_file: str,
    level: int = 0,
    time_step: int | list[int] = 0,
    out_file: str = "",
) -> None:
    time_step = (time_step,) if isinstance(time_step, int) else time_step

    max_wind_speed = 15.0  # m/s
    h_min = 0.0  # m
    wind_scale_height = 3000.0  # m
    # hardcoded time interval
    initial_time_interval, time_inteval = 59.0, 60.0 # seconds
    with xr.open_dataset(data_file) as data:
        half_lvl_height = data["height"].values
        height = 0.5 * (half_lvl_height[1:] + half_lvl_height[:-1])
        thickness = -np.mean(np.diff(half_lvl_height))
        wind_speed = max_wind_speed * (
            np.tanh((height - h_min) / (wind_scale_height - h_min)) - 0.45
        )
        liquid_water_content = (data["qc"] + data["qr"]) * data["air_density"] * 1000.0  # g/kg
        ice_water_content = (data["qi"] + data["qs"] + data["qg"]) * data["air_density"] * 1000.0  # g/kg
        w = data["upward_air_velocity"]
        if "time" in liquid_water_content.dims and "time" in w.dims:
            lwc_local_field = liquid_water_content.isel(time=list(time_step))
            iwc_local_field = ice_water_content.isel(time=list(time_step))
            w_local_field = w.isel(time=list(time_step))
        else:
            raise ValueError("Variable 'liquid_water_content' or 'w' is not time-dependent.")
        if "level" in liquid_water_content.dims and "level" in ice_water_content.dims:
            lwc_local_field = lwc_local_field.isel(level=level)
            iwc_local_field = iwc_local_field.isel(level=level)
        else:
            raise ValueError("Variable 'liquid_water_content' is not vertically dependent.")
        liquid_water_content_plot = lwc_local_field.values
        ice_water_content_plot = iwc_local_field.values
        w_plot = w_local_field.values
        w_plot = 0.5 * (w_plot[:,:-1,:] + w_plot[:,1:,:])  # average to cell centers
        w_plot = w_plot[:, level, :]

        assert liquid_water_content_plot.ndim == 2, "Variable 'liquid_water_content' must be 2D (time, cell)."
        assert ice_water_content_plot.ndim == 2, "Variable 'ice_water_content' must be 2D (time, cell)."
        assert w_plot.ndim == 2, "Variable 'w' must be 2D (time, cell)."
        
        combined_vertex_x = np.zeros((len(time_step), vertex[0].shape[0]), dtype=vertex[0].dtype)
        combined_vertex_y = np.zeros((len(time_step), vertex[1].shape[0]), dtype=vertex[1].dtype)
        total_water_content = liquid_water_content_plot + ice_water_content_plot
        ice_water_content_plot[liquid_water_content_plot < 0.01] = np.nan
        w_plot[liquid_water_content_plot < 0.01] = np.nan
        liquid_water_content_plot[liquid_water_content_plot < 0.01] = np.nan
        extra_speed = (1.0, 1.5, 1.5, 2.3)
        for i, time in enumerate(time_step):
            ref_lev = np.argmin(np.abs(height - 3800.0))
            distance_traveled = int((initial_time_interval + time * time_inteval) * wind_speed[ref_lev] * extra_speed[i] / mean_edge_length) * mean_edge_length
            combined_vertex_x[i] = (vertex[0] + distance_traveled) / 1000.0  # km
            combined_vertex_y[i] = vertex[1] / 1000.0  # km
        tri = [
            create_mpl_triangulation(
                c2v_connectivity=c2v_connectivity,
                node_x=combined_vertex_x[i],
                node_y=combined_vertex_y[i],
                length_max=1.5 * mean_edge_length / 1000.0,  # km
            ) for i in range(len(time_step)) 
        ]

        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 13,
                "axes.labelsize": 15,
                "axes.titlesize": 15,
                "axes.linewidth": 1.5,
                "xtick.direction": "out",
                "ytick.direction": "out",
                "xtick.top": False,
                "ytick.right": False,
                "savefig.dpi": 400,
                "savefig.bbox": "tight",
                "axes.grid": False,
                # spine alpha is set to 0.5 to make the grid lines more visible
            }
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.spines['bottom'].set_alpha(0.3)
        ax.spines['left'].set_alpha(0.3)
        ax.spines['top'].set_alpha(0.3)
        ax.spines['right'].set_alpha(0.3)
        ax.set_axisbelow(True)

        # colorbar limits
        lwc_min, lwc_max = liquid_water_content_plot[~np.isnan(liquid_water_content_plot)].min(), liquid_water_content_plot[~np.isnan(liquid_water_content_plot)].max()
        hotmap = plt.get_cmap("hot").reversed()
        # extract only half of the colormap to avoid too bright colors
        hotmap = mpl.colors.LinearSegmentedColormap.from_list(
            "hot_half", hotmap(np.linspace(0.3, 1.0, 128))
        )
        for i in range(len(time_step)):
            values = liquid_water_content_plot[i][valid_cells]
            # vertex_values = np.zeros(combined_vertex_x[i].shape[0])
            # vertex_counts = np.zeros(combined_vertex_x[i].shape[0])
            # for cell_idx, verts in enumerate(c2v_connectivity):
            #     vertex_values[verts] += values[cell_idx]
            #     vertex_counts[verts] += 1
            # vertex_counts = np.maximum(vertex_counts, 1)
            # vertex_values = vertex_values / vertex_counts
            # vertex_values[vertex_values < 0.1] = np.nan
            # vertex_values = np.ma.masked_invalid(vertex_values)
            # vertex_values = vertex_values.filled(fill_value=-999)
            # tpc = ax.tricontourf(tri[i], vertex_values, cmap=hotmap, vmin=lwc_min, vmax=lwc_max, levels=100)
            tpc = ax.tripcolor(tri[i], values, edgecolor="none", shading="flat", cmap=hotmap, vmin=lwc_min, vmax=lwc_max)
            # line contour plot of liquid water content
            # ax.contour(tri[i], values, colors="black", linewidths=0.1, levels=np.linspace(lwc_min, lwc_max, 5))
        
        
        cbar = fig.colorbar(tpc, ax=ax)
        # cbar.formatter.set_powerlimits((0, 0))  # type: ignore[attr-defined]
        # cbar.formatter.set_useMathText(True)  # type: ignore[attr-defined]
        # ax.triplot(triang, color="k", lw=0.3, alpha=0.4)
        ax.grid("both")  # type: ignore[arg-type]
        ax.set_xlim(40.0, 200.0)
        ax.set_ylim(20.0, 80.0)
        # ax.set_xlabel("$x$ (km)")
        # ax.set_ylabel("$y$ (km)")
        # set clabels for the colorbar
        cbar.set_label("Liquid water content (g m$^{-3}$)", rotation=270, labelpad=20)
        # set the alpha of bounding box of the colorbar to 0.3
        # cbar.ax.patch.set_alpha(0.3)
        # title
        ax.set_title(f"Liquid water content evolution at z = {(height.shape[0] - level) * thickness / 1000.0:.1f} km")
        # turn off the axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        # write the time steps on the plot
        print_out_test = ["15 min", "30 min", "80 min", "120 min"]
        for i, time in enumerate(time_step):
            ref_lev = np.argmin(np.abs(height - 3800.0))
            distance_traveled = int((initial_time_interval + time * time_inteval) * wind_speed[ref_lev] * extra_speed[i] / mean_edge_length) * mean_edge_length
            x_pos = 0.5 * (combined_vertex_x[i].min() + combined_vertex_x[i].max()) if i != 1 else 0.5 * (combined_vertex_x[i].min() + combined_vertex_x[i].max()) + 3.0
            ax.text(
                x_pos,
                25.0,
                print_out_test[i],
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
            )
        finalize_plot(fig=fig, out_file=f"{out_file}_rain_evolution_l{level}.png")


        fig, ax = plt.subplots(figsize=(10, 6))
        ax.spines['bottom'].set_alpha(0.3)
        ax.spines['left'].set_alpha(0.3)
        ax.spines['top'].set_alpha(0.3)
        ax.spines['right'].set_alpha(0.3)
        ax.set_axisbelow(True)

        # colorbar limits
        iwc_min, iwc_max = ice_water_content_plot[~np.isnan(ice_water_content_plot)].min(), ice_water_content_plot[~np.isnan(ice_water_content_plot)].max()
        wintermap = plt.get_cmap("winter").reversed()
        # extract only half of the colormap to avoid too bright colors
        wintermap = mpl.colors.LinearSegmentedColormap.from_list(
            "winter_half", wintermap(np.linspace(0.3, 1.0, 128))
        )
        for i in range(len(time_step)):
            values = ice_water_content_plot[i][valid_cells]
            tpc = ax.tripcolor(tri[i], values, edgecolor="none", shading="flat", cmap=wintermap, vmin=iwc_min, vmax=iwc_max)
        
        cbar = fig.colorbar(tpc, ax=ax)
        ax.grid("both")  # type: ignore[arg-type]
        ax.set_xlim(40.0, 200.0)
        ax.set_ylim(20.0, 80.0)
        cbar.set_label("Ice water content (g m$^{-3}$)", rotation=270, labelpad=20)
        ax.set_title(f"Ice water content evolution at z = {(height.shape[0] - level) * thickness / 1000.0:.1f} km")
        # turn off the axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        # write the time steps on the plot
        print_out_test = ["15 min", "30 min", "80 min", "120 min"]
        for i, time in enumerate(time_step):
            ref_lev = np.argmin(np.abs(height - 3800.0))
            distance_traveled = int((initial_time_interval + time * time_inteval) * wind_speed[ref_lev] * extra_speed[i] / mean_edge_length) * mean_edge_length
            x_pos = 0.5 * (combined_vertex_x[i].min() + combined_vertex_x[i].max()) if i != 1 else 0.5 * (combined_vertex_x[i].min() + combined_vertex_x[i].max()) + 3.0
            ax.text(
                x_pos,
                25.0,
                print_out_test[i],
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
            )
        finalize_plot(fig=fig, out_file=f"{out_file}_solid_evolution_l{level}.png")

        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.spines['bottom'].set_alpha(0.3)
        ax.spines['left'].set_alpha(0.3)
        ax.spines['top'].set_alpha(0.3)
        ax.spines['right'].set_alpha(0.3)
        ax.set_axisbelow(True)

        # colorbar limits
        w_min, w_max = w_plot[~np.isnan(w_plot)].min(), w_plot[~np.isnan(w_plot)].max()
        brmap = plt.get_cmap("RdBu_r")
        for i in range(len(time_step)):
            values = w_plot[i][valid_cells]
            tpc = ax.tripcolor(tri[i], values, edgecolor="none", shading="flat", cmap=brmap, vmin=w_min, vmax=w_max)
        cbar = fig.colorbar(tpc, ax=ax)
        ax.grid("both")  # type: ignore[arg-type]
        ax.set_xlim(40.0, 200.0)
        ax.set_ylim(20.0, 80.0)
        cbar.set_label("W (m s$^{-1}$)", rotation=270, labelpad=20)
        # set the alpha of bounding box of the colorbar to 0.3
        # cbar.ax.patch.set_alpha(0.3)
        # title
        ax.set_title(f"Vertical wind speed evolution at z = {(height.shape[0] - level) * thickness / 1000.0:.1f} km")
        # turn off the axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        # write the time steps on the plot
        print_out_test = ["15 min", "30 min", "80 min", "120 min"]
        for i, time in enumerate(time_step):
            ref_lev = np.argmin(np.abs(height - 3800.0))
            distance_traveled = int((initial_time_interval + time * time_inteval) * wind_speed[ref_lev] * extra_speed[i] / mean_edge_length) * mean_edge_length
            x_pos = 0.5 * (combined_vertex_x[i].min() + combined_vertex_x[i].max()) if i != 1 else 0.5 * (combined_vertex_x[i].min() + combined_vertex_x[i].max()) + 3.0
            ax.text(
                x_pos,
                25.0,
                print_out_test[i],
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
            )
        finalize_plot(fig=fig, out_file=f"{out_file}_w_evolution_l{level}.png")

        # still in progress, need to figure out how to combine the fields at different time steps into one plot
        # combined_lwc = np.zeros(liquid_water_content_plot.shape, dtype=liquid_water_content_plot.dtype)
        # combined_w = np.zeros(w_plot.shape, dtype=w_plot.dtype)
        # one_to_one_mapping = np.arange(cell[0].shape[0]) * domain_length
        # for i, time in enumerate(time_step):
        #     distance_traveled = int((initial_time_interval + time * time_inteval) * wind_speed[level] / edge_length) * edge_length
        #     local_cell_x = cell[0] + distance_traveled
        #     locel_cell_y = cell[1]
        #     # local_cell_x = np.mod(local_cell_x, domain_length)
        #     # new_location_indices = np.searchsorted(cell[0], local_cell_x)
        #     # rearranged_lwc = liquid_water_content[time][new_location_indices]
        #     # rearranged_w = w[time][new_location_indices]
        #     # rearranged_w[rearranged_lwc < 0.001] = 0.0
        #     # rearranged_lwc[rearranged_lwc < 0.001] = 0.0
        #     combined_lwc += rearranged_lwc
        #     combined_w += rearranged_w
        # fig, ax = plot_mpl_triangulation(
        #     tri=tri,
        #     values=combined_lwc[valid_cells],
        #     cmap="plasma",
        # )
        # finalize_plot(fig=fig, out_file=f"{out_file}_rain_evolution_l{level}.png")
        # fig, ax = plot_mpl_triangulation(
        #     tri=tri,
        #     values=combined_w[valid_cells],
        #     cmap="seismic",
        # )
        # finalize_plot(fig=fig, out_file=f"{out_file}_w_evolution_l{level}.png")
        

def plot_cloud_snapshots_in_wk82exp_from_file(
    *,
    tri: mpl.tri.Triangulation,
    mean_edge_length: float,
    c2v_connectivity: np.ndarray,
    valid_cells: np.ndarray,
    cell: tuple[np.ndarray, np.ndarray],
    vertex: tuple[np.ndarray, np.ndarray],
    data_file: str,
    level: int = 0,
    time_step: int | list[int] = 0,
    out_file: str = "",
) -> None:
    time_step = (time_step,) if isinstance(time_step, int) else time_step

    max_wind_speed = 15.0  # m/s
    h_min = 0.0  # m
    wind_scale_height = 3000.0  # m
    # hardcoded time interval
    initial_time_interval, time_inteval = 59.0, 60.0 # seconds
    with xr.open_dataset(data_file) as data:
        half_lvl_height = data["height"].values
        height = 0.5 * (half_lvl_height[1:] + half_lvl_height[:-1])
        thickness = -np.mean(np.diff(half_lvl_height))
        liquid_water_content = data["qr"] * 1000.0  # g/kg
        w = data["upward_air_velocity"]
        u = data["eastward_wind"]
        v = data["northward_wind"]
        if "time" in liquid_water_content.dims and "time" in w.dims:
            lwc_local_field = liquid_water_content.isel(time=list(time_step))
            w_local_field = w.isel(time=list(time_step))
            u_local_field = u.isel(time=list(time_step))
            v_local_field = v.isel(time=list(time_step))
        else:
            raise ValueError("Variable 'liquid_water_content' or 'w' is not time-dependent.")
        if "level" in liquid_water_content.dims:
            lwc_local_field = lwc_local_field.isel(level=level)
            u_local_field = u_local_field.isel(level=level)
            v_local_field = v_local_field.isel(level=level)
        else:
            raise ValueError("Variable 'liquid_water_content' is not vertically dependent.")
        liquid_water_content_plot = lwc_local_field.values
        u_plot = u_local_field.values
        v_plot = v_local_field.values
        w_plot = w_local_field.values
        w_plot = 0.5 * (w_plot[:,:-1,:] + w_plot[:,1:,:])  # average to cell centers
        w_plot = w_plot[:, level, :]

        assert liquid_water_content_plot.ndim == 2, "Variable 'liquid_water_content' must be 2D (time, cell)."
        assert w_plot.ndim == 2, "Variable 'w' must be 2D (time, cell)."
        
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 13,
                "axes.labelsize": 15,
                "axes.titlesize": 15,
                "axes.linewidth": 1.5,
                "xtick.direction": "out",
                "ytick.direction": "out",
                "xtick.top": False,
                "ytick.right": False,
                "savefig.dpi": 400,
                "savefig.bbox": "tight",
                "axes.grid": False,
                # spine alpha is set to 0.5 to make the grid lines more visible
            }
        )

        # regrid to a regular grid for plotting
        domain_length = 100000.0  # m
        x_reg = np.linspace(0.5 * mean_edge_length, domain_length - 0.5 * mean_edge_length, int(domain_length / mean_edge_length))
        y_reg = np.linspace(0.5 * mean_edge_length, domain_length - 0.5 * mean_edge_length, int(domain_length / mean_edge_length))
        
        x2d, y2d = np.meshgrid(x_reg, y_reg)

        w_abs_max = max(np.abs(w_plot[~np.isnan(w_plot)]).min(), np.abs(w_plot[~np.isnan(w_plot)]).max())
        anurag_colors = [(38, 50, 98), (62, 101, 172), (93, 147, 197), (158, 197, 223), (213, 229, 238), (247, 247, 247), (246, 220, 200), (233, 168, 129), (202, 98, 79), (163, 35, 47), (94, 11, 34)]
        anurag_colors_norm = [(r/255, g/255, b/255) for r, g, b in anurag_colors]
        anurag_cmap = mpl.colors.LinearSegmentedColormap.from_list("anurag_list", anurag_colors_norm, N=len(anurag_colors))
        anurag_color_boundaries = [-0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55]
        anurag_color_ticks = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45]
        # coolwarmmap = plt.get_cmap("coolwarm")
        # new_tri = create_mpl_triangulation(
        #     c2v_connectivity=c2v_connectivity,
        #     node_x=vertex[0] / 1000.0,
        #     node_y=vertex[1] / 1000.0,
        #     length_max=1.5 * mean_edge_length / 1000.0,  # km
        # )
        for i in range(len(time_step)):
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.spines['bottom'].set_alpha(0.3)
            ax.spines['left'].set_alpha(0.3)
            ax.spines['top'].set_alpha(0.3)
            ax.spines['right'].set_alpha(0.3)
            ax.set_axisbelow(True)
            w_interp = griddata(
                (cell[0], cell[1]),
                w_plot[i],
                (x2d, y2d),
                method="linear"
            )
            qr_interp = griddata(
                (cell[0], cell[1]),
                liquid_water_content_plot[i],
                (x2d, y2d),
                method="linear"
            )
            cp = ax.contourf(x2d / 1000.0, y2d / 1000.0, w_interp, cmap=anurag_cmap, levels=anurag_color_boundaries, vmin=anurag_color_boundaries[0], vmax=anurag_color_boundaries[-1], extend='both')
            ax.contour(x2d / 1000.0, y2d / 1000.0, qr_interp, colors="green", linewidths=0.5, levels=[0.05, 0.2])
            cb = plt.colorbar(
                cp,
                ax=ax,
                orientation="horizontal",
                pad=0.18,
                ticks=anurag_color_ticks,
            )
            cb.set_label("W (m s$^{-1}$)", rotation=0, labelpad=20)
            ax.set_xlim(0.0, 100.0)
            ax.set_ylim(0.0, 100.0)
            ax.set_xlabel("$x$ (km)")
            ax.set_ylabel("$y$ (km)")
            ax.set_title(f"W at t = {int(time_step[i] + 1)} min, z = {thickness * 0.5 + (height.shape[0] - 1 - level) * thickness:.1f} m")
            # set square aspect ratio
            ax.set_aspect('equal', adjustable='box')
            finalize_plot(fig=fig, out_file=f"{out_file}_snapshot_t{time_step[i]}_l{level}.png")
            
            
            cmap = plt.get_cmap("seismic")
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.spines['bottom'].set_alpha(0.3)
            ax.spines['left'].set_alpha(0.3)
            ax.spines['top'].set_alpha(0.3)
            ax.spines['right'].set_alpha(0.3)
            ax.set_axisbelow(True)
            u_interp = griddata(
                (cell[0], cell[1]),
                u_plot[i],
                (x2d, y2d),
                method="linear"
            )
            cp = ax.contourf(x2d / 1000.0, y2d / 1000.0, u_interp, cmap=cmap,vmin=u_interp.min(), vmax=u_interp.max(), extend='both')
            cb = plt.colorbar(
                cp,
                ax=ax,
                orientation="horizontal",
                pad=0.18,
            )
            cb.set_label("U (m s$^{-1}$)", rotation=0, labelpad=20)
            ax.set_xlim(0.0, 100.0)
            ax.set_ylim(0.0, 100.0)
            ax.set_xlabel("$x$ (km)")
            ax.set_ylabel("$y$ (km)")
            ax.set_title(f"U at t = {int(time_step[i] + 1)} min, z = {thickness * 0.5 + (height.shape[0] - 1 - level) * thickness:.1f} m")
            # set square aspect ratio
            ax.set_aspect('equal', adjustable='box')
            finalize_plot(fig=fig, out_file=f"{out_file}_snapshot_U_t{time_step[i]}_l{level}.png")
            
            cmap = plt.get_cmap("seismic")
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.spines['bottom'].set_alpha(0.3)
            ax.spines['left'].set_alpha(0.3)
            ax.spines['top'].set_alpha(0.3)
            ax.spines['right'].set_alpha(0.3)
            ax.set_axisbelow(True)
            v_interp = griddata(
                (cell[0], cell[1]),
                v_plot[i],
                (x2d, y2d),
                method="linear"
            )
            cp = ax.contourf(x2d / 1000.0, y2d / 1000.0, v_interp, cmap=cmap,vmin=v_interp.min(), vmax=v_interp.max(), extend='both')
            cb = plt.colorbar(
                cp,
                ax=ax,
                orientation="horizontal",
                pad=0.18,
            )
            cb.set_label("V (m s$^{-1}$)", rotation=0, labelpad=20)
            ax.set_xlim(0.0, 100.0)
            ax.set_ylim(0.0, 100.0)
            ax.set_xlabel("$x$ (km)")
            ax.set_ylabel("$y$ (km)")
            ax.set_title(f"V at t = {int(time_step[i] + 1)} min, z = {thickness * 0.5 + (height.shape[0] - 1 - level) * thickness:.1f} m")
            # set square aspect ratio
            ax.set_aspect('equal', adjustable='box')
            finalize_plot(fig=fig, out_file=f"{out_file}_snapshot_V_t{time_step[i]}_l{level}.png")
            # values = w_plot[i][valid_cells]
            # tpc = ax.tripcolor(new_tri, values, edgecolor="none", shading="flat", cmap=coolwarmmap, vmin=-w_abs_max, vmax=w_abs_max)
            # cbar = fig.colorbar(tpc, ax=ax)
            # # cbar.formatter.set_powerlimits((0, 0))  # type: ignore[attr-defined]
            # # cbar.formatter.set_useMathText(True)  # type: ignore[attr-defined]
            # # ax.triplot(triang, color="k", lw=0.3, alpha=0.4)
            # # ax.grid("both")  # type: ignore[arg-type]
            # ax.set_xlim(25.0, 75.0)
            # ax.set_ylim(25.0, 75.0)
            # ax.set_xlabel("$x$ (km)")
            # ax.set_ylabel("$y$ (km)")
            # cbar.set_label("W (m s$^{-1}$)", rotation=270, labelpad=20)
            # # set the alpha of bounding box of the colorbar to 0.3
            # # cbar.ax.patch.set_alpha(0.3)
            # # title
            # ax.set_title(f"W at t = {int(time_step[i] + 1)} min, z = {thickness * 0.5 + (height.shape[0] - 1 - level) * thickness:.1f} m")
            # finalize_plot(fig=fig, out_file=f"{out_file}_snapshot_t{time_step[i]}_l{level}.png")


def section_plot(liquid_data, ice_data, w_data, xx, zz, title: str, output_file: str = ""):
    fig, ax = plt.subplots(figsize=(23, 6))
    ax.set_axisbelow(True)
    ax.grid("both")  # type: ignore[arg-type]
    ax.set_xlabel("$x$ (km)")
    ax.set_ylabel("$y$ (km)")
    red_cmap = plt.get_cmap("Reds")
    blue_cmap = plt.get_cmap("Blues")
    red_boundaries = np.linspace(liquid_data.min(), liquid_data.max(), 101)
    blue_boundaries = np.linspace(ice_data.min(), ice_data.max(), 101)
    red_lnorm = colors.BoundaryNorm(red_boundaries, red_cmap.N, clip=True)
    blue_lnorm = colors.BoundaryNorm(blue_boundaries, blue_cmap.N, clip=True)
    blue_cp = ax.contourf(xx, zz, ice_data, cmap=blue_cmap, levels=blue_boundaries, norm=blue_lnorm, alpha=0.5)
    red_cp = ax.contourf(xx, zz, liquid_data, cmap=red_cmap, levels=red_boundaries, norm=red_lnorm, alpha=0.5)
    # put contour lines for w_data showing values,
    # w_contours = np.linspace(w_data.min(), w_data.max(), 11)
    w_contours = [-14.0, -8.0, -2.0, 2.0, 8.0, 14.0]
    w_cp = ax.contour(xx, zz, w_data, colors="black", linewidths=0.5, levels=w_contours)
    ax.clabel(w_cp, inline=True, fontsize=7, fmt="%.1f")
    cb1 = fig.colorbar(red_cp, location="right")
    cb2 = fig.colorbar(blue_cp, location="left")
    # set extra space in left and right of the figures
    fig.subplots_adjust(left=0.3, right=0.7)
    cb1.set_label("Liquid water content (g m$^{-3}$)", rotation=270, labelpad=20)
    cb2.set_label("Ice water content (g m$^{-3}$)", rotation=270, labelpad=20)
    # set colorbar ticks to be 5 ticks
    cb1.set_ticks(np.linspace(liquid_data.min(), liquid_data.max(), 11))
    cb2.set_ticks(np.linspace(ice_data.min(), ice_data.max(), 11))
    # one decimal point for clabel
    cb1.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    cb2.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    # set the alpha of bounding box of the colorbar to 0.3
    ax.spines['bottom'].set_alpha(0.3)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['top'].set_alpha(0.3)
    ax.spines['right'].set_alpha(0.3)
    ax.set_axisbelow(True)
    # cb1.ax.patch.set_alpha(0.3)
    # cb2.ax.patch.set_alpha(0.3)
    ax.set_xlim(20.0, 80.0)
    ax.set_ylim(0, 6.0)
    ax.set_title(title)
    finalize_plot(fig=fig, out_file=output_file)


def plot_cross_section_in_wk82exp_from_file(
    *,
    mean_edge_length: float,
    data_file: str,
    cell: tuple[np.ndarray, np.ndarray],
    time_step: int | list[int] = 0,
    out_file: str = "",
) -> None:
    time_step = (time_step,) if isinstance(time_step, int) else time_step
    domain_length = 100000.0 # meters
    with xr.open_dataset(data_file) as data:
        half_lvl_height = data["height"].values
        height = 0.5 * (half_lvl_height[1:] + half_lvl_height[:-1])
        cell_x = cell[0]
        cell_y = cell[1]
        center = 0.5 * domain_length
        mask = (cell_x > center - 0.4 * mean_edge_length) & (cell_x <= center + 0.4 * mean_edge_length)
        if len(np.unique(cell_x[mask])) > 1:
            raise ValueError("There are more than one layer to be plotted, please reduce the cell x range.")
        # cell_data = np.transpose(cell_data)
        xx = cell_y[mask]
        argsort = xx.argsort()
        xx = xx[argsort[::-1]]
        # for i in range(len(xx)-1):
        #     print(xx[i+1] - xx[i])
        
        liquid_water_content = (data["qc"] + data["qr"]) * data["air_density"] * 1000.0  # g/kg
        ice_water_content = (data["qi"] + data["qs"] + data["qg"]) * data["air_density"] * 1000.0  # g/kg
        w = data["upward_air_velocity"]
        if "time" in liquid_water_content.dims and "time" in w.dims:
            lwc_local_field = liquid_water_content.isel(time=list(time_step))
            iwc_local_field = ice_water_content.isel(time=list(time_step))
            w_local_field = w.isel(time=list(time_step))
        else:
            raise ValueError("Variable 'liquid_water_content' or 'ice_water_content' or 'w' is not time-dependent.")
        liquid_water_content_plot = lwc_local_field.values[:,:,mask]
        ice_water_content_plot = iwc_local_field.values[:,:,mask]
        w_plot = w_local_field.values
        w_plot = 0.5 * (w_plot[:,:-1,:] + w_plot[:,1:,:])  # average to cell centers
        w_plot = w_plot[:, :, mask]

        for time in range(len(time_step)):
            for k in range(height.shape[0]):
                liquid_water_content_plot[time, k, :] = liquid_water_content_plot[time, k, argsort[::-1]]
                ice_water_content_plot[time, k, :] = ice_water_content_plot[time, k, argsort[::-1]]
                w_plot[time, k, :] = w_plot[time, k, argsort[::-1]]

        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 13,
                "axes.labelsize": 15,
                "axes.titlesize": 15,
                "axes.linewidth": 1.5,
                "xtick.direction": "out",
                "ytick.direction": "out",
                "xtick.top": False,
                "ytick.right": False,
                "savefig.dpi": 400,
                "savefig.bbox": "tight",
                "axes.grid": False,
                # spine alpha is set to 0.5 to make the grid lines more visible
            }
        )

        xx = np.repeat(np.expand_dims(xx, axis=0), height.shape[0], axis=0) / 1000.0 # (k, cell)
        zz = np.repeat(np.expand_dims(height, axis=1), xx.shape[1], axis=1) / 1000.0 # (k, cell)
        for i, time in enumerate(time_step):
            title = f"Cloud cross section at time {int(time + 1.0)} min at x = 50 km"
            assert xx.shape == zz.shape == liquid_water_content_plot[i, :, :].shape == ice_water_content_plot[i, :, :].shape == w_plot[i, :, :].shape, f"Shapes of xx ({xx.shape}), zz ({zz.shape}), liquid_water_content ({liquid_water_content_plot[i, :, :].shape}), ice_water_content and w do not match."
            section_plot(
                liquid_data=liquid_water_content_plot[i, :, :],
                ice_data=ice_water_content_plot[i, :, :],
                w_data=w_plot[i, :, :],
                xx=xx,
                zz=zz,
                title=title,
                output_file=f"{out_file}_t{time}.png",
            )


def plot_kinetic_energy_in_wk82exp_from_file(
    *,
    mean_edge_length: float,
    data_file: str,
    out_file: str = "",
) -> None:
    import csv, pathlib
    csv_base_dir = pathlib.Path("/capstor/scratch/cscs/cong/icon4py/warm_bubble_runs/data/reference")
    u15_csv_file = csv_base_dir / "wk82_u15.csv"
    u10_1_csv_file = csv_base_dir / "wk82_u10_1.csv"
    u10_2_csv_file = csv_base_dir / "wk82_u10_2.csv"
    u20_1_csv_file = csv_base_dir / "wk82_u20_1.csv"
    u20_2_csv_file = csv_base_dir / "wk82_u20_2.csv"
    with open(u15_csv_file, "r") as f:
        reader = csv.reader(f)
        u15_time, u15_data = zip(*[(float(row[0]), float(row[1])) for row in reader])
    with open(u10_1_csv_file, "r") as f:
        reader = csv.reader(f)
        u10_1_time, u10_1_data = zip(*[(float(row[0]), float(row[1])) for row in reader])
    with open(u10_2_csv_file, "r") as f:
        reader = csv.reader(f)
        u10_2_time, u10_2_data = zip(*[(float(row[0]), float(row[1])) for row in reader])
    with open(u20_1_csv_file, "r") as f:
        reader = csv.reader(f)
        u20_1_time, u20_1_data = zip(*[(float(row[0]), float(row[1])) for row in reader])
    with open(u20_2_csv_file, "r") as f:
        reader = csv.reader(f)
        u20_2_time, u20_2_data = zip(*[(float(row[0]), float(row[1])) for row in reader])
    # throw away u10_1 data after 43min and u_10_2 before 43min, and join together
    ind1 = next(i for i, t in enumerate(u10_1_time) if t > 43.0)
    ind2 = next(i for i, t in enumerate(u10_2_time) if t >= 43.0)
    u10_time = list(u10_1_time[:ind1]) + list(u10_2_time[ind2:])
    u10_data = list(u10_1_data[:ind1]) + list(u10_2_data[ind2:])
    ind1 = next(i for i, t in enumerate(u20_1_time) if t > 52.0)
    ind2 = next(i for i, t in enumerate(u20_2_time) if t >= 52.0)
    u20_time = list(u20_1_time[:ind1]) + list(u20_2_time[ind2:])
    u20_data = list(u20_1_data[:ind1]) + list(u20_2_data[ind2:])

    max_wind_speed = 15.0  # m/s
    h_min = 0.0  # m
    wind_scale_height = 3000.0  # m
    # read csv file to get reference maximum vertical wind speed
    with xr.open_dataset(data_file) as data:
        half_lvl_height = data["height"].values
        height = 0.5 * (half_lvl_height[1:] + half_lvl_height[:-1])
        thickness = -np.mean(np.diff(half_lvl_height))
        wind_speed = max_wind_speed * (
            np.tanh((height - h_min) / (wind_scale_height - h_min)) - 0.45
        )
        initial_kinetic_energy = 0.5 * (wind_speed[:, None] ** 2) * data["air_density"].values[0, :, :]  # J/m^3
        w = data["upward_air_velocity"].values
        w = 0.5 * (w[:,:-1,:] + w[:,1:,:])  # average to cell centers
        kinetic_energy = 0.5 * (w ** 2 + data["eastward_wind"] ** 2 + data["northward_wind"] ** 2) * data["air_density"]
        volume = np.sqrt(3.0) / 4.0 * mean_edge_length ** 2 * thickness # m^3
        diff_kinetic_energy = (kinetic_energy.values - initial_kinetic_energy) * volume # J
        diff_kinetic_energy = np.sum(diff_kinetic_energy, axis=(1, 2)) # J
        time = (data["time"].values - data["time"].values[0])
        time = time / np.timedelta64(1, "m")  # convert to minutes
        max_w = np.zeros_like(time)
        for i in range(len(time)):
            max_w[i] = np.max(np.abs(w[i, :, :]))
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 15,
                "axes.labelsize": 17,
                "axes.titlesize": 17,
                "axes.linewidth": 1.5,
                "xtick.direction": "out",
                "ytick.direction": "out",
                "xtick.top": False,
                "ytick.right": False,
                "savefig.dpi": 400,
                "savefig.bbox": "tight",
                "axes.grid": False,
                "legend.framealpha": 0.3,
                "legend.fontsize": 12,
                "grid.alpha": 0.3,
                # spine alpha is set to 0.5 to make the grid lines more visible
            }
        )

        title = f"Integrated kinetic energy time series"
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.spines['bottom'].set_alpha(0.3)
        ax.spines['left'].set_alpha(0.3)
        ax.spines['top'].set_alpha(0.3)
        ax.spines['right'].set_alpha(0.3)
        ax.set_axisbelow(True)
        ax.grid("both")  # type: ignore[arg-type]
        ax.plot(time, diff_kinetic_energy, color="blue", lw=1.5)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Kinetic energy (J)")
        ax.set_title(title)
        finalize_plot(fig=fig, out_file=f"{out_file}_kinetic_energy.png")

        title = f"Maximum vertical wind speed time series"
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.spines['bottom'].set_alpha(0.3)
        ax.spines['left'].set_alpha(0.3)
        ax.spines['top'].set_alpha(0.3)
        ax.spines['right'].set_alpha(0.3)
        ax.set_axisbelow(True)
        ax.grid("both")  # type: ignore[arg-type]
        ax.plot(time, max_w, color="black", lw=1.5, label="icon4py U=15 m s$^{-1}$")
        ax.plot(u10_time, u10_data, color="blue", lw=1.5, label="WK82 U=10 m s$^{-1}$")
        ax.plot(u15_time, u15_data, color="red", lw=1.5, label="WK82 U=15 m s$^{-1}$")
        ax.plot(u20_time, u20_data, color="green", lw=1.5, label="WK82 U=20 m s$^{-1}$")
        ax.legend()
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("W (m s$^{-1}$)")
        ax.set_title(title)
        finalize_plot(fig=fig, out_file=f"{out_file}_max_w.png")
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.spines['bottom'].set_alpha(0.3)
        ax.spines['left'].set_alpha(0.3)
        ax.spines['top'].set_alpha(0.3)
        ax.spines['right'].set_alpha(0.3)
        # plot wind speed profile
        ax.plot(wind_speed, height / 1000.0, color="black", lw=1.5)
        ax.set_xlabel("Wind speed (m s$^{-1}$)")
        ax.set_ylabel("Height (km)")
        ax.set_title("Wind speed profile")
        finalize_plot(fig=fig, out_file=f"{out_file}_initial_u.png")


def plot_1d(
    *,
    x: np.ndarray,
    y: np.ndarray,
    x_axis_label: str = "",
    y_axis_label: str = "",
    out_file: str = "",
) -> None:
    """
    Plot a 1D profile.

    Args:
        x: array that contains the x-coordinates
        y: array that contains the y-coordinates
        label_name: label of plotting curve
        out_file: passed to savefig if present, else plot is shown instead

    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axisbelow(True)
    ax.plot(x, y, color="blue", lw=1.0)

    ax.grid(True, which="both", ls=":", lw=0.25)
    ax.legend()
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)

    if out_file != "":
        fig.savefig(out_file, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_convergence(
    *,
    x: list[float],
    y: list[float],
    label_name: str = "",
    theoretical_orders: list[float] | None = None,
    linestyles: list[str] | None = None,
    out_file: str = "",
) -> None:
    """
    Plot convergence on log-log scales.

    Args:
        x: list that contains the cell sizes
        y: list that contains the errors
        label_name: label of plotting curve
        theoretical_orders: list of slopes to plot
        linestyles: list of linestyles for slopes to plot
        out_file: passed to savefig if present, else plot is shown instead

    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axisbelow(True)
    ax.plot(x, y, marker="o", label=label_name)

    # add theoretical orders if present
    if theoretical_orders is not None:
        assert linestyles is not None, (
            "linestyles must be provided if theoretical_orders is provided"
        )
        assert len(theoretical_orders) == len(linestyles), (
            "theoretical_orders and linestyles must have the same length"
        )
        for i in range(len(theoretical_orders)):
            order = theoretical_orders[i]
            x_min, x_max = np.min(x), np.max(x)
            y_min = np.min(y)
            ax.axline(
                (x_min, y_min),
                (x_max, y_min * (x_max / x_min) ** order),
                ls="--" if len(linestyles) == 0 else linestyles[i],
                c="black",
                lw=1.0,
                label=rf"$p={order}$",
                zorder=1.9,
            )

    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend()
    ax.set_xlabel("$h$")
    ax.set_ylabel("error")
    ax.set_xscale("log")
    ax.set_yscale("log")

    if out_file != "":
        fig.savefig(out_file, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
