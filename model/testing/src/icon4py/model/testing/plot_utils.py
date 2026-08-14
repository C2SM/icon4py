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
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Plot values on a matplotlib triangulation.

    Args:
        tri: matplotlib triangulation
        values: array that contains the values on the triangulation

    """
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)

    tpc = ax.tripcolor(tri, values, edgecolor="none", shading="flat", cmap="viridis")
    cbar = fig.colorbar(tpc, ax=ax)
    cbar.formatter.set_powerlimits((0, 0))  # type: ignore[attr-defined]
    cbar.formatter.set_useMathText(True)  # type: ignore[attr-defined]

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
