# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging

import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.interpolate import RBFInterpolator

from icon4py.model.common import dimension as dims
from icon4py.model.common.settings import xp

# prevent matplotlib logging spam
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# flake8: noqa
log = logging.getLogger(__name__)


def create_mpl_triangulation(grid, node_x, node_y, length_max):
    """
    Create a matplotlib triangulation object for torus grids.

    Args:
        grid: input argument, IconGrid that entails a torus grid
        node_x: input argument, array that contains the vertex x-coordinates
        node_y: input argument, array that contains the vertex y-coordinates
        length_max: input argument, maximum edge length to plot

    """
    # create a matplotlib triangulation from the grid connectivity
    face_node_connectivity = grid.connectivities[dims.C2VDim]
    tri = mpl.tri.Triangulation(node_x, node_y, triangles=face_node_connectivity)

    # remove triangles with edge length smaller greater than some max length
    # note: this is necessary to avoid plotting artifacts due to the periodicity of torus grids
    triangles = tri.triangles
    node_x_diff = node_x[triangles] - xp.roll(node_x[triangles], 1, axis=1)
    node_y_diff = node_y[triangles] - xp.roll(node_y[triangles], 1, axis=1)
    node_dist_max = xp.max(xp.sqrt(node_x_diff**2 + node_y_diff**2), axis=1)
    tri.set_mask(node_dist_max > length_max)

    return tri


def plot_mpl_triangulation(tri, values):
    """
    Plot values on a matplotlib triangulation.

    Args:
        tri: input argument, matplotlib triangulation
        values: input argument, array that contains the values on the triangulation

    """
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)

    tpc = ax.tripcolor(tri, values, edgecolor="none", shading="flat", cmap="viridis")
    cbar = fig.colorbar(tpc, ax=ax)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)

    ax.grid("both")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    return fig, ax


def finalize_plot(fig, out_file=""):
    """
    Save or show the current figure and close it afterwards.

    Args:
        fig: input argument, matplotlib figure
        out_file: input argument, passed to savefig if present, else plot is shown instead

    """
    if out_file != "":
        fig.savefig(out_file, bbox_inches="tight")
        log.debug(f"Saved {out_file}")
    else:
        plt.show()
    plt.close(fig)


def plot_torus_plane(grid, node_x, node_y, values, length_max, out_file=""):
    """
    Plot a single horizontal plane for torus grids.

    Args:
        grid: input argument, IconGrid that entails a torus grid
        node_x: input argument, array that contains the vertex x-coordinates
        node_y: input argument, array that contains the vertex y-coordinates
        values: input argument, array that contains the horizontal values on a single level to plot
        length_max: input argument, maximum edge length to plot
        out_file: input argument, passed to savefig if present, else plot is shown instead

    """
    tri = create_mpl_triangulation(grid, node_x, node_y, length_max)
    fig, ax = plot_mpl_triangulation(tri, values)
    finalize_plot(fig, out_file=out_file)


def plot_torus_plane_quad(grid, node_x, node_y, values, length_max, weights, nodes, out_file=""):
    """
    Plot a single horizontal plane for torus grids including weighted quadrature nodes.

    Args:
        grid: input argument, IconGrid that entails a torus grid
        node_x: input argument, array that contains the vertex x-coordinates
        node_y: input argument, array that contains the vertex y-coordinates
        values: input argument, array that contains the horizontal values on a single level to plot
        length_max: input argument, maximum edge length to plot
        weights: input argument, array that contains torus numerical quadrature weights
        nodes: input argument, array that contains torus numerical quadrature nodes
        out_file: input argument, passed to savefig if present, else plot is shown instead

    """
    tri = create_mpl_triangulation(grid, node_x, node_y, length_max)
    fig, ax = plot_mpl_triangulation(tri, values)
    ax.scatter(
        nodes[0, :, :].reshape(-1),
        nodes[1, :, :].reshape(-1),
        marker=".",
        s=0.1,
        c=weights.reshape(-1),
        cmap="cool",
    )
    finalize_plot(fig, out_file=out_file)


def plot_torus_plane_grid(
    grid,
    node_x,
    node_y,
    edges_center_x,
    edges_center_y,
    cell_center_x,
    cell_center_y,
    values,
    length_max,
    out_file="",
):
    """
    Plot a single horizontal plane for torus grids including vertices, edge centers and cell centers.

    Args:
        grid: input argument, IconGrid that entails a torus grid
        node_x: input argument, array that contains the vertex x-coordinates
        node_y: input argument, array that contains the vertex y-coordinates
        edges_center_x: input argument, array that contains the edge center x-coordinates
        edges_center_y: input argument, array that contains the edge center y-coordinates
        cell_center_x: input argument, array that contains the cell center x-coordinates
        cell_center_y: input argument, array that contains the cell center y-coordinates
        values: input argument, array that contains the horizontal values on a single level to plot
        length_max: input argument, maximum edge length to plot
        out_file: input argument, passed to savefig if present, else plot is shown instead

    """
    tri = create_mpl_triangulation(grid, node_x, node_y, length_max)
    fig, ax = plot_mpl_triangulation(tri, values)
    ax.scatter(node_x, node_y, marker=".", s=0.1, c="r")
    ax.scatter(edges_center_x, edges_center_y, marker=".", s=0.1, c="g")
    ax.scatter(cell_center_x, cell_center_y, marker=".", s=0.1, c="b")
    finalize_plot(fig, out_file=out_file)


def plot_convergence(x, y, name="", theoretical_orders=[], linestyles=[], out_file=""):
    """
    Plot convergence on log-log scales.

    Args:
        x: input argument, array that contains the cell sizes
        y: input argument, array that contains the errors
        name: input argument, label of plotting curve
        theoretical_orders: input argument, list of slopes to plot
        linestyles: input argument, list of linestyles for slopes to plot
        out_file: input argument, passed to savefig if present, else plot is shown instead

    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axisbelow(True)
    ax.plot(x, y, marker="o", label=name)

    # add theoretical orders if present
    for i in range(len(theoretical_orders)):
        order = theoretical_orders[i]
        x_min, x_max = xp.min(x), xp.max(x)
        y_min, y_max = xp.min(y), xp.max(y)
        ax.axline(
            (x_min, y_min),
            (x_max, y_min * (x_max / x_min) ** order),
            ls="--" if len(linestyles) == 0 else linestyles[i],
            c="black",
            lw=1.0,
            label=(r"$p=%s$") % str(order),
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
        log.debug(f"Saved {out_file}")
    else:
        plt.show()
    plt.close(fig)


def interpolate_torus_plane(
    cell_center_x_high,
    cell_center_y_high,
    vals_high,
    node_x_low,
    node_y_low,
    weights,
    nodes,
):
    """
    Interpolate high-resolution to low-resolution values on torus grids.

    Args:
        cell_center_x_high: input argument, array that contains the high-resolution cell center x-coordinates
        cell_center_y_high: input argument, array that contains the high-resolution cell center y-coordinates
        vals_high: input argument, array that contains the high-resolution values to interpolate
        node_x_low: input argument, array that contains the low-resolution vertex x-coordinates
        node_y_low: input argument, array that contains the low-resolution vertex y-coordinates
        weights: input argument, array that contains torus numerical quadrature weights
        nodes: input argument, array that contains torus numerical quadrature nodes

    """
    # 2D function fitting
    fit = RBFInterpolator(
        xp.stack((cell_center_x_high, cell_center_y_high), axis=-1),
        vals_high,
        neighbors=25,
        smoothing=0.0,
        kernel="cubic",
        degree=4,
    )

    # evaluate fit on low-res grid
    nq = weights.shape[0]
    nc = nodes.shape[2]
    vals_low = xp.zeros(nc)
    for i in range(nq):
        nodes_flat = xp.stack((nodes[0, i, :], nodes[1, i, :]), axis=1)
        vals_low += weights[i, :] * fit(nodes_flat)

    return vals_low


def prepare_torus_quadratic_quadrature(
    grid, node_x, node_y, cell_center_x, cell_center_y, length_min
):
    """
    Prepare three-point quadrature rule on torus grids.

    Args:
        grid: input argument, IconGrid that entails a torus grid
        node_x: input argument, array that contains the vertex x-coordinates
        node_y: input argument, array that contains the vertex y-coordinates
        cell_center_x: input argument, array that contains the cell center x-coordinates
        cell_center_y: input argument, array that contains the cell center y-coordinates
        length_min: input argument, the smallest edge length in the grid

    Usage:
        The return values of this function are meant to be used for setting cell averages on torus grids.
        A two-dimensional scalar function f(x,y) can be projected onto a torus plane array arr as follows:
            arr = xp.sum(weights * f(nodes[0,:,:], nodes[1,:,:]), axis=0)

    """
    alpha = xp.array([[0.5, 0, 0.5], [0.5, 0.5, 0], [0, 0.5, 0.5]])
    weights_single = xp.array([1 / 3, 1 / 3, 1 / 3])

    n_cells = grid.num_cells
    n_points = weights_single.size

    weights = xp.tile(weights_single[:, None], (1, n_cells))
    nodes = xp.zeros((2, n_points, n_cells))

    c2v = grid.connectivities[dims.C2VDim]
    node_x_c2v = node_x[c2v]
    node_y_c2v = node_y[c2v]

    nodes[0, :, :] = xp.matmul(alpha, node_x_c2v.T)
    nodes[1, :, :] = xp.matmul(alpha, node_y_c2v.T)

    # revert to cell centers for degenerate triangles at the domain boundary due to periodicity
    node_x_diff = node_x_c2v - xp.roll(node_x_c2v, 1, axis=1)
    node_y_diff = node_y_c2v - xp.roll(node_y_c2v, 1, axis=1)
    node_dist_max = xp.max(xp.sqrt(node_x_diff**2 + node_y_diff**2), axis=1)
    mask = node_dist_max > 2.0 * length_min
    weights[:, mask] = 1 / n_points
    nodes[:, :, mask] = xp.stack((cell_center_x[None, mask], cell_center_y[None, mask]))

    return weights, nodes
