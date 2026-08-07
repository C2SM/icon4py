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

from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.utils import data_allocation as data_alloc


def create_mpl_triangulation(
    *,
    grid: icon_grid.IconGrid,
    node_x: data_alloc.NDArray,
    node_y: data_alloc.NDArray,
    length_max: float,
) -> mpl.tri.Triangulation:
    """
    Create a matplotlib triangulation object for torus grids.

    Args:
        grid: IconGrid that entails a torus grid
        node_x: array that contains the vertex x-coordinates
        node_y: array that contains the vertex y-coordinates
        length_max: maximum edge length to plot

    """
    # create a matplotlib triangulation from the grid connectivity
    face_node_connectivity = grid.connectivities["C2V"].asnumpy()
    tri = mpl.tri.Triangulation(node_x, node_y, triangles=face_node_connectivity)

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
    values: data_alloc.NDArray,
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
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)

    ax.grid("both")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    return fig, ax


def plot_torus_plane(
    *,
    grid: icon_grid.IconGrid,
    node_x: data_alloc.NDArray,
    node_y: data_alloc.NDArray,
    values: data_alloc.NDArray,
    length_max: float,
    out_file: str = "",
) -> None:
    """
    Plot a single horizontal plane for torus grids.

    Args:
        grid: IconGrid that entails a torus grid
        node_x: array that contains the vertex x-coordinates
        node_y: array that contains the vertex y-coordinates
        values: array that contains the horizontal values on a single level to plot
        length_max: maximum edge length to plot
        out_file: passed to savefig if present, else plot is shown instead

    """
    tri = create_mpl_triangulation(
        grid=grid,
        node_x=node_x,
        node_y=node_y,
        length_max=length_max,
    )
    fig, _ = plot_mpl_triangulation(
        tri=tri,
        values=values,
    )
    finalize_plot(fig=fig, out_file=out_file)


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
    else:
        plt.show()
    plt.close(fig)
