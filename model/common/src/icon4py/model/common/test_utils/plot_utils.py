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
import numpy as np

from icon4py.model.common import dimension as dims

# prevent matplotlib logging spam
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# flake8: noqa
log = logging.getLogger(__name__)


def plot_1D(x, y, xlabel="", ylabel="", out_file=""):
    """
    Plot 1D array.
    Args:
        x: input argument, array that contains x-values
        y: input argument, array that contains y-values
        xlabel: input argument, label of x-axis
        ylabel: input argument, label of y-axis
        out_file: input argument, passed to savefig if present, else plot is shown instead
    """
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)

    ax.plot(x, y)

    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.set_xlabel("$z$")
    ax.set_ylabel("$q$")

    if out_file != "":
        fig.savefig(out_file, bbox_inches="tight")
        log.debug(f"Saved {out_file}")
    else:
        plt.show()
    plt.close(fig)


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
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
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
