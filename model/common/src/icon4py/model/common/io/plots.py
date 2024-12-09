import logging
import pickle
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.test_utils import serialbox_utils as sb
from icon4py.model.common import dimension as dims
import icon4py.model.common.grid.states as grid_states
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr

from icon4py.model.common.settings import xp

# prevent matplotlib logging spam
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# flake8: noqa
log = logging.getLogger(__name__)

X_BOUNDARY = xp.pi
Y_BOUNDARY = 15/2*xp.pi/180
X_LIMS = (-X_BOUNDARY*1.02, X_BOUNDARY*1.02)
Y_LIMS = (-Y_BOUNDARY*1.02, Y_BOUNDARY*1.02)

def remove_boundary_triangles(
        tri: mpl.tri.Triangulation,
        criterion: str = 'wrapping',
        mask_edges: bool = False,
) -> mpl.tri.Triangulation:
    """
    Remove boundary triangles from a triangulation.

    This function examines each triangle in the provided triangulation and
    determines if it is elongated based on the ratio of its longest edge to
    its shortest edge. If the ratio exceeds `ratio`, the triangle is considered
    elongated and is masked out.
    Also save the original set of edges

    Args:
        tri: The input triangulation to be processed.

    Returns:
        The modified triangulation with elongated triangles masked out.
    """

    def check_wrapping(vert_x, vert_y):
        #return (check_three_numbers(vert_x) or check_three_numbers(vert_y)) \
        #        and ((xp.abs(vert_x) == X_BOUNDARY).any() or (xp.abs(vert_y) == Y_BOUNDARY).any())
        return (check_three_numbers(vert_x) and (xp.abs(vert_x) == X_BOUNDARY).any()) \
            or (check_three_numbers(vert_y) and (xp.abs(vert_y) == Y_BOUNDARY).any())

    def check_three_numbers(numbers):
        positive_count = sum(1 for x in numbers if x > 0)
        negative_count = sum(1 for x in numbers if x < 0)

        return (positive_count == 2 and negative_count == 1) or (positive_count == 1 and negative_count == 2)


    tri.all_edges = tri.edges.copy()

    tri.n_all_triangles = tri.triangles.shape[0]
    tri.n_all_edges = tri.edges.shape[0]

    boundary_triangles_mask = []
    if criterion == 'wrapping':
        # Remove wrapping triangles
        for triangle in tri.triangles:
            if check_wrapping(tri.x[triangle], tri.y[triangle]):
                boundary_triangles_mask.append(True)
            else:
                boundary_triangles_mask.append(False)
    elif criterion == 'elongated':
        # Remove elongated triangles
        ratio = 4
        for triangle in tri.triangles:
            node_x_diff = tri.x[triangle] - xp.roll(tri.x[triangle], 1)
            node_y_diff = tri.y[triangle] - xp.roll(tri.y[triangle], 1)
            edges = xp.sqrt(node_x_diff**2 + node_y_diff**2)
            if xp.max(edges) > ratio*xp.min(edges):
                boundary_triangles_mask.append(True)
            else:
                boundary_triangles_mask.append(False)

    tri.set_mask(boundary_triangles_mask)

    if mask_edges:
        # Mask out edges that are part of boundary triangles
        edges_mask = xp.ones(tri.all_edges.shape[0], dtype=bool)
        for i, edge in enumerate(tri.all_edges):
            if any(xp.array_equal(edge, filtered_edge) for filtered_edge in tri.edges):
                edges_mask[i] = False
        tri.edges_mask = edges_mask
    else:
        tri.edges_mask = None

    return tri

def create_torus_triangulation_from_savepoint(
    savepoint_path: str,
) -> mpl.tri.Triangulation:
    """
    Create a triangulation for a torus from a savepoint.
    Remove elongated triangles from the triangulation.
    Take care of elongated edges.

    Args:


    Returns:
        A triangulation object created from the grid file.
    """
    grid_savepoint = sb.IconSerialDataProvider("icon_pydycore", savepoint_path).from_savepoint_grid('aa', 0, 2)

    vert_x = grid_savepoint.v_lon().ndarray
    vert_y = grid_savepoint.v_lat().ndarray
    edge_x = grid_savepoint.edge_center_lon().ndarray
    edge_y = grid_savepoint.edge_center_lat().ndarray
    c2v = grid_savepoint.c2v()

    # clean up the grid
    # Adjust x values to coincide with the periodic boundary
    vert_x = xp.where(xp.abs(vert_x - X_BOUNDARY) < 1e-14,  X_BOUNDARY, vert_x)
    vert_x = xp.where(xp.abs(vert_x + X_BOUNDARY) < 1e-14, -X_BOUNDARY, vert_x)
    # shift all to -X_BOUNDARY
    vert_x = xp.where(vert_x == X_BOUNDARY, -X_BOUNDARY, vert_x)
    # Adjust y values to coincide with the periodic boundary
    vert_y = xp.where(xp.abs(vert_y - Y_BOUNDARY) < 1e-14,  Y_BOUNDARY, vert_y)
    vert_y = xp.where(xp.abs(vert_y + Y_BOUNDARY) < 1e-14, -Y_BOUNDARY, vert_y)

    tri = mpl.tri.Triangulation(
        vert_x,
        vert_y,
        triangles=c2v,
        )
    tri.edge_x = edge_x
    tri.edge_y = edge_y

    tri = remove_boundary_triangles(tri, mask_edges=False)

    return tri

def plot_grid(tri: mpl.tri.Triangulation) -> None:

    #plt.close('all')
    fig = plt.figure(1); plt.clf(); plt.show(block=False)
    ax = fig.subplots(nrows=1, ncols=1)
    ax.set_xlim(X_LIMS)
    ax.set_ylim(Y_LIMS)
    ax.triplot(tri, color='k', linewidth=0.25)
    ax.scatter(tri.x,      tri.y,      c='red',  s=3**2)
    ax.scatter(tri.edge_x, tri.edge_y, c='blue', s=3**2)
    plt.draw()

def plot_data(tri: mpl.tri.Triangulation, data, nlev: int, save_to_file: bool = False) -> None:
    """
    Plot data on a triangulation.
    """
    nax_per_col = 10
    if type(data) is not xp.ndarray:
        data = data.ndarray

    cmin = data.min()
    cmax = data.max()

    ntriangles = tri.n_all_triangles
    nedges = tri.n_all_edges
    if data.shape[0] == ntriangles:
        plot_lev = lambda data, i: axs[i].tripcolor(tri, data[:, -1-i], edgecolor='none', shading='flat', cmap='viridis') #, vmin=cmin, vmax=cmax)
    elif data.shape[0] == nedges:
        plot_lev = lambda data, i: axs[i].scatter(tri.edge_x, tri.edge_y, c=data[:, -1-i], s=4**2, cmap='viridis') #, vmin=cmin, vmax=cmax)

    plt.close('all')
    fig = plt.figure(1, figsize=(14,min(13,4*nlev))); plt.clf()
    axs = fig.subplots(nrows=min(nax_per_col, nlev), ncols=max(1,int(xp.ceil(nlev/nax_per_col))), sharex=True, sharey=True)
    if nlev > 1:
        axs = axs.flatten()
    else:
        axs = [axs]
    for i in range(nlev):
        plot_lev(data, i)
        #axs[i].set_aspect('equal')
        axs[i].set_xlim(X_LIMS)
        axs[i].set_ylim(Y_LIMS)
        #axs[i].set_xlabel(f"Level {-i}")
        axs[i].triplot(tri, color='k', linewidth=0.25)

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.draw()

    if save_to_file:
        fig.savefig('plot.png', dpi=600, bbox_inches='tight')
        log.debug(f"Saved plot.png")
    else:
        plt.show(block=False)
        plt.pause(1)


if __name__ == '__main__':
    # Example usage and testing

    state_fname = 'testdata/prognostic_state.torus_small.pkl'
    savepoint_path = 'testdata/ser_icondata/mpitask1/torus_small.flat_and_zeros/ser_data'

    tri = create_torus_triangulation_from_savepoint(savepoint_path=savepoint_path)

    with open(state_fname, 'rb') as f:
        state = pickle.load(f)[0]

    plot_data(tri, state.theta_v, 2)

