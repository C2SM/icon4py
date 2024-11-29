import logging
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr

from icon4py.model.common.settings import xp

# prevent matplotlib logging spam
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# flake8: noqa
log = logging.getLogger(__name__)

def create_triangulation(gridfname: str) -> mpl.tri.Triangulation:
    """
    Create a triangulation from a grid file.

    Args:
        gridfname: The filename of the grid file to open.

    Returns:
        A triangulation object created from the grid file.
    """

    grid = xr.open_dataset(gridfname)

    return mpl.tri.Triangulation(
        grid.cartesian_x_vertices.values,
        grid.cartesian_y_vertices.values,
        triangles=grid.vertex_of_cell.values.T-1,
        )

def remove_elongated_triangles(
        tri: mpl.tri.Triangulation,
) -> mpl.tri.Triangulation:
    """
    Remove elongated triangles from a triangulation.

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
    ratio = 4

    tri.all_edges = tri.edges.copy()

    tri.n_all_triangles = tri.triangles.shape[0]
    tri.n_all_edges = tri.edges.shape[0]

    # Remove elongated triangles
    elongated_mask = []
    for triangle in tri.triangles:
        node_x_diff = tri.x[triangle] - xp.roll(tri.x[triangle], 1)
        node_y_diff = tri.y[triangle] - xp.roll(tri.y[triangle], 1)
        edges = xp.sqrt(node_x_diff**2 + node_y_diff**2)
        if xp.max(edges) > ratio*xp.min(edges):
            elongated_mask.append(True)
        else:
            elongated_mask.append(False)
    tri.set_mask(elongated_mask)

    # Mask out edges that are part of elongated triangles
    edges_mask = xp.ones(tri.all_edges.shape[0], dtype=bool)
    for i, edge in enumerate(tri.all_edges):
        if any(xp.array_equal(edge, filtered_edge) for filtered_edge in tri.edges):
            edges_mask[i] = False

    tri.edges_mask = edges_mask

    return tri

def create_edge_centers(tri):
    edge_centers_x = []
    edge_centers_y = []
    for edge in tri.all_edges:
        x1, y1 = tri.x[edge[0]], tri.y[edge[0]]
        x2, y2 = tri.x[edge[1]], tri.y[edge[1]]
        edge_centers_x.append((x1 + x2) / 2)
        edge_centers_y.append((y1 + y2) / 2)
    return xp.array(edge_centers_x), xp.array(edge_centers_y)

def create_torus_triangulation(grid_fname: str) -> mpl.tri.Triangulation:
    """
    Create a triangulation for a torus from a grid file.
    Remove elongated triangles from the triangulation.
    Take care of elongated edges.

    Args:
        grid_fname: The filename of the grid file to open.

    Returns:
        A triangulation object created from the grid file.
    """
    tri = create_triangulation(grid_fname)
    tri = remove_elongated_triangles(tri)
    tri.edge_x, tri.edge_y = create_edge_centers(tri)
    return tri

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
    fig = plt.figure(1, figsize=(17,min(13,4*nlev))); plt.clf()
    axs = fig.subplots(nrows=min(nax_per_col, nlev), ncols=max(1,int(xp.ceil(nlev/nax_per_col))), sharex=True, sharey=True)
    if nlev > 1:
        axs = axs.flatten()    
    else:
        axs = [axs]
    for i in range(nlev):
        plot_lev(data, i)
        axs[i].set_aspect('equal')
        #axs[i].set_xlabel(f"Level {-i}")
        axs[i].triplot(tri, color='k', linewidth=0.25)

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.draw()

    if save_to_file:
        fig.savefig('plot.png', dpi=600, bbox_inches='tight')
        log.debug(f"Saved plot.png")
    else:
        plt.show()
        plt.pause(1)


if __name__ == '__main__':
    # Example usage and testing
    
    grid_fname = 'testdata/grids/Torus_Triangles_50000m_x_5000m_res500m.nc'
    state_fname = 'testdata/prognostic_state.pkl'
    
    tri = create_torus_triangulation(grid_fname)
    with open(state_fname, 'rb') as f:
        state = pickle.load(f)
    
    #data = state.theta_v.ndarray
    data = state.vn.ndarray
    
    plot_data(tri, data, 2)