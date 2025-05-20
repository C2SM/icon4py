import logging, os

import gt4py.next as gtx
from icon4py.model.testing import serialbox as sb
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.interpolation.stencils.edge_2_cell_vector_rbf_interpolation import edge_2_cell_vector_rbf_interpolation
from icon4py.model.common.dimension import Koff
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.atmosphere.dycore.stencils.compute_tangential_wind import compute_tangential_wind
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pickle
import numpy as np
import xarray as xr

#mpl.use('tkagg')

# Prevent matplotlib logging spam
logging.getLogger("matplotlib").setLevel(logging.WARNING)
# get the the logger with the name 'PIL'
pil_logger = logging.getLogger('PIL')
# override the logger logging level to INFO
pil_logger.setLevel(logging.INFO)

# flake8: noqa
log = logging.getLogger(__name__)

DO_PLOTS = True
PLOT_IMGS_DIR = "imgs"


@gtx.field_operator
def _interpolate_from_half_to_full_levels(
    half_field: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    full_field = 0.5 * (half_field + half_field(Koff[1]))
    return full_field
@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_from_half_to_full_levels(
    half_field: fa.CellKField[ta.wpfloat],
    full_field: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _interpolate_from_half_to_full_levels(
        half_field,
        out=full_field,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

def pickle_data(state, label: str = "") -> None:
    if not os.path.isdir(PLOT_IMGS_DIR):
        os.makedirs(PLOT_IMGS_DIR)
    file_name = f"{PLOT_IMGS_DIR}/{label}.pkl"
    with open(file_name, "wb") as f:
        state_np = {
            "vn": state.vn.asnumpy(),
            "w": state.w.asnumpy(),
            "rho": state.rho.asnumpy(),
            "exner": state.exner.asnumpy(),
            "theta_v": state.theta_v.asnumpy(),
        }
        pickle.dump(state_np, f)


class Plot:
    def __init__(
            self,
            savepoint_path: str,
            grid_file_path: str,
            n_levels_to_plot: int = 2,
            backend: gtx.backend.Backend = gtx.gtfn_cpu,
        ):

        self.DO_PLOTS = DO_PLOTS
        if not self.DO_PLOTS:
            return

        data_provider = sb.IconSerialDataProvider(
            backend=backend,
            fname_prefix="icon_pydycore",
            path=savepoint_path
        )
        self.grid_savepoint = data_provider.from_savepoint_grid('aa', 0, 2)
        self.metrics_savepoint = data_provider.from_metrics_savepoint()
        self._num_levels_to_plot = n_levels_to_plot
        self._backend = backend

        self.grid = self.grid_savepoint.construct_icon_grid(on_gpu=False)
        self.interpolation_savepoint = data_provider.from_interpolation_savepoint()

        self._edge_2_cell_vector_rbf_interpolation = edge_2_cell_vector_rbf_interpolation.with_backend(self._backend)
        self._interpolate_to_full_levels = interpolate_from_half_to_full_levels.with_backend(self._backend)
        self._compute_tangential_wind = compute_tangential_wind.with_backend(self._backend)

        if grid_file_path != "":
            self.grid_file = xr.open_dataset(grid_file_path)
        else:
            raise NotImplementedError(
                "Only grid files are supported for now, too much stuff is missing from the savepoint"
            )
            self.grid_file = None

        # Constants
        self.DO_PLOTS = DO_PLOTS
        self.PLOT_IMGS_DIR = PLOT_IMGS_DIR
        self.NUM_AXES_PER_COLUMN = 2
        self.DOMAIN_LENGTH = self.grid_file.domain_length
        self.DOMAIN_HEIGHT = self.grid_file.domain_height
        self.X_BOUNDARY_RAD = np.pi
        self.Y_BOUNDARY_RAD = 15/2*np.pi/180 # Hardcoded in the grid generation script (could get from vertex lat)
        self.PLOT_X_LIMS = (-self.X_BOUNDARY_RAD*1.02, self.X_BOUNDARY_RAD*1.02)
        self.PLOT_Y_LIMS = (-self.Y_BOUNDARY_RAD*1.02, self.Y_BOUNDARY_RAD*1.02)

        self.tri = self._create_torus_triangulation(
            grid_savepoint = self.grid_savepoint,
            grid_file = self.grid_file,
        )

        self.primal_normal = np.array([
            self.grid_savepoint.primal_normal_v1().asnumpy(),
            self.grid_savepoint.primal_normal_v2().asnumpy(),
        ])
        self.primal_tangent = np.array([
             self.primal_normal[1],
            -self.primal_normal[0],
        ])
        self.half_level_heights = self.metrics_savepoint.z_ifc().asnumpy()
        self.full_level_heights = self.metrics_savepoint.z_mc().asnumpy()

        if not os.path.isdir(self.PLOT_IMGS_DIR):
            os.makedirs(self.PLOT_IMGS_DIR)
        self.plot_counter = 0

    def rad2cart(self, vert_lon: np.ndarray, vert_lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        vert_x = vert_lon * self.DOMAIN_LENGTH / (2.0 * self.X_BOUNDARY_RAD) + self.DOMAIN_LENGTH / 2.0
        vert_y = vert_lat * self.DOMAIN_HEIGHT / (2.0 * self.Y_BOUNDARY_RAD) + self.DOMAIN_HEIGHT / 2.0

        return vert_x, vert_y

    def _remove_boundary_triangles(
        self,
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

        def check_three_numbers(numbers):
            positive_count = sum(1 for x in numbers if x > 0)
            negative_count = sum(1 for x in numbers if x < 0)
            return (positive_count == 2 and negative_count == 1) or (positive_count == 1 and negative_count == 2)
        def check_wrapping(vert_x, vert_y):
            #return (check_three_numbers(vert_x) or check_three_numbers(vert_y)) \
            #        and ((np.abs(vert_x) == X_BOUNDARY).any() or (np.abs(vert_y) == Y_BOUNDARY).any())
            return (check_three_numbers(vert_x) and (np.abs(vert_x) == self.X_BOUNDARY_RAD).any()) \
                or (check_three_numbers(vert_y) and (np.abs(vert_y) == self.Y_BOUNDARY_RAD).any())

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
            ratio = 1.5
            for triangle in tri.triangles:
                node_x_diff = tri.x[triangle] - np.roll(tri.x[triangle], 1)
                node_y_diff = tri.y[triangle] - np.roll(tri.y[triangle], 1)
                edges = np.sqrt(node_x_diff**2 + node_y_diff**2)
                if np.max(edges) > ratio*np.min(edges):
                    boundary_triangles_mask.append(True)
                else:
                    boundary_triangles_mask.append(False)

        tri.set_mask(boundary_triangles_mask)

        if mask_edges:
            # Mask out edges that are part of boundary triangles
            edges_mask = np.ones(tri.all_edges.shape[0], dtype=bool)
            for i, edge in enumerate(tri.all_edges):
                if any(np.array_equal(edge, filtered_edge) for filtered_edge in tri.edges):
                    edges_mask[i] = False
            tri.edges_mask = edges_mask
        else:
            tri.edges_mask = None

        return tri

    def _create_torus_triangulation(
        self,
        grid_savepoint,
        grid_file = None,
    ) -> mpl.tri.Triangulation:
        """
        Create a triangulation for a torus from a savepoint and possibly grid file.
        Remove elongated triangles from the triangulation.

        Returns:
            A triangulation object created from the grid savepoint / file.
        """

        if grid_file is None:
            # lat/lon coordinates
            #
            vert_x = grid_savepoint.v_lon().asnumpy()
            vert_y = grid_savepoint.v_lat().asnumpy()
            edge_x = grid_savepoint.edge_center_lon().asnumpy()
            edge_y = grid_savepoint.edge_center_lat().asnumpy()
            cell_x = grid_savepoint.cell_center_lon().asnumpy()
            cell_y = grid_savepoint.cell_center_lat().asnumpy()
            # clean up the grid
            # Adjust x values to coincide with the periodic boundary
            vert_x = np.where(np.abs(vert_x - self.X_BOUNDARY_RAD) < 1e-14,  self.X_BOUNDARY_RAD, vert_x)
            vert_x = np.where(np.abs(vert_x + self.X_BOUNDARY_RAD) < 1e-14, -self.X_BOUNDARY_RAD, vert_x)
            # shift all to -X_BOUNDARY
            vert_x = np.where(vert_x == self.X_BOUNDARY_RAD, -self.X_BOUNDARY_RAD, vert_x)
            # Adjust y values to coincide with the periodic boundary
            vert_y = np.where(np.abs(vert_y - self.Y_BOUNDARY_RAD) < 1e-14,  self.Y_BOUNDARY_RAD, vert_y)
            vert_y = np.where(np.abs(vert_y + self.Y_BOUNDARY_RAD) < 1e-14, -self.Y_BOUNDARY_RAD, vert_y)
        else:
            # cartesian coordinates
            vert_x = grid_file.cartesian_x_vertices.values
            vert_y = grid_file.cartesian_y_vertices.values
            edge_x = grid_file.edge_middle_cartesian_x.values
            edge_y = grid_file.edge_middle_cartesian_y.values
            cell_x = grid_file.cell_circumcenter_cartesian_x.values
            cell_y = grid_file.cell_circumcenter_cartesian_y.values
            # clean up the grid
            # Adjust x values to coincide with the periodic boundary
            vert_x = np.where(np.abs(vert_x - self.DOMAIN_LENGTH) < 1e-9,  0, vert_x)
            edge_x = np.where(np.abs(edge_x - self.DOMAIN_LENGTH) < 1e-9,  0, edge_x)
            cell_x = np.where(np.abs(cell_x - self.DOMAIN_LENGTH) < 1e-9,  0, cell_x)
            mean_area = grid_file.cell_area.values.mean()
            edge_length = np.sqrt(mean_area*2)
            height_length = np.sqrt(3)/2 * edge_length

        tri = mpl.tri.Triangulation(
            vert_x,
            vert_y,
            triangles= grid_savepoint.c2v(),
            )
        tri.edge_x = edge_x
        tri.edge_y = edge_y
        tri.cell_x = cell_x
        tri.cell_y = cell_y

        tri.mean_area = mean_area
        tri.edge_length = edge_length
        tri.height_length = height_length

        if grid_file is None:
            tri = self._remove_boundary_triangles(tri)
        else:
            tri = self._remove_boundary_triangles(
                tri,
                criterion="elongated",
            )

        return tri

    def plot_grid(self, ax=None) -> None:

        if ax is None:
            fig = plt.figure(1); plt.show(block=False)
            ax = fig.subplots(nrows=1, ncols=1)
        ax.triplot(self.tri, color='k', linewidth=0.25)
        ax.plot(self.tri.x,      self.tri.y,      'vr')
        ax.plot(self.tri.edge_x, self.tri.edge_y, 'sg')
        ax.plot(self.tri.cell_x, self.tri.cell_y, 'ob')
        ax.set_aspect("equal")
        plt.draw()

    def pickle_data(self, state, label: str = "") -> None:
        if not self.DO_PLOTS:
            return
        pickle_data(state=state, label=f"{self.plot_counter:06d}_{label}")
        self.plot_counter += 1

    def _vec_interpolate_to_cell_center(self, vn_gtx):
        u_gtx = data_alloc.zero_field(self.grid, dims.CellDim, dims.KDim, backend=self._backend)
        v_gtx = data_alloc.zero_field(self.grid, dims.CellDim, dims.KDim, backend=self._backend)
        self._edge_2_cell_vector_rbf_interpolation(
            p_e_in=vn_gtx,
            ptr_coeff_1=self.interpolation_savepoint.rbf_vec_coeff_c1(),
            ptr_coeff_2=self.interpolation_savepoint.rbf_vec_coeff_c2(),
            p_u_out=u_gtx,
            p_v_out=v_gtx,
            horizontal_start=0,
            horizontal_end=self.grid.num_cells,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )
        return u_gtx.asnumpy(), v_gtx.asnumpy()

    def _compute_vt(self, vn_gtx):
        vt_gtx = data_alloc.zero_field(self.grid, dims.EdgeDim, dims.KDim, backend=self._backend)
        self._compute_tangential_wind(
            vn=vn_gtx,
            rbf_vec_coeff_e=self.interpolation_savepoint.rbf_vec_coeff_e(),
            vt=vt_gtx,
            horizontal_start=0,
            horizontal_end=self.grid.num_edges,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )
        return vt_gtx.asnumpy()

    def _scal_interpolate_to_full_levels(self, w_half_gtx):
        w_full_gtx = data_alloc.zero_field(self.grid, dims.CellDim, dims.KDim, backend=self._backend)
        self._interpolate_to_full_levels(
            half_field=w_half_gtx,
            full_field=w_full_gtx,
            horizontal_start=0,
            horizontal_end=self.grid.num_cells,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )
        return w_full_gtx.asnumpy()

    def _make_axes(self, num_axes: int = -1, fig_num: int = 1) -> tuple[mpl.figure.Figure, list[mpl.axes.Axes], list[mpl.axes.Axes]]:
        fig = plt.figure(fig_num); plt.clf()
        axs = fig.subplots(nrows=min(self.NUM_AXES_PER_COLUMN, num_axes), ncols=max(1,int(np.ceil(num_axes/self.NUM_AXES_PER_COLUMN))), sharex=True, sharey=True)
        if num_axes > 1:
            axs = axs.flatten()
        else:
            axs = [axs]
        caxs = [make_axes_locatable(ax).append_axes('right', size='3%', pad=0.02) for ax in axs]
        return fig, axs, caxs

    def plot_levels(self, data, num_levels: int = -1, label: str = "", fig_num: int = 1, qscale=40) -> mpl.axes.Axes:
        """
        Plot data defined on a triangulation on horizontal levels.
        """
        if not self.DO_PLOTS:
            return None

        if num_levels == -1:
            num_levels = self._num_levels_to_plot
        file_name = f"{self.PLOT_IMGS_DIR}/{self.plot_counter:05d}_{label}"

        if "vvec_cell" in file_name:
            # quiver-plot *v* at cell centres (data is vn)
            if type(data) is np.ndarray:
                data  = gtx.as_field((dims.EdgeDim, dims.KDim), data)
            u, v = self._vec_interpolate_to_cell_center(data)
            data = (u**2 + v**2)**0.5
        elif "vvec_edge" in file_name:
            # quiver-plot *vn* and *vt* at edge centres (data is vn)
            if type(data) is np.ndarray:
                data  = gtx.as_field((dims.EdgeDim, dims.KDim), data)
            vt = self._compute_vt(data)
            vn = data.asnumpy()
            data = (vn**2 + vt**2)**0.5

        if type(data) is not np.ndarray:
            data = data.asnumpy()

        cmin = data.min()
        cmax = data.max()
        if cmin < 0 and cmax > 0: # and np.abs(cmax + cmin) < cmax/3:
            cmap = "seismic"
            norm = lambda cmin, cmax: colors.TwoSlopeNorm(vmin=min(-1e-9, cmin), vcenter=0, vmax=max(1e-9,cmax))
        else:
            if cmax > -cmin:
                cmap = "YlOrRd"
                #cmap = "gist_rainbow"
            else:
                cmap = "YlOrRd_r"
                #cmap = "gist_rainbow_r"
            norm = lambda cmin, cmax: colors.Normalize(vmin=min(-1e-9, cmin), vmax=max(1e-9,cmax))

        match data.shape[0]:
            case self.grid.num_cells:
                plot_lev = lambda data, i: axs[i].tripcolor(self.tri, data[:, -1-i], edgecolor='none', shading='flat', cmap=cmap, norm=norm(data[:, -1-i].min(), data[:, -1-i].max()))
            case self.grid.num_edges:
                plot_lev = lambda data, i: axs[i].scatter(self.tri.edge_x, self.tri.edge_y, c=data[:, -1-i], s=6**2, cmap=cmap, norm=norm(data[:, -1-i].min(), data[:, -1-i].max()))
            case self.grid.num_vertices:
                plot_lev = lambda data, i: axs[i].scatter(self.tri.x, self.tri.y, c=data[:, -1-i], s=6**2, cmap=cmap, norm=norm(data[:, -1-i].min(), data[:, -1-i].max()))
            case _: raise ValueError("Invalid data shape")

        fig, axs, caxs = self._make_axes(num_axes=num_levels, fig_num=fig_num)

        for i in range(num_levels):
            im = plot_lev(data, i)
            cbar = fig.colorbar(im, cax=caxs[i], orientation='vertical')
            cbar.set_ticks(np.linspace(cbar.vmin, cbar.vmax, 5))
            axs[i].triplot(self.tri, color='k', linewidth=0.25)
            if "vvec_cell" in file_name:
                axs[i].quiver(self.tri.cell_x, self.tri.cell_y, u[:, -1-i], v[:, -1-i], scale=qscale)
            elif "vvec_edge" in file_name:
                u = vn[:, -1-i]*self.primal_normal[0] + vt[:, -1-i]*self.primal_tangent[0]
                v = vn[:, -1-i]*self.primal_normal[1] + vt[:, -1-i]*self.primal_tangent[1]
                axs[i].quiver(self.tri.edge_x, self.tri.edge_y, u, v, scale=qscale)
            axs[i].set_aspect('equal')
            axs[i].set_title(f"Level {-i}")

        fig.subplots_adjust(wspace=0.12, hspace=0.1)
        plt.draw()

        plt.show(block=False)
        if file_name != '':
            fig.savefig(f"{file_name}.png", bbox_inches='tight')
            log.debug(f"Saved {file_name}")
        else:
            plt.pause(1)

        self.plot_counter += 1
        if "vvec_cell" in file_name:
            return axs, self.tri.cell_x, self.tri.cell_y, u[:,-num_levels:], v[:,-num_levels:]
        elif "vvec_edge" in file_name:
            return axs, self.tri.edge_x, self.tri.edge_y, vn[:,-num_levels:], vt[:,-num_levels:]
        else:
            return axs, None, None, None, None


    def plot_sections(self, data, data2=None, sections_x: list[float] = [], sections_y: list[float] = [], plot_every=1, qscale=40, label: str = "", fig_num: int = 1) -> mpl.axes.Axes:
        """
        Plot data defined on a triangulation on vertical sections.
        """
        if not self.DO_PLOTS:
            return None
        if sections_x == [] and sections_y == []:
            return None
        num_sections = len(sections_x) + len(sections_y)

        pever = plot_every

        file_name = f"{self.PLOT_IMGS_DIR}/{self.plot_counter:05d}_{label}"

        if "vvec_cell" in file_name:
            if type(data) is np.ndarray:
                data  = gtx.as_field((dims.EdgeDim, dims.KDim), data)
            if type(data2) is np.ndarray:
                data2 = gtx.as_field((dims.CellDim, dims.KDim), data2)
            # quiver-plot *(u,v,w)* at cell centres (data is [vn, w])
            u, v = self._vec_interpolate_to_cell_center(data)
            w = self._scal_interpolate_to_full_levels(data2)
            data = (u**2 + v**2 + w**2)**0.5

        if type(data) is not np.ndarray:
            data = data.asnumpy()

        cmin = data.min()
        cmax = data.max()
        if cmin < 0 and cmax > 0: # and np.abs(cmax + cmin) < cmax/3:
            cmap = "seismic"
            norm = lambda cmin, cmax: colors.TwoSlopeNorm(vmin=min(-1e-9, cmin), vcenter=0, vmax=max(1e-9,cmax))
        else:
            if cmax > -cmin:
                cmap = "YlOrRd"
                #cmap = "gist_rainbow"
            else:
                cmap = "YlOrRd_r"
                #cmap = "gist_rainbow_r"
            norm = lambda cmin, cmax: colors.Normalize(vmin=min(-1e-9, cmin), vmax=max(1e-9,cmax))

        match data.shape[0]:
            case self.grid.num_cells:
                coords_x = self.tri.cell_x
                coords_y = self.tri.cell_y
            case self.grid.num_edges:
                coords_x = self.tri.edge_x
                coords_y = self.tri.edge_y
            case self.grid.num_vertices:
                coords_x = self.tri.x
                coords_y = self.tri.y
            case _: raise ValueError("Invalid data shape")

        fig, axs, caxs = self._make_axes(num_axes=num_sections, fig_num=fig_num)

        quiver_sec = lambda x, y, u, v, i: axs[i].quiver(x, y, u, v, scale=qscale)
        if "vvec_cell" in file_name:
            plot_sec = lambda x, y, data, i: axs[i].scatter(x, y, c=data, s=6**2, cmap=cmap, norm=norm(data.min(), data.max()))
        else:
            plot_sec = lambda x, y, data, i: axs[i].pcolormesh(x, y, data[:-1, :-1], cmap=cmap, norm=norm(data.min(), data.max()), shading='flat')

        for i in range(num_sections):
            if i < len(sections_x):
                if "vvec_cell" in file_name:
                    data = (v**2 + w**2)**0.5
                idxs = self._get_section_indexes(coords_x, coords_y, s_x=sections_x[i], dist=self.tri.height_length*2/3)
                x_coords = np.tile(coords_y[idxs], (self.grid.num_levels, 1)).T
                y_coords = self.full_level_heights[idxs,:]
                im = plot_sec(x_coords, y_coords, data[idxs, :], i)
                if "vvec_cell" in file_name:
                    quiver_sec(x_coords, y_coords, v[idxs, :], w[idxs, :], i)
                axs[i].set_title(f"Section at x = {sections_x[i]}")
            else:
                if "vvec_cell" in file_name:
                    data = (u**2 + w**2)**0.5
                idxs = self._get_section_indexes(coords_x, coords_y, s_y=sections_y[i-len(sections_x)], dist=self.tri.height_length*2/3)
                if data.shape[1] == self.grid.num_levels:
                    y_coords = self.full_level_heights[idxs,:]
                else:
                    y_coords = self.half_level_heights[idxs,:]
                x_coords = np.tile(coords_x[idxs], (y_coords.shape[1], 1)).T
                im = plot_sec(x_coords[::pever,::pever], y_coords[::pever,::pever], data[idxs, :][::pever,::pever], i)
                if "vvec_cell" in file_name:
                    quiver_sec(x_coords[::pever,::pever], y_coords[::pever,::pever], u[idxs, :][::pever,::pever], w[idxs, :][::pever,::pever], i)
                axs[i].set_title(f"Section at y = {sections_y[i-len(sections_x)]}")
            cbar = fig.colorbar(im, cax=caxs[i], orientation='vertical')
            cbar.set_ticks(np.linspace(cbar.vmin, cbar.vmax, 5))
            cbar.set_ticklabels([f"{c:.2f}" for c in np.linspace(cbar.vmin, cbar.vmax, 5)])
            #axs[i].set_aspect('equal')

        fig.subplots_adjust(wspace=0.12, hspace=0.1)
        plt.draw()

        plt.show(block=False)
        if file_name != '':
            fig.savefig(f"{file_name}.png", bbox_inches='tight')
            log.debug(f"Saved {file_name}")
        else:
            plt.pause(1)

        self.plot_counter += 1
        if "vvec_cell" in file_name:
            return axs, x_coords, y_coords, u[idxs,:], w[idxs,:], idxs
        else:
            return axs, x_coords, y_coords, data[idxs,:], None, idxs

    def export_vtk(self, filename: str, data: dict):
        """
        Export data to a VTK UnstructuredGrid (.vtu, binary) file for ParaView/VisIt.

        Args:
            filename: Output file path (should end with .vtu).
            data: Dictionary of {varname: ndarray} to export.
                  The location (cell, edge, vertex) and vertical coordinate (full/half) are inferred from array shape.
        """
        import struct
        # --- Shorthands for sizes ---
        num_vertices = len(self.tri.x)
        num_edges = len(self.tri.edge_x)
        num_cells = len(self.tri.cell_x)
        num_full_levels = self.full_level_heights.shape[1]
        num_half_levels = self.half_level_heights.shape[1]
        # Triangles (cells): only unmasked
        mask = self.tri.mask if self.tri.mask is not None else np.zeros(self.tri.triangles.shape[0], dtype=bool)
        triangles = self.tri.triangles[~mask]
        n_cells = triangles.shape[0]
        # VTK cell types: 5 = triangle
        cell_types = np.full(n_cells, 5, dtype=np.uint8)
        # Cell offsets for VTK
        cell_offsets = np.arange(1, n_cells+1, dtype=np.int32) * 3
        # Points: always vertices
        points = np.stack([self.tri.x, self.tri.y, np.zeros_like(self.tri.x)], axis=1).astype(np.float32)
        n_points = num_vertices

        # --- Organize data by inferred location and Z coordinate ---
        point_data = {}
        cell_data = {}
        edge_point_data = {}
        edge_points_needed = False
        z_coords = {}  # {varname: z_array}

        for k, arr in data.items():
            arr = np.ascontiguousarray(arr)
            arr_shape0 = arr.shape[0]
            arr_shape1 = arr.shape[1] if arr.ndim > 1 else 1
            # Infer location
            if arr_shape0 == num_vertices:
                point_data[k] = arr
            elif arr_shape0 == num_cells:
                # Only use unmasked cells
                if arr.shape[0] == len(mask):
                    arr = arr[~mask]
                cell_data[k] = arr
            elif arr_shape0 == num_edges:
                edge_point_data[k] = arr
                edge_points_needed = True
            else:
                raise ValueError(f"Cannot infer location for variable '{k}' with shape {arr.shape}")
            # Infer Z
            if arr_shape1 == num_full_levels:
                z_coords[k] = self.full_level_heights[0]
            elif arr_shape1 == num_half_levels:
                z_coords[k] = self.half_level_heights[0]
            elif arr_shape1 == 1:
                z_coords[k] = None
            else:
                raise ValueError(f"Cannot infer Z for variable '{k}' with shape {arr.shape}")

        # If edge data present, add edge points after vertex points
        if edge_points_needed:
            edge_coords = np.stack([self.tri.edge_x, self.tri.edge_y, np.zeros_like(self.tri.edge_x)], axis=1).astype(np.float32)
            points = np.concatenate([points, edge_coords], axis=0)
            n_points = points.shape[0]
            # Pad all vertex point_data to new n_points
            for k in point_data:
                arr = point_data[k]
                if arr.ndim == 1:
                    arr = np.pad(arr, (0, num_edges), mode='constant')
                else:
                    arr = np.pad(arr, ((0, num_edges), (0,0)), mode='constant')
                point_data[k] = arr
            # Place edge data after vertex data
            for k in edge_point_data:
                arr = edge_point_data[k]
                if arr.ndim == 1:
                    arr = np.pad(arr, (num_vertices, 0), mode='constant')
                else:
                    arr = np.pad(arr, ((num_vertices,0),(0,0)), mode='constant')
                point_data[k] = arr

        # --- Prepare appended binary data ---
        appended = b""
        offsets = {}
        offset = 0

        def append_array(arr):
            nonlocal appended, offset
            arr = np.ascontiguousarray(arr)
            raw = arr.tobytes()
            header = struct.pack("<Q", len(raw))
            appended += header + raw
            prev = offset
            offset += len(header) + len(raw)
            return prev

        # Points
        offsets['points'] = append_array(points)
        # Cells
        offsets['connectivity'] = append_array(triangles.astype(np.int32))
        offsets['offsets'] = append_array(cell_offsets)
        offsets['types'] = append_array(cell_types)
        # Data arrays
        point_offsets = {}
        for k, arr in point_data.items():
            arr = np.ascontiguousarray(arr)
            if arr.ndim == 1:
                arr = arr[:, None]
            point_offsets[k] = append_array(arr.astype(np.float32))
            # Add Z coordinate if present
            if z_coords[k] is not None:
                z_arr = np.broadcast_to(z_coords[k], arr.shape)
                point_offsets[k + "_z"] = append_array(z_arr.astype(np.float32))
        cell_offsets_dict = {}
        for k, arr in cell_data.items():
            arr = np.ascontiguousarray(arr)
            if arr.ndim == 1:
                arr = arr[:, None]
            cell_offsets_dict[k] = append_array(arr.astype(np.float32))
            # Add Z coordinate if present
            if z_coords[k] is not None:
                z_arr = np.broadcast_to(z_coords[k], arr.shape)
                cell_offsets_dict[k + "_z"] = append_array(z_arr.astype(np.float32))

        # --- Write XML ---
        def vtk_data_array(name, dtype, ncomp=1, fmt="appended", offset=0):
            vtk_type = {
                np.dtype('float32'): "Float32",
                np.dtype('float64'): "Float64",
                np.dtype('int32'): "Int32",
                np.dtype('uint8'): "UInt8",
            }[np.dtype(dtype)]
            return f'<DataArray type="{vtk_type}" Name="{name}" NumberOfComponents="{ncomp}" format="{fmt}" offset="{offset}"/>\n'

        with open(filename, "wb") as f:
            f.write(b'<?xml version="1.0"?>\n')
            f.write(b'<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
            f.write(b'  <UnstructuredGrid>\n')
            f.write(f'    <Piece NumberOfPoints="{n_points}" NumberOfCells="{n_cells}">\n'.encode())
            # Points
            f.write(b'      <Points>\n')
            f.write(vtk_data_array("Points", np.float32, 3, offset=offsets['points']).encode())
            f.write(b'      </Points>\n')
            # Cells
            f.write(b'      <Cells>\n')
            f.write(vtk_data_array("connectivity", np.int32, 1, offset=offsets['connectivity']).encode())
            f.write(vtk_data_array("offsets", np.int32, 1, offset=offsets['offsets']).encode())
            f.write(vtk_data_array("types", np.uint8, 1, offset=offsets['types']).encode())
            f.write(b'      </Cells>\n')
            # PointData
            f.write(b'      <PointData>\n')
            for k, arr in point_data.items():
                ncomp = arr.shape[1] if arr.ndim > 1 else 1
                f.write(vtk_data_array(k, np.float32, ncomp, offset=point_offsets[k]).encode())
                if z_coords[k] is not None:
                    f.write(vtk_data_array(k + "_z", np.float32, ncomp, offset=point_offsets[k + "_z"]).encode())
            f.write(b'      </PointData>\n')
            # CellData
            f.write(b'      <CellData>\n')
            for k, arr in cell_data.items():
                ncomp = arr.shape[1] if arr.ndim > 1 else 1
                f.write(vtk_data_array(k, np.float32, ncomp, offset=cell_offsets_dict[k]).encode())
                if z_coords[k] is not None:
                    f.write(vtk_data_array(k + "_z", np.float32, ncomp, offset=cell_offsets_dict[k + "_z"]).encode())
            f.write(b'      </CellData>\n')
            # Appended data
            f.write(b'    <AppendedData encoding="raw">\n_')
            f.write(appended)
            f.write(b'\n    </AppendedData>\n')
            f.write(b'    </Piece>\n')
            f.write(b'  </UnstructuredGrid>\n')
            f.write(b'</VTKFile>\n')