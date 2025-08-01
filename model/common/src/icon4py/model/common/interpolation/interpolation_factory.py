# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import logging
from typing import Optional

import gt4py.next as gtx
from gt4py.next import backend as gtx_backend

import icon4py.model.common.metrics.compute_nudgecoeffs as common_metrics
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import (
    geometry,
    geometry_attributes as geometry_attrs,
    horizontal as h_grid,
    icon,
    refinement,
)
from icon4py.model.common.interpolation import (
    interpolation_attributes as attrs,
    interpolation_fields,
    rbf_interpolation as rbf,
)
from icon4py.model.common.states import factory, model
from icon4py.model.common.utils import data_allocation as data_alloc


cell_domain = h_grid.domain(dims.CellDim)
edge_domain = h_grid.domain(dims.EdgeDim)
vertex_domain = h_grid.domain(dims.VertexDim)

log = logging.getLogger(__name__)


class InterpolationFieldsFactory(factory.FieldSource, factory.GridProvider):
    def __init__(
        self,
        grid: icon.IconGrid,
        decomposition_info: definitions.DecompositionInfo,
        geometry_source: geometry.GridGeometry,
        backend: Optional[gtx_backend.Backend],
        metadata: dict[str, model.FieldMetaData],
    ):
        self._backend = backend
        self._xp = data_alloc.import_array_ns(backend)
        self._allocator = gtx.constructors.zeros.partial(allocator=backend)
        self._grid = grid
        self._decomposition_info = decomposition_info
        self._attrs = metadata
        self._providers: dict[str, factory.FieldProvider] = {}
        self._geometry = geometry_source
        characteristic_length = self._grid.global_properties.characteristic_length
        # TODO @halungge: Dummy config dict -  to be replaced by real configuration
        self._config = {
            "divavg_cntrwgt": 0.5,
            "weighting_factor": 0.0,
            "max_nudging_coefficient": 0.375,
            "nudge_efold_width": 2.0,
            "nudge_zone_width": 10,
            "rbf_kernel_cell": rbf.DEFAULT_RBF_KERNEL[rbf.RBFDimension.CELL],
            "rbf_kernel_edge": rbf.DEFAULT_RBF_KERNEL[rbf.RBFDimension.EDGE],
            "rbf_kernel_vertex": rbf.DEFAULT_RBF_KERNEL[rbf.RBFDimension.VERTEX],
            "rbf_scale_cell": rbf.compute_default_rbf_scale(
                characteristic_length, rbf.RBFDimension.CELL
            ),
            "rbf_scale_edge": rbf.compute_default_rbf_scale(
                characteristic_length, rbf.RBFDimension.EDGE
            ),
            "rbf_scale_vertex": rbf.compute_default_rbf_scale(
                characteristic_length, rbf.RBFDimension.VERTEX
            ),
        }
        log.info(
            f"initialized interpolation factory for backend = '{self._backend_name()}' and grid = '{self._grid}'"
        )
        log.debug(f"using array_ns {self._xp} ")

        self.register_provider(
            factory.PrecomputedFieldProvider(
                {
                    "refinement_control_at_edges": self._grid.refinement_control[dims.EdgeDim],
                }
            )
        )

        self._register_computed_fields()

    def __repr__(self):
        return f"{self.__class__.__name__} on (grid={self._grid!r}) providing fields f{self.metadata.keys()}"

    @property
    def _sources(self) -> factory.FieldSource:
        return factory.CompositeSource(self, (self._geometry,))

    def _register_computed_fields(self):
        nudging_coefficients_for_edges = factory.ProgramFieldProvider(
            func=common_metrics.compute_nudgecoeffs.with_backend(None),
            domain={
                dims.EdgeDim: (
                    edge_domain(h_grid.Zone.NUDGING_LEVEL_2),
                    edge_domain(h_grid.Zone.END),
                )
            },
            fields={attrs.NUDGECOEFFS_E: attrs.NUDGECOEFFS_E},
            deps={
                "refin_ctrl": "refinement_control_at_edges",
            },
            params={
                "grf_nudge_start_e": refinement.refine_control_value(
                    dims.EdgeDim, h_grid.Zone.NUDGING
                ).value,
                "max_nudging_coefficient": self._config["max_nudging_coefficient"],
                "nudge_efold_width": self._config["nudge_efold_width"],
                "nudge_zone_width": self._config["nudge_zone_width"],
            },
        )
        self.register_provider(nudging_coefficients_for_edges)

        geofac_div = factory.EmbeddedFieldOperatorProvider(
            # needs to be computed on fieldview-embedded backend
            func=interpolation_fields.compute_geofac_div.with_backend(None),
            domain=(dims.CellDim, dims.C2EDim),
            fields={attrs.GEOFAC_DIV: attrs.GEOFAC_DIV},
            deps={
                "primal_edge_length": geometry_attrs.EDGE_LENGTH,
                "edge_orientation": geometry_attrs.CELL_NORMAL_ORIENTATION,
                "area": geometry_attrs.CELL_AREA,
            },
        )
        self.register_provider(geofac_div)

        geofac_rot = factory.EmbeddedFieldOperatorProvider(
            # needs to be computed on fieldview-embedded backend
            func=interpolation_fields.compute_geofac_rot.with_backend(None),
            domain=(dims.VertexDim, dims.V2EDim),
            fields={attrs.GEOFAC_ROT: attrs.GEOFAC_ROT},
            deps={
                "dual_edge_length": geometry_attrs.DUAL_EDGE_LENGTH,
                "edge_orientation": geometry_attrs.VERTEX_EDGE_ORIENTATION,
                "dual_area": geometry_attrs.DUAL_AREA,
                "owner_mask": "vertex_owner_mask",
            },
        )
        self.register_provider(geofac_rot)

        geofac_n2s = factory.NumpyFieldsProvider(
            func=functools.partial(interpolation_fields.compute_geofac_n2s, array_ns=self._xp),
            fields=(attrs.GEOFAC_N2S,),
            domain=(dims.CellDim, dims.C2E2CODim),
            deps={
                "dual_edge_length": geometry_attrs.DUAL_EDGE_LENGTH,
                "geofac_div": attrs.GEOFAC_DIV,
            },
            connectivities={"c2e": dims.C2EDim, "e2c": dims.E2CDim, "c2e2c": dims.C2E2CDim},
            params={
                "horizontal_start": self._grid.start_index(
                    cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                )
            },
        )
        self.register_provider(geofac_n2s)

        geofac_grdiv = factory.NumpyFieldsProvider(
            func=functools.partial(interpolation_fields.compute_geofac_grdiv, array_ns=self._xp),
            fields=(attrs.GEOFAC_GRDIV,),
            domain=(dims.EdgeDim, dims.E2C2EODim),
            deps={
                "geofac_div": attrs.GEOFAC_DIV,
                "inv_dual_edge_length": f"inverse_of_{geometry_attrs.DUAL_EDGE_LENGTH}",
                "owner_mask": "edge_owner_mask",
            },
            connectivities={"c2e": dims.C2EDim, "e2c": dims.E2CDim, "e2c2e": dims.E2C2EDim},
            params={
                "horizontal_start": self._grid.start_index(
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                )
            },
        )

        self.register_provider(geofac_grdiv)

        cell_average_weight = factory.NumpyFieldsProvider(
            func=functools.partial(
                interpolation_fields.compute_mass_conserving_bilinear_cell_average_weight,
                array_ns=self._xp,
            ),
            fields=(attrs.C_BLN_AVG,),
            domain=(dims.CellDim, dims.C2E2CODim),
            deps={
                "lat": geometry_attrs.CELL_LAT,
                "lon": geometry_attrs.CELL_LON,
                "cell_areas": geometry_attrs.CELL_AREA,
                "cell_owner_mask": "cell_owner_mask",
            },
            connectivities={"c2e2c0": dims.C2E2CODim},
            params={
                "horizontal_start": self.grid.start_index(
                    cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                ),
                "horizontal_start_level_3": self.grid.start_index(
                    cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
                ),
                "divavg_cntrwgt": self._config["divavg_cntrwgt"],
            },
        )
        self.register_provider(cell_average_weight)

        c_lin_e = factory.NumpyFieldsProvider(
            func=functools.partial(interpolation_fields.compute_c_lin_e, array_ns=self._xp),
            fields=(attrs.C_LIN_E,),
            domain=(dims.EdgeDim, dims.E2CDim),
            deps={
                "edge_cell_length": geometry_attrs.EDGE_CELL_DISTANCE,
                "inv_dual_edge_length": f"inverse_of_{geometry_attrs.DUAL_EDGE_LENGTH}",
                "edge_owner_mask": "edge_owner_mask",
            },
            params={
                "horizontal_start": self._grid.start_index(
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                )
            },
        )
        self.register_provider(c_lin_e)

        geofac_grg = factory.NumpyFieldsProvider(
            func=functools.partial(interpolation_fields.compute_geofac_grg, array_ns=self._xp),
            fields=(attrs.GEOFAC_GRG_X, attrs.GEOFAC_GRG_Y),
            domain=(dims.CellDim, dims.C2E2CODim),
            deps={
                "primal_normal_cell_x": geometry_attrs.EDGE_NORMAL_CELL_U,
                "primal_normal_cell_y": geometry_attrs.EDGE_NORMAL_CELL_V,
                "owner_mask": "cell_owner_mask",
                "geofac_div": attrs.GEOFAC_DIV,
                "c_lin_e": attrs.C_LIN_E,
            },
            connectivities={"c2e": dims.C2EDim, "e2c": dims.E2CDim, "c2e2c": dims.C2E2CDim},
            params={
                "horizontal_start": self.grid.start_index(
                    cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                )
            },
        )
        self.register_provider(geofac_grg)

        e_flx_avg = factory.NumpyFieldsProvider(
            func=functools.partial(interpolation_fields.compute_e_flx_avg, array_ns=self._xp),
            fields=(attrs.E_FLX_AVG,),
            domain=(dims.EdgeDim, dims.E2C2EODim),
            deps={
                "c_bln_avg": attrs.C_BLN_AVG,
                "geofac_div": attrs.GEOFAC_DIV,
                "owner_mask": "edge_owner_mask",
                "primal_cart_normal_x": geometry_attrs.EDGE_NORMAL_X,
                "primal_cart_normal_y": geometry_attrs.EDGE_NORMAL_Y,
                "primal_cart_normal_z": geometry_attrs.EDGE_NORMAL_Z,
            },
            connectivities={
                "e2c": dims.E2CDim,
                "c2e": dims.C2EDim,
                "c2e2c": dims.C2E2CDim,
                "e2c2e": dims.E2C2EDim,
            },
            params={
                "horizontal_start_p3": self.grid.start_index(
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
                ),
                "horizontal_start_p4": self.grid.start_index(
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
                ),
            },
        )
        self.register_provider(e_flx_avg)

        e_bln_c_s = factory.NumpyFieldsProvider(
            func=functools.partial(interpolation_fields.compute_e_bln_c_s, array_ns=self._xp),
            fields=(attrs.E_BLN_C_S,),
            domain=(dims.CellDim, dims.C2EDim),
            deps={
                "cells_lat": geometry_attrs.CELL_LAT,
                "cells_lon": geometry_attrs.CELL_LON,
                "edges_lat": geometry_attrs.EDGE_LAT,
                "edges_lon": geometry_attrs.EDGE_LON,
            },
            connectivities={"c2e": dims.C2EDim},
            params={"weighting_factor": self._config["weighting_factor"]},
        )
        self.register_provider(e_bln_c_s)

        pos_on_tplane_e_x_y = factory.NumpyFieldsProvider(
            func=functools.partial(
                interpolation_fields.compute_pos_on_tplane_e_x_y, array_ns=self._xp
            ),
            fields=(attrs.POS_ON_TPLANE_E_X, attrs.POS_ON_TPLANE_E_Y),
            domain=(dims.ECDim,),
            deps={
                "primal_normal_v1": geometry_attrs.EDGE_NORMAL_U,
                "primal_normal_v2": geometry_attrs.EDGE_NORMAL_V,
                "dual_normal_v1": geometry_attrs.EDGE_DUAL_U,
                "dual_normal_v2": geometry_attrs.EDGE_DUAL_V,
                "cells_lon": geometry_attrs.CELL_LON,
                "cells_lat": geometry_attrs.CELL_LAT,
                "edges_lon": geometry_attrs.EDGE_LON,
                "edges_lat": geometry_attrs.EDGE_LAT,
                "vertex_lon": geometry_attrs.VERTEX_LON,
                "vertex_lat": geometry_attrs.VERTEX_LAT,
                "owner_mask": "edge_owner_mask",
            },
            connectivities={"e2c": dims.E2CDim, "e2v": dims.E2VDim, "e2c2e": dims.E2C2EDim},
            params={
                "grid_sphere_radius": constants.EARTH_RADIUS,
                "horizontal_start": self.grid.start_index(
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                ),
            },
        )
        self.register_provider(pos_on_tplane_e_x_y)

        cells_aw_verts = factory.NumpyFieldsProvider(
            func=functools.partial(interpolation_fields.compute_cells_aw_verts, array_ns=self._xp),
            fields=(attrs.CELL_AW_VERTS,),
            domain=(dims.VertexDim, dims.V2CDim),
            deps={
                "dual_area": geometry_attrs.DUAL_AREA,
                "edge_vert_length": geometry_attrs.EDGE_VERTEX_DISTANCE,
                "edge_cell_length": geometry_attrs.EDGE_CELL_DISTANCE,
            },
            connectivities={
                "v2e": dims.V2EDim,
                "e2v": dims.E2VDim,
                "v2c": dims.V2CDim,
                "e2c": dims.E2CDim,
            },
            params={
                "horizontal_start": self.grid.start_index(
                    vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                )
            },
        )
        self.register_provider(cells_aw_verts)

        rbf_vec_coeff_c = factory.NumpyFieldsProvider(
            func=functools.partial(rbf.compute_rbf_interpolation_coeffs_cell, array_ns=self._xp),
            fields=(attrs.RBF_VEC_COEFF_C1, attrs.RBF_VEC_COEFF_C2),
            domain=(dims.CellDim, dims.C2E2C2EDim),
            deps={
                "cell_center_lat": geometry_attrs.CELL_LAT,
                "cell_center_lon": geometry_attrs.CELL_LON,
                "cell_center_x": geometry_attrs.CELL_CENTER_X,
                "cell_center_y": geometry_attrs.CELL_CENTER_Y,
                "cell_center_z": geometry_attrs.CELL_CENTER_Z,
                "edge_center_x": geometry_attrs.EDGE_CENTER_X,
                "edge_center_y": geometry_attrs.EDGE_CENTER_Y,
                "edge_center_z": geometry_attrs.EDGE_CENTER_Z,
                "edge_normal_x": geometry_attrs.EDGE_NORMAL_X,
                "edge_normal_y": geometry_attrs.EDGE_NORMAL_Y,
                "edge_normal_z": geometry_attrs.EDGE_NORMAL_Z,
            },
            connectivities={"rbf_offset": dims.C2E2C2EDim},
            params={
                "rbf_kernel": self._config["rbf_kernel_cell"].value,
                "scale_factor": self._config["rbf_scale_cell"],
                "horizontal_start": self.grid.start_index(
                    cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                ),
            },
        )
        self.register_provider(rbf_vec_coeff_c)

        rbf_vec_coeff_e = factory.NumpyFieldsProvider(
            func=functools.partial(rbf.compute_rbf_interpolation_coeffs_edge, array_ns=self._xp),
            fields=(attrs.RBF_VEC_COEFF_E,),
            domain=(dims.CellDim, dims.E2C2EDim),
            deps={
                "edge_lat": geometry_attrs.EDGE_LAT,
                "edge_lon": geometry_attrs.EDGE_LON,
                "edge_center_x": geometry_attrs.EDGE_CENTER_X,
                "edge_center_y": geometry_attrs.EDGE_CENTER_Y,
                "edge_center_z": geometry_attrs.EDGE_CENTER_Z,
                "edge_normal_x": geometry_attrs.EDGE_NORMAL_X,
                "edge_normal_y": geometry_attrs.EDGE_NORMAL_Y,
                "edge_normal_z": geometry_attrs.EDGE_NORMAL_Z,
                "edge_dual_normal_u": geometry_attrs.EDGE_DUAL_U,
                "edge_dual_normal_v": geometry_attrs.EDGE_DUAL_V,
            },
            connectivities={"rbf_offset": dims.E2C2EDim},
            params={
                "rbf_kernel": self._config["rbf_kernel_edge"].value,
                "scale_factor": self._config["rbf_scale_edge"],
                "horizontal_start": self.grid.start_index(
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                ),
            },
        )
        self.register_provider(rbf_vec_coeff_e)

        rbf_vec_coeff_v = factory.NumpyFieldsProvider(
            func=functools.partial(rbf.compute_rbf_interpolation_coeffs_vertex, array_ns=self._xp),
            fields=(attrs.RBF_VEC_COEFF_V1, attrs.RBF_VEC_COEFF_V2),
            domain=(dims.VertexDim, dims.V2EDim),
            deps={
                "vertex_lat": geometry_attrs.VERTEX_LAT,
                "vertex_lon": geometry_attrs.VERTEX_LON,
                "vertex_x": geometry_attrs.VERTEX_X,
                "vertex_y": geometry_attrs.VERTEX_Y,
                "vertex_z": geometry_attrs.VERTEX_Z,
                "edge_center_x": geometry_attrs.EDGE_CENTER_X,
                "edge_center_y": geometry_attrs.EDGE_CENTER_Y,
                "edge_center_z": geometry_attrs.EDGE_CENTER_Z,
                "edge_normal_x": geometry_attrs.EDGE_NORMAL_X,
                "edge_normal_y": geometry_attrs.EDGE_NORMAL_Y,
                "edge_normal_z": geometry_attrs.EDGE_NORMAL_Z,
            },
            connectivities={"rbf_offset": dims.V2EDim},
            params={
                "rbf_kernel": self._config["rbf_kernel_vertex"].value,
                "scale_factor": self._config["rbf_scale_vertex"],
                "horizontal_start": self.grid.start_index(
                    vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                ),
            },
        )
        self.register_provider(rbf_vec_coeff_v)

    @property
    def metadata(self) -> dict[str, model.FieldMetaData]:
        return self._attrs

    @property
    def backend(self) -> gtx_backend.Backend:
        return self._backend

    @property
    def grid(self):
        return self._grid

    @property
    def vertical_grid(self):
        return None
