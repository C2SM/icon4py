import pathlib

import icon4py.model.common.states.factory as factory
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal
from icon4py.model.common.metrics import metric_fields as mf
from icon4py.model.common.test_utils import datatest_utils as dt_utils, serialbox_utils as sb

import gt4py.next as gtx
from icon4py.model.common.io import cf_utils
from icon4py.model.common.settings import xp
import math

from icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro import (
    HorizontalPressureDiscretizationType,
)

# we need to register a couple of fields from the serializer. Those should get replaced one by one.

dt_utils.TEST_DATA_ROOT = pathlib.Path(__file__).parent / "testdata"
properties = decomposition.get_processor_properties(decomposition.get_runtype(with_mpi=False))
path = dt_utils.get_datapath_for_experiment(dt_utils.get_ranked_data_path(dt_utils.SERIALIZED_DATA_PATH, properties))

data_provider = sb.IconSerialDataProvider(
        "icon_pydycore", str(path.absolute()), False, mpi_rank=properties.rank
    )

# z_ifc (computable from vertical grid for model without topography)
metrics_savepoint = data_provider.from_metrics_savepoint()

#interpolation fields also for now passing as precomputed fields
interpolation_savepoint = data_provider.from_interpolation_savepoint()
#can get geometry fields as pre computed fields from the grid_savepoint
grid_savepoint = data_provider.from_savepoint_grid()
#######

# start build up factory:

# used for vertical domain below: should go away once vertical grid provids start_index and end_index like interface
grid = grid_savepoint.global_grid_params

interface_model_height = metrics_savepoint.z_ifc()
c_lin_e = interpolation_savepoint.c_lin_e()
k_index = gtx.as_field((dims.KDim,), xp.arange(grid.num_levels + 1, dtype=gtx.int32))
vct_a = grid_savepoint.vct_a()
theta_ref_mc = metrics_savepoint.theta_ref_mc()
exner_ref_mc = metrics_savepoint.exner_ref_mc()
wgtfac_c = metrics_savepoint.wgtfac_c()

fields_factory = factory.FieldsFactory()

fields_factory.register_provider(
    factory.PrecomputedFieldsProvider(
        {
            "height_on_interface_levels": interface_model_height,
            "cell_to_edge_interpolation_coefficient": c_lin_e,
            cf_utils.INTERFACE_LEVEL_STANDARD_NAME: k_index,
            "vct_a": vct_a,
            "theta_ref_mc": theta_ref_mc,
            "exner_ref_mc": exner_ref_mc,
            "wgtfac_c": wgtfac_c
        }
    )
)


height_provider = factory.ProgramFieldProvider(
        func=mf.compute_z_mc,
        domain={
            dims.CellDim: (
                horizontal.HorizontalMarkerIndex.local(dims.CellDim),
                horizontal.HorizontalMarkerIndex.end(dims.CellDim),
            ),
            dims.KDim: (0, grid.num_levels),
        },
        fields={"z_mc": "height"},
        deps={"z_ifc": "height_on_interface_levels"},
    )


ddqz_z_full_and_inverse_provider = factory.ProgramFieldProvider(
    func=mf.compute_ddqz_z_full_and_inverse,
    deps={
        "z_ifc": "height_on_interface_levels",
    },
    domain={
        dims.CellDim: (
            horizontal.HorizontalMarkerIndex.local(dims.CellDim),
            horizontal.HorizontalMarkerIndex.end(dims.CellDim),
        ),
        dims.KDim: (0, grid.num_levels),
    },
    fields={"ddqz_z_full": "ddqz_z_full", "inv_ddqz_z_full": "inv_ddqz_z_full"},
)


compute_ddqz_z_half_provider = factory.ProgramFieldProvider(
    func=mf.compute_ddqz_z_half,
    deps={
        "z_ifc": "height_on_interface_levels",
        "z_mc": "height",
        "k_index": cf_utils.INTERFACE_LEVEL_STANDARD_NAME
    },
    domain={
        dims.CellDim: (
            horizontal.HorizontalMarkerIndex.local(dims.CellDim),
            horizontal.HorizontalMarkerIndex.end(dims.CellDim),
        ),
        dims.KDim: (0, grid.num_levels+1),
    },
    fields={"ddqz_z_half": "ddqz_z_half"},
    params={"nlev": grid.num_levels},
)


# TODO: this should include experiment param as in test_metric_fields
damping_height = 50000.0 if dt_utils.GLOBAL_EXPERIMENT else 12500.0
rayleigh_coeff = 0.1 if dt_utils.GLOBAL_EXPERIMENT else 5.0
vct_a_1 = grid_savepoint.vct_a().asnumpy()[0]

compute_rayleigh_w_provider = factory.ProgramFieldProvider(
    func=mf.compute_rayleigh_w,
    deps={
        "vct_a": "vct_a",
    },
    domain={
        dims.KDim: (0, grid_savepoint.nrdmax().item() + 1),
    },
    fields={"rayleigh_w": "rayleigh_w"},
    params={
        "damping_height": damping_height,
        "rayleigh_type": 2,
        "rayleigh_classic": constants.RayleighType.CLASSIC,
        "rayleigh_klemp": constants.RayleighType.KLEMP,
        "rayleigh_coeff": rayleigh_coeff,
        "vct_a_1": vct_a_1,
        "pi_const": math.pi},
)

compute_coeff_dwdz_provider = factory.ProgramFieldProvider(
    func=mf.compute_coeff_dwdz,
    deps={
        "ddqz_z_full": "ddqz_z_full",
        "z_ifc": "height_on_interface_levels",
    },
    domain={
        dims.CellDim: (
            horizontal.HorizontalMarkerIndex.local(dims.CellDim),
            horizontal.HorizontalMarkerIndex.end(dims.CellDim),
        ),
        dims.KDim: (1, grid.num_levels),
    },
    fields={"coeff1_dwdz_full": "coeff1_dwdz_full",
            "coeff2_dwdz_full": "coeff2_dwdz_full"},
)

compute_d2dexdz2_fac_mc_provider = factory.ProgramFieldProvider(
    func=mf.compute_d2dexdz2_fac_mc,
    deps={
        "theta_ref_mc": "theta_ref_mc",
        "inv_ddqz_z_full": "inv_ddqz_z_full",
        "exner_ref_mc": "exner_ref_mc",
        "z_mc": "height",
    },
    domain={
        dims.CellDim: (
            horizontal.HorizontalMarkerIndex.local(dims.CellDim),
            horizontal.HorizontalMarkerIndex.end(dims.CellDim),
        ),
        dims.KDim: (0, grid.num_levels),
    },
    fields={"d2dexdz2_fac1_mc": "d2dexdz2_fac1_mc",
            "d2dexdz2_fac2_mc": "d2dexdz2_fac2_mc"},
    params={
        "cpd": constants.CPD,
        "grav": constants.GRAV,
        "del_t_bg": constants.DEL_T_BG,
        "h_scal_bg": constants._H_SCAL_BG,
        "igradp_method": 3,
        "igradp_constant": HorizontalPressureDiscretizationType.TAYLOR_HYDRO,
    }
)


# # TODO: need to do compute_vwind_impl_wgt first
# compute_vwind_expl_wgt_provider = factory.ProgramFieldProvider(
#     func=mf.compute_vwind_expl_wgt,
#     deps={
#         "vwind_impl_wgt": "vwind_impl_wgt",
#     },
#     domain={
#         dims.CellDim: (
#             horizontal.HorizontalMarkerIndex.local(dims.CellDim),
#             horizontal.HorizontalMarkerIndex.end(dims.CellDim),
#         ),
#     },
#     fields={"vwind_expl_wgt": "vwind_expl_wgt"},
# )


compute_wgtfac_e_provider = factory.ProgramFieldProvider(
    func=mf.compute_wgtfac_e,
    deps={
        "wgtfac_c": "wgtfac_c",
        "c_lin_e": "cell_to_edge_interpolation_coefficient",
    },
    domain={
        dims.CellDim: (
            horizontal.HorizontalMarkerIndex.local(dims.EdgeDim),
            horizontal.HorizontalMarkerIndex.end(dims.EdgeDim), # TODO: check this bound
        ),
        dims.KDim: (0, grid.num_levels+1),
    },
    fields={"wgtfac_e": "wgtfac_e"},
)
