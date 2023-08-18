import pytest
from icon4py.model.atmosphere.diffusion.diffusion import DiffusionConfig, DiffusionType

from icon4py.model.common.test_utils.fixtures import (  # noqa F401
    interpolation_savepoint,
    metrics_savepoint,
    damping_height,
    data_provider,
    datapath,
    grid_savepoint,
    icon_grid,
    setup_icon_data,
    linit,
    ndyn_substeps,
    step_date_exit,
    step_date_init,
    mesh,
    backend
)
@pytest.fixture
def r04b09_diffusion_config(ndyn_substeps) -> DiffusionConfig:
    """
    Create DiffusionConfig matching MCH_CH_r04b09_dsl.

    Set values to the ones used in the  MCH_CH_r04b09_dsl experiment where they differ
    from the default.
    """
    return DiffusionConfig(
        diffusion_type=DiffusionType.SMAGORINSKY_4TH_ORDER,
        hdiff_w=True,
        hdiff_vn=True,
        type_t_diffu=2,
        type_vn_diffu=1,
        hdiff_efdt_ratio=24.0,
        hdiff_w_efdt_ratio=15.0,
        smagorinski_scaling_factor=0.025,
        zdiffu_t=True,
        velocity_boundary_diffusion_denom=150.0,
        max_nudging_coeff=0.075,
        n_substeps=ndyn_substeps,
    )


@pytest.fixture
def diffusion_savepoint_init(data_provider, linit, step_date_init):  # noqa F811
    """
    Load data from ICON savepoint at start of diffusion module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_date_init'
    fixture, passing 'step_date_init=<iso_string>'

    linit flag can be set by overriding the 'linit' fixture
    """
    return data_provider.from_savepoint_diffusion_init(linit=linit, date=step_date_init)


@pytest.fixture
def diffusion_savepoint_exit(data_provider, linit, step_date_exit):  # noqa F811
    """
    Load data from ICON savepoint at exist of diffusion module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    sp = data_provider.from_savepoint_diffusion_exit(linit=linit, date=step_date_exit)
    return sp
