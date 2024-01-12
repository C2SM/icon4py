from icon4py.model.common.test_utils.data_handling import download_and_extract
from icon4py.model.common.test_utils.datatest_utils import TEST_DATA_ROOT

MCH_GRID_FILE = "mch_ch_r04b09_dsl"
R02B04_GLOBAL = "r02b04_global"

grids_path = TEST_DATA_ROOT.joinpath("grids")
r04b09_dsl_grid_path = grids_path.joinpath(MCH_GRID_FILE)
r04b09_dsl_data_file = r04b09_dsl_grid_path.joinpath("mch_ch_r04b09_dsl_grids_v1.tar.gz").name
r02b04_global_grid_path = grids_path.joinpath(R02B04_GLOBAL)
r02b04_global_data_file = r02b04_global_grid_path.joinpath("icon_grid_0013_R02B04_R.tar.gz").name


mch_ch_r04b09_dsl_grid_uri = "https://polybox.ethz.ch/index.php/s/hD232znfEPBh4Oh/download"
r02b04_global_grid_uri = "https://polybox.ethz.ch/index.php/s/AKAO6ImQdIatnkB/download"


def resolve_file_from_gridfile_name(name: str):
    if name == MCH_GRID_FILE:
        if not r04b09_dsl_grid_path.joinpath("grid.nc").exists():
            download_and_extract(
                mch_ch_r04b09_dsl_grid_uri,
                r04b09_dsl_grid_path,
                r04b09_dsl_grid_path,
                r04b09_dsl_data_file,
            )
        return r04b09_dsl_grid_path.joinpath("grid.nc")
    elif name == R02B04_GLOBAL:
        if not r02b04_global_grid_path.joinpath("icon_grid_0013_R02B04_R.nc").exists():
            download_and_extract(
                r02b04_global_grid_uri,
                r02b04_global_grid_path,
                r02b04_global_grid_path,
                r02b04_global_data_file,
            )
        return r02b04_global_grid_path.joinpath("icon_grid_0013_R02B04_R.nc")
