from icon4py.diffusion.icon_grid import IconGrid
from icon4py.testutils import serialbox_utils


def read_icon_grid(path=".", type = "serialbox")->IconGrid:

    if type == "serialbox":
        return serialbox_utils.IconSerialDataProvider("icon_pydycore", path, False).from_savepoint_grid().to_icon_grid()
