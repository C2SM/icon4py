from typing import Annotated

GAS_CONSTANT_DRY_AIR: Annotated[float, "gas constant for dry air [J/K/kg], called 'rd' in ICON (mo_physical_constants.f90), https://glossary.ametsoc.org/wiki/Gas_constant"] = 287.04
CPD: Annotated[float, "specific heat at constant pressure [J/K/kg]"] = 1004.64
GAS_CONSTANT_WATER_VAPOR: Annotated[float, "gas constant for water vapor [J/K/kg], rv in Icon"] = 461.51

