from dataclasses import dataclass

from functional.common import Field

from icon4py.common.dimension import KDim, CellDim


@dataclass
class DiagnosticState:
    # fields for 3D elements in turbdiff
    hdef_ic: Field[[CellDim, KDim], float] #! divergence at half levels(nproma,nlevp1,nblks_c)     [1/s]
    div_ic = Field[[CellDim, KDim], float] #! horizontal wind field deformation (nproma,nlevp1,nblks_c)     [1/s^2]
    dwdx = None #zonal gradient of vertical wind speed (nproma,nlevp1,nblks_c)     [1/s]
    dwdy = None #meridional gradient of vertical wind speed (nproma,nlevp1,nblks_c)
