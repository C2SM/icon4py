from dataclasses import dataclass

from functional.common import Field

from icon4py.common.dimension import KDim, EdgeDim, CellDim, C2E2CDim


@dataclass
class MetricState:
    theta_ref_mc: Field[[CellDim, KDim], float]
    enhfac_diffu: Field[[KDim], float]  #Enhancement factor for nabla4 background diffusion TODO check dimension
    wgtfac_e: Field[[EdgeDim, KDim], float] #weighting factor for interpolation from full to half levels (nproma,nlevp1,nblks_e)
    wgtfac_c: Field[[CellDim, KDim], float]  #weighting factor for interpolation from full to half levels (nproma,nlevp1,nblks_c)
    wgtfacq1_e: Field[[EdgeDim, ], float]  #weighting factor for quadratic interpolation to model top (nproma,3,nblks_e)
    wgtfacq_e: Field[[EdgeDim, ], float]  #weighting factor for quadratic interpolation to surface (nproma,3,nblks_e)
    ddqz_z_full_e:  Field[[EdgeDim, KDim], float] #functional determinant of the metrics [sqrt(gamma)] (nproma,nlev,nblks_e)
    mask_hdiff: Field[[CellDim, KDim], int]
    zd_vertidx_dsl: Field[[CellDim,  C2E2CDim, KDim], int]
    zd_diffcoef: Field[[CellDim, KDim], float]
    zd_intcoef: Field[[CellDim, KDim], float]




