
class NonhydroStepping:
    # def __init__(self, run_program=True):

    def __init__(self):


    def init(
        self,
        grid: IconGrid,
        config: NonHydrostaticConfig,
        params: NonHydrostaticParams,
        metric_state: MetricState,
        metric_state_nonhydro: MetricStateNonHydro,
        interpolation_state: InterpolationState,
        vertical_params: VerticalModelParams,
        a_vec: Field[[KDim], float],
        enh_smag_fac: Field[[KDim], float],
        fac: tuple,
        z: tuple,
    ):
        """
        Initialize NonHydrostatic granule with configuration.

        calculates all local fields that are used in nh_solve within the time loop
        """


    @property
    def initialized(self):
        return self._initialized

    def _allocate_local_fields(self):

    def perform_nh_stepping(self):

    def _perform_nh_timeloop(self):

    def _integrate(self):


    def _perform_dyn_substepping(self):

        for nstep=1,ndyn_substeps_var:

            if nstep == 1:
                lclean_mflx=True
            lrecompute = lclean_mflx


            SolveNonhydro.time_step()

            diffusion.time_step()


        compute_airmass()


