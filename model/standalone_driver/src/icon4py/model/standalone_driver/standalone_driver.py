# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import functools
import logging
import pathlib
from collections.abc import Callable

import gt4py.next as gtx
from gt4py.next import config as gtx_config, metrics as gtx_metrics

import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as solve_nh
from icon4py.model.common import dimension as dims, model_backends, model_options, type_alias as ta
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import grid_manager as gm, vertical as v_grid
from icon4py.model.common.initialization import jablonowski_williamson_topography
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc, device_utils
from icon4py.model.standalone_driver import config as driver_config, driver_states, driver_utils


log = logging.getLogger(__name__)


class Icon4pyDriver:
    def __init__(
        self,
        config: driver_config.DriverConfig,
        backend: model_backends.BackendLike,
        grid_manager: gm.GridManager,
        static_field_factories: driver_states.StaticFieldFactories,
        diffusion_granule: diffusion.Diffusion,
        solve_nonhydro_granule: solve_nh.SolveNonhydro,
    ):
        self.config = config
        self.backend = backend
        self.grid_manager = grid_manager
        self.static_field_factories = static_field_factories
        self.diffusion = diffusion_granule
        self.solve_nonhydro = solve_nonhydro_granule
        self.model_time_var = driver_states.ModelTimeVariables(config=config)
        self.timer_collection = driver_states.TimerCollection(
            [timer.value for timer in driver_states.DriverTimers]
        )

        driver_utils.display_driver_setup_in_log_file(
            self.model_time_var.n_time_steps,
            self.solve_nonhydro._vertical_params,
            self.config,
        )

    @functools.cached_property
    def _allocator(self):
        return model_backends.get_allocator(self.backend)

    @functools.cached_property
    def _xp(self):
        return data_alloc.import_array_ns(self._allocator)

    @functools.cached_property
    def _concrete_backend(self):
        return model_options.customize_backend(program=None, backend=self.backend)

    def _is_last_substep(self, step_nr: int) -> bool:
        return step_nr == (self.model_time_var.ndyn_substeps_var - 1)

    @staticmethod
    def _is_first_substep(step_nr: int) -> bool:
        return step_nr == 0

    def _full_name(self, func: Callable) -> str:
        return f"{self.__class__.__name__}:{func.__name__}"

    def time_integration(
        self,
        ds: driver_states.DriverStates,
        do_prep_adv: bool,
    ) -> None:
        diffusion_diagnostic_state = ds.diffusion_diagnostic
        solve_nonhydro_diagnostic_state = ds.solve_nonhydro_diagnostic
        prognostic_states = ds.prognostics
        prep_adv = ds.prep_advection_prognostic

        log.debug(
            f"starting time loop for dtime = {self.model_time_var.dtime_in_seconds} s, substep_timestep = {self.model_time_var.substep_timestep} s, n_timesteps = {self.model_time_var.n_time_steps}, substep_timestep = {self.model_time_var.substep_timestep}"
        )

        # TODO(OngChia): Initialize vn tendencies that are used in solve_nh and advection to zero (init_ddt_vn_diagnostics subroutine)

        actual_starting_time = datetime.datetime.now()

        for time_step in range(self.model_time_var.n_time_steps):
            if self.config.profiling_stats is not None:
                if not self.config.profiling_stats.skip_first_timestep or time_step > 0:
                    gtx_config.COLLECT_METRICS_LEVEL = (
                        self.config.profiling_stats.gt4py_metrics_level
                    )

            log.info(
                f"\n"
                f"simulation date : {self.model_time_var.simulation_date}, at run timestep : {time_step}, with actual time spent: {(datetime.datetime.now() - actual_starting_time).total_seconds()}"
                f"\n"
            )

            self.model_time_var.next_simulation_date()

            self._integrate_one_time_step(
                diffusion_diagnostic_state,
                solve_nonhydro_diagnostic_state,
                prognostic_states,
                prep_adv,
                do_prep_adv,
            )
            device_utils.sync(self._concrete_backend)

            self.model_time_var.is_first_step_in_simulation = False

            self._adjust_ndyn_substeps_var(solve_nonhydro_diagnostic_state)

            # TODO(OngChia): simple IO enough for JW test

        self._compute_mean_at_final_time_step(prognostic_states.current)

        self.timer_collection.show_timer_report()
        if (
            self.config.profiling_stats is not None
            and self.config.profiling_stats.gt4py_metrics_level > gtx_metrics.DISABLED
        ):
            print(gtx_metrics.dumps())
            gtx_metrics.dump_json(self.config.profiling_stats.gt4py_metrics_output_file)

    def _integrate_one_time_step(
        self,
        diffusion_diagnostic_state: diffusion_states.DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        prep_adv: dycore_states.PrepAdvection,
        do_prep_adv: bool,
    ):
        log.debug(f"Running {self.solve_nonhydro.__class__}")
        self._do_dyn_substepping(
            solve_nonhydro_diagnostic_state,
            prognostic_states,
            prep_adv,
            do_prep_adv,
        )

        if self.diffusion.config.apply_to_horizontal_wind:
            log.debug(f"Running {self.diffusion.__class__}")
            timer_diffusion = (
                self.timer_collection.timers[driver_states.DriverTimers.diffusion_first_step.value]
                if self.model_time_var.is_first_step_in_simulation
                else self.timer_collection.timers[driver_states.DriverTimers.diffusion.value]
            )
            timer_diffusion.start()
            self.diffusion.run(
                diffusion_diagnostic_state,
                prognostic_states.next,
                self.model_time_var.dtime_in_seconds,
            )
            timer_diffusion.capture()

        prognostic_states.swap()

    def _update_time_levels_for_velocity_tendencies(
        self,
        diagnostic_state_nh: dycore_states.DiagnosticStateNonHydro,
        at_first_substep: bool,
        at_initial_timestep: bool,
    ):
        """
        Set time levels of advective tendency fields for call to velocity_tendencies.

        When using `TimeSteppingScheme.MOST_EFFICIENT` (itime_scheme=4 in ICON Fortran),
        `vertical_wind_advective_tendency.predictor` (advection term in vertical momentum equation in
        predictor step) is not computed in the predictor step of each substep.
        Instead, the advection term computed in the corrector step during the
        previous substep is reused for efficiency (except, of course, in the
        very first substep of the initial time step).
        `normal_wind_advective_tendency.predictor` (advection term in horizontal momentum equation in
        predictor step) is only computed in the predictor step of the first
        substep and the advection term in the corrector step during the previous
        substep is reused for `normal_wind_advective_tendency.predictor` from the second substep onwards.
        Additionally, in this scheme the predictor and corrector outputs are kept
        in separate elements of the pair (.predictor for the predictor step and
        .corrector for the corrector step) and interpoolated at the end of the
        corrector step to get the final output.

        No other time stepping schemes are currently supported.

        Args:
            diagnostic_state_nh: Diagnostic fields calculated in the dynamical core (SolveNonHydro)
            at_first_substep: Flag indicating if this is the first substep of the time step.
            at_initial_timestep: Flag indicating if this is the first time step.

        Returns:
            The index of the pair element to be used for the corrector output.
        """
        if not (at_initial_timestep and at_first_substep):
            diagnostic_state_nh.vertical_wind_advective_tendency.swap()
        if not at_first_substep:
            diagnostic_state_nh.normal_wind_advective_tendency.swap()

    def _do_dyn_substepping(
        self,
        solve_nonhydro_diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        prep_adv: dycore_states.PrepAdvection,
        do_prep_adv: bool,
    ):
        # TODO(OngChia): compute airmass for prognostic_state here

        timer_solve_nh = (
            self.timer_collection.timers[driver_states.DriverTimers.solve_nh_first_step.value]
            if self.model_time_var.is_first_step_in_simulation
            else self.timer_collection.timers[driver_states.DriverTimers.solve_nh.value]
        )
        for dyn_substep in range(self.model_time_var.ndyn_substeps_var):
            self._compute_statistics(dyn_substep, prognostic_states.current)

            self._update_time_levels_for_velocity_tendencies(
                solve_nonhydro_diagnostic_state,
                at_first_substep=self._is_first_substep(dyn_substep),
                at_initial_timestep=self.model_time_var.is_first_step_in_simulation,
            )

            timer_solve_nh.start()
            self.solve_nonhydro.time_step(
                solve_nonhydro_diagnostic_state,
                prognostic_states,
                prep_adv=prep_adv,
                second_order_divdamp_factor=self._update_spinup_second_order_divergence_damping(),
                dtime=self.model_time_var.substep_timestep,
                ndyn_substeps_var=self.model_time_var.ndyn_substeps_var,
                at_initial_timestep=self.model_time_var.is_first_step_in_simulation,
                lprep_adv=do_prep_adv,
                at_first_substep=self._is_first_substep(dyn_substep),
                at_last_substep=self._is_last_substep(dyn_substep),
            )
            timer_solve_nh.capture()

            if not self._is_last_substep(dyn_substep):
                prognostic_states.swap()
        # self._compute_total_mass_and_energy(prognostic_states.next)

        # TODO(OngChia): compute airmass for prognostic_state here

    # watch_mode is true if step is <= 1 or cfl already near or exceeding threshold.
    # omit spinup feature and the option that if the model starts from IFS or COSMO data
    # horizontal cfl is not ported
    def _adjust_ndyn_substeps_var(
        self,
        solve_nonhydro_diagnostic_state: dycore_states.DiagnosticStateNonHydro,
    ) -> None:
        # TODO (Chia Rui): perform a global max operation in multinode run
        global_max_vertical_cfl = solve_nonhydro_diagnostic_state.max_vertical_cfl[()]

        if (
            global_max_vertical_cfl > ta.wpfloat("0.81") * self.config.vertical_cfl_threshold
            and not self.model_time_var.cfl_watch_mode
        ):
            log.warning(
                "High CFL number for vertical advection in dynamical core, entering watch mode"
            )
            self.model_time_var.cfl_watch_mode = True

        if self.model_time_var.cfl_watch_mode:
            substep_fraction = ta.wpfloat(
                self.model_time_var.ndyn_substeps_var / self.config.ndyn_substeps
            )
            if (
                global_max_vertical_cfl * substep_fraction
                > ta.wpfloat("0.9") * self.config.vertical_cfl_threshold
            ):
                log.warning(
                    f"Maximum vertical CFL number {global_max_vertical_cfl} is close to critical threshold"
                )

            vertical_cfl_threshold_for_increment = self.config.vertical_cfl_threshold
            vertical_cfl_threshold_for_decrement = (
                ta.wpfloat("0.9") * self.config.vertical_cfl_threshold
            )

            if global_max_vertical_cfl > vertical_cfl_threshold_for_increment:
                if self._xp.isfinite(global_max_vertical_cfl):
                    ndyn_substeps_increment = max(
                        1,
                        round(
                            self.model_time_var.ndyn_substeps_var
                            * (global_max_vertical_cfl - vertical_cfl_threshold_for_increment)
                            / vertical_cfl_threshold_for_increment
                        ),
                    )
                    new_ndyn_substeps_var = min(
                        self.model_time_var.ndyn_substeps_var + ndyn_substeps_increment,
                        self.model_time_var.max_ndyn_substeps,
                    )
                else:
                    log.warning(
                        f"WARNING: max cfl {global_max_vertical_cfl} is not a number! Number of substeps is set to the max value! "
                    )
                    new_ndyn_substeps_var = self.model_time_var.max_ndyn_substeps
                self.model_time_var.update_ndyn_substeps(new_ndyn_substeps_var)
                # TODO (Chia Rui): check if we need to set ndyn_substeps_var in advection_config as in ICON when tracer advection is implemented
                log.warning(
                    f"The number of dynamics substeps is increased to {self.model_time_var.ndyn_substeps_var}"
                )
            if (
                self.model_time_var.ndyn_substeps_var > self.config.ndyn_substeps
                and global_max_vertical_cfl
                * ta.wpfloat(
                    self.model_time_var.ndyn_substeps_var
                    / (self.model_time_var.ndyn_substeps_var - 1)
                )
                < vertical_cfl_threshold_for_decrement
            ):
                self.model_time_var.update_ndyn_substeps(self.model_time_var.ndyn_substeps_var - 1)
                # TODO (Chia Rui): check if we need to set ndyn_substeps_var in advection_config as in ICON when tracer advection is implemented
                log.warning(
                    f"The number of dynamics substeps is decreased to {self.model_time_var.ndyn_substeps_var}"
                )

                if (
                    self.model_time_var.ndyn_substeps_var == self.config.ndyn_substeps
                    and global_max_vertical_cfl
                    < ta.wpfloat("0.76") * self.config.vertical_cfl_threshold
                ):
                    log.warning(
                        "CFL number for vertical advection in dynamical core has decreased, leaving watch mode"
                    )
                    self.model_time_var.cfl_watch_mode = False

        # reset max_vertical_cfl to zero
        solve_nonhydro_diagnostic_state.max_vertical_cfl = data_alloc.scalar_like_array(
            0.0, self._allocator
        )

    def _update_spinup_second_order_divergence_damping(self) -> ta.wpfloat:
        if self.config.apply_extra_second_order_divdamp:
            fourth_order_divdamp_factor = self.solve_nonhydro._config.fourth_order_divdamp_factor
            if self.model_time_var.elapse_time_in_seconds <= ta.wpfloat("1800.0"):
                return ta.wpfloat("0.8") * fourth_order_divdamp_factor
            elif self.model_time_var.elapse_time_in_seconds <= ta.wpfloat("7200.0"):
                return (
                    ta.wpfloat("0.8")
                    * fourth_order_divdamp_factor
                    * (self.model_time_var.elapse_time_in_seconds - ta.wpfloat("1800.0"))
                    / ta.wpfloat("5400.0")
                )
            else:
                return ta.wpfloat("0.0")
        else:
            return ta.wpfloat("0.0")

    def _compute_statistics(
        self, current_dyn_substep: int, prognostic_states: prognostics.PrognosticState
    ) -> None:
        """
        Compute relevant statistics of prognostic variables at the beginning of every time step. The statistics include:
        absolute maximum value of rho, vn, and w, as well as the levels at which their maximum value is found.
        """
        if self.config.enable_statistics_output:
            # TODO (Chia Rui): Do global max when multinode is ready
            rho_arg_max, max_rho = driver_utils.find_maximum_from_field(
                prognostic_states.rho,
                self._xp,
            )
            vn_arg_max, max_vn = driver_utils.find_maximum_from_field(
                prognostic_states.vn,
                self._xp,
            )
            w_arg_max, max_w = driver_utils.find_maximum_from_field(prognostic_states.w, self._xp)

            def _determine_sign(input_number: float) -> str:
                return " " if input_number >= 0.0 else "-"

            rho_sign = _determine_sign(max_rho)
            vn_sign = _determine_sign(max_vn)
            w_sign = _determine_sign(max_w)

            log.info(
                f"substep / n_substeps : {current_dyn_substep:3d} / {self.model_time_var.ndyn_substeps_var:3d} == "
                f"MAX RHO: {rho_sign}{abs(max_rho):.5e} at lvl {rho_arg_max[1]:4d}, MAX VN: {vn_sign}{abs(max_vn):.5e} at lvl {vn_arg_max[1]:4d}, MAX W: {w_sign}{abs(max_w):.5e} at lvl {w_arg_max[1]:4d}"
            )
        else:
            log.info(
                f"substep / n_substeps : {current_dyn_substep:3d} / {self.model_time_var.ndyn_substeps_var:3d}"
            )

    # def _compute_total_mass_and_energy(
    #     self, prognostic_states: prognostics.PrognosticState
    # ) -> None:
    #     if self.config.enable_statistics_output:
    #         rho_ndarray = prognostic_states.rho.ndarray
    #         cell_area_ndarray = self.grid_manager.geometry_fields[
    #             gridfile.GeometryName.CELL_AREA.value
    #         ].ndarray
    #         cell_thickness_ndarray = self.static_field_factories.metrics_field_source.get(
    #             metrics_attr.DDQZ_Z_FULL
    #         ).ndarray
    #         total_mass = self._xp.sum(rho_ndarray * cell_area_ndarray * cell_thickness_ndarray)
    #         self._logger.info(f"TOTAL MASS: {total_mass:.10e}, TOTAL ENERGY")

    def _compute_mean_at_final_time_step(
        self, prognostic_states: prognostics.PrognosticState
    ) -> None:
        if self.config.enable_statistics_output:
            rho_ndarray = prognostic_states.rho.ndarray
            vn_ndarray = prognostic_states.vn.ndarray
            w_ndarray = prognostic_states.w.ndarray
            theta_v_ndarray = prognostic_states.theta_v.ndarray
            exner_ndarray = prognostic_states.exner.ndarray
            interface_physical_height_ndarray = (
                self.solve_nonhydro._vertical_params.interface_physical_height.ndarray
            )
            log.info("")
            log.info(
                "Global mean of    rho         vn           w          theta_v     exner      at model levels:"
            )
            for k in range(rho_ndarray.shape[1]):
                log.info(
                    f"{interface_physical_height_ndarray[k]:12.3f}: {self._xp.mean(rho_ndarray[:, k]):.5e} "
                    f"{self._xp.mean(vn_ndarray[:, k]):.5e} "
                    f"{self._xp.mean(w_ndarray[:, k+1]):.5e} "
                    f"{self._xp.mean(theta_v_ndarray[:, k]):.5e} "
                    f"{self._xp.mean(exner_ndarray[:, k]):.5e} "
                )


# TODO (Chia Rui): this should be replaced by real configuration reader when the configuration PR is merged
def _read_config(
    output_path: pathlib.Path,
    enable_profiling: bool,
) -> tuple[
    driver_config.DriverConfig,
    v_grid.VerticalGridConfig,
    diffusion.DiffusionConfig,
    solve_nh.NonHydrostaticConfig,
]:
    vertical_grid_config = v_grid.VerticalGridConfig(
        num_levels=35,
        rayleigh_damping_height=45000.0,
    )

    diffusion_config = diffusion.DiffusionConfig(
        diffusion_type=diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER,
        hdiff_w=True,
        hdiff_vn=True,
        hdiff_temp=False,
        n_substeps=5,
        type_t_diffu=2,
        type_vn_diffu=1,
        hdiff_efdt_ratio=10.0,
        hdiff_w_efdt_ratio=15.0,
        smagorinski_scaling_factor=0.025,
        zdiffu_t=False,
        velocity_boundary_diffusion_denom=200.0,
    )

    nonhydro_config = solve_nh.NonHydrostaticConfig(
        fourth_order_divdamp_factor=0.0025,
    )

    profiling_stats = driver_config.ProfilingStats() if enable_profiling else None

    icon4py_driver_config = driver_config.DriverConfig(
        experiment_name="Jablonowski_Williamson",
        output_path=output_path,
        dtime=datetime.timedelta(seconds=300.0),
        end_date=datetime.datetime(1, 1, 1, 1, 0, 0),
        apply_extra_second_order_divdamp=False,
        ndyn_substeps=5,
        vertical_cfl_threshold=ta.wpfloat("0.85"),
        enable_statistics_output=True,
        profiling_stats=profiling_stats,
    )

    return (
        icon4py_driver_config,
        vertical_grid_config,
        diffusion_config,
        nonhydro_config,
    )


def initialize_driver(
    configuration_file_path: pathlib.Path,
    output_path: pathlib.Path,
    grid_file_path: pathlib.Path,
    log_level: str,
    backend_name: str,
) -> Icon4pyDriver:
    """
    Initialize the driver:
    - load the configuration
    - load the grid manager and decomposition info
    - load the topography (eventually all external parameters)
    - create the static field factories
    - initialize the components selected by the configuration (diffusion and solve_nh)
    - create the driver object
    Args:
        configuration_file_path: path to the configuration file
        output_path: path where to store the simulation output
        grid_file_path: path of the grid file
        log_level: logging level
        backend: GT4Py backend-like
    Returns:
        Driver: driver object
    """

    parallel_props = decomposition_defs.get_processor_properties(
        decomposition_defs.get_runtype(with_mpi=False)
    )
    driver_utils.configure_logging(
        logging_level=log_level,
        processor_procs=parallel_props,
    )

    configuration_file_path = pathlib.Path(configuration_file_path)
    grid_file_path = pathlib.Path(grid_file_path)
    if pathlib.Path(output_path).exists():
        current_time = datetime.datetime.now()
        log.warning(f"output path {output_path} already exists, a time stamp will be added")
        output_path = pathlib.Path(
            output_path
            + f"_{datetime.date.today()}_{current_time.hour}h_{current_time.minute}m_{current_time.second}s"
        )
    else:
        output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=False)

    backend = driver_utils.get_backend_from_name(backend_name)

    allocator = model_backends.get_allocator(backend)

    log.info("Initializing the driver")

    driver_config, vertical_grid_config, diffusion_config, solve_nh_config = _read_config(
        output_path=output_path,
        enable_profiling=False,
    )

    log.info(f"initializing the grid manager from '{grid_file_path}'")
    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=vertical_grid_config,
        allocator=allocator,
    )

    log.info("creating the decomposition info")

    decomposition_info = driver_utils.create_decomposition_info(
        grid_manager=grid_manager,
        allocator=allocator,
    )
    exchange = decomposition_defs.create_exchange(parallel_props, decomposition_info)

    log.info("initializing the vertical grid")
    vertical_grid = driver_utils.create_vertical_grid(
        vertical_grid_config=vertical_grid_config,
        allocator=allocator,
    )

    log.info("initializing the JW topography")
    cell_topography = jablonowski_williamson_topography.jablonowski_williamson_topography(
        cell_lat=grid_manager.coordinates[dims.CellDim]["lat"].ndarray,
        u0=35.0,
        array_ns=data_alloc.import_array_ns(allocator=allocator),
    )

    log.info("initializing the static-field factories")
    static_field_factories = driver_utils.create_static_field_factories(
        grid_manager=grid_manager,
        decomposition_info=decomposition_info,
        vertical_grid=vertical_grid,
        cell_topography=gtx.as_field((dims.CellDim,), data=cell_topography, allocator=allocator),
        backend=backend,
    )

    log.info("initializing granules")
    (
        diffusion_granule,
        solve_nonhydro_granule,
    ) = driver_utils.initialize_granules(
        grid=grid_manager.grid,
        vertical_grid=vertical_grid,
        diffusion_config=diffusion_config,
        solve_nh_config=solve_nh_config,
        static_field_factories=static_field_factories,
        exchange=exchange,
        owner_mask=gtx.as_field(
            (dims.CellDim,),
            decomposition_info.owner_mask(dims.CellDim),
            allocator=allocator,  # type: ignore[arg-type]
        ),
        backend=backend,
    )

    icon4py_driver = Icon4pyDriver(
        config=driver_config,
        backend=backend,
        grid_manager=grid_manager,
        static_field_factories=static_field_factories,
        diffusion_granule=diffusion_granule,
        solve_nonhydro_granule=solve_nonhydro_granule,
    )

    return icon4py_driver
