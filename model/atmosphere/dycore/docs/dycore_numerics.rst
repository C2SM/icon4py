Dycore numerical documentation
==============================

This page contains the numerical documentation for the dycore. Each "macro" step
(e.g. predictor, corrector) is broken down into its inner steps, which are then
described in detail.

The advection terms are computed separately in dycore/advection

.. toctree::
   :maxdepth: 2
   :caption: Dycore subcomponents:

   dycore_numerics_advection.rst

.. autofull:: icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro.SolveNonhydro.run_predictor_step
   :no-index:
