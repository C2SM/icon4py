# icon4py-atmosphere-physics-driver

The physics driver (`PhysicsDriver`) for ICON4Py, modelled on ICON-MPIM's AES physics interface (`aes_phy_main`).

`PhysicsDriver` and the components it runs (`Muphys`) are typed components; the driver runs each of these in a `PhysicsProcess`, under a `ProcessTimeControl`; see [ADR 0004](../../../../docs/development/ADRs/0004_typed_states_and_components.md).
