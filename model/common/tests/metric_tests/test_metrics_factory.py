
import icon4py.model.common.settings as settings
from icon4py.model.common.metrics import metrics_factory as mf

from icon4py.model.common.states import factory as states_factory
from icon4py.model.common.io import cf_utils
import icon4py.model.common.test_utils.helpers as helpers

def test_factory(icon_grid, metrics_savepoint):

    factory = mf.fields_factory
    factory.with_grid(icon_grid).with_allocator(settings.backend)
    factory.get("height_on_interface_levels", states_factory.RetrievalType.FIELD)
    factory.get("height", states_factory.RetrievalType.FIELD)
    factory.get(cf_utils.INTERFACE_LEVEL_STANDARD_NAME, states_factory.RetrievalType.FIELD)

    inv_ddqz_full_ref = metrics_savepoint.inv_ddqz_z_full()
    factory.register_provider(mf.ddqz_z_full_and_inverse_provider)
    inv_ddqz_z_full = factory.get(
        "inv_ddqz_z_full", states_factory.RetrievalType.FIELD
    )
    assert helpers.dallclose(inv_ddqz_z_full.asnumpy(), inv_ddqz_full_ref.asnumpy())

    ddq_z_half_ref = metrics_savepoint.ddqz_z_half()
    factory.register_provider(mf.compute_ddqz_z_half_provider)
    ddqz_z_half_full = factory.get(
        "ddqz_z_half", states_factory.RetrievalType.FIELD
    )
    assert helpers.dallclose(ddqz_z_half_full.asnumpy(), ddq_z_half_ref.asnumpy())
