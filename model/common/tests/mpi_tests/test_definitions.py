from icon4py.model.common.decomposition.definitions import create_exchange, SingleNodeExchange, \
    DecompositionInfo
def test_create_single_node_runtime_without_mpi(processor_props):
    decomposition_info = DecompositionInfo(klevels=10)
    exchange = create_exchange(processor_props, decomposition_info)

    assert isinstance(exchange, SingleNodeExchange)
