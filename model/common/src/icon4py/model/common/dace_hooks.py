# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence

import dace
from dace import nodes as dace_nodes, sdfg as dace_sdfg, symbolic as dace_sym
from gt4py.next import config as gtx_config
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


_graupel_scan_inout_copies = {
    "if_stmt_92": ["__ct_el_24", "__ct_el_25", "__ct_el_26", "__ct_el_27"]
}


def _cleanup_local_self_update(
    scan_sdfg: dace.SDFG,
    if_stmt_node: dace.sdfg.state.ConditionalBlock,
    if_stmt_conn: str,
    compute_src_node: dace_nodes.AccessNode,
    compute_dst_node: dace_nodes.AccessNode,
    update_src_node: dace_nodes.AccessNode,
    update_dst_node: dace_nodes.AccessNode,
    scan_compute_st: dace.SDFGState,
    scan_update_st: dace.SDFGState,
) -> None:
    assert isinstance(compute_dst_node.desc(scan_sdfg), dace.data.Scalar)
    assert compute_dst_node.desc(scan_sdfg) == compute_src_node.desc(scan_sdfg)
    assert compute_dst_node.desc(scan_sdfg) == update_dst_node.desc(scan_sdfg)

    # reroute the write edge in the compute state
    new_compute_dst_node = scan_compute_st.add_access(compute_src_node.data)
    scan_compute_st.add_edge(
        if_stmt_node,
        if_stmt_conn,
        new_compute_dst_node,
        None,
        dace.Memlet(data=new_compute_dst_node.data, subset="0"),
    )
    for edge in scan_compute_st.out_edges(compute_dst_node):
        scan_compute_st.add_edge(
            new_compute_dst_node,
            None,
            edge.dst,
            edge.dst_conn,
            dace.Memlet(
                data=new_compute_dst_node.data,
                subset=edge.data.get_src_subset(edge, scan_compute_st),
                other_subset=edge.data.get_dst_subset(edge, scan_compute_st),
            ),
        )
    scan_compute_st.remove_node(compute_dst_node)

    # reroute the write edge in the update state
    scan_update_st.add_nedge(
        scan_update_st.add_access(compute_src_node.data),
        update_dst_node,
        dace.Memlet(
            data=compute_src_node.data,
            subset="0",
            other_subset="0",
        ),
    )
    scan_update_st.remove_node(update_src_node)
    print(
        f"Removed self-copy in {if_stmt_node.label}: {compute_src_node.data} -> {compute_dst_node.data}"
    )


def _replace_scan_input(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    old_node: dace_nodes.AccessNode,
    new_node: dace_nodes.AccessNode,
    new_node_offsets: Sequence[dace_sym.SymbolicType],
) -> None:
    reconfigured_neighbour: set[tuple[dace_nodes.Node, str | None]] = set()

    for producer_edge in list(state.in_edges(old_node)):
        producer: dace_nodes.Node = producer_edge.src
        producer_conn = producer_edge.src_conn
        new_producer_edge = gtx_transformations.utils.reroute_edge(
            is_producer_edge=True,
            current_edge=producer_edge,
            ss_offset=new_node_offsets,
            state=state,
            sdfg=sdfg,
            old_node=old_node,
            new_node=new_node,
        )
        if (producer, producer_conn) not in reconfigured_neighbour:
            gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
                is_producer_edge=True,
                new_edge=new_producer_edge,
                sdfg=sdfg,
                state=state,
                ss_offset=new_node_offsets,
                old_node=old_node,
                new_node=new_node,
            )
            reconfigured_neighbour.add((producer, producer_conn))

    for consumer_edge in list(state.out_edges(old_node)):
        consumer: dace_nodes.Node = consumer_edge.dst
        consumer_conn = consumer_edge.dst_conn
        new_consumer_edge = gtx_transformations.utils.reroute_edge(
            is_producer_edge=False,
            current_edge=consumer_edge,
            ss_offset=new_node_offsets,
            state=state,
            sdfg=sdfg,
            old_node=old_node,
            new_node=new_node,
        )
        if (consumer, consumer_conn) not in reconfigured_neighbour:
            gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
                is_producer_edge=False,
                new_edge=new_consumer_edge,
                sdfg=sdfg,
                state=state,
                ss_offset=new_node_offsets,
                old_node=old_node,
                new_node=new_node,
            )
            reconfigured_neighbour.add((consumer, consumer_conn))

    state.remove_node(old_node)
    sdfg.remove_data(old_node.data, validate=gtx_config.DEBUG)

    gtx_transformations.gt_propagate_strides_from_access_node(
        sdfg=sdfg,
        state=state,
        outer_node=new_node,
    )


def _cleanup_global_self_update(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    scan_node: dace_nodes.NestedSDFG,
    if_stmt_node: dace_nodes.NestedSDFG,
    if_stmt_output: str,
    compute_src_node: dace_nodes.AccessNode,
    compute_dst_node: dace_nodes.AccessNode,
    update_src_node: dace_nodes.AccessNode,
    update_dst_node: dace_nodes.AccessNode,
    scan_compute_st: dace.SDFGState,
    scan_update_st: dace.SDFGState,
):
    scan_sdfg = scan_node.sdfg
    assert isinstance(compute_dst_node.desc(scan_sdfg), dace.data.Scalar)
    assert compute_dst_node.desc(scan_sdfg) == update_dst_node.desc(scan_sdfg)

    # retrieve the source data outside the scan map scope
    assert len(list(state.in_edges_by_connector(scan_node, compute_src_node.data))) == 1
    top_level_input_edge = next(state.in_edges_by_connector(scan_node, compute_src_node.data))
    assert isinstance(top_level_input_edge.src, dace_nodes.MapEntry)
    map_entry_in_edge_conn = "IN_" + top_level_input_edge.src_conn[4:]
    assert (
        len(list(state.in_edges_by_connector(top_level_input_edge.src, map_entry_in_edge_conn)))
        == 1
    )
    top_level_src_node = next(
        state.in_edges_by_connector(top_level_input_edge.src, map_entry_in_edge_conn)
    ).src
    assert isinstance(top_level_src_node, dace_nodes.AccessNode)
    assert top_level_src_node.desc(sdfg).transient

    # retrieve the outer write to destination buffer in the compute state
    assert scan_compute_st.out_degree(compute_dst_node) == 1
    output_write_edge = scan_compute_st.out_edges(compute_dst_node)[0]
    assert isinstance(output_write_edge.dst, dace_nodes.AccessNode)
    output_node = output_write_edge.dst
    output_data = output_node.data
    assert output_data.startswith("__gtir_scan_output")
    assert not output_write_edge.dst.desc(scan_sdfg).transient
    output_subset = output_write_edge.data.get_dst_subset(output_write_edge, scan_compute_st)

    # retrieve the destination data outside the scan map scope
    assert len(list(state.out_edges_by_connector(scan_node, output_data))) == 1
    map_exit_in_edge = next(state.out_edges_by_connector(scan_node, output_data))
    assert isinstance(map_exit_in_edge.dst, dace_nodes.MapExit)
    map_exit_out_edge_conn = "OUT_" + map_exit_in_edge.dst_conn[3:]
    assert (
        len(list(state.out_edges_by_connector(map_exit_in_edge.dst, map_exit_out_edge_conn))) == 1
    )
    map_exit_out_edge = next(
        state.out_edges_by_connector(map_exit_in_edge.dst, map_exit_out_edge_conn)
    )
    top_level_dst_node = map_exit_out_edge.dst
    assert isinstance(top_level_dst_node, dace_nodes.AccessNode)
    top_level_dst_node_subset = map_exit_out_edge.data.get_dst_subset(map_exit_out_edge, state)

    if_stmt_sdfg = if_stmt_node.sdfg
    if_stmt_sdfg.arrays[if_stmt_output] = output_node.desc(scan_sdfg).clone()

    # reroute the write edge in the compute state
    scan_compute_st.add_edge(
        if_stmt_node,
        if_stmt_output,
        output_node,
        None,
        scan_sdfg.make_array_memlet(output_data),
    )

    # update the writes inside the if-stmt node to write to the global field
    for if_stmt_state in if_stmt_sdfg.states():
        branch_sink_nodes = [
            node for node in if_stmt_state.sink_nodes() if node.data == if_stmt_output
        ]
        assert len(branch_sink_nodes) <= 1
        if branch_sink_nodes:
            sink_node = branch_sink_nodes[0]
            assert if_stmt_state.out_degree(sink_node) == 0
            for edge in if_stmt_state.in_edges(sink_node):
                src_subset = edge.data.get_src_subset(edge, if_stmt_state)
                edge.data = dace.Memlet(
                    data=if_stmt_output, subset=output_subset, other_subset=src_subset
                )

    # map free symbol in output subset
    for sym in output_subset.free_symbols:
        if sym in if_stmt_sdfg.symbols:
            assert if_stmt_node.symbol_mapping[sym] == sym
        else:
            if_stmt_sdfg.add_symbol(sym, dace.int32)
            if_stmt_node.symbol_mapping[sym] = sym

    scan_compute_st.remove_node(compute_dst_node)
    scan_sdfg.remove_data(compute_dst_node.data, validate=gtx_config.DEBUG)

    gtx_transformations.gt_propagate_strides_from_access_node(
        sdfg=scan_sdfg,
        state=scan_compute_st,
        outer_node=output_node,
    )

    # reroute the write edge in the update state
    scan_update_st.add_nedge(
        scan_update_st.add_access(output_data),
        update_dst_node,
        dace.Memlet(data=output_data, subset=output_subset, other_subset="0"),
    )
    scan_update_st.remove_node(update_src_node)

    # we still need to replace the source node outside the scan map with the output node
    new_top_level_src_node = state.add_access(top_level_dst_node.data)
    _replace_scan_input(
        sdfg=sdfg,
        state=state,
        old_node=top_level_src_node,
        new_node=new_top_level_src_node,
        new_node_offsets=top_level_dst_node_subset.min_element(),
    )
    print(
        f"Removed self-copy in {if_stmt_node.label}: {top_level_src_node.data} -> {compute_src_node.data} -> {compute_dst_node.data} -> {output_data} -> {top_level_dst_node.data}"
    )


def _graupel_run_self_copy_removal_inside_if_stmt(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    scan_node: dace_nodes.NestedSDFG,
    scan_compute_st: dace.SDFGState,
    scan_update_st: dace.SDFGState,
    if_stmt_node: dace_nodes.NestedSDFG,
) -> None:
    scan_inout_copies = _graupel_scan_inout_copies.get(if_stmt_node.label, [])
    scan_sdfg = scan_node.sdfg
    nsdfg = if_stmt_node.sdfg
    assert len(nsdfg.nodes()) == 1 and isinstance(
        nsdfg.nodes()[0], dace_sdfg.state.ConditionalBlock
    )
    if_region = nsdfg.nodes()[0]
    assert len(list(br[1] for br in if_region.branches if br[0] is None)) == 1
    else_br = next(br[1] for br in if_region.branches if br[0] is None)
    assert isinstance(else_br.start_block, dace.SDFGState)
    assert len(if_region.out_degree(else_br.start_block)) == 0
    else_st = else_br.start_block
    src_nodes = [node for node in else_st.source_nodes() if isinstance(node, dace_nodes.AccessNode)]

    nodes_to_remove_inside_else_branch = []
    for src_node in src_nodes:
        assert not src_node.desc(nsdfg).transient
        if else_st.out_degree(src_node) != 1:
            continue
        self_copy_edge = else_st.out_edges(src_node)[0]
        dst_node = self_copy_edge.dst
        if else_st.out_degree(dst_node) != 0:
            continue
        assert not dst_node.desc(nsdfg).transient

        # retrieve the source data to copy in the compute state
        assert len(list(scan_compute_st.in_edges_by_connector(if_stmt_node, src_node.data))) == 1
        compute_read_edge = next(scan_compute_st.in_edges_by_connector(if_stmt_node, src_node.data))
        compute_src_node = compute_read_edge.src
        assert isinstance(compute_src_node, dace_nodes.AccessNode)

        # retrieve the destination node in the compute state, where the data is written
        assert len(list(scan_compute_st.out_edges_by_connector(if_stmt_node, dst_node.data))) == 1
        compute_write_edge = next(
            scan_compute_st.out_edges_by_connector(if_stmt_node, dst_node.data)
        )
        compute_dst_node = compute_write_edge.dst
        assert (
            isinstance(compute_dst_node, dace_nodes.AccessNode)
            and scan_compute_st.in_degree(compute_dst_node) == 1
        )
        if not compute_dst_node.desc(scan_sdfg).transient:
            continue
        temp_data_name = compute_dst_node.data

        # retrieve the data access inside the scan update state
        update_src_nodes = [
            node for node in scan_update_st.source_nodes() if node.data == temp_data_name
        ]
        assert len(update_src_nodes) <= 1
        if not update_src_nodes:
            continue
        update_src_node = update_src_nodes[0]
        assert scan_update_st.out_degree(update_src_node) == 1
        update_write_edge = scan_update_st.out_edges(update_src_node)[0]
        update_dst_node = update_write_edge.dst
        assert isinstance(update_dst_node, dace_nodes.AccessNode)
        assert (
            scan_update_st.in_degree(update_dst_node) == 1
            and scan_update_st.out_degree(update_dst_node) == 0
        )

        if compute_src_node.desc(scan_sdfg).transient:
            _cleanup_local_self_update(
                scan_sdfg=scan_sdfg,
                if_stmt_node=if_stmt_node,
                if_stmt_conn=dst_node.data,
                compute_src_node=compute_src_node,
                compute_dst_node=compute_dst_node,
                update_src_node=update_src_node,
                update_dst_node=update_dst_node,
                scan_compute_st=scan_compute_st,
                scan_update_st=scan_update_st,
            )
        # TODO(edopao): if we make 't' an inout field, this branch can become an else
        elif src_node.data in scan_inout_copies:
            _cleanup_global_self_update(
                sdfg=sdfg,
                state=state,
                scan_node=scan_node,
                if_stmt_node=if_stmt_node,
                if_stmt_output=dst_node.data,
                compute_src_node=compute_src_node,
                compute_dst_node=compute_dst_node,
                update_src_node=update_src_node,
                update_dst_node=update_dst_node,
                scan_compute_st=scan_compute_st,
                scan_update_st=scan_update_st,
            )
        else:
            continue

        # now it is safe to remove the data descriptor
        scan_sdfg.remove_data(temp_data_name, validate=gtx_config.DEBUG)
        nodes_to_remove_inside_else_branch.extend([src_node, dst_node])

    else_st.remove_nodes_from(nodes_to_remove_inside_else_branch)
    if else_st.is_empty():
        if_region.remove_branch(else_br)


def graupel_run_self_copy_removal_inside_scan(sdfg: dace.SDFG) -> None:
    sdfg.save("graupel_run_self_copy_removal_inside_scan_at_entry.sdfg")
    assert len(sdfg.states()) == 1
    st = sdfg.states()[0]
    assert (
        len(
            list(
                node
                for node in st.nodes()
                if isinstance(node, dace_nodes.NestedSDFG) and node.label.startswith("scan_")
            )
        )
        == 1
    )
    scan_nsdfg_node = next(
        node
        for node in st.nodes()
        if isinstance(node, dace_nodes.NestedSDFG) and node.label.startswith("scan_")
    )
    scan_sdfg = scan_nsdfg_node.sdfg
    assert len(scan_sdfg.nodes()) == 2
    assert isinstance(scan_sdfg.nodes()[1], dace_sdfg.state.LoopRegion)
    scan_loop = scan_sdfg.nodes()[1]
    assert len(scan_loop.nodes()) == 2 and all(
        isinstance(node, dace.SDFGState) for node in scan_loop.nodes()
    )
    if scan_loop.nodes()[0].label.startswith("scan_compute"):
        assert scan_loop.nodes()[1].label.startswith("scan_update")
        scan_compute_st, scan_update_st = scan_loop.nodes()
    else:
        assert scan_loop.nodes()[0].label.startswith("scan_update")
        scan_update_st, scan_compute_st = scan_loop.nodes()

    if_stmt_nodes = [
        node
        for node in scan_compute_st.nodes()
        if isinstance(node, dace_nodes.NestedSDFG) and node.label.startswith("if_stmt_")
    ]
    for if_stmt_node in if_stmt_nodes:
        _graupel_run_self_copy_removal_inside_if_stmt(
            sdfg, st, scan_nsdfg_node, scan_compute_st, scan_update_st, if_stmt_node
        )

    sdfg.validate()

    sdfg.save("graupel_run_self_copy_removal_inside_scan_at_exit.sdfg")
