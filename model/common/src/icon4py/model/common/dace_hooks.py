# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence

import dace
from dace import nodes as dace_nodes, sdfg as dace_sdfg
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


def _graupel_run_self_copy_removal_inside_if_stmt_generic(
    scan_sdfg: dace.SDFG,
    scan_compute_st: dace.SDFGState,
    scan_update_st: dace.SDFGState,
    if_stmt_nodes: Sequence[dace_nodes.NestedSDFG],
) -> list[str]:
    data_to_remove = []
    for if_stmt_node in if_stmt_nodes:
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
        src_nodes = [
            node for node in else_st.source_nodes() if isinstance(node, dace_nodes.AccessNode)
        ]
        nodes_to_remove = []
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
            assert (
                len(list(scan_compute_st.in_edges_by_connector(if_stmt_node, src_node.data))) == 1
            )
            outer_read_edge = next(
                scan_compute_st.in_edges_by_connector(if_stmt_node, src_node.data)
            )
            outer_src_node = outer_read_edge.src
            assert isinstance(outer_src_node, dace_nodes.AccessNode)
            if not outer_src_node.desc(scan_sdfg).transient:
                continue
            # retrive the destination node in the compute state, where the data is written
            assert (
                len(list(scan_compute_st.out_edges_by_connector(if_stmt_node, dst_node.data))) == 1
            )
            outer_write_edge = next(
                scan_compute_st.out_edges_by_connector(if_stmt_node, dst_node.data)
            )
            outer_dst_node = outer_write_edge.dst
            assert (
                isinstance(outer_dst_node, dace_nodes.AccessNode)
                and scan_compute_st.in_degree(outer_dst_node) == 1
            )
            if not outer_dst_node.desc(scan_sdfg).transient:
                continue
            assert outer_dst_node.desc(scan_sdfg) == outer_src_node.desc(scan_sdfg)
            # check if the destination node is used in the scan update as state variable
            assert all(
                isinstance(node, dace_nodes.AccessNode) for node in scan_update_st.source_nodes()
            )
            if all(node.data != outer_dst_node.data for node in scan_update_st.source_nodes()):
                continue
            update_src_node = next(
                node for node in scan_update_st.source_nodes() if node.data == outer_dst_node.data
            )
            new_update_src_node = scan_update_st.add_access(outer_src_node.data)
            for edge in scan_update_st.out_edges(update_src_node):
                scan_update_st.add_edge(
                    new_update_src_node,
                    None,
                    edge.dst,
                    edge.dst_conn,
                    dace.Memlet(
                        data=new_update_src_node.data,
                        subset=edge.data.get_src_subset(edge, scan_update_st),
                        other_subset=edge.data.get_dst_subset(edge, scan_update_st),
                    ),
                )
            scan_update_st.remove_node(update_src_node)
            # reroute the write edges in the update state
            new_compute_dst_node = scan_compute_st.add_access(outer_src_node.data)
            scan_compute_st.add_edge(
                if_stmt_node,
                dst_node.data,
                new_compute_dst_node,
                None,
                dace.Memlet(
                    data=new_compute_dst_node.data,
                    subset=outer_write_edge.data.get_dst_subset(outer_write_edge, else_st),
                ),
            )
            for edge in scan_compute_st.out_edges(outer_dst_node):
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
            scan_compute_st.remove_node(outer_dst_node)
            # now it is safe to remove the data descriptor
            # now it is safe to remove the data descriptor
            data_to_remove.append(outer_dst_node.data)
            nodes_to_remove.extend([src_node, dst_node])
        else_st.remove_nodes_from(nodes_to_remove)
        if else_st.is_empty():
            if_region.remove_branch(else_br)
    return data_to_remove


def _graupel_run_self_copy_removal_inside_if_stmt(
    scan_sdfg: dace.SDFG,
    scan_compute_st: dace.SDFGState,
    scan_update_st: dace.SDFGState,
    if_stmt_nodes: Sequence[dace_nodes.NestedSDFG],
) -> list[str]:
    data_to_remove = []
    graupel_scan_inout_copies = {
        "if_stmt_92": ["__ct_el_24", "__ct_el_25", "__ct_el_26", "__ct_el_27"]
    }
    for if_stmt_node in if_stmt_nodes:
        if if_stmt_node.label not in graupel_scan_inout_copies:
            continue
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
        src_nodes = [
            node for node in else_st.source_nodes() if isinstance(node, dace_nodes.AccessNode)
        ]
        nodes_to_remove = []
        for source_data in graupel_scan_inout_copies[if_stmt_node.label]:
            assert len(list(node for node in src_nodes if node.data == source_data)) == 1
            src_node = next(node for node in src_nodes if node.data == source_data)
            assert not src_node.desc(nsdfg).transient
            assert else_st.out_degree(src_node) == 1
            self_copy_edge = else_st.out_edges(src_node)[0]
            dst_node = self_copy_edge.dst
            assert else_st.out_degree(dst_node) == 0
            assert else_st.in_degree(dst_node) == 1
            assert not dst_node.desc(nsdfg).transient
            # retrieve the outer write to destination buffer
            assert (
                len(list(scan_compute_st.out_edges_by_connector(if_stmt_node, dst_node.data))) == 1
            )
            outer_write_edge = next(
                scan_compute_st.out_edges_by_connector(if_stmt_node, dst_node.data)
            )
            outer_dst_node = outer_write_edge.dst
            assert (
                isinstance(outer_dst_node, dace_nodes.AccessNode)
                and scan_compute_st.in_degree(outer_dst_node) == 1
            )
            assert outer_dst_node.desc(scan_sdfg).transient
            assert scan_compute_st.out_degree(outer_dst_node) == 1
            output_write_edge = scan_compute_st.out_edges(outer_dst_node)[0]
            assert isinstance(output_write_edge.dst, dace_nodes.AccessNode)
            output_node = output_write_edge.dst
            output_data = output_node.data
            assert output_data.startswith("__gtir_scan_output")
            assert not output_write_edge.dst.desc(scan_sdfg).transient
            output_subset = output_write_edge.data.get_dst_subset(
                output_write_edge, scan_compute_st
            )
            # bypass the temporary, write directly to output
            scan_compute_st.add_edge(
                if_stmt_node,
                dst_node.data,
                output_node,
                None,
                dace.Memlet(data=output_data, subset=output_subset),
            )
            scan_compute_st.remove_node(outer_dst_node)
            # replace the data access in scan update
            assert all(
                isinstance(node, dace_nodes.AccessNode) for node in scan_update_st.source_nodes()
            )
            assert (
                len(
                    list(
                        node
                        for node in scan_update_st.source_nodes()
                        if node.data == outer_dst_node.data
                    )
                )
                == 1
            )
            update_src_node = next(
                node for node in scan_update_st.source_nodes() if node.data == outer_dst_node.data
            )
            assert scan_update_st.out_degree(update_src_node) == 1
            update_write_edge = scan_update_st.out_edges(update_src_node)[0]
            update_dst_node = update_write_edge.dst
            assert isinstance(update_dst_node, dace_nodes.AccessNode)
            assert (
                scan_update_st.in_degree(update_dst_node) == 1
                and scan_update_st.out_degree(update_dst_node) == 0
            )
            scan_update_st.add_nedge(
                scan_update_st.add_access(output_data),
                update_dst_node,
                dace.Memlet(
                    data=output_data,
                    subset=output_subset,
                    other_subset=update_write_edge.data.get_dst_subset(
                        update_write_edge, scan_update_st
                    ),
                ),
            )
            scan_update_st.remove_node(update_src_node)
            # now it is safe to remove the data descriptor
            data_to_remove.append(outer_dst_node.data)
            nodes_to_remove.extend([src_node, dst_node])
        else_st.remove_nodes_from(nodes_to_remove)
        if else_st.is_empty():
            if_region.remove_branch(else_br)
    return data_to_remove


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
    data_to_remove = []
    data_to_remove += _graupel_run_self_copy_removal_inside_if_stmt_generic(
        scan_sdfg, scan_compute_st, scan_update_st, if_stmt_nodes
    )
    data_to_remove += _graupel_run_self_copy_removal_inside_if_stmt(
        scan_sdfg, scan_compute_st, scan_update_st, if_stmt_nodes
    )
    for data in data_to_remove:
        scan_sdfg.remove_data(data)
    print(f"Removed {','.join(sorted(data_to_remove))}")
    sdfg.validate()

    sdfg.save("graupel_run_self_copy_removal_inside_scan_at_exit.sdfg")

    gtx_transformations.gt_simplify(
        sdfg=scan_sdfg,
        skip=gtx_transformations.constants.GT_SIMPLIFY_DEFAULT_SKIP_SET | {"FuseStates"},
        validate=True,
        validate_all=False,
    )
