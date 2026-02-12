# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dace
from dace import nodes as dace_nodes, sdfg as dace_sdfg
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


def _graupel_run_self_copy_removal_in_scan(scan_sdfg):
    assert len(scan_sdfg.nodes()) == 2
    assert isinstance(scan_sdfg.nodes()[1], dace_sdfg.state.LoopRegion)
    scan_loop = scan_sdfg.nodes()[1]
    assert len(scan_loop.nodes()) == 2 and all(
        isinstance(node, dace.SDFGState) for node in scan_loop.nodes()
    )
    if scan_loop.nodes()[0].label.startswith("scan_compute_"):
        assert scan_loop.nodes()[1].label.startswith("scan_update_")
        scan_compute_st, scan_update_st = scan_loop.nodes()
    else:
        assert scan_loop.nodes()[0].label.startswith("scan_update_")
        scan_update_st, scan_compute_st = scan_loop.nodes()
    if_stmt_nodes = [
        node
        for node in scan_compute_st.nodes()
        if isinstance(node, dace_nodes.NestedSDFG) and node.label.startswith("if_stmt_")
    ]
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
            nodes_to_remove.extend([src_node, dst_node])
            # reroute the outer write, by using inout data
            assert (
                len(list(scan_compute_st.out_edges_by_connector(if_stmt_node, dst_node.data))) == 1
            )
            outer_read_edge = next(
                scan_compute_st.in_edges_by_connector(if_stmt_node, src_node.data)
            )
            outer_src_node = outer_read_edge.src
            assert isinstance(outer_src_node, dace_nodes.AccessNode)
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
            new_outer_dst_node = scan_compute_st.add_access(outer_src_node.data)
            dst_subset = outer_write_edge.data.get_dst_subset(outer_write_edge, else_st)
            scan_compute_st.add_edge(
                if_stmt_node,
                dst_node.data,
                new_outer_dst_node,
                None,
                dace.Memlet(data=new_outer_dst_node.data, subset=dst_subset),
            )
            for edge in scan_compute_st.out_edges(outer_dst_node):
                scan_compute_st.add_edge(
                    new_outer_dst_node,
                    None,
                    edge.dst,
                    edge.dst_conn,
                    dace.Memlet(
                        data=new_outer_dst_node.data,
                        subset=edge.data.get_src_subset(edge, scan_compute_st),
                        other_subset=edge.data.get_dst_subset(edge, scan_compute_st),
                    ),
                )
            scan_compute_st.remove_node(outer_dst_node)
            # check in the scan update state if this data was used
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
            # now it is safe to remove the data descriptor
            scan_sdfg.remove_data(outer_dst_node.data)
            print(f"Remove write {self_copy_edge} in {nsdfg.label}.")
        else_st.remove_nodes_from(nodes_to_remove)
        if else_st.is_empty():
            if_region.remove_branch(else_br)


def graupel_run_top_level_dataflow_pre(sdfg: dace.SDFG) -> None:
    sdfg.save("graupel_run_top_level_dataflow_pre_at_entry.sdfg")
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
    scan_sdfg = next(
        node
        for node in st.nodes()
        if isinstance(node, dace_nodes.NestedSDFG) and node.label.startswith("scan_")
    ).sdfg
    _graupel_run_self_copy_removal_in_scan(scan_sdfg)
    sdfg.validate()

    gtx_transformations.gt_simplify(
        sdfg=sdfg,
        validate=False,
        skip=gtx_transformations.constants._GT_AUTO_OPT_INITIAL_STEP_SIMPLIFY_SKIP_LIST,
        validate_all=False,
    )

    sdfg.save("graupel_run_top_level_dataflow_pre_at_exit.sdfg")
