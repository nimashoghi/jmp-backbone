from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import (
    TYPE_CHECKING,
    Protocol,
    TypedDict,
    cast,
    runtime_checkable,
)

import nshconfig as C
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import BaseData
from torch_geometric.nn import radius_graph
from torch_geometric.utils import sort_edge_index
from torch_scatter import segment_coo
from typing_extensions import NotRequired, override

from .radius_graph import get_pbc_distances, radius_graph_pbc
from .torch_scatter_polyfill import segment_coo
from .utils import get_edge_id, get_max_neighbors_mask, mask_neighbors, repeat_blocks

if TYPE_CHECKING:
    from .config import BackboneConfig


class Graph(TypedDict):
    edge_index: torch.Tensor  # 2 e
    distance: torch.Tensor  # e
    vector: torch.Tensor  # e 3
    cell_offset: torch.Tensor  # e 3
    num_neighbors: torch.Tensor  # b

    cutoff: torch.Tensor  # b
    max_neighbors: torch.Tensor  # b

    id_swap_edge_index: NotRequired[torch.Tensor]  # e


class CutoffsConfig(C.Config):
    main: float
    aeaint: float
    qint: float
    aint: float

    @classmethod
    def from_constant(cls, value: float):
        return cls(main=value, aeaint=value, qint=value, aint=value)

    def __mul__(self, other: float):
        return self.__class__(
            main=self.main * other,
            aeaint=self.aeaint * other,
            qint=self.qint * other,
            aint=self.aint * other,
        )


class MaxNeighborsConfig(C.Config):
    main: int
    aeaint: int
    qint: int
    aint: int

    @classmethod
    def from_goc_base_proportions(cls, max_neighbors: int):
        """
        GOC base proportions:
            max_neighbors: 30
            max_neighbors_qint: 8
            max_neighbors_aeaint: 20
            max_neighbors_aint: 1000
        """
        return cls(
            main=max_neighbors,
            aeaint=int(max_neighbors * 20 / 30),
            qint=int(max_neighbors * 8 / 30),
            aint=int(max_neighbors * 1000 / 30),
        )

    def __mul__(self, other: int):
        return self.__class__(
            main=self.main * other,
            aeaint=self.aeaint * other,
            qint=self.qint * other,
            aint=self.aint * other,
        )


def _select_symmetric_edges(tensor, mask, reorder_idx, opposite_neg):
    """Use a mask to remove values of removed edges and then
    duplicate the values for the correct edge direction.

    Arguments
    ---------
    tensor: torch.Tensor
        Values to symmetrize for the new tensor.
    mask: torch.Tensor
        Mask defining which edges go in the correct direction.
    reorder_idx: torch.Tensor
        Indices defining how to reorder the tensor values after
        concatenating the edge values of both directions.
    opposite_neg: bool
        Whether the edge in the opposite direction should use the
        negative tensor value.

    Returns
    -------
    tensor_ordered: torch.Tensor
        A tensor with symmetrized values.
    """
    # Mask out counter-edges
    tensor_directed = tensor[mask]
    # Concatenate counter-edges after normal edges
    sign = 1 - 2 * opposite_neg
    tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
    # Reorder everything so the edges of every image are consecutive
    tensor_ordered = tensor_cat[reorder_idx]
    return tensor_ordered


def symmetrize_edges(graph: Graph, num_atoms: int):
    """
    Symmetrize edges to ensure existence of counter-directional edges.

    Some edges are only present in one direction in the data,
    since every atom has a maximum number of neighbors.
    We only use i->j edges here. So we lose some j->i edges
    and add others by making it symmetric.
    """
    new_graph = graph.copy()

    # Generate mask
    mask_sep_atoms = graph["edge_index"][0] < graph["edge_index"][1]
    # Distinguish edges between the same (periodic) atom by ordering the cells
    cell_earlier = (
        (graph["cell_offset"][:, 0] < 0)
        | ((graph["cell_offset"][:, 0] == 0) & (graph["cell_offset"][:, 1] < 0))
        | (
            (graph["cell_offset"][:, 0] == 0)
            & (graph["cell_offset"][:, 1] == 0)
            & (graph["cell_offset"][:, 2] < 0)
        )
    )
    mask_same_atoms = graph["edge_index"][0] == graph["edge_index"][1]
    mask_same_atoms &= cell_earlier
    mask = mask_sep_atoms | mask_same_atoms

    # Mask out counter-edges
    edge_index_directed = graph["edge_index"][mask[None, :].expand(2, -1)].view(2, -1)

    # Concatenate counter-edges after normal edges
    edge_index_cat = torch.cat(
        [edge_index_directed, edge_index_directed.flip(0)],
        dim=1,
    )

    # Count remaining edges per image
    batch_edge = torch.repeat_interleave(
        torch.arange(
            graph["num_neighbors"].size(0),
            device=graph["edge_index"].device,
        ),
        graph["num_neighbors"],
    )
    batch_edge = batch_edge[mask]
    # segment_coo assumes sorted batch_edge
    # Factor 2 since this is only one half of the edges
    ones = batch_edge.new_ones(1).expand_as(batch_edge)
    new_graph["num_neighbors"] = 2 * segment_coo(
        ones, batch_edge, dim_size=graph["num_neighbors"].size(0)
    )

    # Create indexing array
    edge_reorder_idx = repeat_blocks(
        torch.div(new_graph["num_neighbors"], 2, rounding_mode="floor"),
        repeats=2,
        continuous_indexing=True,
        repeat_inc=edge_index_directed.size(1),
    )

    # Reorder everything so the edges of every image are consecutive
    new_graph["edge_index"] = edge_index_cat[:, edge_reorder_idx]
    new_graph["cell_offset"] = _select_symmetric_edges(
        graph["cell_offset"], mask, edge_reorder_idx, True
    )
    new_graph["distance"] = _select_symmetric_edges(
        graph["distance"], mask, edge_reorder_idx, False
    )
    new_graph["vector"] = _select_symmetric_edges(
        graph["vector"], mask, edge_reorder_idx, True
    )

    # Indices for swapping c->a and a->c (for symmetric MP)
    # To obtain these efficiently and without any index assumptions,
    # we get order the counter-edge IDs and then
    # map this order back to the edge IDs.
    # Double argsort gives the desired mapping
    # from the ordered tensor to the original tensor.
    edge_ids = get_edge_id(new_graph["edge_index"], new_graph["cell_offset"], num_atoms)
    order_edge_ids = torch.argsort(edge_ids)
    inv_order_edge_ids = torch.argsort(order_edge_ids)
    edge_ids_counter = get_edge_id(
        new_graph["edge_index"].flip(0),
        -new_graph["cell_offset"],
        num_atoms,
    )
    order_edge_ids_counter = torch.argsort(edge_ids_counter)
    id_swap_edge_index = order_edge_ids_counter[inv_order_edge_ids]

    new_graph["id_swap_edge_index"] = id_swap_edge_index

    return cast(Graph, new_graph)


def tag_mask(data: BaseData, graph: Graph, *, tags: list[int]):
    tags_ = torch.tensor(tags, dtype=torch.long, device=data.tags.device)

    # Only use quadruplets for certain tags
    tags_s = data.tags[graph["edge_index"][0]]
    tags_t = data.tags[graph["edge_index"][1]]
    tag_mask_s = (tags_s[..., None] == tags_).any(dim=-1)
    tag_mask_t = (tags_t[..., None] == tags_).any(dim=-1)
    tag_mask = tag_mask_s | tag_mask_t

    graph["edge_index"] = graph["edge_index"][:, tag_mask]
    graph["cell_offset"] = graph["cell_offset"][tag_mask, :]
    graph["distance"] = graph["distance"][tag_mask]
    graph["vector"] = graph["vector"][tag_mask, :]

    return graph


def _radius_graph_pbc(
    radius,
    max_num_neighbors_threshold,
    data: BaseData,
    enforce_max_neighbors_strictly: bool = False,
    pbc=None,
    per_graph: bool = False,
):
    if not per_graph:
        return radius_graph_pbc(
            radius,
            max_num_neighbors_threshold,
            data.pos,
            data.natoms,
            data.cell,
            enforce_max_neighbors_strictly=enforce_max_neighbors_strictly,
            pbc=pbc,
        )

    assert (
        ptr := getattr(data, "ptr", None)
    ) is not None, "`data.ptr` is required for per-graph radius graph"
    pos: torch.Tensor = data.pos  # n 3
    cell: torch.Tensor = data.cell  # b 3 3
    natoms: torch.Tensor = data.natoms  # b

    edge_index, cell_offsets, neighbors = [], [], []
    atom_index_offset = 0
    for i in range(ptr.size(0) - 1):
        pos_i = pos[ptr[i] : ptr[i + 1]]
        natoms_i = natoms[i]
        cell_i = cell[i]
        edge_index_i, cell_offsets_i, neighbors_i = radius_graph_pbc(
            radius,
            max_num_neighbors_threshold,
            pos_i,
            natoms_i[None],
            cell_i[None],
            enforce_max_neighbors_strictly=enforce_max_neighbors_strictly,
            pbc=pbc,
        )
        edge_index.append(edge_index_i + atom_index_offset)
        cell_offsets.append(cell_offsets_i)
        neighbors.append(neighbors_i)
        atom_index_offset += pos_i.shape[0]

    edge_index = torch.cat(edge_index, dim=1)
    cell_offsets = torch.cat(cell_offsets, dim=0)
    neighbors = torch.cat(neighbors, dim=0)

    return edge_index, cell_offsets, neighbors


def _generate_graph(
    data: BaseData,
    *,
    cutoff: float,
    max_neighbors: int,
    pbc: bool,
    per_graph: bool = False,
):
    if pbc:
        edge_index, cell_offsets, neighbors = _radius_graph_pbc(
            cutoff,
            max_neighbors,
            data,
            per_graph=per_graph,
        )

        out = get_pbc_distances(
            data.pos,
            edge_index,
            data.cell,
            cell_offsets,
            neighbors,
            return_offsets=True,
            return_distance_vec=True,
        )

        edge_index: torch.Tensor = out["edge_index"]
        edge_dist: torch.Tensor = out["distances"]
        cell_offset_distances: torch.Tensor = out["offsets"]
        distance_vec: torch.Tensor = out["distance_vec"]
    else:
        edge_index = radius_graph(
            data.pos,
            r=cutoff,
            batch=data.batch,
            max_num_neighbors=max_neighbors,
        )

        j, i = edge_index
        distance_vec = data.pos[j] - data.pos[i]

        edge_dist = distance_vec.norm(dim=-1)
        cell_offsets = torch.zeros(edge_index.shape[1], 3, device=data.pos.device)
        cell_offset_distances = torch.zeros_like(cell_offsets, device=data.pos.device)
        neighbors = edge_index.shape[1]

    return (
        edge_index,
        edge_dist,
        distance_vec,
        cell_offsets,
        cell_offset_distances,
        neighbors,
    )


def generate_graph(
    data: BaseData,
    *,
    cutoff: float,
    max_neighbors: int,
    pbc: bool,
    symmetrize: bool = False,
    filter_tags: list[int] | None = None,
    sort_edges: bool = False,
    per_graph: bool = False,
):
    (
        edge_index,
        edge_dist,
        distance_vec,
        cell_offsets,
        _,  # cell offset distances
        num_neighbors,
    ) = _generate_graph(
        data,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        pbc=pbc,
        per_graph=per_graph,
    )
    # These vectors actually point in the opposite direction.
    # But we want to use col as idx_t for efficient aggregation.
    edge_vector = -distance_vec / edge_dist[:, None]
    # cell_offsets = -cell_offsets  # a - c + offset

    graph: Graph = {
        "edge_index": edge_index,
        "distance": edge_dist,
        "vector": edge_vector,
        "cell_offset": cell_offsets,
        "num_neighbors": num_neighbors,
        "cutoff": torch.tensor(cutoff, dtype=data.pos.dtype, device=data.pos.device),
        "max_neighbors": torch.tensor(
            max_neighbors, dtype=torch.long, device=data.pos.device
        ),
    }

    if symmetrize:
        graph = symmetrize_edges(graph, data.pos.shape[0])

    if filter_tags is not None:
        graph = tag_mask(data, graph, tags=filter_tags)

    if sort_edges:
        (
            graph["edge_index"],
            [
                graph["distance"],
                graph["vector"],
                graph["cell_offset"],
            ],
        ) = sort_edge_index(
            graph["edge_index"],
            [
                graph["distance"],
                graph["vector"],
                graph["cell_offset"],
            ],
            num_nodes=data.pos.shape[0],
            sort_by_row=False,
        )

        graph["num_neighbors"] = torch.full_like(
            graph["num_neighbors"], graph["edge_index"].shape[1]
        )

    return graph


def _subselect_edges(
    data: BaseData,
    graph: Graph,
    cutoff: float | None = None,
    max_neighbors: int | None = None,
):
    """Subselect edges using a stricter cutoff and max_neighbors."""
    subgraph = graph.copy()

    if cutoff is not None:
        edge_mask = subgraph["distance"] <= cutoff

        subgraph["edge_index"] = subgraph["edge_index"][:, edge_mask]
        subgraph["cell_offset"] = subgraph["cell_offset"][edge_mask]
        subgraph["num_neighbors"] = mask_neighbors(subgraph["num_neighbors"], edge_mask)
        subgraph["distance"] = subgraph["distance"][edge_mask]
        subgraph["vector"] = subgraph["vector"][edge_mask]

    if max_neighbors is not None:
        subgraph["max_neighbors"] = torch.tensor(
            max_neighbors, dtype=torch.long, device=data.pos.device
        )
        edge_mask, subgraph["num_neighbors"] = get_max_neighbors_mask(
            natoms=torch.tensor([data.natoms], dtype=torch.long, device=data.pos.device)
            if not torch.is_tensor(data.natoms)
            else data.natoms.view(-1),
            index=subgraph["edge_index"][1],
            atom_distance=subgraph["distance"],
            max_num_neighbors_threshold=max_neighbors,
        )
        if not torch.all(edge_mask):
            subgraph["edge_index"] = subgraph["edge_index"][:, edge_mask]
            subgraph["cell_offset"] = subgraph["cell_offset"][edge_mask]
            subgraph["distance"] = subgraph["distance"][edge_mask]
            subgraph["vector"] = subgraph["vector"][edge_mask]

    empty_image = subgraph["num_neighbors"] == 0
    if torch.any(empty_image):
        raise ValueError(f"An image has no neighbors: {data}")
    return subgraph


def subselect_graph(
    data: BaseData,
    graph: Graph,
    cutoff: float,
    max_neighbors: int,
    cutoff_orig: float,
    max_neighbors_orig: int,
):
    """If the new cutoff and max_neighbors is different from the original,
    subselect the edges of a given graph.
    """
    # Check if embedding edges are different from interaction edges
    if np.isclose(cutoff, cutoff_orig):
        select_cutoff = None
    else:
        select_cutoff = cutoff
    if max_neighbors == max_neighbors_orig:
        select_neighbors = None
    else:
        select_neighbors = max_neighbors

    graph = _subselect_edges(
        data=data,
        graph=graph,
        cutoff=select_cutoff,
        max_neighbors=select_neighbors,
    )
    return graph


def generate_graphs(
    data: BaseData,
    *,
    cutoffs: CutoffsConfig | Callable[[BaseData], CutoffsConfig],
    max_neighbors: MaxNeighborsConfig | Callable[[BaseData], MaxNeighborsConfig],
    pbc: bool,
    symmetrize_main: bool = False,
    qint_tags: list[int] | None = [1, 2],
):
    """
    Data needs the following attributes:
        - cell
        - pos
        - natoms
        - batch
        - tags
    """

    if callable(cutoffs):
        cutoffs = cutoffs(data)
    if callable(max_neighbors):
        max_neighbors = max_neighbors(data)

    assert cutoffs.main <= cutoffs.aint
    assert cutoffs.aeaint <= cutoffs.aint
    assert cutoffs.qint <= cutoffs.aint

    assert max_neighbors.main <= max_neighbors.aint
    assert max_neighbors.aeaint <= max_neighbors.aint
    assert max_neighbors.qint <= max_neighbors.aint

    main_graph = generate_graph(
        data,
        cutoff=cutoffs.main,
        max_neighbors=max_neighbors.main,
        pbc=pbc,
        symmetrize=symmetrize_main,
    )
    a2a_graph = generate_graph(
        data,
        cutoff=cutoffs.aint,
        max_neighbors=max_neighbors.aint,
        pbc=pbc,
    )
    a2ee2a_graph = generate_graph(
        data,
        cutoff=cutoffs.aeaint,
        max_neighbors=max_neighbors.aeaint,
        pbc=pbc,
    )
    qint_graph = generate_graph(
        data,
        cutoff=cutoffs.qint,
        max_neighbors=max_neighbors.qint,
        pbc=pbc,
        filter_tags=qint_tags,
    )

    graphs = {
        "main": main_graph,
        "a2a": a2a_graph,
        "a2ee2a": a2ee2a_graph,
        "qint": qint_graph,
    }
    return graphs


class Graphs(TypedDict):
    main: Graph
    a2a: Graph
    a2ee2a: Graph
    qint: Graph


GRAPH_TYPES = ["main", "a2a", "a2ee2a", "qint"]


def graphs_from_batch(data: BaseData | Batch) -> Graphs:
    global GRAPH_TYPES

    graphs = {
        graph_type: {
            "edge_index": getattr(data, f"{graph_type}_edge_index"),
            "distance": getattr(data, f"{graph_type}_distance"),
            "vector": getattr(data, f"{graph_type}_vector"),
            "cell_offset": getattr(data, f"{graph_type}_cell_offset"),
            "num_neighbors": getattr(data, f"{graph_type}_num_neighbors", None),
            "cutoff": getattr(data, f"{graph_type}_cutoff", None),
            "max_neighbors": getattr(data, f"{graph_type}_max_neighbors", None),
            "id_swap_edge_index": getattr(
                data, f"{graph_type}_id_swap_edge_index", None
            ),
        }
        for graph_type in GRAPH_TYPES
    }
    # remove None values
    graphs = {
        graph_type: {key: value for key, value in graph.items() if value is not None}
        for graph_type, graph in graphs.items()
    }
    return cast(Graphs, graphs)


def write_graphs_to_batch_(data: BaseData | Batch, graphs: Graphs):
    global GRAPH_TYPES

    for graph_type in GRAPH_TYPES:
        for key, value in graphs[graph_type].items():
            setattr(data, f"{graph_type}_{key}", value)


@runtime_checkable
class AintGraphTransformProtocol(Protocol):
    def __call__(self, graph: Graph, training: bool) -> Graph: ...


class GraphComputerConfig(C.Config):
    pbc: bool
    """Whether to use periodic boundary conditions."""

    cutoffs: CutoffsConfig = CutoffsConfig.from_constant(12.0)
    """The cutoff for the radius graph."""

    max_neighbors: MaxNeighborsConfig = MaxNeighborsConfig.from_goc_base_proportions(30)
    """The maximum number of neighbors for the radius graph."""

    per_graph_radius_graph: bool = False
    """Whether to compute the radius graph per graph."""


class GraphComputer(nn.Module):
    def __init__(
        self,
        config: GraphComputerConfig,
        backbone_config: BackboneConfig,
        *,
        process_aint_graph_transform: AintGraphTransformProtocol | None = None,
    ):
        super().__init__()

        self.config = config
        del config

        self.backbone_config = backbone_config
        del backbone_config

        self._process_aint_graph_transform = process_aint_graph_transform

    @override
    def forward(
        self,
        data: BaseData,
        # cutoffs: CutoffsConfig,
        # max_neighbors: MaxNeighborsConfig,
        # pbc: bool,
        # training: bool,
    ):
        cutoffs = self.config.cutoffs
        max_neighbors = self.config.max_neighbors
        pbc = self.config.pbc
        training = self.training

        aint_graph = generate_graph(
            data,
            cutoff=cutoffs.aint,
            max_neighbors=max_neighbors.aint,
            pbc=pbc,
            per_graph=self.config.per_graph_radius_graph,
        )
        if self._process_aint_graph_transform is not None:
            aint_graph = self._process_aint_graph_transform(
                aint_graph, training=training
            )
        subselect = partial(
            subselect_graph,
            data,
            aint_graph,
            cutoff_orig=cutoffs.aint,
            max_neighbors_orig=max_neighbors.aint,
        )
        main_graph = subselect(cutoffs.main, max_neighbors.main)
        aeaint_graph = subselect(cutoffs.aeaint, max_neighbors.aeaint)
        qint_graph = subselect(cutoffs.qint, max_neighbors.qint)

        # We can't do this at the data level: This is because the batch collate_fn doesn't know
        # that it needs to increment the "id_swap" indices as it collates the data.
        # So we do this at the graph level (which is done in the GemNetOC `get_graphs_and_indices` method).
        # main_graph = symmetrize_edges(main_graph, num_atoms=data.pos.shape[0])
        qint_graph = tag_mask(data, qint_graph, tags=self.backbone_config.qint_tags)

        write_graphs_to_batch_(
            data,
            {
                "main": main_graph,
                "a2a": aint_graph,
                "a2ee2a": aeaint_graph,
                "qint": qint_graph,
            },
        )

        return data
