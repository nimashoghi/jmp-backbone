from __future__ import annotations

from logging import getLogger
from typing import Literal

import nshconfig as C
import nshtrainer as nt
import nshutils.typecheck as tc
import torch
import torch.nn as nn
from einops import rearrange
from jmppeft.modules.torch_scatter_polyfill import scatter
from torch_geometric.data.data import BaseData
from typing_extensions import TypedDict, override

from ..models.gemnet.backbone import GOCBackboneOutput

log = getLogger(__name__)


class OutputHeadInput(TypedDict):
    data: BaseData
    backbone_output: GOCBackboneOutput


class EnergyTargetConfig(C.Config):
    max_atomic_number: int
    """The max atomic number in the dataset."""

    reduction: Literal["mean", "sum"] = "sum"
    """The reduction to use for the output."""

    edge_level_energies: bool = False
    """Whether to use edge level energies."""

    num_mlps: int = 5
    """Number of MLPs in the output layer."""

    def create_model(
        self,
        d_model: int,
        d_model_edge: int,
        activation_cls: type[nn.Module],
    ):
        return EnergyOutputHead(
            config=self,
            d_model=d_model,
            d_model_edge=d_model_edge,
            activation_cls=activation_cls,
        )


class EnergyOutputHead(nn.Module):
    @override
    def __init__(
        self,
        config: EnergyTargetConfig,
        d_model: int,
        d_model_edge: int,
        activation_cls: type[nn.Module],
    ):
        super().__init__()

        self.config = config
        self.out_mlp_node = nt.nn.MLP(
            ([d_model] * self.config.num_mlps) + [1],
            activation=activation_cls,
        )

        self.per_atom_scales = nn.Embedding(
            self.config.max_atomic_number + 1,
            1,
            padding_idx=0,
        )
        nn.init.ones_(self.per_atom_scales.weight)

        self.per_atom_shifts = nn.Embedding(
            self.config.max_atomic_number + 1,
            1,
            padding_idx=0,
        )
        nn.init.zeros_(self.per_atom_shifts.weight)

        if self.config.edge_level_energies:
            self.out_mlp_edge = nt.nn.MLP(
                ([d_model_edge] * self.config.num_mlps) + [1],
                activation=activation_cls,
            )

            num_atom_pairs = (self.config.max_atomic_number + 1) ** 2
            self.pairwise_scales = nn.Embedding(num_atom_pairs, 1, padding_idx=0)
            nn.init.ones_(self.pairwise_scales.weight)

    @override
    def forward(self, input: OutputHeadInput) -> torch.Tensor:
        data = input["data"]
        backbone_output = input["backbone_output"]

        atomic_numbers = data.atomic_numbers
        tc.tassert(tc.Int[torch.Tensor, "n"], atomic_numbers)

        # Compute node-level energies from node embeddings
        per_atom_energies = backbone_output["energy"]
        tc.tassert(tc.Float[torch.Tensor, "n d_model"], per_atom_energies)
        per_atom_energies = self.out_mlp_node(per_atom_energies)
        tc.tassert(tc.Float[torch.Tensor, "n 1"], per_atom_energies)

        if self.config.edge_level_energies:
            # Compute edge-level energies from edge embeddings
            per_edge_energies = backbone_output["forces"]
            tc.tassert(tc.Float[torch.Tensor, "e d_model_edge"], per_edge_energies)
            per_edge_energies = self.out_mlp_edge(per_edge_energies)
            tc.tassert(tc.Float[torch.Tensor, "e 1"], per_edge_energies)

            # Multiply edge energies by pairwise scales
            # Compute the pairwise indices
            idx_s, idx_t = backbone_output["idx_s"], backbone_output["idx_t"]
            tc.tassert(tc.Int[torch.Tensor, "e"], (idx_s, idx_t))
            pair_idx = (
                atomic_numbers[idx_s] * (self.config.max_atomic_number + 1)
                + atomic_numbers[idx_t]
            )
            tc.tassert(tc.Int[torch.Tensor, "e"], pair_idx)

            # Get the pairwise scales
            pairwise_scales = self.pairwise_scales(pair_idx)
            tc.tassert(tc.Float[torch.Tensor, "e 1"], pairwise_scales)

            # Multiply edge energies by pairwise scales
            per_edge_energies = per_edge_energies * pairwise_scales

            # Add to node energies
            per_atom_energies_per_edge = scatter(
                per_edge_energies,
                idx_t,
                dim=0,
                dim_size=atomic_numbers.shape[0],
                reduce=self.config.reduction,
            )
            tc.tassert(tc.Float[torch.Tensor, "n 1"], per_atom_energies_per_edge)
            per_atom_energies = per_atom_energies + per_atom_energies_per_edge

        per_atom_scales = self.per_atom_scales(atomic_numbers)
        per_atom_shifts = self.per_atom_shifts(atomic_numbers)
        tc.tassert(tc.Float[torch.Tensor, "n 1"], (per_atom_scales, per_atom_shifts))

        per_atom_energies = per_atom_energies * per_atom_scales + per_atom_shifts
        tc.tassert(tc.Float[torch.Tensor, "n 1"], per_atom_energies)

        per_system_energies = scatter(
            per_atom_energies,
            data.batch,
            dim=0,
            dim_size=data.num_graphs,
            reduce=self.config.reduction,
        )
        tc.tassert(tc.Float[torch.Tensor, "b 1"], per_system_energies)

        per_system_energies = rearrange(per_system_energies, "b 1 -> b")
        return per_system_energies
