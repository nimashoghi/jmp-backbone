from __future__ import annotations

from logging import getLogger
from typing import Literal

import nshconfig as C
import nshtrainer as nt
import torch
import torch.nn as nn
from jmppeft.modules.torch_scatter_polyfill import scatter
from torch_geometric.data.data import BaseData
from typing_extensions import TypedDict, override

from ..models.gemnet.backbone import GOCBackboneOutput

log = getLogger(__name__)


class ForceTargetConfig(C.Config):
    reduction: Literal["sum", "mean"] = "sum"
    """
    The reduction method for the target. This refers to how the target is computed.
    For example, for graph scalar targets, this refers to how the scalar targets are
    computed from each node's scalar prediction.
    """

    num_mlps: int = 2
    """Number of MLPs in the output layer."""

    def create_model(
        self,
        d_model_edge: int,
        activation_cls: type[nn.Module],
    ):
        return ForceOutputHead(
            hparams=self,
            d_model_edge=d_model_edge,
            activation_cls=activation_cls,
        )


class ForceOutputHeadInput(TypedDict):
    data: BaseData
    backbone_output: GOCBackboneOutput


class ForceOutputHead(nn.Module):
    @override
    def __init__(
        self,
        hparams: ForceTargetConfig,
        d_model_edge: int,
        activation_cls: type[nn.Module],
    ):
        super().__init__()

        self.hparams = hparams
        del hparams

        self.d_model_edge = d_model_edge
        self.out_mlp = nt.nn.MLP(
            ([self.d_model_edge] * self.hparams.num_mlps) + [1],
            activation=activation_cls,
        )

    @override
    def forward(self, input: ForceOutputHeadInput) -> torch.Tensor:
        data = input["data"]
        backbone_output = input["backbone_output"]

        n_atoms = data.atomic_numbers.shape[0]

        output = self.out_mlp(backbone_output["forces"])
        output = output * backbone_output["V_st"]  # (n_edges, 3)
        output = scatter(
            output,
            backbone_output["idx_t"],
            dim=0,
            dim_size=n_atoms,
            reduce=self.hparams.reduction,
        )
        return output
