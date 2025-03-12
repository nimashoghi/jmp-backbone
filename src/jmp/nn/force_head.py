from __future__ import annotations

import contextlib
from logging import getLogger
from typing import Literal

import nshconfig as C
import nshtrainer as nt
import torch
import torch.nn as nn
from torch_geometric.data.data import BaseData
from torch_scatter import scatter
from typing_extensions import TypedDict, override

from ..models.gemnet.backbone import GOCBackboneOutput
from .base import OutputHeadBase, OutputHeadInput, TargetConfigBase
from .utils.force_scaler import ForceScaler
from .utils.tensor_grad import enable_grad

log = getLogger(__name__)


class ForceTargetConfig(TargetConfigBase):
    reduction: Literal["sum", "mean"] = "sum"
    """
    The reduction method for the target. This refers to how the target is computed.
    For example, for graph scalar targets, this refers to how the scalar targets are
    computed from each node's scalar prediction.
    """

    num_mlps: int = 1
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


class ForceOutputHead(OutputHeadBase):
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
    def forward(self, input: OutputHeadInput) -> torch.Tensor:
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

    @override
    @contextlib.contextmanager
    def forward_context(self, data: BaseData):
        yield


class ConservativeForceTargetConfig(TargetConfigBase):
    energy_prop_name: str = "energy"
    """The name of the energy property."""

    def create_model(
        self,
    ):
        return ConservativeForceOutputHead(hparams=self)


class ConservativeForceOutputHead(OutputHeadBase):
    @override
    def __init__(self, hparams: ConservativeForceTargetConfig):
        super().__init__()
        self.hparams = hparams
        self.force_saler = ForceScaler()

    @override
    def forward(self, input: OutputHeadInput) -> torch.Tensor:
        if "_stress_precomputed_forces" in input["predicted_props"]:
            return input["predicted_props"]["_stress_precomputed_forces"]
        else:
            predicted_props = input["predicted_props"]
            if self.hparams.energy_prop_name not in predicted_props:
                raise ValueError(
                    f"Predicted props does not contain {self.hparams.energy_prop_name}, check energy prop name and make sure energy is predicted before forces."
                )
            energy = predicted_props[self.hparams.energy_prop_name]
            natoms_in_batch = input["data"].pos.shape[0]
            assert (
                energy.requires_grad
            ), "Energy must require grad to compute conservative forces."
            assert (
                input["data"].pos.requires_grad
            ), "Positions must require grad to compute conservative forces."
            forces = self.force_saler.calc_forces(
                energy=energy,
                pos=input["data"].pos,
            )
            assert forces.shape == (
                natoms_in_batch,
                3,
            ), f"forces.shape={forces.shape} != [num_nodes_in_batch, 3]"
            return forces

    @override
    @contextlib.contextmanager
    def forward_context(self, data: BaseData):
        with contextlib.ExitStack() as stack:
            enable_grad(stack)

            if not data.pos.requires_grad:
                data.pos.requires_grad = True
            yield
