from __future__ import annotations

import contextlib
from typing import Literal

import nshconfig as C
import nshutils.typecheck as tc
import torch
import torch.nn as nn
from e3nn import o3
from einops import rearrange
from torch_geometric.data.data import BaseData
from torch_scatter import scatter
from typing_extensions import TypedDict, assert_never, override

from ..models.gemnet.backbone import GOCBackboneOutput
from .base import OutputHeadBase, OutputHeadInput, TargetConfigBase
from .utils.force_scaler import ForceStressScaler
from .utils.tensor_grad import enable_grad


class _Rank2DecompositionEdgeBlock(nn.Module):
    r"""Prediction of rank 2 tensor
    Decompose rank 2 tensor with irreps
    since it is symmetric we need just irrep degree 0 and 2
    Parameters
    ----------
    emb_size : int
        size of edge embedding used to compute outer products
    num_layers : int
        number of layers of the MLP
    --------
    """

    change_mat: tc.Float[torch.Tensor, "9 9"]

    def __init__(
        self,
        emb_size,
        edge_level,
        extensive=False,
        num_layers=2,
        activation_cls: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.edge_level = edge_level
        self.extensive = extensive
        self.scalar_nonlinearity = activation_cls()
        self.scalar_MLP = nn.ModuleList()
        self.irrep2_MLP = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                self.scalar_MLP.append(nn.Linear(emb_size, emb_size))
                self.irrep2_MLP.append(nn.Linear(emb_size, emb_size))
                self.scalar_MLP.append(self.scalar_nonlinearity)
                self.irrep2_MLP.append(self.scalar_nonlinearity)
            else:
                self.scalar_MLP.append(nn.Linear(emb_size, 1))
                self.irrep2_MLP.append(nn.Linear(emb_size, 1))

        # Change of basis obtained by stacking the C-G coefficients in the right way

        self.register_buffer(
            "change_mat",
            torch.transpose(
                torch.tensor(
                    [
                        [3 ** (-0.5), 0, 0, 0, 3 ** (-0.5), 0, 0, 0, 3 ** (-0.5)],
                        [0, 0, 0, 0, 0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0],
                        [0, 0, -(2 ** (-0.5)), 0, 0, 0, 2 ** (-0.5), 0, 0],
                        [0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0, 0, 0, 0, 0],
                        [0, 0, 0.5**0.5, 0, 0, 0, 0.5**0.5, 0, 0],
                        [0, 2 ** (-0.5), 0, 2 ** (-0.5), 0, 0, 0, 0, 0],
                        [
                            -(6 ** (-0.5)),
                            0,
                            0,
                            0,
                            2 * 6 ** (-0.5),
                            0,
                            0,
                            0,
                            -(6 ** (-0.5)),
                        ],
                        [0, 0, 0, 0, 0, 2 ** (-0.5), 0, 2 ** (-0.5), 0],
                        [-(2 ** (-0.5)), 0, 0, 0, 0, 0, 0, 0, 2 ** (-0.5)],
                    ]
                ).detach(),
                0,
                1,
            ),
            persistent=False,
        )

    @override
    def forward(
        self,
        x_edge: tc.Float[torch.Tensor, "num_edges emb_size"],
        edge_vec: tc.Float[torch.Tensor, "num_edges 3"],
        idx_t: tc.Int[torch.Tensor, "num_edges"],
        batch_idx: tc.Int[torch.Tensor, "num_nodes"],
        batch_size: int,
        return_decomposed: bool = True,
    ):
        """evaluate
        Parameters
        ----------
        x_edge : `torch.Tensor`
            tensor of shape ``(nEdges, emb_size)``
        edge_vec : `torch.Tensor`
            tensor of shape ``(nEdges, 3)``
        data : ``LMDBDataset sample``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., 3, 3)``
        """
        # Calculate spherical harmonics of degree 2 of the points sampled
        sphere_irrep2 = o3.spherical_harmonics(
            2, edge_vec, True
        ).detach()  # (nEdges, 5)

        if self.edge_level:
            # Irrep 0 prediction
            edge_scalar = x_edge
            for i, module in enumerate(self.scalar_MLP):
                edge_scalar = module(edge_scalar)

            # Irrep 2 prediction
            edge_irrep2 = x_edge  # (nEdges, 5, emb_size)
            for i, module in enumerate(self.irrep2_MLP):
                edge_irrep2 = module(edge_irrep2)
            edge_irrep2 = sphere_irrep2[:, :, None] * edge_irrep2[:, None, :]

            node_scalar = scatter(
                edge_scalar,
                idx_t,
                dim=0,
                dim_size=batch_idx.shape[0],
                reduce="mean",
            )
            node_irrep2 = scatter(
                edge_irrep2,
                idx_t,
                dim=0,
                dim_size=batch_idx.shape[0],
                reduce="mean",
            )
        else:
            edge_irrep2 = (
                sphere_irrep2[:, :, None] * x_edge[:, None, :]
            )  # (nAtoms, 5, emb_size)

            node_scalar = scatter(x_edge, idx_t, dim=0, reduce="mean")
            node_irrep2 = scatter(edge_irrep2, idx_t, dim=0, reduce="mean")

            # Irrep 0 prediction
            for i, module in enumerate(self.scalar_MLP):
                if i == 0:
                    node_scalar = module(node_scalar)
                else:
                    node_scalar = module(node_scalar)

            # Irrep 2 prediction
            for i, module in enumerate(self.irrep2_MLP):
                if i == 0:
                    node_irrep2 = module(node_irrep2)
                else:
                    node_irrep2 = module(node_irrep2)

        if self.extensive:
            scalar = scatter(
                node_scalar.view(-1),
                batch_idx,
                dim=0,
                dim_size=batch_size,
                reduce="sum",
            )
            irrep2 = scatter(
                node_irrep2.view(-1, 5),
                batch_idx,
                dim=0,
                dim_size=batch_size,
                reduce="sum",
            )
        else:
            irrep2 = scatter(
                node_irrep2.view(-1, 5),
                batch_idx,
                dim=0,
                dim_size=batch_size,
                reduce="mean",
            )
            scalar = scatter(
                node_scalar.view(-1),
                batch_idx,
                dim=0,
                dim_size=batch_size,
                reduce="mean",
            )

        if return_decomposed:
            return scalar, irrep2

        return self.combine_scalar_irrep2(scalar, irrep2)

    def combine_scalar_irrep2(self, scalar, irrep2):
        # Change of basis to compute a rank 2 symmetric tensor

        vector = torch.zeros((scalar.shape[0], 3), device=scalar.device).detach()
        flatten_irreps = torch.cat([scalar.reshape(-1, 1), vector, irrep2], dim=1)
        stress = torch.einsum(
            "ab, cb->ca", self.change_mat.to(flatten_irreps.device), flatten_irreps
        )
        tc.tassert(tc.Float[torch.Tensor, "bsz nine"], stress)

        stress = rearrange(
            stress,
            "b (three1 three2) -> b three1 three2",
            three1=3,
            three2=3,
        ).contiguous()

        return stress


class StressTargetConfig(TargetConfigBase):
    reduction: Literal["sum", "mean"] = "sum"
    """
    The reduction method for the target. This refers to how the target is computed.
    For example, for graph scalar targets, this refers to how the scalar targets are
    computed from each node's scalar prediction.
    """

    num_layers: int = 1
    """The number of layers in the output head"""

    edge_level: bool = True
    """Whether to predict at the edge level"""

    @property
    def extensive(self):
        match self.reduction:
            case "sum":
                return True
            case "mean":
                return False
            case _:
                assert_never(self.reduction)

    def create_model(
        self,
        d_model_edge: int,
        activation_cls: type[nn.Module],
    ):
        return StressOutputHead(
            hparams=self,
            d_model_edge=d_model_edge,
            activation_cls=activation_cls,
        )


class StressOutputHead(OutputHeadBase):
    @override
    def __init__(
        self,
        hparams: StressTargetConfig,
        d_model_edge: int,
        activation_cls: type[nn.Module],
    ):
        super().__init__()

        self.hparams = hparams
        del hparams

        self.block = _Rank2DecompositionEdgeBlock(
            d_model_edge,
            edge_level=self.hparams.edge_level,
            extensive=self.hparams.extensive,
            num_layers=self.hparams.num_layers,
            activation_cls=activation_cls,
        )

    @override
    def forward(
        self,
        input: OutputHeadInput,
        return_decomposed: bool = True,
    ) -> tc.Float[torch.Tensor, "bsz 3 3"]:
        return self.block(
            input["backbone_output"]["forces"],
            input["backbone_output"]["V_st"],
            input["backbone_output"]["idx_t"],
            input["data"].batch,
            input["data"].cell.shape[0],
            return_decomposed=return_decomposed,
        )

    @override
    @contextlib.contextmanager
    def forward_context(self, data: BaseData):
        yield

    def combine_scalar_irrep2(self, scalar, irrep2):
        return self.block.combine_scalar_irrep2(scalar, irrep2)


class ConservativeStressTargetConfig(TargetConfigBase):
    energy_prop_name: str = "energy"
    """The name of the energy property."""

    def create_model(
        self,
    ):
        return ConservativeStressOutputHead(
            hparams=self,
        )


class ConservativeStressOutputHead(OutputHeadBase):
    @override
    def __init__(
        self,
        hparams: ConservativeStressTargetConfig,
    ):
        super().__init__()
        self.hparams = hparams
        self.force_stress_scaler = ForceStressScaler()

    @override
    def forward(self, input: OutputHeadInput) -> tc.Float[torch.Tensor, "bsz 3 3"]:
        predicted_props = input["predicted_props"]
        if self.hparams.energy_prop_name not in predicted_props:
            raise ValueError(
                f"Predicted props does not contain {self.hparams.energy_prop_name}, check energy prop name and make sure energy is predicted before stress."
            )
        energy = predicted_props[self.hparams.energy_prop_name]
        strain = input["data"].strain
        cell = input["data"].cell
        pos = input["data"].pos
        forces, stress = self.force_stress_scaler.calc_forces_and_update(
            energy, pos, strain, cell
        )
        input["predicted_props"]["_stress_precomputed_forces"] = forces
        return stress

    @override
    @contextlib.contextmanager
    def forward_context(self, data: BaseData):
        with contextlib.ExitStack() as stack:
            enable_grad(stack)

            if not data.pos.requires_grad:
                data.pos.requires_grad_(True)

            num_graphs = int(torch.max(data.batch).item() + 1)
            data.strain = torch.zeros(
                (num_graphs, 3, 3),
                dtype=data.pos.dtype,
                device=data.pos.device,
            )
            data.strain.requires_grad_(True)
            symmetric_displacement = 0.5 * (data.strain + data.strain.transpose(-1, -2))

            data.pos = data.pos + torch.bmm(
                data.pos.unsqueeze(-2), symmetric_displacement[data.batch]
            ).squeeze(-2)
            data.cell = data.cell + torch.bmm(data.cell, symmetric_displacement)

            yield
