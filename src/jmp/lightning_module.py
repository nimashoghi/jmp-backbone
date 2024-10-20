from __future__ import annotations

from typing import cast

import nshconfig as C
import nshconfig_extra as CE
import nshtrainer as nt
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch_geometric.data import Batch
from typing_extensions import override

from .metrics import ForceFieldMetrics
from .models.gemnet.backbone import GemNetOCBackbone
from .models.gemnet.graph import GraphComputer, GraphComputerConfig
from .nn.energy_head import EnergyTargetConfig
from .nn.force_head import ForceTargetConfig
from .nn.stress_head import StressTargetConfig
from .types import Predictions


class TargetsConfig(C.Config):
    energy: EnergyTargetConfig
    """Energy target configuration."""
    force: ForceTargetConfig
    """Force target configuration."""
    stress: StressTargetConfig
    """Stress target configuration."""

    energy_loss_coefficient: float
    """Coefficient for the energy loss."""
    force_loss_coefficient: float
    """Coefficient for the force loss."""
    stress_loss_coefficient: float
    """Coefficient for the stress loss."""


class SeparateLRMultiplierConfig(C.Config):
    backbone_multiplier: float
    """Learning rate multiplier for the backbone."""

    rest_multiplier: float = 1.0
    """Learning rate multiplier for the rest of the model (heads)."""


class OptimizationConfig(C.Config):
    optimizer: nt.config.OptimizerConfig
    """Optimizer configuration."""

    lr_scheduler: nt.config.LRSchedulerConfig | None
    """Learning rate scheduler configuration."""

    separate_lr_multiplier: SeparateLRMultiplierConfig | None = None
    """Separate learning rate multipliers for the backbone and heads."""


class Config(nt.BaseConfig):
    pretrained_ckpt: CE.CachedPath
    """Path to the pretrained checkpoint."""

    graph_computer: GraphComputerConfig
    """Graph computer configuration."""

    targets: TargetsConfig
    """Targets configuration."""

    optimization: OptimizationConfig
    """Optimization configuration."""


class Module(nt.LightningModuleBase[Config]):
    @override
    @classmethod
    def config_cls(cls):
        return Config

    @override
    def __init__(self, hparams):
        super().__init__(hparams)

        # Backbone
        self.backbone = GemNetOCBackbone.from_pretrained_ckpt(
            self.config.pretrained_ckpt.resolve()
        )
        d_model = self.backbone.config.emb_size_atom
        d_model_edge = self.backbone.config.emb_size_edge
        match self.backbone.config.activation:
            case "scaled_silu" | "scaled_swish":
                from .models.gemnet.layers.base_layers import ScaledSiLU

                activation_cls = ScaledSiLU
            case "silu" | "swish":
                activation_cls = nn.SiLU
            case _:
                raise ValueError(
                    f"Unknown activation: {self.backbone.config.activation}"
                )

        # Output heads
        self.energy_head = self.config.targets.energy.create_model(
            d_model=d_model,
            d_model_edge=d_model_edge,
            activation_cls=activation_cls,
        )
        self.force_head = self.config.targets.force.create_model(
            d_model_edge=d_model_edge,
            activation_cls=activation_cls,
        )
        self.stress_head = self.config.targets.stress.create_model(
            d_model_edge=d_model_edge,
            activation_cls=activation_cls,
        )

        # Graph computer
        self.graph_computer = GraphComputer(
            self.config.graph_computer,
            self.backbone.config,
            # self._process_aint_graph_transform,
        )

        # Metrics
        self.train_metrics = ForceFieldMetrics()
        self.val_metrics = ForceFieldMetrics()
        self.test_metrics = ForceFieldMetrics()

    @override
    def forward(self, data: Batch):
        backbone_output = self.backbone(data)

        output_head_input = {"backbone_output": backbone_output, "data": data}
        outputs: Predictions = {
            "energy": self.energy_head(output_head_input),
            "forces": self.force_head(output_head_input),
            "stress": self.stress_head(output_head_input),
        }
        return outputs

    def _compute_loss(self, prediction: Predictions, data: Batch):
        energy_hat, forces_hat, stress_hat = (
            prediction["energy"],
            prediction["forces"],
            prediction["stress"],
        )
        energy_true, forces_true, stress_true = (
            data.y,
            data.force,
            data.stress,
        )

        losses: list[torch.Tensor] = []

        # Energy loss
        energy_loss = (
            F.l1_loss(energy_hat, energy_true)
            * self.config.targets.energy_loss_coefficient
        )
        losses.append(energy_loss)

        # Force loss
        force_loss = (
            F.l1_loss(forces_hat, forces_true)
            * self.config.targets.force_loss_coefficient
        )
        losses.append(force_loss)

        # Stress loss
        stress_loss = (
            F.l1_loss(stress_hat, stress_true)
            * self.config.targets.stress_loss_coefficient
        )
        losses.append(stress_loss)

        # Total loss
        loss = cast(torch.Tensor, sum(losses))
        return loss

    def _common_step(self, data: Batch, metrics: ForceFieldMetrics):
        # Compute graphs
        data = self.graph_computer(data)

        # Forward pass
        outputs = self(data)
        outputs = cast(Predictions, outputs)

        # Compute loss
        loss = self._compute_loss(outputs, data)
        self.log("loss", loss)

        # Compute metrics
        self.log_dict(metrics(outputs, data))

        return loss

    @override
    def training_step(self, batch: Batch, batch_idx: int):
        with self.log_context(prefix="train/"):
            loss = self._common_step(batch, self.train_metrics)

        return loss

    @override
    def validation_step(self, batch: Batch, batch_idx: int):
        with self.log_context(prefix="val/"):
            _ = self._common_step(batch, self.val_metrics)

    @override
    def test_step(self, batch: Batch, batch_idx: int):
        with self.log_context(prefix="test/"):
            _ = self._common_step(batch, self.test_metrics)

    @override
    def configure_optimizers(self):
        config = self.config.optimization

        if (lr_mult := config.separate_lr_multiplier) is None:
            optimizer = config.optimizer.create_optimizer(self.parameters())
        else:
            backbone_params = list(self.backbone.parameters())
            backbone_param_set = set(backbone_params)

            rest_params: list[nn.Parameter] = []
            for param in self.parameters():
                if param not in backbone_param_set:
                    rest_params.append(param)

            optimizer = config.optimizer.create_optimizer(
                [
                    {
                        "params": backbone_params,
                        "lr": config.optimizer.lr * lr_mult.backbone_multiplier,
                        "name": "backbone",
                    },
                    {
                        "params": rest_params,
                        "lr": config.optimizer.lr * lr_mult.rest_multiplier,
                        "name": "rest",
                    },
                ]
            )

        output: OptimizerLRSchedulerConfig = {"optimizer": optimizer}
        if config.lr_scheduler is not None:
            output["lr_scheduler"] = config.lr_scheduler.create_scheduler(
                output["optimizer"], self
            )

        return output
