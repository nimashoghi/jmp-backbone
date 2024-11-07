from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import nshconfig as C
import nshconfig_extra as CE
import nshtrainer as nt
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.fabric.utilities.apply_func import move_data_to_device
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch_geometric.data import Batch
from typing_extensions import assert_never, override

from .loss import l2mae_loss
from .metrics import ForceFieldMetrics
from .models.gemnet.backbone import GemNetOCBackbone, GOCBackboneOutput
from .models.gemnet.graph import GraphComputer, GraphComputerConfig
from .nn.energy_head import EnergyTargetConfig
from .nn.force_head import ForceTargetConfig
from .nn.stress_head import StressTargetConfig
from .referencing import IdentityReferencerConfig, ReferencerConfig
from .types import Predictions

log = logging.getLogger(__name__)


class ModelOutput(TypedDict):
    energy: torch.Tensor
    forces: torch.Tensor
    stress_isotropic: torch.Tensor
    stress_anisotropic: torch.Tensor


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
    stress_loss_coefficient: float | tuple[float, float]
    """Coefficient for the stress loss (isotropic, anisotropic)."""

    @property
    def _isotropic_stress_loss_coefficient(self):
        return (
            self.stress_loss_coefficient
            if not isinstance(self.stress_loss_coefficient, tuple)
            else self.stress_loss_coefficient[0]
        )

    @property
    def _anisotropic_stress_loss_coefficient(self):
        return (
            self.stress_loss_coefficient
            if not isinstance(self.stress_loss_coefficient, tuple)
            else self.stress_loss_coefficient[1]
        )


class SeparateLRMultiplierConfig(C.Config):
    backbone_multiplier: float
    """Learning rate multiplier for the backbone."""

    rest_multiplier: float = 1.0
    """Learning rate multiplier for the rest of the model (heads)."""


class OptimizationConfig(C.Config):
    optimizer: nt.configs.OptimizerConfig
    """Optimizer configuration."""

    lr_scheduler: nt.configs.LRSchedulerConfig | None
    """Learning rate scheduler configuration."""

    separate_lr_multiplier: SeparateLRMultiplierConfig | None = None
    """Separate learning rate multipliers for the backbone and heads."""


class NormalizationConfig(C.Config):
    mean: float
    """Mean value for normalization."""

    rmsd: float
    """Root mean square deviation for normalization."""

    @torch.autocast(device_type="cuda", enabled=False)
    def norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.rmsd

    @torch.autocast(device_type="cuda", enabled=False)
    def denorm(self, normed_tensor: torch.Tensor) -> torch.Tensor:
        return normed_tensor * self.rmsd + self.mean


class Config(C.Config):
    pretrained_ckpt: CE.CachedPath | None
    """Path to the pretrained checkpoint."""

    graph_computer: GraphComputerConfig
    """Graph computer configuration."""

    ignore_graph_generation_errors: bool = False
    """Whether to ignore errors during graph generation."""

    targets: TargetsConfig
    """Targets configuration."""

    optimization: OptimizationConfig
    """Optimization configuration."""

    energy_referencer: ReferencerConfig = IdentityReferencerConfig()
    """Energy referencing configuration."""

    normalization: dict[str, NormalizationConfig] = {}
    """Normalization configuration for the input data."""


class Module(nt.LightningModuleBase[Config]):
    @override
    @classmethod
    def hparams_cls(cls):
        return Config

    @override
    def __init__(self, hparams):
        super().__init__(hparams)

        # Backbone
        self.backbone = GemNetOCBackbone.from_pretrained_ckpt(
            self.hparams.pretrained_ckpt.resolve()
        )
        d_model = self.backbone.hparams.emb_size_atom
        d_model_edge = self.backbone.hparams.emb_size_edge
        match self.backbone.hparams.activation:
            case "scaled_silu" | "scaled_swish":
                from .models.gemnet.layers.base_layers import ScaledSiLU

                activation_cls = ScaledSiLU
            case "silu" | "swish":
                activation_cls = nn.SiLU
            case _:
                raise ValueError(
                    f"Unknown activation: {self.backbone.hparams.activation}"
                )

        # Output heads
        self.energy_head = self.hparams.targets.energy.create_model(
            d_model=d_model,
            d_model_edge=d_model_edge,
            activation_cls=activation_cls,
        )
        self.force_head = self.hparams.targets.force.create_model(
            d_model_edge=d_model_edge,
            activation_cls=activation_cls,
        )
        self.stress_head = self.hparams.targets.stress.create_model(
            d_model_edge=d_model_edge,
            activation_cls=activation_cls,
        )

        # Graph computer
        self.graph_computer = GraphComputer(
            self.hparams.graph_computer,
            self.backbone.hparams,
            # self._process_aint_graph_transform,
        )

        # Energy referencing
        self.energy_referencer = self.hparams.energy_referencer.create_referencer()
        log.info(f"Energy referencer: {self.energy_referencer}")

        # Metrics
        self.train_metrics = ForceFieldMetrics()
        self.val_metrics = ForceFieldMetrics()
        self.test_metrics = ForceFieldMetrics()

    @override
    def forward(self, data: Batch):
        backbone_output = self.backbone(data)

        output_head_input = {"backbone_output": backbone_output, "data": data}
        energy = self.energy_head(output_head_input)
        forces = self.force_head(output_head_input)
        stress_isotropic, stress_anisotropic = self.stress_head(output_head_input)
        outputs: ModelOutput = {
            "energy": energy,
            "forces": forces,
            "stress_isotropic": stress_isotropic,
            "stress_anisotropic": stress_anisotropic,
        }

        return outputs

    def embeddings(self, batch: Batch):
        # Move the batch to the correct device
        batch = move_data_to_device(batch, self.device)

        # Compute graphs
        batch = self.graph_computer(batch)

        embeddings: GOCBackboneOutput = self.backbone(batch)
        return embeddings

    def _norm_targets(self, targets: ModelOutput) -> ModelOutput:
        return cast(
            ModelOutput,
            {
                key: norm.norm(value)
                if (norm := self.hparams.normalization.get(key))
                else value
                for key, value in targets.items()
            },
        )

    def _denorm_model_output(self, outputs: ModelOutput) -> ModelOutput:
        return cast(
            ModelOutput,
            {
                key: norm.denorm(value)
                if (norm := self.hparams.normalization.get(key))
                else value
                for key, value in outputs.items()
            },
        )

    def _undo_linref(
        self,
        outputs: ModelOutput,
        data: Batch,
        energy_kind: Literal["total", "referenced"] = "total",
    ):
        match energy_kind:
            case "total":
                outputs["energy"] = self.energy_referencer.dereference(
                    outputs["energy"],
                    data.atomic_numbers,
                    data.batch,
                    data.num_graphs,
                )
            case "referenced":
                # Nothing to do here, the energy referencing is already applied in the forward
                pass
            case _:
                assert_never(energy_kind)
        return outputs

    def _model_output_to_predictions(
        self,
        outputs: ModelOutput,
        batch: Batch,
        energy_kind: Literal["total", "referenced"] = "total",
    ) -> Predictions:
        # Denormalize the outputs
        outputs = self._denorm_model_output(outputs)

        # Energy referencing
        outputs = self._undo_linref(outputs, batch, energy_kind)

        # Compute stress from the isotropic and anisotropic components
        stress = self.stress_head.combine_scalar_irrep2(
            outputs["stress_isotropic"], outputs["stress_anisotropic"]
        )

        return {
            "energy": outputs["energy"],
            "forces": outputs["forces"],
            "stress": stress,
        }

    def predict(
        self,
        batch: Batch,
        *,
        energy_kind: Literal["total", "referenced"] = "total",
    ) -> Predictions:
        """
        Perform a forward pass and return the predictions.

        Args:
            batch (Batch): Input batch.
            energy_kind (Literal["total", "referenced"]): Kind of energy to return.
                - "total": Total energy.
                - "referenced": Referenced energy.

        Returns:
            Predictions: Model predictions.
        """
        # Move the batch to the correct device
        batch = move_data_to_device(batch, self.device)

        # Compute graphs
        batch = self.graph_computer(batch)

        # Perform the forward pass
        outputs: ModelOutput = self(batch)

        # ModelOutput -> Predictions
        predictions = self._model_output_to_predictions(outputs, batch, energy_kind)

        return predictions

    def energy_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
    ):
        if True or (world_size := self.trainer.world_size) == 1:
            return F.l1_loss(prediction / natoms, target / natoms, reduction="mean")

        # Multiply by world size since gradients are averaged across DDP replicas
        loss = F.l1_loss(prediction / natoms, target / natoms, reduction="sum")
        num_samples = torch.tensor(
            target.shape[0], dtype=loss.dtype, device=loss.device
        )
        num_samples = self.trainer.strategy.reduce(num_samples, reduce_op="sum")
        loss = (loss * world_size) / num_samples
        return loss

    def forces_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        if True or (world_size := self.trainer.world_size) == 1:
            return l2mae_loss(prediction, target, reduction="mean")

        # Multiply by world size since gradients are averaged across DDP replicas
        loss = l2mae_loss(prediction, target, reduction="sum")
        num_samples = torch.tensor(
            target.shape[0], dtype=loss.dtype, device=loss.device
        )
        num_samples = self.trainer.strategy.reduce(num_samples, reduce_op="sum")
        loss = (loss * world_size) / num_samples
        return loss

    def isotropic_stress_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        if True or (world_size := self.trainer.world_size) == 1:
            return F.l1_loss(prediction, target, reduction="mean")

        # Multiply by world size since gradients are averaged across DDP replicas
        loss = F.l1_loss(prediction, target, reduction="sum")
        num_samples = torch.tensor(
            target.shape[0], dtype=loss.dtype, device=loss.device
        )
        num_samples = self.trainer.strategy.reduce(num_samples, reduce_op="sum")
        loss = (loss * world_size) / num_samples
        return loss

    def anisotropic_stress_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        return F.l1_loss(prediction, target, reduction="mean")

    def _compute_loss(self, model_output: ModelOutput, data: Batch):
        # Create a targets dict with a similar structure to the model output
        targets: ModelOutput = {
            "energy": data.y,
            "forces": data.force,
            "stress_isotropic": data.stress_isotropic.squeeze(dim=-1),
            "stress_anisotropic": data.stress_anisotropic,
        }
        # Normalize targets
        targets = self._norm_targets(targets)

        losses: list[torch.Tensor] = []

        # Energy loss
        energy_loss = self.energy_loss(
            model_output["energy"],
            targets["energy"],
            data.natoms,
        )
        self.log("energy_loss", energy_loss)
        losses.append(energy_loss * self.hparams.targets.energy_loss_coefficient)

        # Force loss
        force_loss = self.forces_loss(model_output["forces"], targets["forces"])
        self.log("force_loss", force_loss)
        losses.append(force_loss * self.hparams.targets.force_loss_coefficient)
        losses.append(force_loss)

        # Isotropic stress loss
        isotropic_stress_loss = self.isotropic_stress_loss(
            model_output["stress_isotropic"], targets["stress_isotropic"]
        )
        self.log("isotropic_stress_loss", isotropic_stress_loss)
        losses.append(
            isotropic_stress_loss
            * self.hparams.targets._isotropic_stress_loss_coefficient
        )

        # Anisotropic stress loss
        anisotropic_stress_loss = self.anisotropic_stress_loss(
            model_output["stress_anisotropic"], targets["stress_anisotropic"]
        )
        self.log("anisotropic_stress_loss", anisotropic_stress_loss)
        losses.append(
            anisotropic_stress_loss
            * self.hparams.targets._anisotropic_stress_loss_coefficient
        )

        # Total loss
        loss = cast(torch.Tensor, sum(losses))
        return loss

    def _common_step(self, data: Batch, metrics: ForceFieldMetrics):
        # Compute graphs
        if self.hparams.ignore_graph_generation_errors:
            try:
                data = self.graph_computer(data)
            except Exception as e:
                # If this is a CUDA error, rethrow it
                if "CUDA" in str(data):
                    raise

                # Otherwise, log the error and skip the batch
                log.error(f"Error generating graphs: {e}", exc_info=True)
                return self.zero_loss()
        else:
            data = self.graph_computer(data)

        # Apply energy referencing
        data.y_total = data.y.clone()
        data.y = self.energy_referencer.reference(
            data.y_total,
            data.atomic_numbers,
            data.batch,
            data.num_graphs,
        )

        # Forward pass
        outputs: ModelOutput = self(data)

        # Compute loss
        loss = self._compute_loss(outputs, data)
        self.log("loss", loss)

        # Undo energy referencing
        data.y = data.pop("y_total")

        # ModelOutput -> Predictions for metrics
        predictions = self._model_output_to_predictions(outputs, data, "total")

        # Compute metrics
        self.log_dict(metrics(predictions, data))

        return loss

    @override
    def training_step(self, batch: Batch, batch_idx: int):
        with self.log_context(prefix="train/"):
            return self._common_step(batch, self.train_metrics)

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
        config = self.hparams.optimization

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

    @classmethod
    def load_ckpt(
        cls,
        path: Path,
        update_hparams: Callable[[Config], Config] | None = None,
        update_hparams_dict: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        ckpt = torch.load(path, map_location="cpu")

        hparams_dict = ckpt[cls.CHECKPOINT_HYPER_PARAMS_KEY]
        if update_hparams_dict is not None:
            hparams_dict = update_hparams_dict(hparams_dict)

        hparams = cls.hparams_cls().model_validate(hparams_dict)
        if update_hparams is not None:
            hparams = update_hparams(hparams)

        model = cls(hparams)
        model.load_state_dict(ckpt["state_dict"])
        return model
