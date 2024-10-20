from __future__ import annotations

import torch.nn as nn
import torchmetrics
from torch_geometric.data import Batch
from typing_extensions import override

from .types import Predictions


class ForceFieldMetrics(nn.Module):
    @override
    def __init__(self):
        super().__init__()

        # Energy metrics
        self.energy_mae = torchmetrics.MeanAbsoluteError()
        self.energy_mse = torchmetrics.MeanSquaredError(squared=True)
        self.energy_rmse = torchmetrics.MeanSquaredError(squared=False)

        # Force metrics
        self.force_mae = torchmetrics.MeanAbsoluteError()
        self.force_mse = torchmetrics.MeanSquaredError(squared=True)
        self.force_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.force_cos = torchmetrics.CosineSimilarity(reduction="mean")

        # Stress metrics
        self.stress_mae = torchmetrics.MeanAbsoluteError()
        self.stress_mse = torchmetrics.MeanSquaredError(squared=True)
        self.stress_rmse = torchmetrics.MeanSquaredError(squared=False)

    @override
    def forward(self, prediction: Predictions, data: Batch):
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

        # Energy metrics
        self.energy_mae(energy_hat, energy_true)
        self.energy_mse(energy_hat, energy_true)
        self.energy_rmse(energy_hat, energy_true)

        # Force metrics
        self.force_mae(forces_hat, forces_true)
        self.force_mse(forces_hat, forces_true)
        self.force_rmse(forces_hat.float(), forces_true.float())
        self.force_cos(forces_hat, forces_true)

        # Stress metrics
        self.stress_mae(stress_hat, stress_true)
        self.stress_mse(stress_hat, stress_true)
        self.stress_rmse(stress_hat, stress_true)

        return {
            "energy_mae": self.energy_mae,
            "energy_mse": self.energy_mse,
            "energy_rmse": self.energy_rmse,
            "force_mae": self.force_mae,
            "force_mse": self.force_mse,
            "force_rmse": self.force_rmse,
            "force_cos": self.force_cos,
            "stress_mae": self.stress_mae,
            "stress_mse": self.stress_mse,
            "stress_rmse": self.stress_rmse,
        }
