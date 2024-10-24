from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from torch_geometric.data import Batch
from typing_extensions import override

from ..lightning_module import Module
from ..types import Predictions


@torch.inference_mode()
@torch.no_grad()
def default_predict(data: Batch, lightning_module: Module) -> Predictions:
    # Make sure the expected properties are in the right format
    if "tags" not in data or data.tags is None or (data.tags == 0).all():
        data.tags = torch.full_like(data.atomic_numbers, 2, dtype=torch.long)

    data.atomic_numbers = data.atomic_numbers.long()
    data.natoms = data.natoms.long()
    data.tags = data.tags.long()
    data.fixed = data.fixed.bool()

    predictions = lightning_module.predict(data, energy_kind="total")
    return predictions


@runtime_checkable
class PredictCallableProtocol(Protocol):
    def __call__(self, data: Batch, /, lightning_module: Module) -> Predictions: ...


class JMPCalculator(Calculator):
    """ASE based calculator using an OCP model"""

    def __init__(
        self,
        lightning_module: Module,
        predict: PredictCallableProtocol = default_predict,
    ):
        super().__init__()

        self.lightning_module = lightning_module.eval()
        del lightning_module

        self.predict = predict
        del predict

        from fairchem.core.preprocessing import AtomsToGraphs

        self.a2g = AtomsToGraphs(
            r_energy=False,
            r_forces=False,
            r_distances=False,
            r_edges=False,
            r_pbc=True,
        )

        # Determine which properties are implemented
        self.implemented_properties: list[str] = []
        targets = self.lightning_module.config.targets
        if targets.energy:
            self.implemented_properties.append("energy")
        if targets.force:
            self.implemented_properties.append("forces")
        if targets.stress:
            self.implemented_properties.append("stress")

    @override
    def calculate(
        self,
        atoms: Atoms | Batch | None = None,
        properties=["energy"],
        system_changes=all_changes,
    ) -> None:
        """Calculate implemented properties for a single Atoms object or a Batch of them."""
        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError("Atoms object or Batch required")

        if isinstance(atoms, Atoms):
            data_object = self.a2g.convert(atoms)
            batch: Batch = Batch.from_data_list([data_object])
        else:
            batch = atoms

        predictions = self.predict(batch, self.lightning_module)

        for key, pred in predictions.items():
            pred = pred.item() if pred.numel() == 1 else pred.cpu().numpy()
            self.results[key] = pred
