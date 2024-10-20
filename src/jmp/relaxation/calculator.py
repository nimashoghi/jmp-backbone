from __future__ import annotations

from types import MappingProxyType

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.geometry import wrap_positions
from fairchem.core.preprocessing import AtomsToGraphs
from torch_geometric.data import Batch
from typing_extensions import override

from ..lightning_module import Module

# system level model predictions have different shapes than expected by ASE
ASE_PROP_RESHAPE = MappingProxyType(
    {"stress": (-1, 3, 3), "dielectric_tensor": (-1, 3, 3)}
)


def batch_to_atoms(
    batch: Batch,
    results: dict[str, torch.Tensor] | None = None,
    wrap_pos: bool = True,
    eps: float = 1e-7,
) -> list[Atoms]:
    """Convert a data batch to ase Atoms

    Args:
        batch: data batch
        results: dictionary with predicted result tensors that will be added to a SinglePointCalculator. If no results
            are given no calculator will be added to the atoms objects.
        wrap_pos: wrap positions back into the cell.
        eps: Small number to prevent slightly negative coordinates from being wrapped.

    Returns:
        list of Atoms
    """
    n_systems = batch.natoms.shape[0]
    natoms = batch.natoms.tolist()
    numbers = torch.split(batch.atomic_numbers, natoms)
    fixed = torch.split(batch.fixed.to(torch.bool), natoms)
    if results is not None:
        results = {
            key: val.view(ASE_PROP_RESHAPE.get(key, -1)).tolist()
            if len(val) == len(batch)
            else [v.cpu().detach().numpy() for v in torch.split(val, natoms)]
            for key, val in results.items()
        }

    positions = torch.split(batch.pos, natoms)
    tags = torch.split(batch.tags, natoms)
    cells = batch.cell

    atoms_objects = []
    for idx in range(n_systems):
        pos = positions[idx].cpu().detach().numpy()
        cell = cells[idx].cpu().detach().numpy()

        # TODO take pbc from data
        if wrap_pos:
            pos = wrap_positions(pos, cell, pbc=[True, True, True], eps=eps)

        atoms = Atoms(
            numbers=numbers[idx].tolist(),
            cell=cell,
            positions=pos,
            tags=tags[idx].tolist(),
            constraint=FixAtoms(mask=fixed[idx].tolist()),
            pbc=[True, True, True],
        )

        if results is not None:
            calc = SinglePointCalculator(
                atoms=atoms, **{key: val[idx] for key, val in results.items()}
            )
            atoms.set_calculator(calc)

        atoms_objects.append(atoms)

    return atoms_objects


class JMPCalculator(Calculator):
    """ASE based calculator using an OCP model"""

    _reshaped_props = ASE_PROP_RESHAPE

    def __init__(self, lightning_module: Module):
        super().__init__()

        self.lightning_module = lightning_module
        del lightning_module

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
    def calculate(self, atoms: Atoms | Batch, properties, system_changes) -> None:
        """Calculate implemented properties for a single Atoms object or a Batch of them."""
        super().calculate(atoms, properties, system_changes)

        if isinstance(atoms, Atoms):
            data_object = self.a2g.convert(atoms)
            batch = Batch.from_data_list([data_object])
        else:
            batch = atoms

        predictions = self.lightning_module.predict(batch)
        for key, pred in predictions.items():
            pred = pred.item() if pred.numel() == 1 else pred.cpu().numpy()
            self.results[key] = pred
