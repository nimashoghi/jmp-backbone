from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import dill
import nshconfig as C
from ase import Atoms
from ase.filters import FrechetCellFilter, UnitCellFilter
from ase.optimize import BFGS, FIRE, LBFGS
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm
from typing_extensions import NotRequired, TypedDict

from ..lightning_module import Module
from .calculator import JMPCalculator

FILTER_CLS = {"frechet": FrechetCellFilter, "unit": UnitCellFilter}
OPTIM_CLS = {"FIRE": FIRE, "LBFGS": LBFGS, "BFGS": BFGS}


class RelaxerConfig(C.Config):
    results_dir: Path
    """Directory to save the relaxation results."""

    optimizer: Literal["FIRE", "LBFGS", "BFGS"]
    """ASE optimizer to use for relaxation."""

    optimizer_kwargs: dict[str, Any] = {}
    """Keyword arguments to pass to the optimizer."""

    force_max: float
    """Maximum force allowed during relaxation."""

    max_steps: int
    """Maximum number of relaxation steps."""

    cell_filter: Literal["frechet", "unit"] | None = None
    """Cell filter to use for relaxation."""

    optim_log_file: Path = Path("/dev/null")
    """Path to the log file for the optimizer. If None, the log file will be written to /dev/null."""

    def _cell_filter_cls(self):
        if self.cell_filter is None:
            return None
        return FILTER_CLS[self.cell_filter]

    def _optim_cls(self):
        return OPTIM_CLS[self.optimizer]


def _write_result(
    config: RelaxerConfig,
    material_id: str,
    result: dict[str, Any],
):
    result_path = config.results_dir / f"{material_id}.dill"
    with open(result_path, "wb") as f:
        dill.dump(result, f)


class DatasetItem(TypedDict):
    material_id: str
    """Material ID of the structure."""

    atoms: Atoms
    """ase.Atoms object representing the structure."""

    metadata: NotRequired[dict[str, Any]]
    """Metadata associated with the structure, will be saved with the relaxation results."""


def relax(
    config: RelaxerConfig, lightning_module: Module, dataset: Iterable[DatasetItem]
):
    """Run WBM relaxations using an ASE optimizer."""

    # Create the results directory
    config.results_dir.mkdir(parents=True, exist_ok=False)

    # Resolve the optimizer and cell filter classes
    optim_cls = config._optim_cls()
    filter_cls = config._cell_filter_cls()

    # Create the ASE calculator
    calculator = JMPCalculator(lightning_module)

    # Create a set for the relaxed ids
    relaxed: set[str] = set()

    for dataset_item in tqdm(dataset, desc="Relaxing with ASE"):
        material_id = dataset_item["material_id"]
        if material_id in relaxed:
            logging.info(f"Structure {material_id} has already been relaxed.")
            continue

        atoms = dataset_item["atoms"]
        try:
            atoms.calc = calculator

            if filter_cls is not None:
                optim = optim_cls(
                    filter_cls(atoms),
                    logfile="/dev/null",
                    **config.optimizer_kwargs,
                )
            else:
                optim = optim_cls(atoms, logfile="/dev/null", **config.optimizer_kwargs)

            optim.run(fmax=config.force_max, steps=config.max_steps)

            energy = atoms.get_potential_energy()
            structure = AseAtomsAdaptor.get_structure(atoms)

            # Save the results
            result = {
                "material_id": material_id,
                "structure": structure,
                "energy": energy,
            }
            if (metadata := dataset_item.get("metadata")) is not None:
                result["metadata"] = metadata
            _write_result(config, material_id, result)
            relaxed.add(material_id)
        except Exception:
            logging.exception(f"Failed to relax {material_id}")
            continue
