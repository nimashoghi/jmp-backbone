from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import ase
import nshconfig as C
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from tqdm.auto import tqdm
from typing_extensions import Self

from ..lightning_module import Config, Module
from .calculator import JMPCalculator
from .dataset_relaxer import DatasetItem, RelaxerConfig, relax_generator, write_result

log = logging.getLogger(__name__)


def load_dataset(
    subset: float | int | None,
    subset_seed: int,
    rank: int,
    world_size: int,
):
    from matbench_discovery.data import DataFiles

    # Load the DataFrames
    df_wbm = pd.read_csv(DataFiles.wbm_summary.path)
    df_wbm_initial = pd.read_json(DataFiles.wbm_initial_structures.path)

    # Split the dataset rows
    rows = np.arange(len(df_wbm_initial))
    rows_split = np.array_split(rows, world_size)

    # Get the rows for this rank
    row_idxs = rows_split[rank]
    df_wbm = df_wbm.iloc[row_idxs]
    df_wbm_initial = df_wbm_initial.iloc[row_idxs]

    # Subset the dataset
    if subset is not None:
        if isinstance(subset, float):
            subset = int(subset * len(df_wbm))

        np.random.seed(subset_seed)
        subset_idxs = np.random.choice(len(df_wbm), subset, replace=False)
        df_wbm = df_wbm.iloc[subset_idxs]
        df_wbm_initial = df_wbm_initial.iloc[subset_idxs]
        log.info(f"Subset the dataset to {len(df_wbm)} rows.")
    else:
        log.info(f"Using the entire dataset with {len(df_wbm)} rows.")

    return df_wbm, df_wbm_initial


class RelaxWBMConfig(C.Config):
    relaxer: RelaxerConfig
    """Relaxer configuration."""

    ckpt_path: Path
    """Path to the checkpoint file."""

    wbm_subset: float | int | None = None
    """
    Subset of the WBM dataset to use.
    - If float, it will be interpreted as a fraction of the dataset to use.
    - If int, it will be interpreted as the number of rows to use.
    - If None, the entire dataset will be used.
    """

    wbm_subset_seed: int = 42
    """Seed to use for the subset of the WBM dataset."""

    rank: int = 0
    """Rank of the current process."""

    world_size: int = 1
    """Number of processes to use for the relaxation."""

    device: str | None = None
    """Device to use for the relaxation. If None, the device will be automatically resolved."""

    tqdm_disable: bool = False
    """Whether to disable tqdm progress bars."""

    def _resolve_device(self):
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    # Dataset builder methods
    def subset_(self, subset: float | int | None, seed: int = 42):
        self.wbm_subset = subset
        self.wbm_subset_seed = seed
        return self

    def distributed(self, world_size: int):
        configs_out: list[Self] = []
        for rank in range(world_size):
            configs_out.append(
                self.model_copy(
                    deep=True,
                    update={"rank": rank, "world_size": world_size},
                )
            )
        return configs_out


def _dataset_generator(df: pd.DataFrame):
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatviz.enums import Key

    for idx, row in df.iterrows():
        # Get the material id
        material_id = row[Key.mat_id.value]
        # Get the initial structure
        atoms = AseAtomsAdaptor.get_atoms(Structure.from_dict(row["initial_structure"]))
        assert isinstance(atoms, ase.Atoms), f"Expected ase.Atoms, got {type(atoms)}"
        # Get everything else, except the initial structure, as a dictionary.
        metadata = row.drop("initial_structure").to_dict()
        # Add the row index to the metadata
        metadata["__row_idx__"] = idx

        # Create the dataset item
        dataset_item: DatasetItem = {
            "material_id": material_id,
            "atoms": atoms,
            "metadata": metadata,
        }

        yield dataset_item


@torch.inference_mode()
@torch.no_grad()
def predict(data: Batch, lightning_module: Module):
    from matbench_discovery.energy import get_e_form_per_atom

    # Make sure the expected properties are in the right format
    if "tags" not in data or data.tags is None or (data.tags == 0).all():
        data.tags = torch.full_like(data.atomic_numbers, 2, dtype=torch.long)

    data.atomic_numbers = data.atomic_numbers.long()
    data.natoms = data.natoms.long()
    data.tags = data.tags.long()
    data.fixed = data.fixed.bool()

    # Run the prediction, converting stress to ev/A^3 and using the total energy
    predictions = lightning_module.predict(
        data,
        convert_stress_to_ev_a3=True,
        energy_kind="total",
    )

    # Compute the formation energy per atom from the total energy
    def _composition(data: Batch):
        return dict(Counter(data.atomic_numbers.tolist()))

    predictions["energy"] = get_e_form_per_atom(
        {"composition": _composition(data), "energy": predictions["energy"]}
    )
    return predictions


def relax_wbm_run_fn(
    config: RelaxWBMConfig,
    results_dir: Path,
    update_lm_config: Callable[[Config], Config] | None = None,
):
    from pymatviz.enums import Key

    # Resolve the current device
    device = config._resolve_device()

    # Load the model from the checkpoint
    lm_config = Config.from_checkpoint(config.ckpt_path)
    if update_lm_config is not None:
        lm_config = update_lm_config(lm_config)
    lightning_module = Module.load_checkpoint(
        config.ckpt_path, lm_config, map_location=device
    )
    lightning_module = lightning_module.to(device)

    # Create the calculator
    calculator = JMPCalculator(lightning_module, predict=predict)

    # Load the dataset for this rank
    df_wbm, df_wbm_initial = load_dataset(
        config.wbm_subset,
        config.wbm_subset_seed,
        config.rank,
        config.world_size,
    )

    # Merge on matching material ids
    df = pd.merge(df_wbm, df_wbm_initial, on=Key.mat_id.value, how="inner")

    # Create the dataset generator and run the relaxation
    dataset = _dataset_generator(df)
    predicted: list[float] = []
    for relax_result in (
        pbar := tqdm(
            relax_generator(config.relaxer, calculator, dataset),
            disable=config.tqdm_disable,
            desc="Relaxing WBM",
            total=len(df),
        )
    ):
        pbar.set_postfix_str(f"Material ID: {relax_result['material_id']}")

        # Write the relaxation result
        write_result(relax_result, results_dir)
        predicted.append(
            relax_result["energy"].item()
            if not isinstance(relax_result["energy"], (float, int))
            else relax_result["energy"]
        )
