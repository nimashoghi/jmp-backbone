from __future__ import annotations

import logging
from pathlib import Path

import ase
import nshconfig as C
import numpy as np
import pandas as pd
import torch
from matbench_discovery.data import DataFiles
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatviz.enums import Key
from tqdm.auto import tqdm
from typing_extensions import Self

from ..lightning_module import Module
from .dataset_relaxer import DatasetItem, RelaxerConfig, relax

log = logging.getLogger(__name__)


def load_dataset(
    subset: float | int | None,
    subset_seed: int,
    rank: int,
    world_size: int,
):
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


def _dataset_generator(df: pd.DataFrame, *, config: RelaxWBMConfig):
    for idx, row in (
        pbar := tqdm(
            df.iterrows(),
            disable=config.tqdm_disable,
            desc="Relaxing WBM",
            total=len(df),
        )
    ):
        # Get the material id
        material_id = row[Key.mat_id.value]
        pbar.set_postfix_str(f"Material ID: {material_id}")
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


def relax_wbm_run_fn(config: RelaxWBMConfig):
    # Resolve the current device
    device = config._resolve_device()

    # Load the model from the checkpoint
    lightning_module = Module.load_checkpoint(config.ckpt_path, map_location=device)
    lightning_module = lightning_module.to(device)

    # Load the dataset for this rank
    df_wbm, df_wbm_initial = load_dataset(
        config.wbm_subset,
        config.wbm_subset_seed,
        config.rank,
        config.world_size,
    )

    # Merge on matching material ids
    df = pd.merge(df_wbm, df_wbm_initial, on=Key.mat_id.value, how="inner")

    # Create the dataset generator
    dataset = _dataset_generator(df, config=config)

    # Run the relaxation
    relax(config.relaxer, lightning_module, dataset)
