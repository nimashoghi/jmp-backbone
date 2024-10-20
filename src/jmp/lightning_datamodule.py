from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Literal

import datasets
import nshconfig as C
import nshtrainer as nt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from typing_extensions import assert_never, override

log = logging.getLogger(__name__)


class DatasetConfig(C.Config):
    enabled: bool = True
    """Whether to include the dataset."""

    hf_name: str
    """Name of the Hugging Face dataset."""

    local_path: str | Path | None = None
    """Path to the local dataset."""

    def __bool__(self):
        return self.enabled


class MPTrjAlexOMAT24DataModuleConfig(C.Config):
    name: Literal["mptrj_alex_omat24"] = "mptrj_alex_omat24"

    batch_size: int
    """Batch size."""

    num_workers: int
    """Number of workers for the data loader."""

    mptrj: DatasetConfig = DatasetConfig(hf_name="nimashoghi/mptrj")
    """Whether to include the MPTrj dataset."""

    salex: DatasetConfig = DatasetConfig(hf_name="nimashoghi/salex")
    """Whether to include the SAlEx dataset."""

    omat24: DatasetConfig = DatasetConfig(hf_name="nimashoghi/omat24")
    """Whether to include the OMAT24 dataset."""

    pin_memory: bool = True
    """Whether to pin memory in the data loader."""

    reference: list[float] | None = None
    """Atomic energy reference values."""

    filter_small_systems: bool = True
    """Whether to filter out small systems (less than 4 atoms)."""

    subsample_val: int | None = 10_000
    """If not `None`, subsample each validation dataset to this number of samples (if the dataset is larger)."""

    def with_linear_reference_(self, reference: Literal["mptrj-salex"]):
        match reference:
            case "mptrj-salex":
                from .linref import PRECOMPUTED_MPTRJ_ALEX

                self.reference = PRECOMPUTED_MPTRJ_ALEX
            case _:
                assert_never(reference)


def _load_dataset(
    data_config: MPTrjAlexOMAT24DataModuleConfig,
    config: DatasetConfig | None,
    split: str,
    subsample: int | None = None,
):
    if not config:
        return None

    if config.local_path is not None:
        dataset = datasets.load_from_disk(str(config.local_path))
        assert isinstance(
            dataset, datasets.DatasetDict
        ), f"Expected a `datasets.DatasetDict` but got {type(dataset)}"
        dataset = dataset[split]
    else:
        dataset = datasets.load_dataset(config.hf_name, split=split)
    assert isinstance(
        dataset, datasets.Dataset
    ), f"Expected a `datasets.Dataset` but got {type(dataset)}"

    # Rename the "num_atoms" column to "natoms"
    if "num_atoms" in dataset.column_names:
        dataset = dataset.rename_column("num_atoms", "natoms")

    # Filter small systems
    if data_config.filter_small_systems:
        dataset = dataset.filter(lambda natoms: natoms >= 4, input_columns=["natoms"])

    if subsample is not None:
        dataset = dataset.shuffle(seed=42).select(range(subsample))

    dataset.set_format("torch")
    return dataset


class MPTrjAlexOMAT24Dataset(Dataset, nt.data.balanced_batch_sampler.DatasetWithSizes):
    @override
    def __init__(
        self,
        data_config: MPTrjAlexOMAT24DataModuleConfig,
        split: Literal["train", "val"],
    ):
        super().__init__()

        self.data_config = data_config
        del data_config

        subsample = None
        if split == "val":
            subsample = self.data_config.subsample_val

        self.mptrj = _load_dataset(
            self.data_config,
            self.data_config.mptrj,
            split=split,
            subsample=subsample,
        )
        self.salex = _load_dataset(
            self.data_config,
            self.data_config.salex,
            split=split,
            subsample=subsample,
        )
        self.omat24 = _load_dataset(
            self.data_config,
            self.data_config.omat24,
            split=split,
            subsample=subsample,
        )

        self.reference = None
        if self.data_config.reference is not None:
            self.reference = torch.tensor(self.data_config.reference, dtype=torch.float)
            log.critical(f"Using reference: {self.reference}")
        else:
            log.critical("No reference provided. Using raw energies.")

    @staticmethod
    def ensure_downloaded(data_config: MPTrjAlexOMAT24DataModuleConfig):
        _ = _load_dataset(data_config, data_config.mptrj, split="train")
        _ = _load_dataset(data_config, data_config.mptrj, split="val")

        _ = _load_dataset(data_config, data_config.salex, split="train")
        _ = _load_dataset(data_config, data_config.salex, split="val")

        _ = _load_dataset(data_config, data_config.omat24, split="train")
        _ = _load_dataset(data_config, data_config.omat24, split="val")

    @functools.cached_property
    def natoms(self):
        mptrj_natoms = self.mptrj["natoms"] if self.mptrj is not None else []
        salex_natoms = self.salex["natoms"] if self.salex is not None else []
        omat24_natoms = self.omat24["natoms"] if self.omat24 is not None else []

        return np.concatenate([mptrj_natoms, salex_natoms, omat24_natoms], axis=0)

    def data_sizes(self, indices: list[int]) -> np.ndarray:
        return self.natoms[indices]

    def __len__(self):
        total_len = 0
        if self.mptrj is not None:
            total_len += len(self.mptrj)
        if self.salex is not None:
            total_len += len(self.salex)
        if self.omat24 is not None:
            total_len += len(self.omat24)
        return total_len

    def _get_data(self, index: int):
        if self.mptrj is not None:
            if index < len(self.mptrj):
                data_dict: dict[str, torch.Tensor] = self.mptrj[index]
                return Data.from_dict(
                    {
                        "pos": data_dict["positions"],
                        "atomic_numbers": data_dict["numbers"].long(),
                        "natoms": data_dict["natoms"].long(),
                        "cell": data_dict["cell"].view(1, 3, 3),
                        "y": data_dict["corrected_total_energy"],
                        "force": data_dict["forces"],
                        "stress": data_dict["stress"].view(1, 3, 3),
                    }
                )
            index -= len(self.mptrj)

        if self.salex is not None:
            if index < len(self.salex):
                data_dict: dict[str, torch.Tensor] = self.salex[index]
                return Data.from_dict(
                    {
                        "pos": data_dict["pos"],
                        "atomic_numbers": data_dict["atomic_numbers"].long(),
                        "natoms": data_dict["natoms"].long(),
                        "tags": data_dict["tags"].long(),
                        "fixed": data_dict["fixed"].to(torch.bool),
                        "cell": data_dict["cell"].view(1, 3, 3),
                        "y": data_dict["energy"],
                        "force": data_dict["forces"],
                        "stress": data_dict["stress"].view(1, 3, 3),
                    }
                )
            index -= len(self.salex)

        if self.omat24 is not None:
            if index < len(self.omat24):
                data_dict: dict[str, torch.Tensor] = self.omat24[index]
                return Data.from_dict(
                    {
                        "pos": data_dict["pos"],
                        "atomic_numbers": data_dict["atomic_numbers"].long(),
                        "natoms": data_dict["natoms"].long(),
                        "tags": data_dict["tags"].long(),
                        "fixed": data_dict["fixed"].to(torch.bool),
                        "cell": data_dict["cell"].view(1, 3, 3),
                        "y": data_dict["energy"],
                        "force": data_dict["forces"],
                        "stress": data_dict["stress"].view(1, 3, 3),
                    }
                )
            index -= len(self.omat24)

        raise IndexError(f"Index {index} out of bounds")

    def __getitem__(self, index: int):
        data = self._get_data(index)

        # Basic stuff that has to be set
        if "tags" not in data:
            data.tags = torch.full_like(data.atomic_numbers, 2, dtype=torch.long)
        if "fixed" not in data:
            data.fixed = torch.zeros_like(data.atomic_numbers, dtype=torch.bool)

        # Apply reference
        if self.reference is not None:
            data.y = data.y - self.reference[data.atomic_numbers].sum()

        return data


class MPTrjAlexOMAT24DataModule(nt.LightningDataModuleBase):
    @override
    def __init__(self, config: MPTrjAlexOMAT24DataModuleConfig):
        super().__init__()

        self.config = config
        del config

    @override
    def prepare_data(self):
        super().prepare_data()

        # Make sure all datasets are downloaded
        MPTrjAlexOMAT24Dataset.ensure_downloaded(self.config)

    def _dataset(self, split: Literal["train", "val"]):
        return MPTrjAlexOMAT24Dataset(self.config, split=split)

    @staticmethod
    def _collate_fn(data_list: list[Data]):
        return Batch.from_data_list(data_list)

    @override
    def train_dataloader(self):
        dataset = MPTrjAlexOMAT24Dataset(self.config, split="train")
        dl = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        return dl

    @override
    def val_dataloader(self):
        dataset = MPTrjAlexOMAT24Dataset(self.config, split="val")
        dl = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=False,
            collate_fn=self._collate_fn,
        )
        return dl
