# %%
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import nshconfig_extra as CE
import nshutils as nu
import rich
import torch

import jmp.configs as jc
from jmp.lightning_module import Module

nu.pretty()

ckpt_path = CE.CachedPath(
    uri="hf://nimashoghi/mptrj-alex-omat24-jmp-s-1our3wgd/checkpoints/last/epoch1-step172332.ckpt"
)


def update_hparams(config):
    config.pretrained_ckpt = CE.CachedPath(uri="/mnt/shared/checkpoints/jmp-s.pt")
    config.energy_referencer = jc.PerAtomReferencerConfig.linear_reference(
        "mptrj-salex"
    )
    return config


def fresh_model_from_ckpt(path: Path):
    ckpt = torch.load(path, map_location="cpu")

    hparams_dict = ckpt[Module.CHECKPOINT_HYPER_PARAMS_KEY]

    hparams = Module.hparams_cls().model_validate(hparams_dict)
    if update_hparams is not None:
        hparams = update_hparams(hparams)

    model = Module(hparams)
    return model


module = fresh_model_from_ckpt(ckpt_path.resolve()).cuda().eval()
module


# %%
import datasets
import torch
from pymatgen.core import Structure
from torch_geometric.data import Batch, Data


def mptrj_dataset():
    dataset = datasets.load_dataset("nimashoghi/mptrj", split="train", streaming=True)
    dataset = dataset.filter(
        lambda extxyz_id: extxyz_id == 0, input_columns=["extxyz_id"]
    )
    dataset_iter = iter(dataset)

    def new_batch():
        data_dict = next(dataset_iter)
        data = Data.from_dict(
            {
                "pos": torch.tensor(data_dict["positions"], dtype=torch.float),
                "atomic_numbers": (
                    atomic_numbers := torch.tensor(
                        data_dict["numbers"], dtype=torch.long
                    )
                ),
                "tags": torch.full_like(atomic_numbers, 2, dtype=torch.long),
                "fixed": torch.zeros_like(atomic_numbers, dtype=torch.bool),
                "natoms": torch.tensor(data_dict["num_atoms"], dtype=torch.long),
                "cell": torch.tensor(data_dict["cell"], dtype=torch.float).view(
                    1, 3, 3
                ),
                "energy": torch.tensor(
                    data_dict["corrected_total_energy"], dtype=torch.float
                ),
                "force": torch.tensor(data_dict["forces"], dtype=torch.float),
                "stress": torch.tensor(data_dict["stress"], dtype=torch.float).view(
                    1, 3, 3
                ),
                "eform": torch.tensor(
                    data_dict["ef_per_atom_relaxed"], dtype=torch.float
                ),
            }
        )
        batch = Batch.from_data_list([data])
        return batch

    return new_batch


def wbm_dataset():
    dataset = datasets.load_dataset("nimashoghi/wbm", split="train", streaming=True)
    dataset_iter = iter(dataset)

    def new_batch():
        data_dict = next(dataset_iter)
        structure = Structure.from_dict(data_dict["initial_structure"])
        data = Data.from_dict(
            {
                "pos": torch.tensor(structure.cart_coords, dtype=torch.float),
                "atomic_numbers": (
                    atomic_numbers := torch.tensor(
                        structure.atomic_numbers, dtype=torch.long
                    )
                ),
                "tags": torch.full_like(atomic_numbers, 2, dtype=torch.long),
                "fixed": torch.zeros_like(atomic_numbers, dtype=torch.bool),
                "natoms": torch.tensor(atomic_numbers.numel(), dtype=torch.long),
                "cell": torch.tensor(structure.lattice.matrix, dtype=torch.float).view(
                    1, 3, 3
                ),
                "energy": torch.tensor(
                    data_dict["uncorrected_energy"], dtype=torch.float
                ),
                "eform": torch.tensor(
                    data_dict["e_form_per_atom_wbm"], dtype=torch.float
                ),
            }
        )
        batch = Batch.from_data_list([data])
        return batch

    return new_batch


def oc20_dataset():
    dataset = datasets.load_dataset("nimashoghi/oc20-s2ef", split="2M", streaming=True)
    dataset_iter = iter(dataset)

    def new_batch():
        data_dict = next(dataset_iter)
        data = Data.from_dict(
            {
                "pos": torch.tensor(data_dict["pos"], dtype=torch.float),
                "atomic_numbers": (
                    atomic_numbers := torch.tensor(
                        data_dict["atomic_numbers"], dtype=torch.long
                    )
                ),
                "tags": torch.full_like(atomic_numbers, 2, dtype=torch.long),
                "fixed": torch.zeros_like(atomic_numbers, dtype=torch.bool),
                "natoms": torch.tensor(data_dict["num_atoms"], dtype=torch.long),
                "cell": torch.tensor(data_dict["cell"], dtype=torch.float).view(
                    1, 3, 3
                ),
                "energy": torch.tensor(data_dict["energy"], dtype=torch.float),
                "force": torch.tensor(data_dict["forces"], dtype=torch.float),
            }
        )
        batch = Batch.from_data_list([data])
        return batch

    return new_batch


def oc22_dataset():
    dataset = datasets.load_dataset("nimashoghi/oc22", split="train", streaming=True)
    dataset_iter = iter(dataset)

    def new_batch():
        data_dict = next(dataset_iter)
        data = Data.from_dict(
            {
                "pos": torch.tensor(data_dict["pos"], dtype=torch.float),
                "atomic_numbers": (
                    atomic_numbers := torch.tensor(
                        data_dict["atomic_numbers"], dtype=torch.long
                    )
                ),
                "tags": torch.full_like(atomic_numbers, 2, dtype=torch.long),
                "fixed": torch.zeros_like(atomic_numbers, dtype=torch.bool),
                "natoms": torch.tensor(data_dict["natoms"], dtype=torch.long),
                "cell": torch.tensor(data_dict["cell"], dtype=torch.float).view(
                    1, 3, 3
                ),
                "energy": torch.tensor(data_dict["y"], dtype=torch.float),
                "force": torch.tensor(data_dict["force"], dtype=torch.float),
            }
        )
        batch = Batch.from_data_list([data])
        return batch

    return new_batch


def ani1x_dataset():
    dataset = datasets.load_dataset("nimashoghi/ani1x", split="train", streaming=True)
    dataset_iter = iter(dataset)

    def new_batch():
        data_dict = next(dataset_iter)
        data = Data.from_dict(
            {
                "pos": torch.tensor(data_dict["pos"], dtype=torch.float),
                "atomic_numbers": (
                    atomic_numbers := torch.tensor(
                        data_dict["atomic_numbers"], dtype=torch.long
                    )
                ),
                "tags": torch.full_like(atomic_numbers, 2, dtype=torch.long),
                "fixed": torch.zeros_like(atomic_numbers, dtype=torch.bool),
                "natoms": torch.tensor(data_dict["natoms"], dtype=torch.long),
                "cell": (torch.eye(3).float() * 1000).view(1, 3, 3),
                "energy": torch.tensor(data_dict["y"], dtype=torch.float),
                "force": torch.tensor(data_dict["force"], dtype=torch.float),
            }
        )
        batch = Batch.from_data_list([data])
        return batch

    return new_batch


def transition1x_dataset():
    dataset = datasets.load_dataset(
        "nimashoghi/transition1x", split="train", streaming=True
    )
    dataset_iter = iter(dataset)

    def new_batch():
        data_dict = next(dataset_iter)
        data = Data.from_dict(
            {
                "pos": torch.tensor(data_dict["pos"], dtype=torch.float),
                "atomic_numbers": (
                    atomic_numbers := torch.tensor(
                        data_dict["atomic_numbers"], dtype=torch.long
                    )
                ),
                "tags": torch.full_like(atomic_numbers, 2, dtype=torch.long),
                "fixed": torch.zeros_like(atomic_numbers, dtype=torch.bool),
                "natoms": torch.tensor(data_dict["natoms"], dtype=torch.long),
                "cell": (torch.eye(3).float() * 1000).view(1, 3, 3),
                "energy": torch.tensor(data_dict["y"], dtype=torch.float),
                "force": torch.tensor(data_dict["force"], dtype=torch.float),
            }
        )
        batch = Batch.from_data_list([data])
        return batch

    return new_batch


# %%
from collections import defaultdict
from collections.abc import Callable

from lightning.fabric.utilities.apply_func import move_data_to_device
from tqdm import trange


@torch.no_grad()
@torch.inference_mode()
def create_embeddings(
    dataset_fns: dict[str, Callable[[], Callable[[], Batch]]],
    num_embeddings: int,
):
    embeddings = defaultdict[str, list[Any]](lambda: [])
    energies = defaultdict[str, list[Any]](lambda: [])

    for dataset_name, dataset_fn in dataset_fns.items():
        new_batch = dataset_fn()
        for i in trange(num_embeddings, desc=dataset_name):
            while True:
                try:
                    batch = new_batch()
                    embedding = module.embeddings(batch)
                    eform = batch.eform if hasattr(batch, "eform") else batch.energy

                    embedding = move_data_to_device(embedding, "cpu")
                    eform = move_data_to_device(eform, "cpu")

                    embeddings[dataset_name].append(embedding)
                    energies[dataset_name].append(eform)
                    break
                except Exception as e:
                    logging.error("Error", exc_info=e)
                    continue

    return embeddings, energies


dataset_fns = {
    "mptrj": mptrj_dataset,
    "wbm": wbm_dataset,
    "oc20": oc20_dataset,
    "oc22": oc22_dataset,
    "ani1x": ani1x_dataset,
    "transition1x": transition1x_dataset,
}

embeddings, energies = create_embeddings(
    dataset_fns,
    num_embeddings=10_000,
)

# %%
import numpy as np

avg_embeddings = {
    dataset_name: np.stack(
        [
            model_output["energy"].mean(dim=0).float().numpy()
            for model_output in model_outputs
        ],
        axis=0,
    )
    for dataset_name, model_outputs in embeddings.items()
}
import pickle

with open("/mnt/datasets/jmpwbm_all_avg_embeddings.pkl", "wb") as f:
    pickle.dump(avg_embeddings, f)

rich.print(avg_embeddings)

# %%
# Get t-SNE embeddings
from sklearn.manifold import TSNE

all_embeddings = np.concatenate(list(avg_embeddings.values()), axis=0)
all_labels = np.concatenate(
    [
        np.full((embedding.shape[0],), i)
        for i, embedding in enumerate(avg_embeddings.values())
    ],
    axis=0,
)
tsne = TSNE(n_components=2, random_state=0)
tsne_embeddings = tsne.fit_transform(all_embeddings)
print(tsne_embeddings.shape)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white")

# Plot t-SNE embeddings
fig, ax = plt.subplots(figsize=(10, 10))

for i, dataset_name in enumerate(avg_embeddings.keys()):
    mask = all_labels == i
    sns.scatterplot(
        x=tsne_embeddings[mask, 0],
        y=tsne_embeddings[mask, 1],
        label=dataset_name,
        ax=ax,
    )

ax.legend()
plt.show()
