from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Annotated, Literal, TypeAlias

import nshconfig as C
import nshutils.typecheck as tc
import torch
import torch.nn as nn
from typing_extensions import override

log = logging.getLogger(__name__)


class ReferencerBaseModule(nn.Module, ABC):
    @abstractmethod
    def reference(
        self,
        total_energy: tc.Float[torch.Tensor, "bsz"],
        atomic_numbers: tc.Int[torch.Tensor, "natoms"],
    ) -> tc.Float[torch.Tensor, "bsz"]:
        pass

    @abstractmethod
    def dereference(
        self,
        referenced_energy: tc.Float[torch.Tensor, "bsz"],
        atomic_numbers: tc.Int[torch.Tensor, "bsz natoms"],
    ) -> tc.Float[torch.Tensor, "bsz"]:
        pass


class IdentityReferencerConfig(C.Config):
    name: Literal["identity_referencer"] = "identity_referencer"

    def create_referencer(self):
        return IdentityReferencer(self)


class IdentityReferencer(ReferencerBaseModule):
    @override
    def __init__(self, hparams: IdentityReferencerConfig):
        super().__init__()

        self.hparams = hparams
        del hparams

    def reference(
        self,
        total_energy: tc.Float[torch.Tensor, "bsz"],
        atomic_numbers: tc.Int[torch.Tensor, "natoms"],
    ) -> tc.Float[torch.Tensor, "bsz"]:
        return total_energy

    def dereference(
        self,
        referenced_energy: tc.Float[torch.Tensor, "bsz"],
        atomic_numbers: tc.Int[torch.Tensor, "bsz natoms"],
    ) -> tc.Float[torch.Tensor, "bsz"]:
        return referenced_energy


class PerAtomReferencerConfig(C.Config):
    name: Literal["per_atom_referencer"] = "per_atom_referencer"

    references: list[float]
    """Per-atom reference energies."""

    def create_referencer(self):
        return PerAtomReferencer(self)

    @classmethod
    def linear_reference(cls, reference: Literal["mptrj-salex"]):
        match reference:
            case "mptrj-salex":
                from .linref import PRECOMPUTED_MPTRJ_ALEX

                return cls(references=PRECOMPUTED_MPTRJ_ALEX)
            case _:
                assert_never(reference)


class PerAtomReferencer(ReferencerBaseModule):
    per_atom_references: tc.Float[torch.Tensor, "max_num_atoms"]

    @override
    def __init__(self, hparams: PerAtomReferencerConfig):
        super().__init__()

        self.hparams = hparams
        del hparams

        per_atom_references = torch.tensor(self.hparams.references, dtype=torch.float)
        self.register_buffer(
            "per_atom_references", per_atom_references, persistent=False
        )
        log.info(f"Per-atom references: {self.per_atom_references}")

    def reference(
        self,
        total_energy: tc.Float[torch.Tensor, "bsz"],
        atomic_numbers: tc.Int[torch.Tensor, "natoms"],
    ) -> tc.Float[torch.Tensor, "bsz"]:
        return total_energy - self.per_atom_references[atomic_numbers].sum()

    def dereference(
        self,
        referenced_energy: tc.Float[torch.Tensor, "bsz"],
        atomic_numbers: tc.Int[torch.Tensor, "bsz natoms"],
    ) -> tc.Float[torch.Tensor, "bsz"]:
        return referenced_energy + self.per_atom_references[atomic_numbers].sum()


ReferencerConfig: TypeAlias = Annotated[
    IdentityReferencerConfig | PerAtomReferencerConfig,
    C.Field(discriminator="name"),
]
