from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod

import nshconfig as C
import torch
import torch.nn as nn
from torch_geometric.data.data import BaseData
from typing_extensions import TypedDict

from ..models.gemnet.backbone import GOCBackboneOutput


class OutputHeadInput(TypedDict):
    data: BaseData
    backbone_output: GOCBackboneOutput
    predicted_props: dict[str, torch.Tensor]


class TargetConfigBase(C.Config, ABC):
    @abstractmethod
    def create_model(
        self,
        *args,
        **kwargs,
    ) -> OutputHeadBase: ...


class OutputHeadBase(nn.Module, ABC):
    @abstractmethod
    @contextlib.contextmanager
    def forward_context(self, data: BaseData):
        yield
