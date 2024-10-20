from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from jmp.lightning_datamodule import DatasetConfig as DatasetConfig
    from jmp.lightning_datamodule import (
        MPTrjAlexOMAT24DataModuleConfig as MPTrjAlexOMAT24DataModuleConfig,
    )
    from jmp.lightning_module import Config as Config
    from jmp.lightning_module import GraphComputerConfig as GraphComputerConfig
    from jmp.lightning_module import OptimizationConfig as OptimizationConfig
    from jmp.lightning_module import (
        SeparateLRMultiplierConfig as SeparateLRMultiplierConfig,
    )
    from jmp.lightning_module import TargetsConfig as TargetsConfig
    from jmp.models.gemnet.bases import BasesConfig as BasesConfig
    from jmp.models.gemnet.config import BackboneConfig as BackboneConfig
    from jmp.models.gemnet.graph import CutoffsConfig as CutoffsConfig
    from jmp.models.gemnet.graph import MaxNeighborsConfig as MaxNeighborsConfig
    from jmp.nn.energy_head import EnergyTargetConfig as EnergyTargetConfig
    from jmp.nn.force_head import ForceTargetConfig as ForceTargetConfig
    from jmp.nn.stress_head import StressTargetConfig as StressTargetConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "DatasetConfig":
            return importlib.import_module("jmp.lightning_datamodule").DatasetConfig
        if name == "MPTrjAlexOMAT24DataModuleConfig":
            return importlib.import_module(
                "jmp.lightning_datamodule"
            ).MPTrjAlexOMAT24DataModuleConfig
        if name == "TargetsConfig":
            return importlib.import_module("jmp.lightning_module").TargetsConfig
        if name == "OptimizationConfig":
            return importlib.import_module("jmp.lightning_module").OptimizationConfig
        if name == "SeparateLRMultiplierConfig":
            return importlib.import_module(
                "jmp.lightning_module"
            ).SeparateLRMultiplierConfig
        if name == "StressTargetConfig":
            return importlib.import_module("jmp.nn.stress_head").StressTargetConfig
        if name == "ForceTargetConfig":
            return importlib.import_module("jmp.nn.force_head").ForceTargetConfig
        if name == "Config":
            return importlib.import_module("jmp.lightning_module").Config
        if name == "EnergyTargetConfig":
            return importlib.import_module("jmp.nn.energy_head").EnergyTargetConfig
        if name == "GraphComputerConfig":
            return importlib.import_module("jmp.lightning_module").GraphComputerConfig
        if name == "BasesConfig":
            return importlib.import_module("jmp.models.gemnet.bases").BasesConfig
        if name == "BackboneConfig":
            return importlib.import_module("jmp.models.gemnet.config").BackboneConfig
        if name == "MaxNeighborsConfig":
            return importlib.import_module("jmp.models.gemnet.graph").MaxNeighborsConfig
        if name == "CutoffsConfig":
            return importlib.import_module("jmp.models.gemnet.graph").CutoffsConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Submodule exports
from . import lightning_datamodule as lightning_datamodule
from . import lightning_module as lightning_module
from . import models as models
from . import nn as nn
