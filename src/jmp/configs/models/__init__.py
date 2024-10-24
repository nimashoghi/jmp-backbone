from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from jmp.models.gemnet.bases import BasesConfig as BasesConfig
    from jmp.models.gemnet.config import BackboneConfig as BackboneConfig
    from jmp.models.gemnet.graph import CutoffsConfig as CutoffsConfig
    from jmp.models.gemnet.graph import GraphComputerConfig as GraphComputerConfig
    from jmp.models.gemnet.graph import MaxNeighborsConfig as MaxNeighborsConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "BackboneConfig":
            return importlib.import_module("jmp.models.gemnet.config").BackboneConfig
        if name == "BasesConfig":
            return importlib.import_module("jmp.models.gemnet.bases").BasesConfig
        if name == "CutoffsConfig":
            return importlib.import_module("jmp.models.gemnet.graph").CutoffsConfig
        if name == "GraphComputerConfig":
            return importlib.import_module(
                "jmp.models.gemnet.graph"
            ).GraphComputerConfig
        if name == "MaxNeighborsConfig":
            return importlib.import_module("jmp.models.gemnet.graph").MaxNeighborsConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Submodule exports
from . import gemnet as gemnet
