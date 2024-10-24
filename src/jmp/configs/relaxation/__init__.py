from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from jmp.relaxation.wbm import Config as Config
    from jmp.relaxation.wbm import RelaxerConfig as RelaxerConfig
    from jmp.relaxation.wbm import RelaxWBMConfig as RelaxWBMConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "Config":
            return importlib.import_module("jmp.relaxation.wbm").Config
        if name == "RelaxWBMConfig":
            return importlib.import_module("jmp.relaxation.wbm").RelaxWBMConfig
        if name == "RelaxerConfig":
            return importlib.import_module("jmp.relaxation.wbm").RelaxerConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Submodule exports
from . import dataset_relaxer as dataset_relaxer
from . import wbm as wbm
