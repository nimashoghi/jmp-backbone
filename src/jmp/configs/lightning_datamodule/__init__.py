from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from jmp.lightning_datamodule import DatasetConfig as DatasetConfig
    from jmp.lightning_datamodule import (
        MPTrjAlexOMAT24DataModuleConfig as MPTrjAlexOMAT24DataModuleConfig,
    )
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
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
