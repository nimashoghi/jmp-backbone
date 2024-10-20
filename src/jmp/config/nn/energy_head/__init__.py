__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from jmp.nn.energy_head import EnergyTargetConfig as EnergyTargetConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "EnergyTargetConfig":
            return importlib.import_module("jmp.nn.energy_head").EnergyTargetConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
