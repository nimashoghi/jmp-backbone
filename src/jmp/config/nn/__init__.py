from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from jmp.nn.energy_head import EnergyTargetConfig as EnergyTargetConfig
    from jmp.nn.force_head import ForceTargetConfig as ForceTargetConfig
    from jmp.nn.stress_head import StressTargetConfig as StressTargetConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "EnergyTargetConfig":
            return importlib.import_module("jmp.nn.energy_head").EnergyTargetConfig
        if name == "ForceTargetConfig":
            return importlib.import_module("jmp.nn.force_head").ForceTargetConfig
        if name == "StressTargetConfig":
            return importlib.import_module("jmp.nn.stress_head").StressTargetConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Submodule exports
from . import energy_head as energy_head
from . import force_head as force_head
from . import stress_head as stress_head
