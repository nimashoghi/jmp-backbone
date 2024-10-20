from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from jmp.lightning_module import Config as Config
    from jmp.lightning_module import EnergyTargetConfig as EnergyTargetConfig
    from jmp.lightning_module import ForceTargetConfig as ForceTargetConfig
    from jmp.lightning_module import GraphComputerConfig as GraphComputerConfig
    from jmp.lightning_module import OptimizationConfig as OptimizationConfig
    from jmp.lightning_module import (
        SeparateLRMultiplierConfig as SeparateLRMultiplierConfig,
    )
    from jmp.lightning_module import StressTargetConfig as StressTargetConfig
    from jmp.lightning_module import TargetsConfig as TargetsConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "OptimizationConfig":
            return importlib.import_module("jmp.lightning_module").OptimizationConfig
        if name == "SeparateLRMultiplierConfig":
            return importlib.import_module(
                "jmp.lightning_module"
            ).SeparateLRMultiplierConfig
        if name == "ForceTargetConfig":
            return importlib.import_module("jmp.lightning_module").ForceTargetConfig
        if name == "TargetsConfig":
            return importlib.import_module("jmp.lightning_module").TargetsConfig
        if name == "EnergyTargetConfig":
            return importlib.import_module("jmp.lightning_module").EnergyTargetConfig
        if name == "Config":
            return importlib.import_module("jmp.lightning_module").Config
        if name == "GraphComputerConfig":
            return importlib.import_module("jmp.lightning_module").GraphComputerConfig
        if name == "StressTargetConfig":
            return importlib.import_module("jmp.lightning_module").StressTargetConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
