from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from jmp.nn.stress_head import StressTargetConfig as StressTargetConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "StressTargetConfig":
            return importlib.import_module("jmp.nn.stress_head").StressTargetConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
