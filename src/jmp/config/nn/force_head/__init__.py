__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from jmp.nn.force_head import ForceTargetConfig as ForceTargetConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "ForceTargetConfig":
            return importlib.import_module("jmp.nn.force_head").ForceTargetConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
