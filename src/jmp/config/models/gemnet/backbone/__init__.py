__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from jmp.models.gemnet.backbone import BackboneConfig as BackboneConfig
    from jmp.models.gemnet.backbone import BasesConfig as BasesConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "BasesConfig":
            return importlib.import_module("jmp.models.gemnet.backbone").BasesConfig
        if name == "BackboneConfig":
            return importlib.import_module("jmp.models.gemnet.backbone").BackboneConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
