from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from jmp.referencing import IdentityReferencerConfig as IdentityReferencerConfig
    from jmp.referencing import PerAtomReferencerConfig as PerAtomReferencerConfig
    from jmp.referencing import ReferencerConfig as ReferencerConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "IdentityReferencerConfig":
            return importlib.import_module("jmp.referencing").IdentityReferencerConfig
        if name == "PerAtomReferencerConfig":
            return importlib.import_module("jmp.referencing").PerAtomReferencerConfig
        if name == "ReferencerConfig":
            return importlib.import_module("jmp.referencing").ReferencerConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
