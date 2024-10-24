from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from jmp.relaxation.dataset_relaxer import RelaxerConfig as RelaxerConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "RelaxerConfig":
            return importlib.import_module(
                "jmp.relaxation.dataset_relaxer"
            ).RelaxerConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
