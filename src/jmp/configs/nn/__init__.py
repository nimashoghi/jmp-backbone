from __future__ import annotations

__codegen__ = True

from jmp.nn.energy_head import EnergyTargetConfig as EnergyTargetConfig
from jmp.nn.force_head import ForceTargetConfig as ForceTargetConfig
from jmp.nn.stress_head import StressTargetConfig as StressTargetConfig

from . import energy_head as energy_head
from . import force_head as force_head
from . import stress_head as stress_head
