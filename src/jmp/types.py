from __future__ import annotations

from typing import TypedDict

import nshutils.typecheck as tc
import torch


class Predictions(TypedDict):
    """
    Prediction is a TypedDict that defines the structure of the prediction output.

    Attributes:
        energy (tc.Float[torch.Tensor, "b"]): A tensor representing the energy with batch dimension 'b'.
        forces (tc.Float[torch.Tensor, "n 3"]): A tensor representing the forces with dimensions 'n' by 3.
        stress (tc.Float[torch.Tensor, "b 3 3"]): A tensor representing the stress with dimensions 'b' by 3 by 3.
    """

    energy: tc.Float[torch.Tensor, "b"]
    forces: tc.Float[torch.Tensor, "n 3"]
    stress: tc.Float[torch.Tensor, "b 3 3"]
