from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from typing_extensions import assert_never


def l2mae_loss(
    y_pred: torch.Tensor,  # N 3
    y_true: torch.Tensor,  # N 3
    reduction: Literal["mean", "sum"] = "mean",
) -> torch.Tensor:
    l2_distances = F.pairwise_distance(y_pred, y_true, p=2)  # N
    match reduction:
        case "mean":
            return l2_distances.mean()
        case "sum":
            return l2_distances.sum()
        case _:
            assert_never(reduction)
