from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_pt_checkpoint(path: str | Path, map_location: str | torch.device) -> dict[str, Any]:
    """Load a training checkpoint dict (state + config). Uses weights_only=False for PyTorch 2.6+."""
    kwargs: dict[str, Any] = {"map_location": map_location}
    try:
        kwargs["weights_only"] = False
        return torch.load(str(path), **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        return torch.load(str(path), **kwargs)
